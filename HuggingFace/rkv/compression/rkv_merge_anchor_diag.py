import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cal_similarity, compute_attention_scores
from .merge_core import constant_gap_merge


class RKVMergeAnchorDiag:
    """R-KV Eviction Scorer + Covariance Merging. Use Diagonal covariance instead of full.
    """

    uses_merge_metadata = True
    uses_query_window = True

    def __init__(
        self,
        budget=128,
        window_size=8,
        first_tokens=4,
        kernel_size=7,
        mix_lambda=0.07,
        retain_ratio=0.1,
        retain_direction="last",
        merge_threshold=1.0,
        **kwargs,
    ):
        assert budget - window_size - first_tokens > 0, (
            "budget must leave at least one slot that is neither sink nor recent, since only those "
            "are eligible merge targets"
        )
        self.budget = budget
        self.window_size = window_size
        self.n_sink = first_tokens
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction
        self.merge_threshold = merge_threshold

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        beta_states,
        n_states,
        mu,
        cov,
    ):
        bsz, num_kv_heads, kv_len, head_dim = key_states.shape  # [B, Hkv, T, D]

        if kv_len < self.budget:
            return key_states, value_states, beta_states, n_states

        n_remove = kv_len - self.budget
        scaling = head_dim**-0.5
        window = self.window_size

        # ================= R-KV scorer, with the merge bias folded in =================
        attn_weights = compute_attention_scores(query_states, key_states)  # [B, Hkv, Tq, Tk]

        # CHANGE 1: beta enters before the softmax. compute_attention_scores already returns
        # q.k/sqrt(d), so beta adds directly to give the true pre-softmax logit. The softmax is
        # normalised over the non-window keys only, which is R-KV's own convention and is kept.
        biased_logits = attn_weights[:, :, -window:, :-window] + beta_states[
            :, :, None, :-window
        ].to(attn_weights.dtype)

        attn_weights_sum = (
            nn.functional.softmax(biased_logits, dim=-1, dtype=torch.float32)
            .mean(dim=-2)
            .to(query_states.dtype)
        )

        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )

        similarity_cos = cal_similarity(
            key_states,
            retain_ratio=self.retain_ratio,
            retain_direction=self.retain_direction,
        )[:, :, :-window]

        final_score = attn_cache * self.mix_lambda - similarity_cos * (1 - self.mix_lambda)
        # ==============================================================================

        # R-KV's kept set and output layout, reproduced exactly (top-k order, then the window).
        keep_pos = final_score.topk(self.budget - window, dim=-1).indices
        is_keep = torch.zeros(
            bsz, num_kv_heads, kv_len - window, dtype=torch.bool, device=key_states.device
        )
        is_keep.scatter_(-1, keep_pos, True)
        # complement via the mask, not a second topk: those agree only absent ties in final_score
        src_pos = (
            (~is_keep).to(torch.int64).argsort(dim=-1, stable=True, descending=True)
        )[..., :n_remove]

        window_pos = torch.arange(kv_len - window, kv_len, device=key_states.device)
        kept_idx = torch.cat(
            [keep_pos, window_pos.view(1, 1, -1).expand(bsz, num_kv_heads, -1)], dim=-1
        )

        flat = bsz * num_kv_heads
        key_flat = key_states.reshape(flat, kv_len, head_dim).float()
        val_flat = value_states.reshape(flat, kv_len, head_dim).float()
        beta_flat = beta_states.reshape(flat, kv_len).float()
        n_flat = n_states.reshape(flat, kv_len).float()
        mu_flat = mu.reshape(flat, head_dim).float()
        cov_flat = cov.reshape(flat, head_dim, head_dim).float()

        ############################ Diagonalization ablation ####################################
        # most of the information is in the diagonal entries
        I = torch.eye(cov_flat.size(-1), device=cov_flat.device)
        cov_flat = cov_flat * I
        ############################ Diagonalization ablation ####################################

        quad = torch.einsum("ftd,fde,fte->ft", key_flat, cov_flat, key_flat)
        mu_dot_k = torch.einsum("ftd,fd->ft", key_flat, mu_flat)

        kept_idx_flat = kept_idx.reshape(flat, self.budget)
        src_idx_flat = src_pos.reshape(flat, n_remove)
        expand_kept = kept_idx_flat.unsqueeze(-1).expand(-1, -1, head_dim)
        expand_src = src_idx_flat.unsqueeze(-1).expand(-1, -1, head_dim)

        # CHANGE 2: a survivor is an eligible target only if it is neither a sink slot (original
        # position < n_sink) nor part of the recent window (the trailing `window` entries of the
        # kept layout, by construction of kept_idx above).
        target_allowed = kept_idx_flat >= self.n_sink
        target_allowed[:, self.budget - window :] = False

        new_key, new_value, new_beta, new_n = constant_gap_merge(
            kept_keys=torch.gather(key_flat, 1, expand_kept),
            kept_values=torch.gather(val_flat, 1, expand_kept),
            kept_beta=torch.gather(beta_flat, 1, kept_idx_flat),
            kept_n=torch.gather(n_flat, 1, kept_idx_flat),
            kept_quad=torch.gather(quad, 1, kept_idx_flat),
            kept_mu_dot=torch.gather(mu_dot_k, 1, kept_idx_flat),
            src_keys=torch.gather(key_flat, 1, expand_src),
            src_values=torch.gather(val_flat, 1, expand_src),
            src_beta=torch.gather(beta_flat, 1, src_idx_flat),
            src_n=torch.gather(n_flat, 1, src_idx_flat),
            src_quad=torch.gather(quad, 1, src_idx_flat),
            src_mu_dot=torch.gather(mu_dot_k, 1, src_idx_flat),
            cov_flat=cov_flat,
            mu_flat=mu_flat,
            scaling=scaling,
            merge_threshold=self.merge_threshold,
            target_allowed=target_allowed,
            representative="anchor",   # CHANGE 3
        )

        new_key = new_key.reshape(bsz, num_kv_heads, self.budget, head_dim).to(key_states.dtype)
        new_value = new_value.reshape(bsz, num_kv_heads, self.budget, head_dim).to(value_states.dtype)
        new_beta = new_beta.reshape(bsz, num_kv_heads, self.budget)
        new_n = new_n.reshape(bsz, num_kv_heads, self.budget)
        return new_key, new_value, new_beta, new_n
