import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cal_similarity, compute_attention_scores
from .merge_core import constant_gap_merge


class RKVMerge:
    """R-KV's eviction scorer, with the constant-gap merge rule instead of outright eviction.

    This exists to ablate the two halves of `CovarianceMerge` against each other. That press pairs an
    expected-attention importance score with covariance merging; `R1KV` pairs R-KV's
    attention-minus-redundancy score with plain eviction. This class is the third corner:

        press              scorer                              removal
        -----------------  ----------------------------------  ----------------
        R1KV               R-KV attention - redundancy         evict
        CovarianceMerge    expected attention under N(mu,Sig)  merge (or evict)
        RKVMerge           R-KV attention - redundancy         merge (or evict)

    Comparing `RKVMerge` against `R1KV` isolates the merge rule with the scorer held fixed;
    comparing it against `CovarianceMerge` isolates the scorer with the removal rule held fixed.

    The scoring block below is copied verbatim from `r1_kv.py` so the kept set is identical, and the
    output layout reproduces R-KV's too (surviving non-window slots in top-k score order, then the
    recent window). Both are deliberate: with `merge_threshold <= 0` nothing merges and this press
    reduces to `R1KV` exactly, which `test_rkv_merge.py` asserts. Preserving the layout also matters
    because `cal_similarity` reads absolute positions, so a different permutation would feed the
    scorer different inputs at the *next* compression event and quietly confound the ablation.
    """

    # consumed by rkv/modeling.py: this press needs the (beta, n) per-slot metadata, the future-query
    # moments, and the additive-bias attention path...
    uses_merge_metadata = True
    # ...and unlike CovarianceMerge it also needs the recent-query window, because R-KV's score is
    # built from attention over those queries.
    uses_query_window = True

    def __init__(
        self,
        budget=128,
        window_size=8,
        kernel_size=7,
        mix_lambda=0.07,
        retain_ratio=0.1,
        retain_direction="last",
        merge_threshold=1.0,
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction
        self.merge_threshold = merge_threshold
        # NOTE: `first_tokens` is accepted via **kwargs and deliberately ignored. R-KV has no sink
        # protection, and adding one here would make this a different scorer, defeating the ablation.

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
        """
        key_states, value_states: (bsz, num_kv_heads, kv_len, head_dim)
        query_states: (bsz, num_query_heads, window_len, head_dim) cached recent queries, post-RoPE
        beta_states, n_states: (bsz, num_kv_heads, kv_len) fp32
        mu: (bsz, num_kv_heads, head_dim); cov: (bsz, num_kv_heads, head_dim, head_dim) fp32
        """
        bsz, num_kv_heads, kv_len, head_dim = key_states.shape

        if kv_len < self.budget:
            return key_states, value_states, beta_states, n_states

        n_remove = kv_len - self.budget
        scaling = head_dim**-0.5
        window = self.window_size

        # ================= R-KV scorer, verbatim from r1_kv.py =================
        attn_weights = compute_attention_scores(query_states, key_states)

        attn_weights_sum = (
            nn.functional.softmax(
                attn_weights[:, :, -window:, :-window],
                dim=-1,
                dtype=torch.float32,
            )
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
        # ======================= end verbatim block ============================

        # R-KV keeps the top-(budget - window) of the non-window region, in top-k order. Reproduce
        # that exactly, then take the complement as the merge sources.
        keep_pos = final_score.topk(self.budget - window, dim=-1).indices
        is_keep = torch.zeros(
            bsz, num_kv_heads, kv_len - window, dtype=torch.bool, device=key_states.device
        )
        is_keep.scatter_(-1, keep_pos, True)
        # Complement via a stable descending argsort on the boolean rather than a second topk:
        # topk(largest=False) is only the exact complement when there are no ties in final_score.
        src_pos = (
            (~is_keep).to(torch.int64).argsort(dim=-1, stable=True, descending=True)
        )[..., :n_remove]

        # indices into the full cache: the window is always kept and stays a suffix
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

        # k^T Sigma k and mu . k for every slot, once; the merge reuses both
        quad = torch.einsum("ftd,fde,fte->ft", key_flat, cov_flat, key_flat)
        mu_dot_k = torch.einsum("ftd,fd->ft", key_flat, mu_flat)

        kept_idx_flat = kept_idx.reshape(flat, self.budget)
        src_idx_flat = src_pos.reshape(flat, n_remove)
        expand_kept = kept_idx_flat.unsqueeze(-1).expand(-1, -1, head_dim)
        expand_src = src_idx_flat.unsqueeze(-1).expand(-1, -1, head_dim)

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
        )

        new_key = new_key.reshape(bsz, num_kv_heads, self.budget, head_dim).to(key_states.dtype)
        new_value = new_value.reshape(bsz, num_kv_heads, self.budget, head_dim).to(value_states.dtype)
        new_beta = new_beta.reshape(bsz, num_kv_heads, self.budget)
        new_n = new_n.reshape(bsz, num_kv_heads, self.budget)
        return new_key, new_value, new_beta, new_n
