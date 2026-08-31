import torch

from .merge_core import constant_gap_merge


class CovarianceMerge:
    """
    Decoding-time query-aware covariance merging, ported from
    `merging/text_handoff/decoding_covariance_merging_{mathematical_exposition,pseudocode}.tex`
    onto the R-KV decoding harness.

    Every cache slot is (key, value, beta, n): `key`/`value` are the slot's centroid (mass-
    weighted mean of the member keys/values it represents); `beta` is the additive attention-
    logit bias that makes attending to the centroid approximate attending to every original
    member; `n` is the slot's mass (sum of the masses of whatever was merged into it -- 1 for
    a never-merged token).

    The future-query mean and covariance are supplied by the caller (see rkv/query_moments.py):
    an EMA over pre-RoPE queries updated every step, projected into the next P RoPE frames.
    Both the importance score and the merge metric read that single model, so the two decisions
    are consistent with each other.

    Remaining scope cut, flagged rather than silently dropped: no persistent cumulative
    importance state. Importance is evaluated fresh at each compression event from the current
    (mu, Sigma) and each slot's own beta, rather than accumulated as a merge-recursive running
    total `I_C = I_A + I_B`.
    """

    # consumed by rkv/modeling.py: needs (beta, n) per-slot metadata, the future-query moments and
    # the additive-bias attention path...
    uses_merge_metadata = True
    # ...but no recent-query window: the moments are maintained online instead.
    uses_query_window = False

    def __init__(
        self,
        budget=128,
        window_size=8,
        first_tokens=0,
        merge_threshold=1.0,
        **kwargs,
    ):
        assert budget - window_size - first_tokens > 0, (
            "budget must leave room for both the protected recent window (window_size) and "
            "the protected sink prefix (first_tokens)"
        )
        self.budget = budget
        self.n_recent = window_size
        self.n_sink = first_tokens
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
        """
        key_states, value_states: (bsz, num_kv_heads, kv_len, head_dim)
        query_states: unused -- the future-query model replaces the recent-query window entirely.
            Kept in the signature because the compression seam passes it positionally.
        beta_states, n_states: (bsz, num_kv_heads, kv_len) -- fp32
        mu: (bsz, num_kv_heads, head_dim), cov: (bsz, num_kv_heads, head_dim, head_dim) -- the
            future-query moments for this layer, fp32
        returns the four per-slot tensors, compressed to `self.budget` slots along dim 2.
        """
        bsz, num_kv_heads, kv_len, head_dim = key_states.shape

        if kv_len < self.budget:
            return key_states, value_states, beta_states, n_states

        n_remove = kv_len - self.budget
        scaling = head_dim**-0.5

        flat = bsz * num_kv_heads
        key_flat = key_states.reshape(flat, kv_len, head_dim).float()
        val_flat = value_states.reshape(flat, kv_len, head_dim).float()
        beta_flat = beta_states.reshape(flat, kv_len).float()
        n_flat = n_states.reshape(flat, kv_len).float()
        mu_flat = mu.reshape(flat, head_dim).float()
        cov_flat = cov.reshape(flat, head_dim, head_dim).float()

        # k^T Sigma k for every slot, computed once: it is both the quadratic term of the
        # importance score and the diagonal of the pairwise distance below.
        quad = torch.einsum("ftd,fde,fte->ft", key_flat, cov_flat, key_flat)
        mu_dot_k = torch.einsum("ftd,fd->ft", key_flat, mu_flat)

        importance = self._importance_score(beta_flat, mu_dot_k, quad, scaling)

        # ---- source selection: lowest importance among unprotected positions ----
        protected = torch.zeros(kv_len, dtype=torch.bool, device=key_states.device)
        protected[: self.n_sink] = True
        protected[kv_len - self.n_recent :] = True
        selection_score = importance.masked_fill(protected.view(1, -1), float("inf"))

        is_src = torch.zeros(flat, kv_len, dtype=torch.bool, device=key_states.device)
        src_pos = selection_score.topk(n_remove, dim=-1, largest=False).indices
        is_src.scatter_(-1, src_pos, True)

        # stable sort on a binary key: all "kept" (False) positions first, in their original
        # relative order, then all "source" (True) positions -- fully vectorized, no loop
        order = is_src.to(torch.int64).argsort(dim=-1, stable=True)
        kept_idx_flat = order[..., : self.budget]
        src_idx_flat = order[..., self.budget :]

        expand_kept = kept_idx_flat.unsqueeze(-1).expand(-1, -1, head_dim)
        expand_src = src_idx_flat.unsqueeze(-1).expand(-1, -1, head_dim)

        kept_keys = torch.gather(key_flat, 1, expand_kept)
        kept_values = torch.gather(val_flat, 1, expand_kept)
        kept_beta = torch.gather(beta_flat, 1, kept_idx_flat)
        kept_n = torch.gather(n_flat, 1, kept_idx_flat)
        kept_q = torch.gather(quad, 1, kept_idx_flat)
        kept_mu_dot = torch.gather(mu_dot_k, 1, kept_idx_flat)

        src_keys = torch.gather(key_flat, 1, expand_src)
        src_values = torch.gather(val_flat, 1, expand_src)
        src_beta = torch.gather(beta_flat, 1, src_idx_flat)
        src_n = torch.gather(n_flat, 1, src_idx_flat)
        src_q = torch.gather(quad, 1, src_idx_flat)
        src_mu_dot = torch.gather(mu_dot_k, 1, src_idx_flat)

        new_key, new_value, new_beta, new_n = constant_gap_merge(
            kept_keys=kept_keys, kept_values=kept_values, kept_beta=kept_beta, kept_n=kept_n,
            kept_quad=kept_q, kept_mu_dot=kept_mu_dot,
            src_keys=src_keys, src_values=src_values, src_beta=src_beta, src_n=src_n,
            src_quad=src_q, src_mu_dot=src_mu_dot,
            cov_flat=cov_flat, mu_flat=mu_flat, scaling=scaling,
            merge_threshold=self.merge_threshold,
        )

        new_key = new_key.reshape(bsz, num_kv_heads, self.budget, head_dim).to(key_states.dtype)
        new_value = new_value.reshape(bsz, num_kv_heads, self.budget, head_dim).to(value_states.dtype)
        new_beta = new_beta.reshape(bsz, num_kv_heads, self.budget)
        new_n = new_n.reshape(bsz, num_kv_heads, self.budget)
        return new_key, new_value, new_beta, new_n

    @staticmethod
    def _importance_score(beta, mu_dot_k, quad, scaling):
        """Expected attention weight under the future-query model, in log space.

        For `q ~ N(mu, Sigma)` the pre-softmax weight of slot i is lognormal, so

            E[exp(beta_i + q.k_i / sqrt(d))] = exp(beta_i + scaling*mu.k_i + scaling^2 k_i^T Sigma k_i / 2)

        (the lognormal MGF `E[e^X] = e^(m + v/2)` with `m = scaling*mu.k_i` and
        `v = scaling^2 k_i^T Sigma k_i`). Returned as the log, which is rank-equivalent and cannot
        overflow when the quadratic term is large.

        Including `beta` is what makes this correct for slots that already represent a cluster:
        beta carries the logsumexp of the members absorbed into the slot, so the score is the
        expected attention mass of the *whole cluster*. The mass `n` must NOT also be multiplied
        in -- that would count the same members twice.
        """
        return beta + scaling * mu_dot_k + 0.5 * (scaling**2) * quad
