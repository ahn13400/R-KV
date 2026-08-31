import torch


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

        # ---- nearest surviving target per source, under the covariance quadratic form ----
        # d(a,b) = a^T Sigma a + b^T Sigma b - 2 a^T Sigma b, avoids materializing (n_remove,
        # budget, head_dim) pairwise differences
        cross = torch.einsum("fsd,fde,fke->fsk", src_keys, cov_flat, kept_keys)
        dist2_raw = src_q.unsqueeze(-1) + kept_q.unsqueeze(1) - 2 * cross

        nearest_dist2_raw, nearest_target = dist2_raw.min(dim=-1)
        merge_ok = (scaling**2) * nearest_dist2_raw <= self.merge_threshold

        # ---- recursive constant-gap merge: every kept slot absorbs whichever sources
        # (0, 1, or many) chose it as their nearest target and passed the threshold ----
        proj_kept = kept_beta + scaling * kept_mu_dot
        proj_src = src_beta + scaling * src_mu_dot

        src_mass = src_n * merge_ok.float()
        added_mass = torch.zeros_like(kept_n).scatter_add_(1, nearest_target, src_mass)
        new_n = kept_n + added_mass

        weighted_src_key = src_keys * src_mass.unsqueeze(-1)
        added_key_sum = torch.zeros_like(kept_keys).scatter_add_(
            1, nearest_target.unsqueeze(-1).expand(-1, -1, head_dim), weighted_src_key
        )
        new_key = (kept_keys * kept_n.unsqueeze(-1) + added_key_sum) / new_n.unsqueeze(-1)

        neg_inf_where_evicted = torch.where(
            merge_ok, proj_src, torch.full_like(proj_src, float("-inf"))
        )
        max_per_target = torch.full_like(kept_beta, float("-inf")).scatter_reduce(
            1, nearest_target, neg_inf_where_evicted, reduce="amax", include_self=True
        )
        running_max = torch.maximum(proj_kept, max_per_target)

        exp_src = torch.where(
            merge_ok,
            torch.exp(proj_src - torch.gather(running_max, 1, nearest_target)),
            torch.zeros_like(proj_src),
        )
        sum_exp_src = torch.zeros_like(kept_beta).scatter_add_(1, nearest_target, exp_src)
        exp_kept_self = torch.exp(proj_kept - running_max)  # == 1 when no source beats kept's own proj
        total_exp = exp_kept_self + sum_exp_src

        weighted_src_val = src_values * exp_src.unsqueeze(-1)
        added_val_sum = torch.zeros_like(kept_values).scatter_add_(
            1, nearest_target.unsqueeze(-1).expand(-1, -1, head_dim), weighted_src_val
        )
        new_value = (kept_values * exp_kept_self.unsqueeze(-1) + added_val_sum) / total_exp.unsqueeze(-1)

        # b_C is *defined* as whatever makes scaling*mu.k_C + b_C exactly equal
        # logsumexp_j(proj_j) -- exact for this specific k_C by construction, regardless of how
        # k_C itself was chosen; reduces to the unmerged slot's own beta unchanged when no
        # source merges into it (running_max==proj_kept, total_exp==1, new_key==kept_keys).
        lse_proj = running_max + torch.log(total_exp)
        new_beta = lse_proj - scaling * torch.einsum("fkd,fd->fk", new_key, mu_flat)

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
