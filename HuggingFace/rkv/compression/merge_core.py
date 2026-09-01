"""
The constant-gap merge, shared by every merging press.

Factored out so that the merge rule and the *choice of which slots to remove* are independent: a
new press only has to decide `kept` vs `src` and can reuse this unchanged. That separation is what
makes an eviction-scorer ablation meaningful --- `CovarianceMerge` and `RKVMerge` differ only in the
scorer, and call identical merge code.

Every cache slot is `(k, v, beta, n)`: key, value, additive attention-logit bias, and cluster mass.
A never-merged token has `beta = 0, n = 1`.
"""

import torch


def constant_gap_merge(
    kept_keys,
    kept_values,
    kept_beta,
    kept_n,
    kept_quad,
    kept_mu_dot,
    src_keys,
    src_values,
    src_beta,
    src_n,
    src_quad,
    src_mu_dot,
    cov_flat,
    mu_flat,
    scaling,
    merge_threshold,
    target_allowed=None,
    representative="centroid",
):
    """Merge each source into its nearest surviving target, or evict it.

    All tensors are flattened over (batch, KV head) into a leading axis `f = bsz * num_kv_heads`:
      * `kept_*`  -- (f, budget, ...) the slots that survive
      * `src_*`   -- (f, n_remove, ...) the slots being removed
      * `*_quad`  -- k^T Sigma k per slot, precomputed by the caller (it is also the diagonal of
                     the pairwise distance, so it is never recomputed here)
      * `*_mu_dot`-- mu . k per slot, likewise precomputed
      * `cov_flat`, `mu_flat` -- (f, D, D) and (f, D), the future-query moments

    `target_allowed` -- optional (f, budget) bool marking which survivors may absorb a source.
    Disallowed columns get an infinite distance, so a source whose only near neighbours are
    ineligible is evicted rather than forced into a bad target. `None` allows every survivor.

    `representative` -- `"centroid"` stores the mass-weighted mean of the member keys;
    `"anchor"` keeps the target's own key verbatim. The value and bias formulas below are
    *independent of this choice*: under the constant-gap approximation,

        sum_j exp(beta_j + s q.k_j) ~= exp(s q.k_C) * sum_j exp(beta_j + s mu.(k_j - k_C)),

    so `v_C = softmax_j(proj_j) . v_j` and `beta_C = LSE_j(proj_j) - s mu.k_C` hold for whatever
    `k_C` is stored -- only the numerical value of `beta_C` differs. Anchor mode keeps every stored
    key a genuine post-RoPE key the model actually produced; centroid mode minimises the expected
    gap to the members it represents.

    Returns `(new_key, new_value, new_beta, new_n)`, each (f, budget, ...).
    """
    assert representative in ("centroid", "anchor"), representative
    head_dim = kept_keys.shape[-1]

    # ---- nearest surviving target per source, under the covariance quadratic form ----
    # d(a,c) = a^T Sigma a + c^T Sigma c - 2 a^T Sigma c, which avoids materializing the
    # (n_remove, budget, head_dim) tensor of pairwise differences
    cross = torch.einsum("fsd,fde,fke->fsk", src_keys, cov_flat, kept_keys)
    dist2_raw = src_quad.unsqueeze(-1) + kept_quad.unsqueeze(1) - 2 * cross
    if target_allowed is not None:
        dist2_raw = dist2_raw.masked_fill(~target_allowed.unsqueeze(1), float("inf"))

    nearest_dist2_raw, nearest_target = dist2_raw.min(dim=-1)
    # units: squared logits, so sqrt(threshold) is a predicted std deviation of the source/target
    # logit gap in nats. A source that fails this is evicted outright rather than merged. An
    # all-ineligible row yields +inf here, which fails for any finite threshold.
    merge_ok = (scaling**2) * nearest_dist2_raw <= merge_threshold

    # ---- recursive merge: every kept slot absorbs whichever sources (0, 1, or many) chose it
    # as their nearest target and passed the threshold ----
    proj_kept = kept_beta + scaling * kept_mu_dot
    proj_src = src_beta + scaling * src_mu_dot

    src_mass = src_n * merge_ok.float()
    added_mass = torch.zeros_like(kept_n).scatter_add_(1, nearest_target, src_mass)
    new_n = kept_n + added_mass

    if representative == "anchor":
        # The target's own key, verbatim. Bit-exact when nothing merged into it, and every stored
        # key stays a real post-RoPE key rather than a synthetic point in key space.
        new_key = kept_keys
    else:
        weighted_src_key = src_keys * src_mass.unsqueeze(-1)
        added_key_sum = torch.zeros_like(kept_keys).scatter_add_(
            1, nearest_target.unsqueeze(-1).expand(-1, -1, head_dim), weighted_src_key
        )
        new_key = (kept_keys * kept_n.unsqueeze(-1) + added_key_sum) / new_n.unsqueeze(-1)

    # log-sum-exp over each cluster's members, computed in a max-shifted form so a large proj
    # cannot overflow. Sources that failed the threshold are excluded via -inf.
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

    # beta_C is *defined* as whatever makes  scaling*mu.k_C + beta_C == logsumexp_j(proj_j)  hold
    # exactly for this specific k_C, whatever k_C turned out to be. That is what lets a centroid
    # representative recurse without accumulating error, and it reduces to the slot's own beta
    # unchanged when nothing merged into it (running_max == proj_kept, total_exp == 1,
    # new_key == kept_keys).
    lse_proj = running_max + torch.log(total_exp)
    new_beta = lse_proj - scaling * torch.einsum("fkd,fd->fk", new_key, mu_flat)

    return new_key, new_value, new_beta, new_n
