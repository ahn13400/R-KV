"""
Online future-query moments for `CovarianceMerge`.

The merge decision needs the mean and covariance of the queries the model *will* issue, per KV
head: the merge threshold is a bound on `Var[q . dk]` over that future population, and the
importance score is `E[exp(q . k / sqrt(d))]` under it. This module maintains that estimate.

Configuration follows the estimator sweep in `qstats/results/figures_prelim/summary.md`, which
scores estimators by `excess_logit_std` -- the extra standard deviation (nats) of the source/target
logit gap incurred by choosing a merge target with the estimated covariance rather than the true
one. Best was an EMA with half-life H=64 decode steps, projected into P=128 future RoPE frames with
uniform weighting (0.0440). For reference, a raw 8-query window -- which is what this kernel used
before -- scored 0.3261, and dropping the RoPE projection cost 2.6x (0.1146).

Two properties this module is careful about, both of which that sweep shows matter:

* **Centering.** `Sigma = E[qq^T] - mu mu^T`. The uncentered second moment scored 0.1361 vs 0.0440:
  it double-penalises the mean offset that the merge bias exists to cancel.
* **Moment-level GQA pooling.** Moments are pooled over the joint (step, query-head) population and
  centred *once*, never averaged as per-head covariances. One merged slot carries one scalar bias
  for its entire GQA group, so between-head mean spread is error the bias cannot absorb. Pooling
  moments and centring once picks it up via the law of total covariance; averaging per-head
  covariances silently drops it (worth 15-58% of the within-head term).

Pre-RoPE formulation
--------------------
qstats' estimator keeps *post-RoPE* moments and advances them by one relative rotation per step.
This module keeps *pre-RoPE* moments and rotates into P future *absolute* positions at readout.
These are algebraically identical whenever RoPE composes (`R_a R_b == R_{a+b}`), because

    R_{L-t+p} q_t == R_{L-t+p} R_t x_t == R_{L+p} x_t,

so a query observed at step t transported to p steps past the current end is the same vector either
way. The pre-RoPE form is preferable here for two implementation reasons: the state is stationary,
so there is no per-step `R S R^T` transport at all, and seeding from a whole prompt collapses to one
einsum instead of a loop over prompt positions.

Composition fails under non-linear RoPE scaling (YaRN and friends), so `assert_rope_composes`
refuses rather than quietly changing what is being estimated.
"""

import torch


def beta_from_half_life(half_life):
    """EMA decay whose weight halves after `half_life` steps. H=64 -> 0.98923."""
    assert half_life >= 1, "ema_half_life must be >= 1"
    return float(2.0 ** (-1.0 / half_life))


def future_weights(horizon, decay, device, dtype=torch.float32):
    """Weights over future offsets h = 1..P, `w_h ~ decay^(h-1)`, normalized to sum to 1.

    `decay=1.0` is uniform and is the default: the sweep found recency weighting only ever hurts
    (gamma=0.999 -> 0.0441, 0.99 -> 0.0463, 0.97 -> 0.0626, against 0.0440 uniform). Exposed as an
    ablation, per the handoff's instruction that it change the objective explicitly rather than
    silently.
    """
    assert horizon >= 1, "future_horizon must be >= 1"
    assert 0.0 < decay <= 1.0, "future_decay must be in (0, 1]"
    w = decay ** torch.arange(horizon, device=device, dtype=dtype)
    return w / w.sum()


def assert_rope_composes(config):
    """The pre-RoPE formulation above requires `R_a R_b == R_{a+b}`, which holds for default and
    linear RoPE scaling but not for non-linear variants."""
    scaling = getattr(config, "rope_scaling", None)
    if scaling is None:
        return
    kind = str(scaling.get("rope_type", scaling.get("type", ""))).lower()
    if kind not in ("default", "linear"):
        raise NotImplementedError(
            f"CovarianceMerge's future-query projection assumes RoPE rotations compose "
            f"(R_a R_b == R_a+b), which fails for rope_scaling type {kind!r}. Supported: "
            f"default, linear."
        )


class FutureRopeOperator:
    """The weighted average over future RoPE frames, `mu -> E_p[R_p mu]` and `M -> E_p[R_p M R_p^T]`.

    Held as four pre-averaged `(D, D)` trig coefficient matrices plus two `(D,)` vectors, which is
    what keeps this `O(D^2)` elementwise per head instead of `O(P D^3)`.

    Why the second moment cannot reuse the averaged rotation: `E[R M R^T] != E[R] M E[R]^T`, exactly
    as `E[X^2] != E[X]^2`. Averaging the rotation *before* applying it to a covariance annihilates
    the variance RoPE itself induces across the horizon -- high-frequency bands have `E[R_p] -> 0`
    over a long window, which manufactures spurious zero-variance directions and therefore false
    constant-gap merge candidates. The mean is linear in `R_p`, so averaging first is exact there.

    HF RoPE (`rotate_half`) pairs dimension `i` with `sigma(i) = i + D/2 (mod D)` and acts as the
    2x2 rotation `[[c, -s], [s, c]]` on each pair. With `eps(i) = -1` for `i < D/2` else `+1`,
    `R_p[i, j] = cos_p[i] delta_ij + eps(i) sin_p[i] delta_{j, sigma(i)}`, hence

        m_q[i,j] = cc[i,j] M[i,j]
                 + eps(j) cs[i,j] M[i,sigma(j)]
                 + eps(i) sc[i,j] M[sigma(i),j]
                 + eps(i) eps(j) ss[i,j] M[sigma(i),sigma(j)],

    and each `M[sigma(.)]` is a `roll` by `D/2` because sigma just swaps the halves of an axis.
    """

    def __init__(self, cos, sin, weights):
        """`cos`, `sin`: (bsz, P, D) from the model's rotary embedding at each row's own P future
        positions, already half-duplicated. `weights`: (P,) summing to 1.

        Coefficients are stored with a singleton KV-head axis inserted so they broadcast against
        `(bsz, num_kv_heads, D)` and `(bsz, num_kv_heads, D, D)` state.
        """
        d = cos.shape[-1]
        assert d % 2 == 0, f"RoPE needs an even head dim, got {d}"
        assert cos.shape[-2] == weights.shape[0], "weights must have one entry per future position"
        self.head_dim = d
        self.half = d // 2

        w = weights.view(-1, 1)
        cw, sw = cos * w, sin * w
        # (bsz, 1, D) weighted means, for the (exact) linear mean transform
        self.cos_mean = (cw.sum(-2)).unsqueeze(-2)
        self.sin_mean = (sw.sum(-2)).unsqueeze(-2)
        # (bsz, 1, D, D) weighted trig outer products, for the second moment
        self.cc = torch.einsum("...pi,...pj->...ij", cos, cw).unsqueeze(-3)
        self.cs = torch.einsum("...pi,...pj->...ij", cos, sw).unsqueeze(-3)
        self.sc = torch.einsum("...pi,...pj->...ij", sin, cw).unsqueeze(-3)
        self.ss = torch.einsum("...pi,...pj->...ij", sin, sw).unsqueeze(-3)
        # sign of the paired term
        self.eps = torch.cat([-cos.new_ones(self.half), cos.new_ones(self.half)])

    def __call__(self, mean, second):
        """`mean` (..., D), `second` (..., D, D) -> the same shapes, averaged over the horizon."""
        f, eps = self.half, self.eps
        mean_q = self.cos_mean * mean + eps * self.sin_mean * mean.roll(f, -1)
        second_q = (
            self.cc * second
            + (self.cs * eps) * second.roll(f, -1)
            + (eps[:, None] * self.sc) * second.roll(f, -2)
            + (eps[:, None] * self.ss * eps) * second.roll((f, f), (-2, -1))
        )
        return mean_q, second_q


def build_future_rope_operator(rotary_emb, position_ids, horizon, decay, reference):
    """Build the operator for the `horizon` positions immediately after each row's current one.

    `position_ids` (bsz, q_len): the positions of this forward's queries. Only the last column is
    used, so a padded batch is handled correctly -- each row's future window starts after *its own*
    last real token, not after the padded end of the batch. HuggingFace derives `position_ids` from
    the 2D attention mask, so for a left-padded row the last column is already `real_length - 1`.

    Called once per compression event in `CausalLM_forward` and shared by every layer: the operator
    depends only on the RoPE frequencies and the horizon, not on the layer.
    """
    offsets = torch.arange(1, horizon + 1, device=position_ids.device)
    future = position_ids[:, -1:].to(torch.long) + offsets.view(1, -1)  # (bsz, P)
    cos, sin = rotary_emb(reference, future)
    weights = future_weights(horizon, decay, position_ids.device)
    return FutureRopeOperator(cos.float(), sin.float(), weights)


def pooled_moments(queries, weights):
    """Moment-level pooling over the GQA group and the time axis together.

    `queries`: (bsz, num_kv_heads, groups, n_steps, head_dim) pre-RoPE queries.
    `weights`: (bsz, n_steps), each row summing to 1 (or to 0 for a row with no real steps).

    Returns `mean` (bsz, num_kv_heads, head_dim) and `second` (bsz, num_kv_heads, D, D), pooled with
    weight `weights[b, t] / groups` on every (group member, step) pair, so total weight per row is
    `sum_t weights[b, t]`.

    Never materializes an (n_steps, D, D) tensor: the second moment is formed by scaling the samples
    by sqrt(weight) and contracting, so peak memory is that of `queries` itself.
    """
    groups = queries.shape[2]
    w = (weights / groups).view(weights.shape[0], 1, 1, weights.shape[1], 1)
    mean = (queries * w).sum(dim=(2, 3))
    scaled = queries * w.sqrt()
    second = torch.einsum("bhgtd,bhgte->bhde", scaled, scaled)
    return mean, second


def prefill_ema_weights(n_steps, valid_lengths, ema_beta, device):
    """EMA weights over prompt positions, as if prefill and decode were one continuous stream.

    Position `i` (0-based, left-padded so the newest real token is at `n_steps - 1`) gets
    `beta^(n_steps - 1 - i) (1 - beta)`, padding gets 0, and each row is renormalized to sum to 1.

    The renormalization is what keeps `Sigma = S - mu mu^T` PSD: a weighted second moment minus the
    outer product of the weighted mean is a genuine weighted covariance only when the weights sum to
    exactly 1. Left un-normalized the sum is `1 - beta^L`, which is visibly short for a short prompt
    and would report a covariance that is too large.
    """
    age = torch.arange(n_steps - 1, -1, -1, device=device, dtype=torch.float32)
    w = (ema_beta**age) * (1.0 - ema_beta)
    w = w.unsqueeze(0).expand(valid_lengths.shape[0], -1).clone()
    positions = torch.arange(n_steps, device=device)
    real = positions.view(1, -1) >= (n_steps - valid_lengths.view(-1, 1))
    w = w * real
    return w / w.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(torch.float32).tiny)


def update_query_moments(
    past_key_value, layer_idx, prerope_query, num_kv_heads, ema_beta, valid_lengths, is_prefill
):
    """Fold this forward's pre-RoPE queries into the layer's running EMA moments.

    `prerope_query`: (bsz, num_query_heads, n_steps, head_dim), pre-RoPE (post q_norm).

    State lives on `past_key_value` in dicts keyed by layer_idx, exactly like the pre-existing
    `query_cache`. Because transformers builds a fresh Cache for every `generate()` call, this resets
    per sequence for free rather than leaking across samples.
    """
    if not hasattr(past_key_value, "q_mean"):
        past_key_value.q_mean = {}
        past_key_value.q_second = {}

    bsz, num_query_heads, n_steps, head_dim = prerope_query.shape
    groups = num_query_heads // num_kv_heads
    # query heads of one KV group are contiguous, so head h serves KV head h // groups
    x = prerope_query.reshape(bsz, num_kv_heads, groups, n_steps, head_dim).float()

    if is_prefill or layer_idx not in past_key_value.q_mean:
        weights = prefill_ema_weights(n_steps, valid_lengths, ema_beta, x.device)
        mean, second = pooled_moments(x, weights)
        past_key_value.q_mean[layer_idx] = mean
        past_key_value.q_second[layer_idx] = second
        return

    weights = x.new_ones(bsz, n_steps) / n_steps
    mean, second = pooled_moments(x, weights)
    past_key_value.q_mean[layer_idx] = (
        ema_beta * past_key_value.q_mean[layer_idx] + (1.0 - ema_beta) * mean
    )
    past_key_value.q_second[layer_idx] = (
        ema_beta * past_key_value.q_second[layer_idx] + (1.0 - ema_beta) * second
    )


def read_future_moments(past_key_value, layer_idx, operator):
    """Project the stored pre-RoPE moments into the future horizon and centre them.

    Returns `(mu, cov)`, each per (batch, KV head): `mu` (bsz, Hkv, D), `cov` (bsz, Hkv, D, D).
    """
    mean = past_key_value.q_mean[layer_idx]
    second = past_key_value.q_second[layer_idx]
    mean_q, second_q = operator(mean, second)
    cov = second_q - mean_q.unsqueeze(-1) * mean_q.unsqueeze(-2)
    # Centring subtracts nearly equal quantities, so symmetrise: the quadratic forms downstream
    # should see a genuinely symmetric matrix rather than one carrying round-off asymmetry.
    return mean_q, 0.5 * (cov + cov.transpose(-1, -2))
