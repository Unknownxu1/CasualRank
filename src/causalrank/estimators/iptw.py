"""IPTW estimation of the multi-treatment potential-outcome means.

Estimand (common target population for every arm)::

    mu_a = E_P[Y(a)],  a = 1..K

Hajek (group-normalised) IPTW, the validated Phase 1 estimator::

    w_i          = 1 / e_{T_i}(X_i)
    mu_a^H       = sum_i 1{T_i=a} w_i Y_i / sum_i 1{T_i=a} w_i

Horvitz-Thompson is implemented as an *internal* cross-check only (it is not
exposed through the public API) and equals ``(1/n) * sum_i 1{T_i=a} Y_i/e_a(X_i)``
— note the ``1/n`` normalisation, which the spec forbids mixing with the
group-normalised form.

Numerical boundaries handled here:
* ``e_{T_i}`` must be strictly positive and finite, otherwise estimation fails
  fast (silent clipping is forbidden; explicit clipping is configured and its
  rate is reported by the diagnostics);
* with binary outcomes the Hajek mean lies in [0,1] by construction;
* the estimate is invariant to scaling all weights of a treatment by a constant.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..exceptions import CausalRankError


@dataclass
class IPTWEstimate:
    """Hajek IPTW point estimates and the weight/ESS bookkeeping."""

    labels: tuple[Any, ...]
    t_idx: np.ndarray  # (n,) int codes of the observed treatment, in `labels` order
    mu: np.ndarray  # (K,) Hajek means
    weights: np.ndarray  # (n,) w_i = 1 / e_{T_i}(X_i) (after any explicit clipping)
    weight_sums: np.ndarray  # (K,) sum of w_i within each arm
    numerators: np.ndarray  # (K,) sum of w_i Y_i within each arm
    ess: np.ndarray  # (K,) effective sample size per arm
    n_treated: np.ndarray  # (K,)
    raw_means: np.ndarray  # (K,) unweighted within-arm outcome means
    observed_propensity: np.ndarray  # (n,) e_{T_i}(X_i) after any clipping
    clipped_fraction: float  # share of observed propensities changed by clipping

    @property
    def n_treatments(self) -> int:
        return len(self.labels)


def observed_propensity(propensity: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
    """Return ``e_{T_i}(X_i)`` for every row."""
    propensity = np.asarray(propensity, dtype=float)
    t_idx = np.asarray(t_idx, dtype=np.int64)
    if propensity.ndim != 2:
        raise CausalRankError("propensity must be an (n, K) array.")
    if t_idx.shape[0] != propensity.shape[0]:
        raise CausalRankError("t_idx and propensity row counts differ.")
    return propensity[np.arange(t_idx.shape[0]), t_idx]


def effective_sample_size(weights: np.ndarray) -> float:
    """``ESS = (sum w)^2 / sum w^2`` for a vector of non-negative weights."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise CausalRankError("weights must be a non-empty 1-D array.")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise CausalRankError("weights must be finite and non-negative.")
    denom = float(np.sum(w * w))
    total = float(np.sum(w))
    if denom == 0.0:
        raise CausalRankError("All weights are zero; ESS is undefined.")
    return total * total / denom


def _clip_observed(
    e_obs: np.ndarray, clip: float | None
) -> tuple[np.ndarray, float]:
    """Explicit, opt-in clipping of the observed-treatment propensity.

    Only the denominator ``e_{T_i}`` is clipped, and only from below:
    ``e <- max(e, eps)``. Values near 1 are left alone because the weight
    ``1/e`` is already bounded there. Clipping changes the finite-sample bias
    of the estimator and the meaning of the weights, so it is never applied
    silently and the clipped share is reported. Exactly-zero propensities are
    still rejected before clipping: that is a positivity violation, not a
    numerical nuisance.
    """
    if clip is None:
        return e_obs, 0.0
    eps = float(clip)
    if not (0.0 < eps < 0.5):
        raise CausalRankError("clip must lie in (0, 0.5).")
    clipped = np.clip(e_obs, eps, None)
    frac = float(np.mean(clipped != e_obs))
    return clipped, frac


def hajek_estimate(
    y: np.ndarray,
    t_idx: np.ndarray,
    propensity: np.ndarray,
    labels: Sequence[Any],
    clip_propensity: float | None = None,
) -> IPTWEstimate:
    """Compute the Hajek IPTW mean of ``Y(a)`` for every treatment."""
    y = np.asarray(y, dtype=float)
    t_idx = np.asarray(t_idx, dtype=np.int64)
    propensity = np.asarray(propensity, dtype=float)
    k = len(labels)
    if y.ndim != 1 or y.shape[0] != t_idx.shape[0]:
        raise CausalRankError("y must be 1-D and aligned with t_idx.")
    if propensity.shape != (y.shape[0], k):
        raise CausalRankError(
            f"propensity must have shape {(y.shape[0], k)}; got {propensity.shape}."
        )
    if not np.all(np.isfinite(y)):
        raise CausalRankError("y contains non-finite values.")
    if np.any(propensity < 0) or not np.all(np.isfinite(propensity)):
        raise CausalRankError("propensity must be finite and non-negative.")

    e_obs = observed_propensity(propensity, t_idx)
    if np.any(e_obs <= 0.0):
        n_bad = int(np.sum(e_obs <= 0.0))
        raise CausalRankError(
            f"{n_bad} observation(s) have zero/negative propensity for their observed "
            "treatment; positivity is violated for these rows and the Hajek weight 1/e is "
            "undefined. Inspect the overlap diagnostics before any clipping."
        )
    e_used, clipped_fraction = _clip_observed(e_obs, clip_propensity)
    w = 1.0 / e_used

    mu = np.empty(k, dtype=float)
    weight_sums = np.empty(k, dtype=float)
    numerators = np.empty(k, dtype=float)
    ess = np.empty(k, dtype=float)
    n_treated = np.empty(k, dtype=np.int64)
    raw_means = np.empty(k, dtype=float)

    for a in range(k):
        mask = t_idx == a
        w_a = w[mask]
        y_a = y[mask]
        n_a = int(mask.sum())
        n_treated[a] = n_a
        if n_a == 0:
            raise CausalRankError(
                f"Treatment {labels[a]!r} has no observations; cannot estimate its mean."
            )
        w_sum = float(w_a.sum())
        weight_sums[a] = w_sum
        numerators[a] = float((w_a * y_a).sum())
        mu[a] = numerators[a] / w_sum
        ess[a] = effective_sample_size(w_a)
        raw_means[a] = float(y_a.mean())

    return IPTWEstimate(
        labels=tuple(labels),
        t_idx=t_idx.copy(),
        mu=mu,
        weights=w,
        weight_sums=weight_sums,
        numerators=numerators,
        ess=ess,
        n_treated=n_treated,
        raw_means=raw_means,
        observed_propensity=e_used,
        clipped_fraction=clipped_fraction,
    )


def horvitz_thompson_estimate(
    y: np.ndarray,
    t_idx: np.ndarray,
    propensity: np.ndarray,
    labels: Sequence[Any],
) -> np.ndarray:
    """Horvitz-Thompson means — *internal* reference for numerical cross-checks.

    ``mu_a^HT = (1/n) sum_i 1{T_i=a} Y_i / e_a(X_i)``. Not part of the public
    API: it is less stable in finite samples and is only used in tests to verify
    that the weighting algebra is implemented consistently.
    """
    y = np.asarray(y, dtype=float)
    t_idx = np.asarray(t_idx, dtype=np.int64)
    propensity = np.asarray(propensity, dtype=float)
    k = len(labels)
    n = y.shape[0]
    out = np.empty(k, dtype=float)
    for a in range(k):
        mask = t_idx == a
        e_a = propensity[mask, a]
        if np.any(e_a <= 0.0):
            raise CausalRankError("Zero propensity for an observed treatment.")
        out[a] = float(np.sum(y[mask] / e_a)) / n
    return out
