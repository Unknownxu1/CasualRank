"""Bonferroni simultaneous pairwise difference intervals (PROJECT_SPEC.md 3.6/3.7).

For every unordered pair ``i < j`` of treatments the difference estimator is

    d_ij = mu_i - mu_j,
    SE(d_ij) = sqrt(Sigma_ii + Sigma_jj - 2 * Sigma_ij),

with ``Sigma`` the **joint** covariance of the K mean estimates, so the
correlation between arms is carried into the difference standard error.

With ``M = K(K-1)/2`` pairs the simultaneous (Bonferroni) interval is

    d_ij +- z_{1 - alpha/(2M)} * SE(d_ij),   alpha = 1 - confidence_level.

A pair is *confirmed* only when this multiplicity-adjusted interval excludes
zero:

* ``CI_lower(d_ij) > 0``  -> ``mu_i`` is (point-estimate-)larger than ``mu_j``
  by a statistically confirmed margin;
* ``CI_upper(d_ij) < 0``  -> confirmed smaller;
* otherwise the comparison is **inconclusive**.

``inconclusive`` never means "equal": failing to reject a zero difference is
not evidence of equality (PROJECT_SPEC.md 3.6 red line). The direction words
``i_better`` / ``j_better`` additionally respect ``higher_is_better`` (a lower
outcome can be the better outcome).

Degenerate variance: when ``SE(d_ij)`` is numerically zero the interval
collapses to the point ``{d_ij}``; the decision then follows the sign of the
point difference, and an exact point tie (``d_ij == 0``) stays inconclusive and
is flagged via ``point_tie``. Point-estimate ties must never be broken by label
order.

K = 2 degeneracy: ``M = 1`` gives ``z_{1 - alpha/2}`` — exactly the unadjusted
two-arm comparison.

Red lines (PROJECT_SPEC.md 3.7):

* pointwise CIs may not substitute for these simultaneous comparisons;
* the intervals rely on the joint covariance and an asymptotic normal
  approximation of the estimator; that is a modelling statement, not a
  universal guarantee (validate by simulation, see ``run_rank_benchmark``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..exceptions import CausalRankError

#: Decision vocabulary. There is deliberately no "equal" / "tie" decision:
#: an inconclusive interval is NOT evidence of equality.
DECISIONS: tuple[str, ...] = ("i_better", "j_better", "inconclusive")

#: Relative tolerance used when detecting a numerically zero difference
#: variance on the scale of the covariance diagonal.
_ZERO_SE_RTOL = 1e-12


def bonferroni_critical_value(confidence_level: float, n_pairs: int) -> float:
    """Return ``z_{1 - alpha/(2M)}`` with ``alpha = 1 - confidence_level``.

    For ``n_pairs = 1`` (two treatments) this is the ordinary two-sided normal
    quantile ``z_{1 - alpha/2}``.
    """
    from scipy.stats import norm

    if not (0.0 < float(confidence_level) < 1.0):
        raise CausalRankError("confidence_level must lie in (0, 1).")
    if int(n_pairs) < 1:
        raise CausalRankError("n_pairs must be >= 1.")
    alpha = 1.0 - float(confidence_level)
    return float(norm.ppf(1.0 - alpha / (2.0 * int(n_pairs))))


def difference_variance(sigma: np.ndarray, i: int, j: int) -> float:
    """Var(mu_i - mu_j) = Sigma_ii + Sigma_jj - 2 * Sigma_ij.

    A tiny negative value from floating-point cancellation is clipped to zero;
    a meaningfully negative value raises (the matrix would not be PSD).
    """
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise CausalRankError(f"covariance must be square 2-D; got shape {sigma.shape}.")
    k = sigma.shape[0]
    if not (0 <= int(i) < k and 0 <= int(j) < k):
        raise CausalRankError(f"index out of range for a {k}x{k} covariance.")
    var = float(sigma[i, i] + sigma[j, j] - 2.0 * sigma[i, j])
    scale = max(float(sigma[i, i]), float(sigma[j, j]), 1e-300)
    if var < 0.0:
        if var >= -_ZERO_SE_RTOL * scale:
            return 0.0
        raise CausalRankError(
            f"var(mu_{i} - mu_{j}) = {var:.3e} is negative; the covariance is not PSD "
            "and the difference SE is undefined. Refusing to silently repair it."
        )
    return var


def difference_se(sigma: np.ndarray, i: int, j: int) -> float:
    """``SE(mu_i - mu_j) = sqrt(Sigma_ii + Sigma_jj - 2 Sigma_ij)``."""
    return float(np.sqrt(difference_variance(sigma, i, j)))


def check_labels(labels: Sequence[Any]) -> tuple[Any, ...]:
    """Normalise ``labels`` to a tuple; reject duplicates and unhashable entries.

    Duplicate labels silently mis-index the ranking statistics (the label ->
    position map keeps only the last occurrence), so they are refused here
    instead of producing a plausible-looking but wrong ranking.
    """
    lab = tuple(labels)
    try:
        unique = set(lab)
    except TypeError as exc:
        raise CausalRankError("labels must be hashable.") from exc
    if len(unique) != len(lab):
        raise CausalRankError(
            f"labels must be unique; got duplicates in {list(lab)}. Duplicate labels "
            "mis-index the pairwise/ranking statistics and are rejected."
        )
    return lab


def _normalize_inputs(
    estimates: Mapping[Any, float] | Sequence[float] | np.ndarray,
    covariance: np.ndarray,
    labels: Sequence[Any] | None,
) -> tuple[tuple[Any, ...], np.ndarray, np.ndarray]:
    """Return ``(labels, mu, sigma)`` in one consistent canonical order.

    When ``estimates`` is a mapping, ``labels`` is a **permutation of its
    keys**: the mean vector is looked up *by label* and reordered accordingly,
    and the covariance is interpreted in that same order. Without this
    reordering the position-based means and the label-based output would drift
    apart and invert the better/worse relations.
    """
    if isinstance(estimates, Mapping):
        keys = tuple(estimates.keys())
        if labels is not None:
            lab = check_labels(labels)
            if len(lab) != len(keys) or set(lab) != set(keys):
                raise CausalRankError(
                    "labels must be a permutation of the estimate keys "
                    f"(got labels={list(lab)}, keys={list(keys)})."
                )
            mu = np.array([float(estimates[key]) for key in lab], dtype=float)
        else:
            lab = keys
            mu = np.array([float(estimates[key]) for key in keys], dtype=float)
    else:
        mu = np.asarray(estimates, dtype=float).reshape(-1)
        lab = check_labels(labels) if labels is not None else tuple(range(mu.size))
    sigma = np.asarray(covariance, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise CausalRankError(f"covariance must be square 2-D; got shape {sigma.shape}.")
    if sigma.shape[0] != mu.size:
        raise CausalRankError(
            f"covariance shape {sigma.shape} does not match {mu.size} estimates."
        )
    if mu.size < 2:
        raise CausalRankError("rank inference needs at least 2 treatments.")
    if not np.all(np.isfinite(mu)):
        raise CausalRankError("estimates must all be finite.")
    if not np.all(np.isfinite(sigma)):
        raise CausalRankError("covariance contains non-finite entries.")
    if len(lab) != mu.size:
        raise CausalRankError("labels must match the number of estimates.")
    return lab, mu, sigma


def pairwise_comparisons(
    estimates: Mapping[Any, float] | Sequence[float] | np.ndarray,
    covariance: np.ndarray,
    confidence_level: float = 0.95,
    labels: Sequence[Any] | None = None,
    higher_is_better: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bonferroni simultaneous intervals for all ``M = K(K-1)/2`` pairs.

    Returns ``(rows, info)`` where ``rows`` is one record per pair ``i < j`` in
    canonical label order and ``info`` carries the multiplicity settings
    (``n_pairs``, ``z_crit``, ``confidence_level``, ``method``,
    ``higher_is_better``) for traceability.

    Row schema::

        label_i, label_j, mu_i, mu_j, diff (= mu_i - mu_j), se_diff,
        ci_low, ci_high, z_crit, n_pairs, confidence_level,
        decision ("i_better" | "j_better" | "inconclusive"),
        confirmed (bool), point_tie (bool), degenerate_se (bool)

    ``decision`` is expressed in *better* terms: ``i_better`` means arm ``i``
    is confirmed better than arm ``j`` (higher outcome if
    ``higher_is_better=True``, lower outcome otherwise). ``inconclusive`` is
    never an equality claim.

    ``labels`` fixes the canonical order: for a mapping ``estimates`` it must
    be a permutation of the mapping keys and the means (and the covariance,
    which is read in the same order) are re-indexed to follow it; for a
    sequence ``estimates`` it names the positions. Duplicate labels are
    rejected.
    """
    lab, mu, sigma = _normalize_inputs(estimates, covariance, labels)
    k = mu.size
    if not isinstance(higher_is_better, bool):
        raise CausalRankError("higher_is_better must be a bool.")
    if not (0.0 < float(confidence_level) < 1.0):
        raise CausalRankError("confidence_level must lie in (0, 1).")

    n_pairs = k * (k - 1) // 2
    z_crit = bonferroni_critical_value(confidence_level, n_pairs)

    rows: list[dict[str, Any]] = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = float(mu[i] - mu[j])
            var_diff = difference_variance(sigma, i, j)
            se_diff = float(np.sqrt(var_diff))
            scale = max(abs(diff), 1.0)
            degenerate = se_diff <= _ZERO_SE_RTOL * scale
            lo = diff - z_crit * se_diff
            hi = diff + z_crit * se_diff

            point_tie = bool(diff == 0.0)
            if lo > 0.0:
                decision = "i_better" if higher_is_better else "j_better"
            elif hi < 0.0:
                decision = "j_better" if higher_is_better else "i_better"
            else:
                decision = "inconclusive"

            rows.append(
                {
                    "label_i": lab[i],
                    "label_j": lab[j],
                    "mu_i": float(mu[i]),
                    "mu_j": float(mu[j]),
                    "diff": diff,
                    "se_diff": se_diff,
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "z_crit": z_crit,
                    "n_pairs": int(n_pairs),
                    "confidence_level": float(confidence_level),
                    "decision": decision,
                    "confirmed": decision != "inconclusive",
                    "point_tie": point_tie,
                    "degenerate_se": bool(degenerate),
                }
            )

    info = {
        "method": "bonferroni",
        "confidence_level": float(confidence_level),
        "n_pairs": int(n_pairs),
        "z_crit": z_crit,
        "higher_is_better": bool(higher_is_better),
        "labels": list(lab),
    }
    return rows, info
