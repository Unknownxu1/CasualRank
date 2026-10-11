"""Covariate balance diagnostics: standardised mean differences (SMD).

Definition used throughout CausalRank (documented in docs/methodology.md):

* numeric covariate -> the covariate itself is one *term* (missing entries are
  dropped term-wise, never allowed to poison the means/SDs);
* categorical covariate -> one 0/1 indicator per level is a *term*
  (plus a ``__missing__`` indicator when the feature has missing values, so the
  missingness pattern itself is checked for imbalance);
* for every unordered treatment pair ``(a, b)``::

      SMD_ab = (m_a - m_b) / s_ref,   s_ref = sqrt((s_a^2 + s_b^2) / 2)

  where ``m_g`` is the mean of the term inside arm ``g`` — raw
  (``smd_unweighted``) or Hajek-weighted with ``w_i = 1/e_{T_i}(X_i)``
  (``smd_weighted``) — and ``s_a, s_b`` are the *unweighted* within-arm sample
  standard deviations (ddof=1) of the term, i.e. a pooled reference SD that the
  weights cannot move (Austin 2009). Using the weighted SD in the denominator
  would let extreme weights shrink the SMD and hide imbalance.

Interpretation limits: ``max|SMD| <= 0.1`` is a conventional screening rule,
not a sufficiency certificate; balancing weak covariates says nothing about
strong confounders that were never measured.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config import CausalRankConfig
from ..results import DiagnosticWarning

_MISSING_LEVEL = "__missing__"


@dataclass
class BalanceResult:
    """Long-form SMD table plus a per-term summary."""

    detail: pd.DataFrame
    summary: pd.DataFrame
    warnings: list[DiagnosticWarning]


def _term_matrix(
    X: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> tuple[list[str], list[str | None], np.ndarray, np.ndarray]:
    """Build the (n x p) term matrix with explicit (feature, level) labels.

    Returns ``(names, levels, Z, valid)`` where ``Z`` holds zero-filled values
    and ``valid`` marks observed entries; every statistic is computed with the
    per-term validity mask so that missing covariates never poison the SMD. A
    ``__missing__`` indicator term is added for any feature that actually has
    missing values, so imbalance in the missingness pattern is visible too.
    """
    names: list[str] = []
    levels: list[str | None] = []
    columns: list[np.ndarray] = []
    valids: list[np.ndarray] = []

    for col in numeric_features:
        values = X[col].to_numpy(dtype=float)
        valid = ~np.isnan(values)
        if not valid.any():
            continue  # entirely-missing columns are rejected earlier; defensive
        names.append(str(col))
        levels.append(None)
        columns.append(np.where(valid, values, 0.0))
        valids.append(valid)
        if not valid.all():
            names.append(str(col))
            levels.append(_MISSING_LEVEL)
            columns.append((~valid).astype(float))
            valids.append(np.ones(values.shape[0], dtype=bool))

    for col in categorical_features:
        s = X[col]
        is_missing = s.isna().to_numpy()
        present = s[~is_missing]
        for level in sorted(pd.unique(present).tolist(), key=str):
            # fillna(False) keeps pd.NA markers from breaking the comparison
            eq = s.eq(level).fillna(False).to_numpy(dtype=bool)
            indicator = (eq & ~is_missing).astype(float)
            names.append(str(col))
            levels.append(str(level))
            columns.append(indicator)
            valids.append(np.ones(indicator.shape[0], dtype=bool))
        if bool(is_missing.any()):
            names.append(str(col))
            levels.append(_MISSING_LEVEL)
            columns.append(is_missing.astype(float))
            valids.append(np.ones(is_missing.shape[0], dtype=bool))

    if not columns:  # pragma: no cover - validation requires >=1 confounder
        raise ValueError("No covariate terms to compare.")
    return names, levels, np.column_stack(columns), np.column_stack(valids)


def _group_stats(
    Z: np.ndarray, valid: np.ndarray, mask: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-term (raw_mean, weighted_mean, sd) inside one arm, missing-aware.

    ``sd`` is the *unweighted* ddof=1 sample standard deviation computed on the
    observed entries of the term — the pooled reference SD that weights cannot
    move. Terms with fewer than 2 observed entries yield NaN.

    Numerics: the variance is computed two-pass, ``ss = sum (x - mean)^2``,
    never as ``sum x^2 - n*mean^2`` — the latter cancels catastrophically when
    the data carry a large offset (e.g. values around 1e9), turning a real SMD
    into ``inf``/``NaN``. The caller additionally centres every column before
    calling this, so the means themselves are also computed near zero.
    """
    Zg = Z[mask]
    Vg = valid[mask]
    wg = w[mask]
    n_obs = Vg.sum(axis=0).astype(float)

    Zv = Zg * Vg
    raw_sum = Zv.sum(axis=0)
    wsum_obs = wg @ Vg
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_mean = np.where(n_obs > 0, raw_sum / n_obs, np.nan)
        w_mean = np.where(wsum_obs > 0, (wg @ Zv) / wsum_obs, np.nan)
        centered = np.where(Vg, Zg - np.where(n_obs > 0, raw_mean, 0.0), 0.0)
        ss = (centered * centered).sum(axis=0)
        sd = np.where(n_obs > 1, np.sqrt(np.maximum(ss, 0.0) / (n_obs - 1)), np.nan)
    return raw_mean, w_mean, sd


def _smd(
    diff: float, sd_a: float, sd_b: float
) -> tuple[float, bool]:
    """Return (SMD, degenerate_flag)."""
    if not (np.isfinite(sd_a) and np.isfinite(sd_b)):
        return float("nan"), True
    ref = float(np.sqrt((sd_a**2 + sd_b**2) / 2.0))
    if ref == 0.0:
        if diff == 0.0:
            return 0.0, False
        return float(np.sign(diff) * np.inf), True
    return diff / ref, False


def standardized_mean_differences(
    X: pd.DataFrame,
    t_idx: np.ndarray,
    labels: Sequence[Any],
    weights: np.ndarray | None,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    config: CausalRankConfig,
) -> BalanceResult:
    """Compute unweighted/weighted SMDs for every term and treatment pair."""
    t_idx = np.asarray(t_idx, dtype=np.int64)
    names, levels, Z, valid = _term_matrix(X, numeric_features, categorical_features)
    n, p = Z.shape
    k = len(labels)
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)

    # Centre every term on its overall valid mean: SMD is shift-invariant by
    # definition, and centring keeps both means and variances numerically
    # stable for data with large offsets (e.g. timestamps in seconds).
    n_valid = valid.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        offset = np.where(n_valid > 0, (Z * valid).sum(axis=0) / n_valid, 0.0)
    Z = np.where(valid, Z - offset, 0.0)

    detail_rows: list[dict[str, Any]] = []
    degenerate = False

    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    for a, b in pairs:
        mask_a, mask_b = t_idx == a, t_idx == b
        w_a, w_b = w[mask_a], w[mask_b]
        wsum_a, wsum_b = float(w_a.sum()), float(w_b.sum())
        if wsum_a <= 0 or wsum_b <= 0:  # pragma: no cover - Hajek weights are positive
            raise ValueError("Zero total weight inside an arm; cannot compute balance.")

        mean_a_raw, mean_a_w, sd_a = _group_stats(Z, valid, mask_a, w)
        mean_b_raw, mean_b_w, sd_b = _group_stats(Z, valid, mask_b, w)

        pair_name = f"{labels[a]} vs {labels[b]}"
        for j in range(p):
            smd_u, deg_u = _smd(float(mean_a_raw[j] - mean_b_raw[j]), float(sd_a[j]), float(sd_b[j]))
            smd_w, deg_w = _smd(float(mean_a_w[j] - mean_b_w[j]), float(sd_a[j]), float(sd_b[j]))
            degenerate = degenerate or deg_u or deg_w
            detail_rows.append(
                {
                    "feature": names[j],
                    "level": levels[j],
                    "pair": pair_name,
                    "mean_a": float(mean_a_raw[j] + offset[j]),
                    "mean_b": float(mean_b_raw[j] + offset[j]),
                    "mean_a_weighted": float(mean_a_w[j] + offset[j]),
                    "mean_b_weighted": float(mean_b_w[j] + offset[j]),
                    "sd_ref": float(np.sqrt((sd_a[j] ** 2 + sd_b[j] ** 2) / 2.0))
                    if np.isfinite(sd_a[j]) and np.isfinite(sd_b[j])
                    else float("nan"),
                    "smd_unweighted": smd_u,
                    "smd_weighted": smd_w,
                }
            )

    detail = pd.DataFrame(detail_rows)

    summary_rows: list[dict[str, Any]] = []
    for (feature, level), grp in detail.groupby(["feature", "level"], dropna=False, sort=False):
        su = grp["smd_unweighted"].abs()
        sw = grp["smd_weighted"].abs()
        summary_rows.append(
            {
                "feature": feature,
                "level": level,
                "max_abs_smd_unweighted": float(su.max(skipna=True)),
                "max_abs_smd_weighted": float(sw.max(skipna=True)),
                "n_pairs_over_threshold": int((sw > config.smd_threshold).sum()),
                "n_pairs_over_severe": int((sw > config.smd_severe_threshold).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values("max_abs_smd_weighted", ascending=False).reset_index(drop=True)

    warnings: list[DiagnosticWarning] = []
    if degenerate:
        warnings.append(
            DiagnosticWarning(
                code="degenerate_smd",
                level="warning",
                message=(
                    "Some covariate terms have zero within-arm variance, so their SMD is 0 or "
                    "infinite by convention. Check constant/rare features manually."
                ),
                details={},
            )
        )
    if not summary.empty:
        max_w = float(summary["max_abs_smd_weighted"].max(skipna=True))
        max_u = float(summary["max_abs_smd_unweighted"].max(skipna=True))
        if max_w > config.smd_severe_threshold:
            warnings.append(
                DiagnosticWarning(
                    code="severe_imbalance",
                    level="critical",
                    message=(
                        f"Residual imbalance after weighting: max |SMD| = {max_w:.3f} "
                        f"(> {config.smd_severe_threshold}). At least one covariate remains "
                        "strongly imbalanced across treatments; the adjusted means may still be "
                        "confounded. Do NOT present these estimates as a statistically "
                        "established ranking."
                    ),
                    details={"max_abs_smd_weighted": max_w, "max_abs_smd_unweighted": max_u},
                )
            )
        elif max_w > config.smd_threshold:
            warnings.append(
                DiagnosticWarning(
                    code="residual_imbalance",
                    level="warning",
                    message=(
                        f"Residual imbalance after weighting: max |SMD| = {max_w:.3f} "
                        f"(> {config.smd_threshold}). The threshold is a screening convention, "
                        "not a guarantee of unbiasedness."
                    ),
                    details={"max_abs_smd_weighted": max_w, "max_abs_smd_unweighted": max_u},
                )
            )

    return BalanceResult(detail=detail, summary=summary, warnings=warnings)
