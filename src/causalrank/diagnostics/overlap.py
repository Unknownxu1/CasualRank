"""Overlap / common-support diagnostics from the estimated propensity scores."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..config import CausalRankConfig
from ..results import DiagnosticWarning


def overlap_diagnostics(
    propensity: np.ndarray,
    t_idx: np.ndarray,
    labels: Sequence[Any],
    config: CausalRankConfig,
) -> tuple[pd.DataFrame, list[DiagnosticWarning]]:
    """Describe how well the arms share common support.

    For every arm ``a`` this reports

    * the distribution of ``e_a(X)`` over **all** rows (the arm's reach in the
      covariate space — the quantity that must stay positive for positivity);
    * the distribution of ``e_{T}(X)`` over the rows actually treated with ``a``
      (the denominator of the IPTW weights);
    * the share of treated rows whose own propensity falls below
      ``config.min_propensity_warn``.

    Overlap is an empirical diagnostic only: a clean table does not prove
    positivity in the population, and a bad table cannot be repaired by
    trimming without changing the estimand.
    """
    propensity = np.asarray(propensity, dtype=float)
    t_idx = np.asarray(t_idx, dtype=np.int64)
    quantiles = list(config.quantiles)
    rows: list[dict[str, Any]] = []
    warnings: list[DiagnosticWarning] = []

    for a, label in enumerate(labels):
        e_arm = propensity[:, a]
        e_obs = propensity[t_idx == a, a]
        q_arm = np.quantile(e_arm, quantiles)
        q_obs = np.quantile(e_obs, quantiles)
        frac_low = float(np.mean(e_obs < config.min_propensity_warn))
        row: dict[str, Any] = {
            "label": label,
            "e_arm_min": float(e_arm.min()),
            "e_arm_max": float(e_arm.max()),
            "e_obs_min": float(e_obs.min()),
            "e_obs_median": float(np.median(e_obs)),
            "frac_obs_below_threshold": frac_low,
        }
        for q, v_arm, v_obs in zip(quantiles, q_arm, q_obs, strict=True):
            tag = f"p{round(q * 100):02d}"
            row[f"e_arm_{tag}"] = float(v_arm)
            row[f"e_obs_{tag}"] = float(v_obs)
        rows.append(row)

        if float(e_obs.min()) < config.min_propensity_warn:
            warnings.append(
                DiagnosticWarning(
                    code="low_propensity",
                    level="warning",
                    message=(
                        f"Treatment {label!r}: minimum propensity for the observed treatment is "
                        f"{float(e_obs.min()):.4g} (< {config.min_propensity_warn}). Positivity is "
                        "weakly satisfied; the corresponding IPTW weights are extreme and the "
                        "estimate for this arm is extrapolation-heavy."
                    ),
                    details={
                        "label": label,
                        "min_observed_propensity": float(e_obs.min()),
                        "frac_below_threshold": frac_low,
                    },
                )
            )

    # Global diagnostic: any row with a near-deterministic treatment assignment.
    max_e = propensity.max(axis=1)
    frac_dominant = float(np.mean(max_e > 1.0 - config.min_propensity_warn))
    if frac_dominant > 0.05:
        warnings.append(
            DiagnosticWarning(
                code="low_propensity",
                level="warning",
                message=(
                    f"{frac_dominant:.1%} of rows have a near-deterministic treatment assignment "
                    f"(max_a e_a(X) > {1.0 - config.min_propensity_warn:.2f}). Weak common support "
                    "makes IPTW estimates for the non-dominant arms unstable and largely "
                    "model-driven."
                ),
                details={"frac_near_deterministic": frac_dominant},
            )
        )

    return pd.DataFrame(rows), warnings
