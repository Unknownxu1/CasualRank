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
      covariate space — the quantity that must stay positive for positivity),
      including ``frac_arm_below_threshold``: the share of the *entire target
      sample* where ``e_a(X)`` falls below ``config.min_propensity_warn``;
    * the distribution of ``e_{T}(X)`` over the rows actually treated with ``a``
      (the denominator of the IPTW weights);
    * the share of treated rows whose own propensity falls below
      ``config.min_propensity_warn``.

    Warning policy (PROJECT_SPEC.md 3.4 / 8):

    * ``weak_common_support`` (warning) when an arm's unsupported share of the
      target sample exceeds ``config.max_unsupported_share`` — the
      population-mean estimand for that arm then rests on extrapolation, even
      if every *treated* row happens to have a moderate propensity;
    * ``no_common_support`` (critical) when some row has ``e_a(X) = 0``
      exactly: that region of the covariate space never receives arm ``a``, so
      ``E[Y(a)]`` is not identified there;
    * ``low_propensity`` (warning) for near-deterministic treatment assignment.

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
        frac_low_obs = float(np.mean(e_obs < config.min_propensity_warn))
        frac_low_arm = float(np.mean(e_arm < config.min_propensity_warn))
        n_unsupported = int(np.sum(e_arm == 0.0))
        row: dict[str, Any] = {
            "label": label,
            "e_arm_min": float(e_arm.min()),
            "e_arm_max": float(e_arm.max()),
            "e_obs_min": float(e_obs.min()),
            "e_obs_median": float(np.median(e_obs)),
            "frac_arm_below_threshold": frac_low_arm,
            "frac_obs_below_threshold": frac_low_obs,
        }
        for q, v_arm, v_obs in zip(quantiles, q_arm, q_obs, strict=True):
            tag = f"p{round(q * 100):02d}"
            row[f"e_arm_{tag}"] = float(v_arm)
            row[f"e_obs_{tag}"] = float(v_obs)
        rows.append(row)

        if n_unsupported > 0:
            warnings.append(
                DiagnosticWarning(
                    code="no_common_support",
                    level="critical",
                    message=(
                        f"Treatment {label!r}: e_a(X) = 0 for {n_unsupported} row(s) of the "
                        "target sample. Positivity is violated in those covariate regions: "
                        "E[Y(a)] is not identified there and the weighted mean for this arm "
                        "extrapolates from other regions. Do NOT report a causal mean for "
                        "this arm without restricting the target population."
                    ),
                    details={
                        "label": label,
                        "n_zero_propensity": n_unsupported,
                        "share_zero": float(n_unsupported) / max(e_arm.size, 1),
                    },
                )
            )
        elif frac_low_arm > config.max_unsupported_share:
            warnings.append(
                DiagnosticWarning(
                    code="weak_common_support",
                    level="warning",
                    message=(
                        f"Treatment {label!r}: {frac_low_arm:.1%} of the target sample has "
                        f"e_a(X) < {config.min_propensity_warn} "
                        f"(> max_unsupported_share = {config.max_unsupported_share:.0%}). "
                        "The population mean for this arm is estimated from a pseudo-population "
                        "that largely lacks observations in those regions — weak common support "
                        "for the population_mean estimand."
                    ),
                    details={
                        "label": label,
                        "frac_arm_below_threshold": frac_low_arm,
                        "min_arm_propensity": float(e_arm.min()),
                    },
                )
            )

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
                        "frac_below_threshold": frac_low_obs,
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
