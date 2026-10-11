"""Weight-distribution and effective-sample-size diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..config import CausalRankConfig
from ..estimators.iptw import IPTWEstimate
from ..results import DiagnosticWarning


def weight_diagnostics(
    estimate: IPTWEstimate,
    config: CausalRankConfig,
) -> tuple[pd.DataFrame, list[DiagnosticWarning]]:
    """Per-treatment weight distribution, ESS and extreme-weight warnings.

    Columns: raw counts, weight sum, ESS, ESS ratio, weight quantiles
    (min / configured quantiles / max) and the share of the arm's total weight
    carried by its single largest weight.
    """
    w = estimate.weights
    t_idx = estimate.t_idx
    labels: Sequence[Any] = estimate.labels
    rows: list[dict[str, Any]] = []
    warnings: list[DiagnosticWarning] = []
    quantiles = list(config.quantiles)

    for a, label in enumerate(labels):
        mask = t_idx == a
        w_a = w[mask]
        ess = float(estimate.ess[a])
        n_a = int(w_a.size)
        q_vals = np.quantile(w_a, quantiles)
        max_share = float(w_a.max() / w_a.sum()) if w_a.sum() > 0 else np.nan
        row: dict[str, Any] = {
            "label": label,
            "n": n_a,
            "weight_sum": float(w_a.sum()),
            "ess": ess,
            "ess_ratio": ess / n_a if n_a else np.nan,
            "w_min": float(w_a.min()),
            "w_max": float(w_a.max()),
            "max_weight_share": max_share,
        }
        for q, val in zip(quantiles, q_vals, strict=True):
            row[f"w_p{round(q * 100):02d}"] = float(val)
        rows.append(row)

        if ess < config.min_ess:
            warnings.append(
                DiagnosticWarning(
                    code="low_ess",
                    level="warning",
                    message=(
                        f"Treatment {label!r}: effective sample size ESS = {ess:.1f} "
                        f"(< min_ess = {config.min_ess}). The weighted mean for this arm is "
                        "dominated by a few observations."
                    ),
                    details={"label": label, "ess": ess, "n": n_a},
                )
            )
        if n_a and ess / n_a < config.min_ess_ratio:
            warnings.append(
                DiagnosticWarning(
                    code="low_ess",
                    level="warning",
                    message=(
                        f"Treatment {label!r}: ESS ratio = {ess / n_a:.3f} "
                        f"(< min_ess_ratio = {config.min_ess_ratio}); extreme weights are "
                        "concentrating the estimate."
                    ),
                    details={"label": label, "ess_ratio": ess / n_a, "n": n_a},
                )
            )
        if max_share > config.max_weight_share_warn:
            warnings.append(
                DiagnosticWarning(
                    code="extreme_weight",
                    level="warning",
                    message=(
                        f"Treatment {label!r}: the largest weight carries "
                        f"{max_share:.1%} of the arm's total weight "
                        f"(> {config.max_weight_share_warn:.0%}). Check overlap for this arm."
                    ),
                    details={"label": label, "max_weight_share": max_share},
                )
            )

    if estimate.clipped_fraction > 0.0:
        warnings.append(
            DiagnosticWarning(
                code="propensity_clipped",
                level="warning",
                message=(
                    f"{estimate.clipped_fraction:.2%} of observed-treatment propensities were "
                    "clipped (explicit clip_propensity setting). Clipping trades bias for "
                    "variance and changes the weighting scheme; report it with any result."
                ),
                details={"clipped_fraction": estimate.clipped_fraction},
            )
        )

    return pd.DataFrame(rows), warnings


def ess_table(estimate: IPTWEstimate) -> pd.DataFrame:
    """Compact ESS table (kept separate for programmatic use)."""
    n = np.maximum(np.asarray(estimate.n_treated, dtype=float), 1.0)
    return pd.DataFrame(
        {
            "label": list(estimate.labels),
            "n": np.asarray(estimate.n_treated, dtype=int),
            "ess": np.asarray(estimate.ess, dtype=float),
            "ess_ratio": np.asarray(estimate.ess, dtype=float) / n,
        }
    )
