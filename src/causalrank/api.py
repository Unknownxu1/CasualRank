"""High-level API: ``CausalRank(...).fit(df) -> CausalRankResult``.

Pipeline (Phase 1):

1. schema / label / missingness validation (:mod:`causalrank.validation`);
2. multi-class GBM propensity scores with in-fold preprocessing and optional
   out-of-fold cross-fitting (:mod:`causalrank.estimators.propensity`);
3. Hajek IPTW estimates ``mu_a = E[Y(a)]`` on the common target population
   (:mod:`causalrank.estimators.iptw`);
4. weight / overlap / balance diagnostics with quality-gate warnings
   (:mod:`causalrank.diagnostics`).

Uncertainty quantification (joint bootstrap covariance, simultaneous pairwise
intervals, rank confidence sets) is *not* part of this release; the result
object exposes the reserved fields but reports ``inference_available=False``.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ._version import __version__
from .config import CausalRankConfig
from .diagnostics.balance import standardized_mean_differences
from .diagnostics.overlap import overlap_diagnostics
from .diagnostics.weights import weight_diagnostics
from .estimators.iptw import hajek_estimate
from .estimators.propensity import fit_propensity
from .results import CausalRankResult, DiagnosticsReport, EstimatesTable
from .validation import ValidatedData, validate_dataframe


class CausalRank:
    """Estimate multi-treatment potential-outcome means from observational data.

    Parameters mirror :class:`causalrank.config.CausalRankConfig`; see that
    class for the exact meaning of every threshold. Phase 1 supports exactly
    ``estimator="iptw"`` with ``propensity_model="gbm"``,
    ``estimand="population_mean"`` and ``weighting="hajek"``.
    """

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: Sequence[str],
        estimator: str = "iptw",
        propensity_model: str = "gbm",
        estimand: str = "population_mean",
        weighting: str = "hajek",
        confidence_level: float = 0.95,
        n_bootstrap: int = 300,
        n_jobs: int = 1,
        random_state: int | None = None,
        cross_fit: int | None = None,
        clip_propensity: float | None = None,
        gbm_params: dict[str, Any] | None = None,
        impute_numeric: str = "median",
        impute_categorical: str = "most_frequent",
        categorical_fill_value: str = "missing",
        max_category_cardinality: int = 25,
        min_group_n: int = 30,
        min_ess: float = 30.0,
        min_ess_ratio: float = 0.2,
        min_propensity_warn: float = 0.01,
        max_weight_share_warn: float = 0.1,
        smd_threshold: float = 0.1,
        smd_severe_threshold: float = 0.25,
        quantiles: tuple[float, ...] = (0.01, 0.05, 0.5, 0.95, 0.99),
    ) -> None:
        self.config = CausalRankConfig(
            treatment=treatment,
            outcome=outcome,
            confounders=tuple(confounders),
            estimator=estimator,
            propensity_model=propensity_model,
            estimand=estimand,
            weighting=weighting,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            cross_fit=cross_fit,
            clip_propensity=clip_propensity,
            gbm_params=gbm_params,
            impute_numeric=impute_numeric,
            impute_categorical=impute_categorical,
            categorical_fill_value=categorical_fill_value,
            max_category_cardinality=max_category_cardinality,
            min_group_n=min_group_n,
            min_ess=min_ess,
            min_ess_ratio=min_ess_ratio,
            min_propensity_warn=min_propensity_warn,
            max_weight_share_warn=max_weight_share_warn,
            smd_threshold=smd_threshold,
            smd_severe_threshold=smd_severe_threshold,
            quantiles=tuple(quantiles),
        )
        self.result_: CausalRankResult | None = None

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> CausalRankResult:
        """Run validation, propensity estimation, IPTW and diagnostics on ``df``."""
        config = self.config
        validated = validate_dataframe(df, config)

        prop_fit = fit_propensity(
            X=validated.X,
            t_idx=validated.t_idx,
            labels=validated.labels,
            numeric_features=validated.numeric_features,
            categorical_features=validated.categorical_features,
            config=config,
        )

        estimate = hajek_estimate(
            y=validated.y,
            t_idx=validated.t_idx,
            propensity=prop_fit.propensity,
            labels=validated.labels,
            clip_propensity=config.clip_propensity,
        )

        # ---- diagnostics -------------------------------------------------
        weights_df, weight_warn = weight_diagnostics(estimate, config)
        overlap_df, overlap_warn = overlap_diagnostics(
            prop_fit.propensity, validated.t_idx, validated.labels, config
        )
        balance = standardized_mean_differences(
            X=validated.X,
            t_idx=validated.t_idx,
            labels=validated.labels,
            weights=estimate.weights,
            numeric_features=validated.numeric_features,
            categorical_features=validated.categorical_features,
            config=config,
        )
        group_summary = pd.DataFrame(
            {
                "label": list(validated.labels),
                "n": np.asarray(validated.group_counts, dtype=int),
                "share": np.asarray(validated.group_counts, dtype=float) / validated.n_rows,
                "raw_outcome_mean": np.asarray(validated.raw_outcome_means, dtype=float),
                "mu_hajek_iptw": np.asarray(estimate.mu, dtype=float),
            }
        )

        warnings = [
            *validated.warnings,
            *prop_fit.warnings,
            *weight_warn,
            *overlap_warn,
            *balance.warnings,
        ]
        diagnostics = DiagnosticsReport(
            group_summary=group_summary,
            weights=weights_df,
            overlap=overlap_df,
            balance=balance.detail,
            balance_summary=balance.summary,
            warnings=warnings,
        )

        effects = EstimatesTable(
            labels=validated.labels,
            n_treated=np.asarray(estimate.n_treated, dtype=np.int64),
            raw_mean=np.asarray(estimate.raw_means, dtype=float),
            weight_sum=np.asarray(estimate.weight_sums, dtype=float),
            ess=np.asarray(estimate.ess, dtype=float),
            mu=np.asarray(estimate.mu, dtype=float),
            se=None,
            ci_low=None,
            ci_high=None,
        )

        metadata = self._build_metadata(validated, prop_fit, diagnostics)

        self.result_ = CausalRankResult(
            effects=effects,
            diagnostics=diagnostics,
            metadata=metadata,
            covariance=None,
            pairwise=None,
            ranks=None,
            inference_available=False,
        )
        return self.result_

    # ------------------------------------------------------------------
    def _build_metadata(
        self,
        validated: ValidatedData,
        prop_fit: Any,
        diagnostics: DiagnosticsReport,
    ) -> dict[str, Any]:
        config = self.config
        import scipy
        import sklearn

        return {
            "package": "causalrank",
            "version": __version__,
            "created_at": datetime.now(UTC).isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "scikit-learn": sklearn.__version__,
            },
            "estimand": config.estimand,
            "estimand_formula": "mu_a = E[Y(a)] over the sample population",
            "estimator": config.estimator,
            "estimator_formula": (
                "mu_a = sum_i 1{T_i=a} Y_i / e_{T_i}(X_i) / sum_i 1{T_i=a} 1/e_{T_i}(X_i) "
                "(Hajek-normalised IPTW)"
            ),
            "propensity_model": config.propensity_model,
            "propensity_formula": "e_a(x) = P(T = a | X = x), multi-class GBM",
            "weighting": config.weighting,
            "n_rows": validated.n_rows,
            "n_treatments": validated.n_treatments,
            "labels": list(validated.labels),
            "group_counts": {
                str(lab): int(validated.group_counts[k]) for k, lab in enumerate(validated.labels)
            },
            "confounders": list(config.confounders),
            "numeric_features": validated.numeric_features,
            "categorical_features": validated.categorical_features,
            "missing_rates": validated.missing_rates,
            "cross_fit": config.cross_fit,
            "propensity_out_of_fold": bool(prop_fit.oof),
            "n_propensity_models": len(prop_fit.models),
            "clip_propensity": config.clip_propensity,
            "random_state": config.random_state,
            "confidence_level": config.confidence_level,
            "n_bootstrap": config.n_bootstrap,
            "n_jobs": config.n_jobs,
            "config": config.to_dict(),
            "quality_gate": diagnostics.gate,
            "n_warnings": len(diagnostics.warnings),
            "inference_available": False,
            "inference_note": (
                "Phase 1 release: point estimates and diagnostics only. Joint bootstrap "
                "covariance (Phase 2) and rank confidence sets (Phase 3) are not computed; "
                "se/ci/ranks are NaN/None by design."
            ),
            "causal_caveat": (
                "Interpretation as causal requires consistency, no unmeasured confounding, "
                "positivity and iid sampling. These assumptions cannot be verified from data; "
                "the diagnostics only flag empirical problems."
            ),
        }
