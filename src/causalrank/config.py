"""Configuration object for the CausalRank estimator.

Every knob that affects estimation, diagnostics or quality gating lives here so
that a run is fully described by ``CausalRankConfig.to_dict()`` (recorded in
``result.metadata`` for reproducibility).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .exceptions import CausalRankValidationError

SUPPORTED_ESTIMATORS: tuple[str, ...] = ("iptw",)
SUPPORTED_PROPENSITY_MODELS: tuple[str, ...] = ("gbm",)
SUPPORTED_ESTIMANDS: tuple[str, ...] = ("population_mean",)
SUPPORTED_WEIGHTING: tuple[str, ...] = ("hajek",)
SUPPORTED_NUMERIC_IMPUTE: tuple[str, ...] = ("median", "mean", "constant")
SUPPORTED_CATEGORICAL_IMPUTE: tuple[str, ...] = ("most_frequent", "constant")

_DEFAULT_GBM_PARAMS: dict[str, Any] = {
    # Deliberately regularised defaults: for propensity scores we want a smooth
    # approximation of P(T=a|X), not a label-noise chaser. Deep/low-leaf trees
    # produce over-confident probabilities, which (a) inflates in-sample
    # propensities and (b) creates spuriously tiny out-of-fold propensities and
    # therefore explosive IPTW weights. Tuned against the synthetic DGP where
    # the true propensity surface is known (see docs/methodology.md).
    "max_iter": 400,
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_samples_leaf": 60,
    "l2_regularization": 2.0,
}


@dataclass(frozen=True)
class CausalRankConfig:
    """Fully-specified configuration for one analysis.

    Notes
    -----
    ``estimator``, ``propensity_model``, ``estimand`` and ``weighting`` are
    restricted to the values implemented and validated in Phase 1. The names
    are kept distinct on purpose: *GBM* is the prediction model used for the
    propensity score, *IPTW* is the causal estimation method that consumes it,
    and ``weighting="hajek"`` is the group-normalised IPTW variant (it does not
    change the estimand, unlike overlap weighting).
    """

    treatment: str
    outcome: str
    confounders: tuple[str, ...]

    # --- method selection (Phase 1 supports exactly one combination) ---
    estimator: str = "iptw"
    propensity_model: str = "gbm"
    estimand: str = "population_mean"
    weighting: str = "hajek"

    # --- inference scaffolding (used from Phase 2 onward) ---
    confidence_level: float = 0.95
    n_bootstrap: int = 300
    n_jobs: int = 1

    # --- randomness ---
    random_state: int | None = None

    # --- propensity estimation ---
    cross_fit: int | None = None
    clip_propensity: float | None = None
    gbm_params: dict[str, Any] | None = None
    impute_numeric: str = "median"
    impute_categorical: str = "most_frequent"
    categorical_fill_value: str = "missing"
    max_category_cardinality: int = 25

    # --- quality gates ---
    min_group_n: int = 30
    min_ess: float = 30.0
    min_ess_ratio: float = 0.2
    min_propensity_warn: float = 0.01
    max_weight_share_warn: float = 0.1
    smd_threshold: float = 0.1
    smd_severe_threshold: float = 0.25
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.5, 0.95, 0.99)

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Validate the configuration; raise ``CausalRankValidationError``."""
        for name, value in (("treatment", self.treatment), ("outcome", self.outcome)):
            if not isinstance(value, str) or not value.strip():
                raise CausalRankValidationError(f"'{name}' must be a non-empty column name.")

        if not isinstance(self.confounders, tuple):
            raise CausalRankValidationError("'confounders' must be a tuple of column names.")
        if len(self.confounders) == 0:
            raise CausalRankValidationError(
                "At least one confounder column is required. CausalRank estimates a "
                "covariate-adjusted contrast; an empty confounder list is not supported."
            )
        for col in self.confounders:
            if not isinstance(col, str) or not col.strip():
                raise CausalRankValidationError("Every confounder must be a non-empty column name.")
        if len(set(self.confounders)) != len(self.confounders):
            raise CausalRankValidationError("Duplicate names in 'confounders'.")
        overlap = set(self.confounders) & {self.treatment, self.outcome}
        if overlap:
            raise CausalRankValidationError(
                f"Columns {sorted(overlap)} cannot be both a confounder and the treatment/outcome."
            )

        if self.estimator not in SUPPORTED_ESTIMATORS:
            raise CausalRankValidationError(
                f"estimator={self.estimator!r} is not supported in Phase 1. "
                f"Supported: {SUPPORTED_ESTIMATORS}. AIPW / overlap weighting / matching are "
                "explicitly out of scope for v0.1 (see PROJECT_SPEC.md section 6)."
            )
        if self.propensity_model not in SUPPORTED_PROPENSITY_MODELS:
            raise CausalRankValidationError(
                f"propensity_model={self.propensity_model!r} is not supported. "
                f"Supported: {SUPPORTED_PROPENSITY_MODELS} (multi-class gradient boosting)."
            )
        if self.estimand not in SUPPORTED_ESTIMANDS:
            raise CausalRankValidationError(
                f"estimand={self.estimand!r} is not supported. Supported: {SUPPORTED_ESTIMANDS}."
            )
        if self.weighting not in SUPPORTED_WEIGHTING:
            raise CausalRankValidationError(
                f"weighting={self.weighting!r} is not supported. Supported: {SUPPORTED_WEIGHTING}. "
                "The Hajek (group-normalised) IPTW form is the validated Phase 1 estimator; "
                "Horvitz-Thompson is kept internal for numerical cross-checks only."
            )

        if not (0.0 < float(self.confidence_level) < 1.0):
            raise CausalRankValidationError("confidence_level must lie in (0, 1).")
        if int(self.n_bootstrap) < 1:
            raise CausalRankValidationError("n_bootstrap must be >= 1.")
        if int(self.n_jobs) < 1:
            raise CausalRankValidationError("n_jobs must be >= 1.")
        if self.random_state is not None and not isinstance(self.random_state, int):
            raise CausalRankValidationError("random_state must be an int or None.")

        if self.cross_fit is not None:
            if not isinstance(self.cross_fit, int) or isinstance(self.cross_fit, bool):
                raise CausalRankValidationError("cross_fit must be an int >= 2 or None.")
            if self.cross_fit < 2:
                raise CausalRankValidationError("cross_fit must be >= 2 (or None to disable).")
        if self.clip_propensity is not None:
            eps = float(self.clip_propensity)
            if not (0.0 < eps < 0.5):
                raise CausalRankValidationError("clip_propensity must lie in (0, 0.5) or be None.")

        if self.impute_numeric not in SUPPORTED_NUMERIC_IMPUTE:
            raise CausalRankValidationError(
                f"impute_numeric must be one of {SUPPORTED_NUMERIC_IMPUTE}."
            )
        if self.impute_categorical not in SUPPORTED_CATEGORICAL_IMPUTE:
            raise CausalRankValidationError(
                f"impute_categorical must be one of {SUPPORTED_CATEGORICAL_IMPUTE}."
            )
        if int(self.max_category_cardinality) < 2:
            raise CausalRankValidationError("max_category_cardinality must be >= 2.")

        if int(self.min_group_n) < 2:
            raise CausalRankValidationError("min_group_n must be >= 2.")
        if float(self.min_ess) <= 0 or not (0.0 < float(self.min_ess_ratio) <= 1.0):
            raise CausalRankValidationError("min_ess must be > 0 and min_ess_ratio in (0, 1].")
        if not (0.0 < float(self.min_propensity_warn) < 0.5):
            raise CausalRankValidationError("min_propensity_warn must lie in (0, 0.5).")
        if not (0.0 < float(self.max_weight_share_warn) <= 1.0):
            raise CausalRankValidationError("max_weight_share_warn must lie in (0, 1].")
        if not (0.0 < float(self.smd_threshold) < float(self.smd_severe_threshold)):
            raise CausalRankValidationError(
                "Require 0 < smd_threshold < smd_severe_threshold."
            )
        if not self.quantiles or any(not (0.0 < q < 1.0) for q in self.quantiles):
            raise CausalRankValidationError("quantiles must be values strictly inside (0, 1).")
        if tuple(sorted(self.quantiles)) != tuple(self.quantiles):
            raise CausalRankValidationError("quantiles must be sorted ascending.")

        if self.gbm_params is not None:
            if not isinstance(self.gbm_params, dict):
                raise CausalRankValidationError("gbm_params must be a dict or None.")
            forbidden = {"random_state"}
            clash = forbidden & set(self.gbm_params)
            if clash:
                raise CausalRankValidationError(
                    f"gbm_params must not override {sorted(clash)}; use config.random_state."
                )

    # ------------------------------------------------------------------
    @property
    def resolved_gbm_params(self) -> dict[str, Any]:
        """Default GBM hyper-parameters merged with user overrides."""
        params = dict(_DEFAULT_GBM_PARAMS)
        if self.gbm_params:
            params.update(self.gbm_params)
        params.setdefault("random_state", self.random_state)
        return params

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict for metadata / reproducibility records."""
        raw = asdict(self)
        raw["confounders"] = list(self.confounders)
        raw["quantiles"] = list(self.quantiles)
        return raw


def default_gbm_params() -> dict[str, Any]:
    """Return a copy of the built-in GBM hyper-parameters."""
    return dict(_DEFAULT_GBM_PARAMS)
