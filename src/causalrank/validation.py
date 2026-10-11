"""Input validation, treatment-label normalisation and pre-flight summaries.

This module is the single place where raw user data is checked against the
Phase 1 contract (PROJECT_SPEC.md sections 3.1 and 7.1). It never estimates
anything; it produces a :class:`ValidatedData` bundle consumed by the
estimators and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import CausalRankConfig
from .exceptions import CausalRankValidationError
from .results import DiagnosticWarning


@dataclass
class ValidatedData:
    """Clean, schema-checked analysis frame."""

    X: pd.DataFrame
    t_idx: np.ndarray  # int64 (n,) codes in canonical label order
    y: np.ndarray  # float64 (n,) values in {0.0, 1.0}
    labels: tuple[Any, ...]  # canonical (sorted) treatment labels
    label_to_index: dict[Any, int]
    group_counts: np.ndarray  # int64 (K,)
    raw_outcome_means: np.ndarray  # float64 (K,)
    numeric_features: list[str]
    categorical_features: list[str]
    missing_rates: dict[str, float]
    warnings: list[DiagnosticWarning] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_treatments(self) -> int:
        return len(self.labels)


# ---------------------------------------------------------------------------
# treatment labels
# ---------------------------------------------------------------------------
def _normalize_treatment_labels(t_raw: pd.Series) -> tuple[list[Any], str]:
    """Return homogeneous, hashable labels plus a description of any casting.

    Accepted label types: ``int`` (including integral floats, cast to ``int``)
    and ``str``. Booleans, mixed types and anything else are rejected: a
    deterministic label order is required so that propensity-score columns can
    be aligned with the public treatment order.
    """
    if isinstance(t_raw.dtype, pd.CategoricalDtype):
        values = t_raw.astype(object).to_numpy().tolist()
    else:
        values = np.asarray(t_raw, dtype=object).tolist()

    kinds: set[str] = set()
    normalized: list[Any] = []
    saw_float = False
    for v in values:
        if isinstance(v, (bool, np.bool_)):
            kinds.add("bool")
            normalized.append(v)
        elif isinstance(v, (int, np.integer)):
            kinds.add("int")
            normalized.append(int(v))
        elif isinstance(v, (float, np.floating)):
            kinds.add("float")
            if not float(v).is_integer():
                raise CausalRankValidationError(
                    "Treatment labels must be integers or strings; found a non-integral "
                    f"float label {v!r}."
                )
            saw_float = True
            normalized.append(int(v))
        elif isinstance(v, str):
            kinds.add("str")
            normalized.append(v)
        else:
            raise CausalRankValidationError(
                f"Unsupported treatment label type {type(v).__name__!r} (value {v!r}). "
                "Treatment labels must be integers or strings."
            )

    if "bool" in kinds:
        raise CausalRankValidationError(
            "Boolean treatment labels are ambiguous; use integer or string labels."
        )
    if kinds == {"str"}:
        return normalized, "str"
    if kinds <= {"int", "float"}:
        cast_note = "int_from_float" if saw_float else "int"
        return normalized, cast_note
    raise CausalRankValidationError(
        "Mixed treatment label types are not allowed "
        f"(found kinds: {sorted(kinds)}). Use one homogeneous type."
    )


def canonical_label_order(t_raw: pd.Series) -> tuple[Any, ...]:
    """Deterministic (sorted) unique treatment labels."""
    normalized, _ = _normalize_treatment_labels(t_raw)
    return tuple(sorted(set(normalized)))


# ---------------------------------------------------------------------------
# outcome
# ---------------------------------------------------------------------------
def _normalize_outcome(y_raw: pd.Series) -> np.ndarray:
    if y_raw.isna().any():
        n_missing = int(y_raw.isna().sum())
        raise CausalRankValidationError(
            f"Outcome column contains {n_missing} missing value(s). Phase 1 rejects missing "
            "outcomes explicitly instead of silently dropping rows."
        )
    if isinstance(y_raw.dtype, pd.CategoricalDtype) or y_raw.dtype == object:
        raise CausalRankValidationError(
            "Outcome must be numeric 0/1 (or boolean); got a non-numeric dtype "
            f"{y_raw.dtype!r}."
        )
    if isinstance(y_raw.dtype, pd.BooleanDtype) or y_raw.dtype == bool:
        return y_raw.to_numpy(dtype=float)
    try:
        y = y_raw.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:  # pragma: no cover - dtype guard above
        raise CausalRankValidationError(f"Outcome must be numeric 0/1: {exc}") from exc
    bad = ~np.isin(y, (0.0, 1.0))
    if bad.any():
        sample = list(np.unique(y[bad]))[:5]
        raise CausalRankValidationError(
            "Outcome must be binary 0/1 in Phase 1; found other values such as "
            f"{sample}. Continuous/count outcomes are a later extension."
        )
    return y.astype(float)


# ---------------------------------------------------------------------------
# covariates
# ---------------------------------------------------------------------------
def _feature_kinds(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for col in X.columns:
        s = X[col]
        if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or s.dtype == bool or isinstance(s.dtype, pd.BooleanDtype):
            categorical.append(str(col))
        elif pd.api.types.is_numeric_dtype(s):
            numeric.append(str(col))
        else:
            raise CausalRankValidationError(
                f"Confounder {col!r} has unsupported dtype {s.dtype!r}. Use numeric dtypes for "
                "continuous covariates and object/category/bool for categorical ones "
                "(e.g. df[col] = df[col].astype('category'))."
            )
    return numeric, categorical


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def validate_dataframe(df: pd.DataFrame, config: CausalRankConfig) -> ValidatedData:
    """Validate ``df`` against ``config`` and return the analysis bundle."""
    if not isinstance(df, pd.DataFrame):
        raise CausalRankValidationError("Input must be a pandas DataFrame.")
    if df.shape[0] == 0:
        raise CausalRankValidationError("Input DataFrame has no rows.")

    required = [config.treatment, config.outcome, *config.confounders]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise CausalRankValidationError(
            f"Missing required column(s) in the input DataFrame: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    warnings: list[DiagnosticWarning] = []

    # ---- treatment ------------------------------------------------------
    t_raw = df[config.treatment]
    if t_raw.isna().any():
        n_missing = int(t_raw.isna().sum())
        raise CausalRankValidationError(
            f"Treatment column {config.treatment!r} contains {n_missing} missing value(s). "
            "Phase 1 rejects missing treatments explicitly."
        )
    normalized, cast_note = _normalize_treatment_labels(t_raw)
    if cast_note == "int_from_float":
        warnings.append(
            DiagnosticWarning(
                code="label_dtype_cast",
                level="info",
                message="Integral float treatment labels were cast to int for a stable order.",
                details={},
            )
        )
    labels = tuple(sorted(set(normalized)))
    if len(labels) < 2:
        raise CausalRankValidationError(
            f"Treatment {config.treatment!r} must contain at least 2 distinct classes; "
            f"found {len(labels)}."
        )
    label_to_index = {lab: i for i, lab in enumerate(labels)}
    t_idx = np.fromiter((label_to_index[v] for v in normalized), dtype=np.int64, count=len(normalized))

    # ---- outcome --------------------------------------------------------
    y = _normalize_outcome(df[config.outcome])

    # ---- covariates -----------------------------------------------------
    X = df.loc[:, list(config.confounders)].copy()
    numeric_features, categorical_features = _feature_kinds(X)

    missing_rates: dict[str, float] = {}
    for col in config.confounders:
        rate = float(X[col].isna().mean())
        missing_rates[str(col)] = rate
        if rate >= 1.0:
            raise CausalRankValidationError(
                f"Confounder {col!r} is entirely missing; drop it or supply values."
            )
    if any(r > 0.0 for r in missing_rates.values()):
        warnings.append(
            DiagnosticWarning(
                code="missing_covariates",
                level="info",
                message=(
                    "Covariates contain missing values; imputation is fitted inside the "
                    "training fold / pipeline only (no pre-fit on the full sample)."
                ),
                details={"missing_rates": {k: v for k, v in missing_rates.items() if v > 0}},
            )
        )

    for col in categorical_features:
        n_unique = int(X[col].nunique(dropna=True))
        if n_unique > config.max_category_cardinality:
            warnings.append(
                DiagnosticWarning(
                    code="high_cardinality_categorical",
                    level="warning",
                    message=(
                        f"Categorical covariate {col!r} has {n_unique} levels "
                        f"(> {config.max_category_cardinality}); one-hot encoding will create "
                        "many columns. Consider grouping rare levels."
                    ),
                    details={"feature": col, "n_levels": n_unique},
                )
            )

    # ---- groups ---------------------------------------------------------
    n_treat = len(labels)
    group_counts = np.bincount(t_idx, minlength=n_treat).astype(np.int64)
    raw_outcome_means = np.array(
        [float(y[t_idx == k].mean()) if group_counts[k] > 0 else np.nan for k in range(n_treat)],
        dtype=float,
    )
    for k, lab in enumerate(labels):
        if group_counts[k] < 2:
            raise CausalRankValidationError(
                f"Treatment {lab!r} has {int(group_counts[k])} observation(s); at least 2 are "
                "required to form a within-group weighted mean."
            )
        if group_counts[k] < config.min_group_n:
            warnings.append(
                DiagnosticWarning(
                    code="small_group",
                    level="warning",
                    message=(
                        f"Treatment {lab!r} has only {int(group_counts[k])} observations "
                        f"(< min_group_n={config.min_group_n}); estimates for this arm are "
                        "unstable."
                    ),
                    details={"label": lab, "n": int(group_counts[k])},
                )
            )
    if int(np.min(group_counts)) < 0.05 * int(np.max(group_counts)):
        warnings.append(
            DiagnosticWarning(
                code="rare_treatment",
                level="warning",
                message=(
                    "Treatment groups are severely unbalanced (smallest group < 5% of the "
                    "largest); propensity estimation and any resampling may be unreliable."
                ),
                details={
                    "counts": {str(lab): int(group_counts[k]) for k, lab in enumerate(labels)}
                },
            )
        )
    if config.cross_fit is not None and int(np.min(group_counts)) < config.cross_fit:
        raise CausalRankValidationError(
            f"cross_fit={config.cross_fit} requires every treatment class to have at least "
            f"{config.cross_fit} observations (smallest group has {int(np.min(group_counts))}). "
            "Reduce cross_fit or collect more data."
        )

    # ---- iid sanity -----------------------------------------------------
    dup_mask = df.duplicated(subset=[config.treatment, config.outcome, *config.confounders])
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        warnings.append(
            DiagnosticWarning(
                code="duplicate_rows",
                level="warning",
                message=(
                    f"{n_dup} row(s) are exact duplicates of (treatment, outcome, confounders). "
                    "If these are repeated records of the same unit, the iid row-bootstrap / "
                    "iid inference assumption is violated and cluster resampling is required."
                ),
                details={"n_duplicates": n_dup},
            )
        )

    return ValidatedData(
        X=X,
        t_idx=t_idx,
        y=y,
        labels=labels,
        label_to_index=label_to_index,
        group_counts=group_counts,
        raw_outcome_means=raw_outcome_means,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        missing_rates=missing_rates,
        warnings=warnings,
    )
