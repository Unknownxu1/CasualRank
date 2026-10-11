"""Input validation and label-order tests (PROJECT_SPEC.md 7.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrank import CausalRankValidationError
from causalrank.config import CausalRankConfig
from causalrank.validation import canonical_label_order, validate_dataframe

from .conftest import make_config


def test_missing_column_raises(simple_df: pd.DataFrame) -> None:
    cfg = make_config(confounders=("x1", "nope"))
    with pytest.raises(CausalRankValidationError, match="nope"):
        validate_dataframe(simple_df, cfg)


def test_missing_treatment_outcome_column_raises(simple_df: pd.DataFrame) -> None:
    cfg = make_config(treatment="nope")
    with pytest.raises(CausalRankValidationError, match="Missing required column"):
        validate_dataframe(simple_df, cfg)


def test_non_binary_outcome_raises(simple_df: pd.DataFrame) -> None:
    df = simple_df.copy()
    df.loc[0, "outcome"] = 2.0
    with pytest.raises(CausalRankValidationError, match="binary 0/1"):
        validate_dataframe(df, make_config())


def test_missing_outcome_raises(simple_df: pd.DataFrame) -> None:
    df = simple_df.copy()
    df.loc[3, "outcome"] = np.nan
    with pytest.raises(CausalRankValidationError, match="missing value"):
        validate_dataframe(df, make_config())


def test_missing_treatment_raises(simple_df: pd.DataFrame) -> None:
    df = simple_df.copy()
    df.loc[7, "treatment"] = None
    with pytest.raises(CausalRankValidationError, match="missing value"):
        validate_dataframe(df, make_config())


def test_single_treatment_raises(simple_df: pd.DataFrame) -> None:
    df = simple_df.copy()
    df["treatment"] = "A"
    with pytest.raises(CausalRankValidationError, match="at least 2 distinct"):
        validate_dataframe(df, make_config())


def test_boolean_treatment_labels_rejected(simple_df: pd.DataFrame) -> None:
    df = simple_df.copy()
    df["treatment"] = (df["treatment"] == "A").astype(bool)
    with pytest.raises(CausalRankValidationError, match="Boolean treatment labels"):
        validate_dataframe(df, make_config())


def test_mixed_label_types_rejected(simple_df: pd.DataFrame) -> None:
    df = simple_df.copy()
    labels = df["treatment"].tolist()
    labels[0] = 3
    df["treatment"] = labels
    with pytest.raises(CausalRankValidationError, match="Mixed treatment label types"):
        validate_dataframe(df, make_config())


def test_non_integral_float_labels_rejected(simple_df: pd.DataFrame) -> None:
    df = simple_df.copy()
    df["treatment"] = df["treatment"].map({"A": 1.0, "B": 2.5, "C": 3.0})
    with pytest.raises(CausalRankValidationError, match="non-integral float"):
        validate_dataframe(df, make_config())


def test_integral_float_labels_cast_to_int(simple_df: pd.DataFrame) -> None:
    df = simple_df.copy()
    df["treatment"] = df["treatment"].map({"A": 2.0, "B": 1.0, "C": 10.0})
    validated = validate_dataframe(df, make_config())
    assert validated.labels == (1, 2, 10)
    codes = {w.code for w in validated.warnings}
    assert "label_dtype_cast" in codes


def test_string_label_order_is_sorted_and_aligned(simple_df: pd.DataFrame) -> None:
    df = simple_df.copy()
    # "B" appears first in the data; the canonical order must still be sorted.
    validated = validate_dataframe(df, make_config())
    assert validated.labels == ("A", "B", "C")
    expected = df["treatment"].map({"A": 0, "B": 1, "C": 2}).to_numpy()
    assert np.array_equal(validated.t_idx, expected)
    assert canonical_label_order(df["treatment"]) == ("A", "B", "C")


def test_integer_label_order_is_numeric_not_lexicographic() -> None:
    s = pd.Series([2, 10, 5, 2, 10, 5])
    assert canonical_label_order(s) == (2, 5, 10)


def test_small_group_warning(simple_df: pd.DataFrame) -> None:
    df = simple_df.copy()
    extra = pd.DataFrame([df.iloc[0].to_dict() for _ in range(3)])
    extra["treatment"] = "D"
    df = pd.concat([df, extra], ignore_index=True)
    validated = validate_dataframe(df, make_config())
    codes = {w.code for w in validated.warnings}
    assert "small_group" in codes


def test_single_observation_group_raises(simple_df: pd.DataFrame) -> None:
    df = simple_df.copy()
    extra = pd.DataFrame([df.iloc[0].to_dict()])
    extra["treatment"] = "D"
    df = pd.concat([df, extra], ignore_index=True)
    with pytest.raises(CausalRankValidationError, match="at least 2"):
        validate_dataframe(df, make_config())


def test_rare_treatment_warning() -> None:
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.integers(0, 2, size=n).astype(float),
            "treatment": ["A"] * (n - 5) + ["B"] * 5,
            "outcome": rng.integers(0, 2, size=n).astype(float),
        }
    )
    validated = validate_dataframe(df, make_config())
    codes = {w.code for w in validated.warnings}
    assert "rare_treatment" in codes
    assert "small_group" in codes


def test_cross_fit_requires_enough_rows_per_class() -> None:
    rng = np.random.default_rng(2)
    n = 8
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.integers(0, 2, size=n).astype(float),
            "treatment": ["A"] * 4 + ["B"] * 4,
            "outcome": rng.integers(0, 2, size=n).astype(float),
        }
    )
    with pytest.raises(CausalRankValidationError, match="cross_fit"):
        validate_dataframe(df, make_config(cross_fit=5))


def test_missing_covariates_are_reported_not_dropped(simple_df: pd.DataFrame) -> None:
    df = simple_df.copy()
    df.loc[0:4, "x1"] = np.nan
    df.loc[0:2, "x3"] = np.nan
    validated = validate_dataframe(df, make_config())
    assert validated.missing_rates["x1"] > 0
    assert validated.n_rows == len(df)  # no silent row dropping
    assert "missing_covariates" in {w.code for w in validated.warnings}


def test_all_missing_confounder_raises(simple_df: pd.DataFrame) -> None:
    df = simple_df.copy()
    df["x1"] = np.nan
    with pytest.raises(CausalRankValidationError, match="entirely missing"):
        validate_dataframe(df, make_config())


def test_high_cardinality_categorical_warning() -> None:
    rng = np.random.default_rng(3)
    n = 120
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": pd.Series([f"lvl{i % 40}" for i in range(n)], dtype=object),
            "treatment": ["A"] * 60 + ["B"] * 60,
            "outcome": rng.integers(0, 2, size=n).astype(float),
        }
    )
    validated = validate_dataframe(df, make_config(confounders=("x1", "x2", "x3")))
    assert "high_cardinality_categorical" in {w.code for w in validated.warnings}
    assert validated.categorical_features == ["x3"]


def test_duplicate_rows_warning() -> None:
    df = pd.DataFrame(
        {
            "x1": [0.0, 0.0, 1.0, 1.0],
            "x2": [1.0, 1.0, 2.0, 2.0],
            "x3": [0.0, 0.0, 1.0, 1.0],
            "treatment": ["A", "A", "B", "B"],
            "outcome": [1.0, 1.0, 0.0, 0.0],
        }
    )
    validated = validate_dataframe(df, make_config())
    assert "duplicate_rows" in {w.code for w in validated.warnings}


def test_unsupported_confounder_dtype_raises(simple_df: pd.DataFrame) -> None:
    df = simple_df.copy()
    df["x2"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(np.arange(len(df)), unit="D")
    with pytest.raises(CausalRankValidationError, match="unsupported dtype"):
        validate_dataframe(df, make_config())


def test_common_pandas_dtypes_are_accepted_and_normalised() -> None:
    """bool / pandas string dtype / object-with-pd.NA / nullable Int64 must work."""
    rng = np.random.default_rng(4)
    n = 60
    df = pd.DataFrame(
        {
            "x_bool": pd.Series(rng.random(n) > 0.5),
            "x_str": pd.Series([["a", "b", "c"][i % 3] for i in range(n)]),
            "x_obj": pd.Series([("m", "n", pd.NA)[i % 3] for i in range(n)], dtype=object),
            "x_int": pd.Series([1, 2, pd.NA] * (n // 3), dtype="Int64"),
            "treatment": pd.Series(["A", "B"] * (n // 2)),
            "outcome": pd.Series(rng.integers(0, 2, size=n).astype(float)),
        }
    )
    validated = validate_dataframe(
        df, make_config(confounders=("x_bool", "x_str", "x_obj", "x_int"))
    )
    assert validated.categorical_features == ["x_bool", "x_str", "x_obj"]
    assert validated.numeric_features == ["x_int"]
    # normalised: no pd.NA survives into the analysis frame, numerics are float64
    assert not validated.X["x_obj"].map(lambda v: v is pd.NA).any()
    assert validated.X["x_int"].dtype == float
    assert validated.X["x_str"].dtype == object
    assert validated.X["x_bool"].dtype == object


# --- configuration validation ------------------------------------------
def test_config_rejects_unsupported_methods() -> None:
    with pytest.raises(CausalRankValidationError, match="not supported in Phase 1"):
        make_config(estimator="aipw")
    with pytest.raises(CausalRankValidationError, match="not supported"):
        make_config(propensity_model="logit")
    with pytest.raises(CausalRankValidationError, match="not supported"):
        make_config(weighting="ht")
    with pytest.raises(CausalRankValidationError, match="not supported"):
        make_config(estimand="att")


def test_config_rejects_bad_thresholds() -> None:
    with pytest.raises(CausalRankValidationError, match="confidence_level"):
        make_config(confidence_level=1.5)
    with pytest.raises(CausalRankValidationError, match="smd_threshold"):
        make_config(smd_threshold=0.5, smd_severe_threshold=0.25)
    with pytest.raises(CausalRankValidationError, match="cross_fit"):
        make_config(cross_fit=1)
    with pytest.raises(CausalRankValidationError, match="clip_propensity"):
        make_config(clip_propensity=0.7)


def test_config_rejects_empty_or_overlapping_confounders(simple_df: pd.DataFrame) -> None:
    with pytest.raises(CausalRankValidationError, match="At least one confounder"):
        make_config(confounders=())
    with pytest.raises(CausalRankValidationError, match="cannot be both"):
        make_config(confounders=("x1", "outcome"))


def test_config_rejects_gbm_random_state_override() -> None:
    with pytest.raises(CausalRankValidationError, match="random_state"):
        make_config(gbm_params={"random_state": 3})


def test_config_to_dict_roundtrips() -> None:
    cfg = make_config(cross_fit=5, random_state=7)
    d = cfg.to_dict()
    assert d["confounders"] == ["x1", "x2", "x3"]
    assert d["cross_fit"] == 5
    assert isinstance(CausalRankConfig(**{**d, "confounders": tuple(d["confounders"])}), CausalRankConfig)
