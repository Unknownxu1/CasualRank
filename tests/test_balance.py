"""Covariate balance (SMD) tests with hand-computed values (PROJECT_SPEC.md 7.4/7.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrank.diagnostics.balance import standardized_mean_differences

from .conftest import make_config

LABELS = ("A", "B")


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": [1.0, 3.0, 2.0, 6.0],
            "x3": ["a", "b", "a", "a"],
        }
    )


def _run(weights: np.ndarray | None, config=None):
    cfg = config or make_config()
    return standardized_mean_differences(
        X=_frame(),
        t_idx=np.array([0, 0, 1, 1]),
        labels=LABELS,
        weights=weights,
        numeric_features=["x1"],
        categorical_features=["x3"],
        config=cfg,
    )


def test_unweighted_numeric_smd_matches_hand_computation() -> None:
    res = _run(weights=None)
    row = res.detail[(res.detail["feature"] == "x1") & (res.detail["level"].isna())].iloc[0]
    # means 2 vs 4 ; sd_a = sqrt(2), sd_b = sqrt(8) ; s_ref = sqrt(5)
    assert row["mean_a"] == pytest.approx(2.0, rel=1e-12)
    assert row["mean_b"] == pytest.approx(4.0, rel=1e-12)
    assert row["sd_ref"] == pytest.approx(np.sqrt(5.0), rel=1e-12)
    assert row["smd_unweighted"] == pytest.approx(-2.0 / np.sqrt(5.0), rel=1e-12)


def test_weighted_numeric_smd_matches_hand_computation() -> None:
    # arm A weights [1, 3] -> weighted mean 2.5 ; arm B weights [1, 1] -> 4.0
    weights = np.array([1.0, 3.0, 1.0, 1.0])
    res = _run(weights)
    row = res.detail[(res.detail["feature"] == "x1") & (res.detail["level"].isna())].iloc[0]
    assert row["mean_a_weighted"] == pytest.approx(2.5, rel=1e-12)
    assert row["mean_b_weighted"] == pytest.approx(4.0, rel=1e-12)
    assert row["smd_weighted"] == pytest.approx((2.5 - 4.0) / np.sqrt(5.0), rel=1e-12)


def test_reference_sd_is_weight_independent() -> None:
    weighted = _run(np.array([1.0, 3.0, 1.0, 1.0]))
    unweighted = _run(None)
    key = ["feature", "level", "pair"]
    m = weighted.detail.merge(unweighted.detail, on=key, suffixes=("_w", "_u"))
    assert np.allclose(m["sd_ref_w"], m["sd_ref_u"])


def test_categorical_indicator_smd() -> None:
    res = _run(weights=None)
    row = res.detail[(res.detail["feature"] == "x3") & (res.detail["level"] == "a")].iloc[0]
    # level 'a' indicator: A = [0, 1] (mean .5, sd sqrt(.5)), B = [1, 1] (mean 1, sd 0)
    assert row["mean_a"] == pytest.approx(0.5, rel=1e-12)
    assert row["mean_b"] == pytest.approx(1.0, rel=1e-12)
    assert row["sd_ref"] == pytest.approx(np.sqrt(0.25), rel=1e-12)
    assert row["smd_unweighted"] == pytest.approx(-1.0, rel=1e-12)


def test_degenerate_zero_variance_feature_is_flagged() -> None:
    df = pd.DataFrame(
        {
            "x1": [1.0, 1.0, 2.0, 2.0],  # constant within arm, different across arms
            "x3": ["a", "a", "a", "a"],
        }
    )
    res = standardized_mean_differences(
        X=df,
        t_idx=np.array([0, 0, 1, 1]),
        labels=LABELS,
        weights=None,
        numeric_features=["x1"],
        categorical_features=["x3"],
        config=make_config(),
    )
    row = res.detail[(res.detail["feature"] == "x1") & (res.detail["level"].isna())].iloc[0]
    assert np.isinf(row["smd_unweighted"])
    assert "degenerate_smd" in {w.code for w in res.warnings}


def test_missing_values_do_not_poison_smd() -> None:
    df = pd.DataFrame(
        {
            "x1": [1.0, 3.0, np.nan, 2.0, 6.0, 7.0],
            "x3": ["a", "b", "a", None, "a", "b"],
        }
    )
    res = standardized_mean_differences(
        X=df,
        t_idx=np.array([0, 0, 0, 1, 1, 1]),
        labels=LABELS,
        weights=None,
        numeric_features=["x1"],
        categorical_features=["x3"],
        config=make_config(),
    )
    num = res.detail[(res.detail["feature"] == "x1") & (res.detail["level"].isna())].iloc[0]
    assert num["mean_a"] == pytest.approx(2.0, rel=1e-12)  # observed [1, 3]
    assert num["mean_b"] == pytest.approx(5.0, rel=1e-12)  # observed [2, 6, 7]
    # missingness indicator terms are part of the balance check
    levels = set(res.detail.loc[res.detail["feature"] == "x1", "level"].dropna())
    assert "__missing__" in levels
    assert not res.detail["smd_weighted"].isna().any()


def test_balance_summary_aggregates_max_over_pairs() -> None:
    res = _run(weights=None)
    assert set(res.summary.columns) >= {
        "feature",
        "level",
        "max_abs_smd_unweighted",
        "max_abs_smd_weighted",
    }
    x1 = res.summary[(res.summary["feature"] == "x1") & (res.summary["level"].isna())].iloc[0]
    detail_x1 = res.detail[(res.detail["feature"] == "x1") & (res.detail["level"].isna())]
    assert x1["max_abs_smd_unweighted"] == pytest.approx(
        detail_x1["smd_unweighted"].abs().max(), rel=1e-12
    )


def test_severe_imbalance_raises_critical_warning() -> None:
    df = pd.DataFrame(
        {
            "x1": [0.0, 0.1, 5.0, 5.1],  # huge separation relative to within-arm sd
            "x3": ["a", "a", "a", "a"],
        }
    )
    res = standardized_mean_differences(
        X=df,
        t_idx=np.array([0, 0, 1, 1]),
        labels=LABELS,
        weights=None,
        numeric_features=["x1"],
        categorical_features=["x3"],
        config=make_config(),
    )
    codes = {w.code: w.level for w in res.warnings}
    assert codes.get("severe_imbalance") == "critical"
