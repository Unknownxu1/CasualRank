"""Diagnostics tests: weights, ESS, overlap and the quality gate."""

from __future__ import annotations

import numpy as np
import pytest

from causalrank import CausalRank
from causalrank.diagnostics.overlap import overlap_diagnostics
from causalrank.diagnostics.weights import ess_table, weight_diagnostics
from causalrank.estimators.iptw import hajek_estimate
from causalrank.simulation import make_dataset

from .conftest import FAST_GBM

LABELS = ("A", "B")


def _estimate(y, t_idx, propensity, labels=LABELS):
    return hajek_estimate(y, t_idx, propensity, labels)


def test_weight_table_ess_and_share() -> None:
    y = np.array([1.0, 0.0, 1.0, 0.0])
    t_idx = np.array([0, 0, 1, 1])
    propensity = np.array([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.6, 0.4]])
    est = _estimate(y, t_idx, propensity)
    from .conftest import make_config

    cfg = make_config()
    table, warnings = weight_diagnostics(est, cfg)
    row_a = table[table["label"] == "A"].iloc[0]
    assert row_a["n"] == 2
    assert row_a["ess"] == pytest.approx(1.8, rel=1e-12)
    assert row_a["ess_ratio"] == pytest.approx(0.9, rel=1e-12)
    assert row_a["w_min"] == pytest.approx(1.25, rel=1e-12)
    assert row_a["w_max"] == pytest.approx(2.5, rel=1e-12)
    assert row_a["max_weight_share"] == pytest.approx(2.5 / 3.75, rel=1e-12)
    # the single large weight triggers the extreme-weight gate
    assert any(w.code == "extreme_weight" for w in warnings)

    compact = ess_table(est)
    assert compact["ess"].tolist() == pytest.approx(list(table["ess"]), rel=1e-12)


def test_low_ess_warning() -> None:
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    t_idx = np.array([0, 0, 0, 1, 1, 1])
    # extreme spread in arm A: one tiny propensity -> weight 100
    propensity = np.array(
        [
            [0.01, 0.99],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.34, 0.66],
            [0.34, 0.66],
            [0.33, 0.67],
        ]
    )
    est = _estimate(y, t_idx, propensity)
    from .conftest import make_config

    cfg = make_config()
    _, warnings = weight_diagnostics(est, cfg)
    codes = [w.code for w in warnings]
    assert codes.count("low_ess") >= 1


def test_common_support_gap_on_target_sample_is_flagged() -> None:
    """Rows where an arm has zero reach must warn even if treated rows look fine.

    Regression test: each arm is treated with moderate propensity on its own
    rows, but every arm is unsupported on one third of the target sample.
    """
    propensity = np.array(
        [
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ]
    )
    t_idx = np.array([0, 1, 2])
    from .conftest import make_config

    table, warnings = overlap_diagnostics(propensity, t_idx, ["A", "B", "C"], make_config())
    assert np.allclose(table["frac_arm_below_threshold"].to_numpy(), 1 / 3)
    codes = {w.code: w.level for w in warnings}
    assert codes.get("no_common_support") == "critical"
    assert sum(1 for w in warnings if w.code == "no_common_support") == 3


def test_weak_common_support_share_triggers_warning() -> None:
    """Non-zero but tiny reach over a large share of the sample also warns."""
    propensity = np.full((20, 2), 0.5)
    propensity[:, 0] = np.array([0.005] * 5 + [0.6] * 15)  # arm A weak on 25% of rows
    propensity[:, 1] = 1.0 - propensity[:, 0]
    t_idx = np.array([0] * 10 + [1] * 10)
    from .conftest import make_config

    table, warnings = overlap_diagnostics(propensity, t_idx, ["A", "B"], make_config())
    row_a = table[table["label"] == "A"].iloc[0]
    assert row_a["frac_arm_below_threshold"] == pytest.approx(0.25)
    codes = {w.code for w in warnings}
    assert "weak_common_support" in codes
    assert "no_common_support" not in codes  # nothing is exactly zero


def test_overlap_table_and_low_propensity_warning() -> None:
    propensity = np.array(
        [
            [0.98, 0.02],
            [0.55, 0.45],
            [0.40, 0.60],
            [0.02, 0.98],
        ]
    )
    t_idx = np.array([0, 0, 1, 1])
    from .conftest import make_config

    cfg = make_config()
    table, warnings = overlap_diagnostics(propensity, t_idx, LABELS, cfg)
    row_a = table[table["label"] == "A"].iloc[0]
    assert row_a["e_obs_min"] == pytest.approx(0.55, rel=1e-12)  # own propensity of A rows
    assert row_a["e_arm_min"] == pytest.approx(0.02, rel=1e-12)
    assert not warnings  # both arms keep e_obs >= threshold


def test_low_propensity_warning_on_extreme_rows() -> None:
    propensity = np.array(
        [
            [0.55, 0.45],
            [0.005, 0.995],
            [0.40, 0.60],
            [0.30, 0.70],
        ]
    )
    t_idx = np.array([0, 0, 1, 1])
    from .conftest import make_config

    table, warnings = overlap_diagnostics(propensity, t_idx, LABELS, make_config())
    codes = {w.code for w in warnings}
    assert "low_propensity" in codes
    row_a = table[table["label"] == "A"].iloc[0]
    assert row_a["e_obs_min"] == pytest.approx(0.005, rel=1e-12)
    assert row_a["frac_obs_below_threshold"] == pytest.approx(0.5, rel=1e-12)


def test_clipping_is_reported_as_warning() -> None:
    y = np.array([1.0, 0.0, 1.0, 0.0])
    t_idx = np.array([0, 0, 1, 1])
    propensity = np.array([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.6, 0.4]])
    est = hajek_estimate(y, t_idx, propensity, LABELS, clip_propensity=0.45)
    from .conftest import make_config

    _, warnings = weight_diagnostics(est, make_config())
    clip_warns = [w for w in warnings if w.code == "propensity_clipped"]
    assert clip_warns and clip_warns[0].details["clipped_fraction"] == pytest.approx(0.5)


def test_weak_overlap_scenario_triggers_quality_gate() -> None:
    ds = make_dataset("weak_overlap", n=1500, random_state=11)
    model = CausalRank(
        treatment="treatment",
        outcome="outcome",
        confounders=["x1", "x2", "x3", "x4"],
        random_state=0,
        gbm_params=dict(FAST_GBM),
    )
    result = model.fit(ds.df)
    codes = {w.code for w in result.warnings}
    assert codes & {"low_propensity", "low_ess", "extreme_weight"}
    assert result.diagnostics.gate in {"caution", "fail"}
    assert result.metadata["quality_gate"] == result.diagnostics.gate
    # the summary must not claim a reliable causal ranking
    text = result.summary()
    assert "reliable causal ranking" not in text
    assert "NOT available in this result" in text


def test_group_summary_reports_raw_counts_and_means() -> None:
    ds = make_dataset("randomized", n=300, random_state=4)
    model = CausalRank(
        treatment="treatment",
        outcome="outcome",
        confounders=["x1", "x2", "x3", "x4"],
        random_state=0,
        gbm_params=dict(FAST_GBM),
    )
    result = model.fit(ds.df)
    gs = result.diagnostics.group_summary
    assert set(gs.columns) == {
        "label",
        "n",
        "share",
        "raw_outcome_mean",
        "mu_hajek_iptw",
    }
    assert gs["n"].sum() == len(ds.df)
    assert np.isclose(gs["share"].sum(), 1.0)
    raw = ds.df.groupby("treatment")["outcome"].mean()
    for _, row in gs.iterrows():
        assert row["raw_outcome_mean"] == pytest.approx(float(raw[row["label"]]), rel=1e-12)
