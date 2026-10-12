"""Repeated-experiment validation of the estimator against known truth.

Marked ``slow``: kept small enough for CI but sufficient to show convergence of
the Hajek IPTW estimator with n, and to compare the confounding bias of raw
group means with the adjusted estimates. The full research grid is run manually
via ``scripts/run_simulation.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from causalrank.simulation import run_estimand_benchmark, summarize_benchmark


@pytest.mark.slow
def test_iptw_converges_to_truth_under_confounding() -> None:
    bench = run_estimand_benchmark(
        "confounded",
        n_values=(1_000, 4_000),
        n_replications=6,
        random_state=2026,
        cross_fit=3,
    )
    summary = bench.summary
    small = summary[summary["n"] == 1_000]
    large = summary[summary["n"] == 4_000]

    # (1) bias shrinks with sample size
    assert large["bias"].abs().mean() < small["bias"].abs().mean()
    assert large["bias"].abs().mean() < 0.035

    # (2) adjustment beats the raw group means where confounding biases them
    assert large["bias"].abs().mean() < 0.6 * large["bias_raw"].abs().mean()

    # (3) RMSE decreases with n for every arm
    merged = small.merge(large, on="label", suffixes=("_s", "_l"))
    assert np.all(merged["rmse_l"] < merged["rmse_s"])

    # provenance of the ground truth is recorded
    assert bench.n_reference == 200_000
    assert bench.reference_seed == 20260101


@pytest.mark.slow
def test_randomized_dgp_unbiased_for_raw_and_adjusted_means() -> None:
    bench = run_estimand_benchmark(
        "randomized",
        n_values=(2_000,),
        n_replications=6,
        random_state=7,
        cross_fit=3,
    )
    summary = bench.summary
    assert summary["bias"].abs().max() < 0.03
    assert summary["bias_raw"].abs().max() < 0.03


def test_summarize_benchmark_is_consistent() -> None:
    bench = run_estimand_benchmark(
        "equal_effects", n_values=(400,), n_replications=2, random_state=1
    )
    assert len(bench.replications) == 2 * 3
    assert len(bench.summary) == 3
    assert set(bench.summary["label"]) == {"A", "B", "C"}
    # summarising twice is idempotent
    again = summarize_benchmark(bench.replications)
    assert np.allclose(again["bias"], bench.summary["bias"])


def test_failed_runs_are_recorded_not_raised() -> None:
    """A fit that throws must be recorded with its error, not abort the batch."""
    bench = run_estimand_benchmark(
        "confounded",
        n_values=(300,),
        n_replications=3,
        random_state=3,
        estimator_kwargs={"cross_fit": 1000},  # impossible fold count -> fit error
    )
    assert len(bench.runs) == 3
    assert (bench.runs["status"] == "failed").all()
    assert bench.runs["error"].str.contains("cross_fit").all()
    assert float(bench.summary["failure_rate"].max()) == 1.0
    assert bench.summary["bias"].isna().all()
    assert float(bench.runs_summary["failure_rate"].iloc[0]) == 1.0


def test_run_diagnostics_capture_gates_and_warnings() -> None:
    bench = run_estimand_benchmark("weak_overlap", n_values=(400,), n_replications=2, random_state=5)
    runs = bench.runs
    assert {"quality_gate", "warning_codes", "min_e_obs", "max_weight_share"} <= set(runs.columns)
    assert (runs["status"] == "ok").all()
    # weak overlap must degrade the gate and raise diagnostic warnings
    assert (runs["quality_gate"] != "pass").any()
    assert runs["warning_codes"].str.len().gt(0).all()

    rs = bench.runs_summary
    assert int(rs["n_runs"].iloc[0]) == 2
    assert float(rs["failure_rate"].iloc[0]) == 0.0
    warn_cols = [c for c in rs.columns if c.startswith("warn_")]
    assert warn_cols, "expected warning trigger-rate columns for weak_overlap"
    assert any(float(rs[c].iloc[0]) > 0 for c in warn_cols)


@pytest.mark.slow
def test_bootstrap_se_matches_empirical_variability() -> None:
    """Phase 2 gate: bootstrap SEs must track the real sampling variability.

    ``se_ratio = mean_se / empirical_sd`` over replications; a value far from 1
    means the covariance is miscalibrated. The tolerance reflects both Monte
    Carlo error (R is small by design) and the known variability of bootstrap
    SEs — a tighter claim would not survive the simulation noise.
    """
    from causalrank.simulation import run_se_benchmark

    res = run_se_benchmark(
        "confounded", n=800, n_replications=16, n_bootstrap=60, random_state=2026
    )
    summary = res.summary
    assert (summary["n_replications"] == 16).all()
    ratio = summary["se_ratio"].to_numpy(dtype=float)
    assert np.all(ratio > 0.55), summary
    assert np.all(ratio < 1.6), summary
    # nominal 95% CI: coverage must be reasonably close given MC error
    coverage = summary["coverage"].to_numpy(dtype=float)
    assert np.all(coverage >= 0.80), summary
    assert np.all(coverage <= 1.0), summary


@pytest.mark.slow
def test_rank_benchmark_reports_tie_aware_coverage() -> None:
    """Phase 3 gate: rank confidence sets vs tie-aware truth on ``equal_effects``.

    With truly equal potential-outcome means every admissible rank lies in
    ``[1, K]``, so the strict coverage definition can only hold when the
    simultaneous intervals confirm nothing -- exactly the red line that noise
    must not produce a determined ranking.
    """
    from causalrank.simulation import run_rank_benchmark

    from .conftest import FAST_GBM

    res = run_rank_benchmark(
        "equal_effects",
        n=500,
        n_replications=3,
        n_bootstrap=10,
        random_state=11,
        bootstrap_min_successes=4,
        estimator_kwargs={"gbm_params": dict(FAST_GBM)},
    )
    summary = res.summary
    assert (summary["true_rank_min"] == 1).all()
    assert (summary["true_rank_max"] == 3).all()
    assert (summary["coverage"] == 1.0).all(), summary
    runs = res.run_summary
    assert int(runs["n_failed"].iloc[0]) == 0
    assert float(runs["any_confirmed_rate"].iloc[0]) == 0.0
    assert float(runs["all_rank_sets_full_rate"].iloc[0]) == 1.0


@pytest.mark.slow
def test_rank_benchmark_schema_and_confirmed_relations() -> None:
    """Separated truths: the best arm must be pinned to rank 1 in the point ranks
    and the benchmark must record the per-arm coverage columns."""
    from causalrank.simulation import run_rank_benchmark

    from .conftest import FAST_GBM

    res = run_rank_benchmark(
        "confounded",
        n=800,
        n_replications=2,
        n_bootstrap=10,
        random_state=12,
        bootstrap_min_successes=4,
        estimator_kwargs={"gbm_params": dict(FAST_GBM)},
    )
    rep = res.replications
    for col in ("true_rank_min", "true_rank_max", "rank_low", "rank_high", "covers"):
        assert col in rep.columns
    # truth is unique here (tau = (0, 0.35, -0.15))
    assert (rep["true_rank_min"] == rep["true_rank_max"]).all()
    best = rep[rep["true_rank_min"] == 1]
    assert set(best["label"]) == {"B"}
    assert res.summary["coverage"].notna().all()


def test_rank_benchmark_returns_recorded_failures_when_all_runs_fail() -> None:
    """Regression (P3): an all-failed batch must return the failure records,
    not crash while summarising an empty table."""
    from causalrank.simulation import run_rank_benchmark

    res = run_rank_benchmark(
        "confounded",
        n=2,  # far too small: every fit raises, recorded as status="failed"
        n_replications=1,
        n_bootstrap=2,
        random_state=3,
        bootstrap_min_successes=2,
    )
    runs = res.runs
    assert list(runs["status"]) == ["failed"]
    assert "error" in runs.columns and str(runs["error"].iloc[0])
    # well-formed empty tables instead of a KeyError on 'label'
    assert res.replications.empty
    assert list(res.summary.columns) == [
        "scenario",
        "n",
        "label",
        "n_replications",
        "mu_true",
        "true_rank_min",
        "true_rank_max",
        "coverage",
        "coverage_mc_se",
        "mean_rank_width",
        "confidence_level",
        "n_bootstrap",
        "higher_is_better",
    ]
    assert res.summary.empty
    run_summary = res.run_summary
    assert int(run_summary["n_failed"].iloc[0]) == 1
    assert int(run_summary["n_replications"].iloc[0]) == 1
    assert np.isnan(float(run_summary["any_confirmed_rate"].iloc[0]))
