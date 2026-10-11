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
