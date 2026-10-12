"""Simulation sub-package: synthetic DGP with known truth + benchmarks."""

from __future__ import annotations

from .benchmark import (
    BenchmarkResult,
    RankBenchmarkResult,
    SEBenchmarkResult,
    run_estimand_benchmark,
    run_rank_benchmark,
    run_se_benchmark,
    summarize_benchmark,
    summarize_runs,
)
from .dgp import (
    CONFOUNDER_COLS,
    OUTCOME_COL,
    TREATMENT_COL,
    DGPConfig,
    SyntheticDataset,
    SyntheticDGP,
    get_scenario,
    make_dataset,
    make_oracle_frame,
    scenario_names,
)

__all__ = [
    "CONFOUNDER_COLS",
    "OUTCOME_COL",
    "TREATMENT_COL",
    "BenchmarkResult",
    "DGPConfig",
    "RankBenchmarkResult",
    "SEBenchmarkResult",
    "SyntheticDGP",
    "SyntheticDataset",
    "get_scenario",
    "make_dataset",
    "make_oracle_frame",
    "run_estimand_benchmark",
    "run_rank_benchmark",
    "run_se_benchmark",
    "scenario_names",
    "summarize_benchmark",
    "summarize_runs",
]
