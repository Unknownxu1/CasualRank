"""Repeated-experiment benchmarks: bias / RMSE of the IPTW estimator vs truth.

These helpers are the validation backbone demanded by PROJECT_SPEC.md section
5.3: point estimates must be compared against the known ``mu_true`` over many
replications (a single lucky sample proves nothing), and the benchmark must
also record **fit failure rates, quality-gate outcomes and warning trigger
rates** so that hard regimes (rare treatments, weak overlap) can be assessed
systematically instead of being silently dropped.

A failing ``fit`` never aborts the batch: the run is recorded with
``status="failed"`` plus the error message and excluded from the bias/RMSE
computations, while the failure rate is reported in the summaries.

Monte Carlo error of a coverage-style proportion ``p_hat`` over ``R``
replications is ``sqrt(p_hat (1 - p_hat) / R)`` and must be reported alongside
any conclusion drawn from the benchmark.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..api import CausalRank
from .dgp import CONFOUNDER_COLS, OUTCOME_COL, TREATMENT_COL, DGPConfig, SyntheticDGP, get_scenario

_RUN_DIAGNOSTIC_KEYS = (
    "quality_gate",
    "warning_codes",
    "n_warnings",
    "max_abs_smd_weighted",
    "min_e_obs",
    "max_weight_share",
    "min_ess_ratio",
)


@dataclass
class BenchmarkResult:
    """Replication-level rows, arm-level summary and run-level diagnostics."""

    replications: pd.DataFrame
    summary: pd.DataFrame
    runs: pd.DataFrame
    runs_summary: pd.DataFrame
    scenario: str
    n_reference: int
    reference_seed: int
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return (
            f"BenchmarkResult(scenario={self.scenario!r}, runs={len(self.runs)}), "
            f"summary=\n{self.summary}\nruns_summary=\n{self.runs_summary}"
        )


def _extract_run_diagnostics(result: Any) -> dict[str, Any]:
    """Pull the diagnostic signals of one fitted result into a flat dict."""
    codes = sorted({w.code for w in result.warnings})
    balance = result.diagnostics.balance_summary
    overlap = result.diagnostics.overlap
    weights = result.diagnostics.weights
    return {
        "quality_gate": result.diagnostics.gate,
        "warning_codes": "|".join(codes),
        "n_warnings": len(result.warnings),
        "max_abs_smd_weighted": float(balance["max_abs_smd_weighted"].max())
        if not balance.empty
        else float("nan"),
        "min_e_obs": float(overlap["e_obs_min"].min()) if not overlap.empty else float("nan"),
        "max_weight_share": float(weights["max_weight_share"].max())
        if not weights.empty
        else float("nan"),
        "min_ess_ratio": float(weights["ess_ratio"].min()) if not weights.empty else float("nan"),
    }


def run_estimand_benchmark(
    scenario: str | DGPConfig = "confounded",
    n_values: Sequence[int] = (1_000, 4_000),
    n_replications: int = 10,
    random_state: int = 2026,
    cross_fit: int | None = None,
    n_reference: int | None = None,
    estimator_kwargs: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Estimate ``mu_a`` repeatedly and compare with the known truth.

    For each ``n`` in ``n_values`` and each replication the DGP is redrawn with
    an independent, *reproducible* child seed (``numpy.random.SeedSequence``
    spawn), the full Phase 1 pipeline runs, and the IPTW estimate, the raw
    (unadjusted) group mean and the run's diagnostics are recorded.

    Failing fits are recorded, not raised: check ``runs``/``runs_summary`` for
    ``failure_rate`` before trusting any bias/RMSE number.
    """
    config = get_scenario(scenario) if isinstance(scenario, str) else scenario
    dgp = SyntheticDGP(config)
    mu_true, mu_se = dgp.population_mean(n_reference=n_reference)
    n_ref = int(n_reference) if n_reference is not None else int(config.reference_size)

    kwargs = dict(estimator_kwargs or {})
    kwargs.setdefault("cross_fit", cross_fit)

    n_values = list(n_values)
    root = np.random.SeedSequence(int(random_state))
    child_seeds = root.spawn(len(n_values) * int(n_replications))
    seed_iter = iter(child_seeds)

    run_rows: list[dict[str, Any]] = []
    arm_rows: list[dict[str, Any]] = []
    for n in n_values:
        for rep in range(int(n_replications)):
            seed = int(np.asarray(next(seed_iter).generate_state(1))[0])
            run: dict[str, Any] = {
                "scenario": config.name,
                "n": int(n),
                "replication": rep,
                "seed": seed,
                "status": "ok",
                "error": "",
            }
            est_by_label: dict[str, tuple[float, float, float]] = {}
            try:
                df = dgp.sample(int(n), random_state=seed)
                model = CausalRank(
                    treatment=TREATMENT_COL,
                    outcome=OUTCOME_COL,
                    confounders=list(CONFOUNDER_COLS),
                    random_state=seed,
                    **kwargs,
                )
                result = model.fit(df)
                run.update(_extract_run_diagnostics(result))
                est = result.effects.to_dataframe().set_index("label")
                for label in config.labels:
                    est_by_label[label] = (
                        float(est.loc[label, "mu"]),
                        float(est.loc[label, "raw_mean"]),
                        float(est.loc[label, "ess"]),
                    )
            except Exception as exc:  # never abort the whole batch
                run["status"] = "failed"
                run["error"] = f"{type(exc).__name__}: {exc}"
                for key in _RUN_DIAGNOSTIC_KEYS:
                    run.setdefault(key, np.nan)
                run["quality_gate"] = "failed"
                run["warning_codes"] = ""
                run["n_warnings"] = 0
            run_rows.append(run)

            for a, label in enumerate(config.labels):
                mu_hat, raw_mean, ess = est_by_label.get(label, (np.nan, np.nan, np.nan))
                arm_rows.append(
                    {
                        "scenario": config.name,
                        "n": int(n),
                        "replication": rep,
                        "seed": seed,
                        "label": label,
                        "status": run["status"],
                        "mu_true": float(mu_true[a]),
                        "mu_true_mc_se": float(mu_se[a]),
                        "mu_hat": mu_hat,
                        "raw_mean": raw_mean,
                        "ess": ess,
                    }
                )

    replications = pd.DataFrame(arm_rows)
    runs = pd.DataFrame(run_rows)
    return BenchmarkResult(
        replications=replications,
        summary=summarize_benchmark(replications),
        runs=runs,
        runs_summary=summarize_runs(runs),
        scenario=config.name,
        n_reference=n_ref,
        reference_seed=int(config.reference_seed),
    )


def summarize_benchmark(replications: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replication rows into bias / RMSE per (scenario, n, arm).

    Only runs with ``status == "ok"`` contribute to the error metrics; the
    number of failed runs is reported next to them so an arm estimated mostly
    from failures cannot look artificially good.
    """
    out: list[dict[str, Any]] = []
    grouped = replications.groupby(["scenario", "n", "label"], sort=True)
    for (scenario, n, label), grp in grouped:
        ok = grp[grp["status"] == "ok"]
        n_failed = int((grp["status"] != "ok").sum())
        mu_true = float(grp["mu_true"].iloc[0])
        err = ok["mu_hat"].to_numpy(dtype=float) - mu_true
        raw_err = ok["raw_mean"].to_numpy(dtype=float) - mu_true
        out.append(
            {
                "scenario": scenario,
                "n": int(n),
                "label": label,
                "n_replications": int(grp.shape[0]),
                "n_ok": int(ok.shape[0]),
                "n_failed": n_failed,
                "failure_rate": n_failed / max(int(grp.shape[0]), 1),
                "mu_true": mu_true,
                "mean_mu_hat": float(ok["mu_hat"].mean()) if not ok.empty else np.nan,
                "bias": float(err.mean()) if err.size else np.nan,
                "rmse": float(np.sqrt(np.mean(err**2))) if err.size else np.nan,
                "sd_across_replications": float(np.std(err, ddof=1)) if err.size > 1 else np.nan,
                "bias_raw": float(raw_err.mean()) if raw_err.size else np.nan,
                "rmse_raw": float(np.sqrt(np.mean(raw_err**2))) if raw_err.size else np.nan,
                "mean_ess": float(ok["ess"].mean()) if not ok.empty else np.nan,
            }
        )
    return pd.DataFrame(out)


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Run-level quality statistics per (scenario, n).

    Reports fit failure rate, quality-gate distribution, per-warning-code
    trigger rates (share of runs whose warnings include the code), and mean
    diagnostics. These are the numbers needed to evaluate hard regimes such as
    rare treatments or weak overlap — and to justify the Phase 2 bootstrap
    failure-rate gates.
    """
    out: list[dict[str, Any]] = []
    grouped = runs.groupby(["scenario", "n"], sort=True)
    for (scenario, n), grp in grouped:
        n_runs = int(grp.shape[0])
        n_failed = int((grp["status"] != "ok").sum())
        row: dict[str, Any] = {
            "scenario": scenario,
            "n": int(n),
            "n_runs": n_runs,
            "n_failed": n_failed,
            "failure_rate": n_failed / max(n_runs, 1),
            "gate_pass_rate": float((grp["quality_gate"] == "pass").mean()),
            "gate_caution_rate": float((grp["quality_gate"] == "caution").mean()),
            "gate_fail_rate": float((grp["quality_gate"] == "fail").mean()),
            "mean_n_warnings": float(grp["n_warnings"].astype(float).mean()),
            "mean_max_abs_smd_weighted": float(
                pd.to_numeric(grp["max_abs_smd_weighted"], errors="coerce").mean()
            ),
            "mean_min_e_obs": float(pd.to_numeric(grp["min_e_obs"], errors="coerce").mean()),
            "mean_max_weight_share": float(
                pd.to_numeric(grp["max_weight_share"], errors="coerce").mean()
            ),
            "mean_min_ess_ratio": float(
                pd.to_numeric(grp["min_ess_ratio"], errors="coerce").mean()
            ),
        }
        code_sets = [
            {c for c in str(raw).split("|") if c} for raw in grp["warning_codes"].fillna("")
        ]
        codes: set[str] = set().union(*code_sets) if code_sets else set()
        for code in sorted(codes):
            rate = float(sum(code in s for s in code_sets) / max(len(code_sets), 1))
            row[f"warn_{code}_rate"] = rate
        out.append(row)
    return pd.DataFrame(out)
