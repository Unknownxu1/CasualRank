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


def run_se_benchmark(
    scenario: str | DGPConfig = "confounded",
    n: int = 2_000,
    n_replications: int = 20,
    n_bootstrap: int = 200,
    random_state: int = 2026,
    cross_fit: int | None = None,
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    n_jobs: int = 1,
) -> SEBenchmarkResult:
    """Compare bootstrap SEs with the empirical sampling variability of ``mu_hat``.

    For each replication the full pipeline runs *with* the joint bootstrap, so
    every replication yields ``mu_hat``, ``se_hat`` and a pointwise CI. The
    summary reports, per arm:

    * ``empirical_sd`` — sd of ``mu_hat`` across replications (the real sampling
      variability at this ``n``);
    * ``mean_se`` and ``se_ratio = mean_se / empirical_sd`` — the calibration of
      the bootstrap covariance;
    * ``coverage`` of the nominal CI for ``mu_true`` with its Monte Carlo error
      ``sqrt(p(1-p)/R)``.

    This is the Phase 2 validation gate for the uncertainty quantification: a
    badly calibrated SE (ratio far from 1) or under-covering CI must be reported
    as a limitation, not hidden behind a plausible-looking interval.
    """
    config = get_scenario(scenario) if isinstance(scenario, str) else scenario
    dgp = SyntheticDGP(config)
    mu_true, _ = dgp.population_mean()

    root = np.random.SeedSequence(int(random_state))
    seeds = [int(np.asarray(c.generate_state(1))[0]) for c in root.spawn(int(n_replications))]

    rows: list[dict[str, Any]] = []
    for rep, seed in enumerate(seeds):
        df = dgp.sample(int(n), random_state=seed)
        model = CausalRank(
            treatment=TREATMENT_COL,
            outcome=OUTCOME_COL,
            confounders=list(CONFOUNDER_COLS),
            random_state=seed,
            cross_fit=cross_fit,
            n_bootstrap=int(n_bootstrap),
            ci_method=ci_method,
            confidence_level=confidence_level,
            n_jobs=n_jobs,
        )
        result = model.fit(df)
        est = result.effects.to_dataframe().set_index("label")
        for a, label in enumerate(config.labels):
            truth = float(mu_true[a])
            lo = float(est.loc[label, "ci_low"])
            hi = float(est.loc[label, "ci_high"])
            rows.append(
                {
                    "scenario": config.name,
                    "n": int(n),
                    "replication": rep,
                    "seed": seed,
                    "label": label,
                    "mu_true": truth,
                    "mu_hat": float(est.loc[label, "mu"]),
                    "se_hat": float(est.loc[label, "se"]),
                    "ci_low": lo,
                    "ci_high": hi,
                    "covers": bool(lo <= truth <= hi),
                    "inference_available": bool(result.inference_available),
                    "bootstrap_success_rate": float(
                        result.metadata.get("bootstrap", {}).get("failure_rate", np.nan)
                    ),
                }
            )

    replications = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for label, grp in replications.groupby("label", sort=True):
        ok = grp[grp["inference_available"]]
        mu_hat = ok["mu_hat"].to_numpy(dtype=float)
        empirical_sd = float(np.std(mu_hat, ddof=1)) if mu_hat.size > 1 else np.nan
        mean_se = float(ok["se_hat"].mean())
        p = float(ok["covers"].mean()) if not ok.empty else np.nan
        mc_se = float(np.sqrt(p * (1 - p) / max(ok.shape[0], 1))) if ok.shape[0] else np.nan
        summary_rows.append(
            {
                "scenario": config.name,
                "n": int(n),
                "label": label,
                "n_replications": int(ok.shape[0]),
                "mu_true": float(grp["mu_true"].iloc[0]),
                "bias": float(mu_hat.mean() - grp["mu_true"].iloc[0]) if mu_hat.size else np.nan,
                "empirical_sd": empirical_sd,
                "mean_se": mean_se,
                "se_ratio": mean_se / empirical_sd if empirical_sd else np.nan,
                "coverage": p,
                "coverage_mc_se": mc_se,
                "ci_method": ci_method,
                "n_bootstrap": int(n_bootstrap),
            }
        )
    return SEBenchmarkResult(
        replications=replications,
        summary=pd.DataFrame(summary_rows),
        scenario=config.name,
        n=int(n),
        n_bootstrap=int(n_bootstrap),
    )


@dataclass
class SEBenchmarkResult:
    """SE calibration: replication rows plus a per-arm summary."""

    replications: pd.DataFrame
    summary: pd.DataFrame
    scenario: str
    n: int
    n_bootstrap: int

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"SEBenchmarkResult(scenario={self.scenario!r}, n={self.n}):\n{self.summary}"


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


# ---------------------------------------------------------------------------
# Phase 3: rank confidence-set coverage benchmark
# ---------------------------------------------------------------------------
@dataclass
class RankBenchmarkResult:
    """Rank-inference validation: per-arm coverage rows and run-level summary.

    ``runs`` keeps one row per replication (``status`` / ``error`` included),
    so a batch where every fit failed still returns its failure records instead
    of crashing while summarising an empty table.
    """

    replications: pd.DataFrame
    summary: pd.DataFrame
    run_summary: pd.DataFrame
    runs: pd.DataFrame
    scenario: str
    n: int
    n_bootstrap: int

    def __str__(self) -> str:  # pragma: no cover - convenience
        return (
            f"RankBenchmarkResult(scenario={self.scenario!r}, n={self.n}, "
            f"n_bootstrap={self.n_bootstrap}):\n{self.summary}\n{self.run_summary}"
        )


#: Stable schemas so all-failed batches still return well-formed empty tables.
_RANK_REPLICATION_COLUMNS: tuple[str, ...] = (
    "scenario",
    "n",
    "replication",
    "seed",
    "label",
    "mu_true",
    "true_rank_min",
    "true_rank_max",
    "inference_available",
    "mu_hat",
    "point_rank",
    "rank_low",
    "rank_high",
    "rank_width",
    "covers",
)

_RANK_RUN_COLUMNS: tuple[str, ...] = (
    "scenario",
    "n",
    "replication",
    "seed",
    "status",
    "error",
    "n_confirmed_pairs",
    "any_confirmed",
    "all_rank_sets_full",
    "inference_available",
)

_RANK_SUMMARY_COLUMNS: tuple[str, ...] = (
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
)

_RANK_RUN_SUMMARY_COLUMNS: tuple[str, ...] = (
    "scenario",
    "n",
    "n_replications",
    "n_failed",
    "blocked_rate",
    "any_confirmed_rate",
    "mean_n_confirmed_pairs",
    "all_rank_sets_full_rate",
    "confidence_level",
    "n_bootstrap",
)


def run_rank_benchmark(
    scenario: str | DGPConfig = "confounded",
    n: int = 2_000,
    n_replications: int = 20,
    n_bootstrap: int = 200,
    random_state: int = 2026,
    cross_fit: int | None = None,
    confidence_level: float = 0.95,
    higher_is_better: bool = True,
    n_jobs: int = 1,
    bootstrap_min_successes: int = 30,
    estimator_kwargs: dict[str, Any] | None = None,
) -> RankBenchmarkResult:
    """Validate the rank confidence sets against the known truth of the DGP.

    Per replication the full pipeline runs *with* the joint bootstrap, so every
    replication yields simultaneous pairwise intervals and the rank confidence
    sets ``[L_i, U_i] = [1 + B_i, K - W_i]``. The truth is tie-aware: for arm
    ``i``

    ``true_rank_min_i = 1 + #{j : mu_true_j strictly better than i}``,
    ``true_rank_max_i = K - #{j : mu_true_j strictly worse than i}``,

    which collapses to a single value whenever the true effects are distinct
    and widens to an interval under true ties (e.g. ``equal_effects``).

    ``covers_i`` is ``L_i <= true_rank_min_i`` and ``true_rank_max_i <= U_i``:
    the set must contain **every** admissible true rank. Under true ties this
    is exactly the red line from PROJECT_SPEC.md 3.6 - noise must not produce
    a determined ranking - and any spurious confirmed pair shows up as a
    non-covering interval and in ``any_confirmed_rate``.

    Monte Carlo error of the coverage proportion is reported as
    ``sqrt(p (1 - p) / R)``. The results are a simulation-based check of the
    joint-bootstrap + asymptotic-normal ranking statement, not a universal
    inference guarantee.
    """
    config = get_scenario(scenario) if isinstance(scenario, str) else scenario
    dgp = SyntheticDGP(config)
    mu_true, _ = dgp.population_mean()
    k = config.n_treatments

    score = np.asarray(mu_true, dtype=float)
    if not higher_is_better:
        score = -score
    true_rank_min = np.array([1 + int(np.sum(score > score[a])) for a in range(k)], dtype=int)
    true_rank_max = np.array([k - int(np.sum(score < score[a])) for a in range(k)], dtype=int)

    root = np.random.SeedSequence(int(random_state))
    seeds = [int(np.asarray(c.generate_state(1))[0]) for c in root.spawn(int(n_replications))]

    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    extra_kwargs: dict[str, Any] = dict(estimator_kwargs or {})
    for rep, seed in enumerate(seeds):
        df = dgp.sample(int(n), random_state=seed)
        model = CausalRank(
            treatment=TREATMENT_COL,
            outcome=OUTCOME_COL,
            confounders=list(CONFOUNDER_COLS),
            random_state=seed,
            cross_fit=cross_fit,
            n_bootstrap=int(n_bootstrap),
            bootstrap_min_successes=int(bootstrap_min_successes),
            confidence_level=confidence_level,
            higher_is_better=higher_is_better,
            n_jobs=n_jobs,
            **extra_kwargs,
        )
        try:
            result = model.fit(df)
        except Exception as exc:  # recorded, never aborts the batch
            run_rows.append(
                {
                    "scenario": config.name,
                    "n": int(n),
                    "replication": rep,
                    "seed": seed,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "n_confirmed_pairs": np.nan,
                    "any_confirmed": np.nan,
                    "all_rank_sets_full": np.nan,
                    "inference_available": False,
                }
            )
            continue

        ranks_obj = result.ranks
        pairs_obj = result.pairwise
        if result.inference_available and ranks_obj is not None and pairs_obj is not None:
            ranks_df = ranks_obj.to_dataframe().set_index("label")
            pairs_df = pairs_obj.to_dataframe()
            n_confirmed = int(pairs_df["confirmed"].astype(bool).sum())
            full = bool((ranks_df["rank_low"] == 1).all() and (ranks_df["rank_high"] == k).all())
            available = True
        else:
            ranks_df = None
            n_confirmed = -1
            full = False
            available = False

        run_rows.append(
            {
                "scenario": config.name,
                "n": int(n),
                "replication": rep,
                "seed": seed,
                "status": "ok",
                "error": "",
                "n_confirmed_pairs": n_confirmed,
                "any_confirmed": bool(n_confirmed > 0),
                "all_rank_sets_full": full,
                "inference_available": available,
            }
        )

        for a, label in enumerate(config.labels):
            row: dict[str, Any] = {
                "scenario": config.name,
                "n": int(n),
                "replication": rep,
                "seed": seed,
                "label": label,
                "mu_true": float(mu_true[a]),
                "true_rank_min": int(true_rank_min[a]),
                "true_rank_max": int(true_rank_max[a]),
                "inference_available": available,
            }
            if available and ranks_df is not None:
                rank_low = int(ranks_df.loc[label, "rank_low"])
                rank_high = int(ranks_df.loc[label, "rank_high"])
                row.update(
                    {
                        "mu_hat": float(ranks_df.loc[label, "mu"]),
                        "point_rank": int(ranks_df.loc[label, "point_rank"]),
                        "rank_low": rank_low,
                        "rank_high": rank_high,
                        "rank_width": rank_high - rank_low + 1,
                        "covers": bool(
                            rank_low <= true_rank_min[a] and true_rank_max[a] <= rank_high
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "mu_hat": np.nan,
                        "point_rank": np.nan,
                        "rank_low": np.nan,
                        "rank_high": np.nan,
                        "rank_width": np.nan,
                        "covers": np.nan,
                    }
                )
            rows.append(row)

    replications = pd.DataFrame(rows)
    if replications.empty:
        replications = pd.DataFrame(columns=_RANK_REPLICATION_COLUMNS)
    runs = pd.DataFrame(run_rows)
    if runs.empty:
        runs = pd.DataFrame(columns=_RANK_RUN_COLUMNS)

    summary_rows: list[dict[str, Any]] = []
    if "label" in replications.columns:
        for label, grp in replications.groupby("label", sort=True):
            ok = grp[grp["inference_available"] == True]  # noqa: E712
            p = float(ok["covers"].mean()) if not ok.empty else np.nan
            mc_se = (
                float(np.sqrt(p * (1 - p) / max(ok.shape[0], 1))) if ok.shape[0] else np.nan
            )
            summary_rows.append(
                {
                    "scenario": config.name,
                    "n": int(n),
                    "label": label,
                    "n_replications": int(ok.shape[0]),
                    "mu_true": float(grp["mu_true"].iloc[0]),
                    "true_rank_min": int(grp["true_rank_min"].iloc[0]),
                    "true_rank_max": int(grp["true_rank_max"].iloc[0]),
                    "coverage": p,
                    "coverage_mc_se": mc_se,
                    "mean_rank_width": (
                        float(ok["rank_width"].mean()) if not ok.empty else np.nan
                    ),
                    "confidence_level": float(confidence_level),
                    "n_bootstrap": int(n_bootstrap),
                    "higher_is_better": bool(higher_is_better),
                }
            )
    summary = pd.DataFrame(summary_rows, columns=list(_RANK_SUMMARY_COLUMNS))

    ok_runs = runs[runs["status"] == "ok"]
    avail = ok_runs[ok_runs["inference_available"] == True]  # noqa: E712
    n_ok = int(ok_runs.shape[0])
    run_summary = pd.DataFrame(
        [
            {
                "scenario": config.name,
                "n": int(n),
                "n_replications": int(runs.shape[0]),
                "n_failed": int((runs["status"] != "ok").sum()),
                "blocked_rate": (
                    float(1.0 - (avail.shape[0] / n_ok)) if n_ok else np.nan
                ),
                "any_confirmed_rate": (
                    float(avail["any_confirmed"].astype(bool).mean())
                    if not avail.empty
                    else np.nan
                ),
                "mean_n_confirmed_pairs": (
                    float(avail["n_confirmed_pairs"].mean()) if not avail.empty else np.nan
                ),
                "all_rank_sets_full_rate": (
                    float(avail["all_rank_sets_full"].astype(bool).mean())
                    if not avail.empty
                    else np.nan
                ),
                "confidence_level": float(confidence_level),
                "n_bootstrap": int(n_bootstrap),
            }
        ],
        columns=list(_RANK_RUN_SUMMARY_COLUMNS),
    )

    return RankBenchmarkResult(
        replications=replications,
        summary=summary,
        run_summary=run_summary,
        runs=runs,
        scenario=config.name,
        n=int(n),
        n_bootstrap=int(n_bootstrap),
    )
