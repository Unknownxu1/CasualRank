"""Repeated-experiment benchmarks: bias / RMSE of the IPTW estimator vs truth.

These helpers are the validation backbone demanded by PROJECT_SPEC.md section
5.3: point estimates must be compared against the known ``mu_true`` over many
replications (a single lucky sample proves nothing). Fast variants run in CI
(``@pytest.mark.slow``-sized); the full research grid is run manually.

Monte Carlo error of a coverage-style proportion ``p_hat`` over ``R``
replications is ``sqrt(p_hat (1 - p_hat) / R)`` and must be reported alongside
any conclusion drawn from the benchmark.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..api import CausalRank
from .dgp import CONFOUNDER_COLS, OUTCOME_COL, TREATMENT_COL, DGPConfig, SyntheticDGP, get_scenario


@dataclass
class BenchmarkResult:
    """Raw replication-level table plus the aggregated summary."""

    replications: pd.DataFrame
    summary: pd.DataFrame
    scenario: str
    n_reference: int
    reference_seed: int

    def __str__(self) -> str:  # pragma: no cover - convenience
        return (
            f"BenchmarkResult(scenario={self.scenario!r}, "
            f"replications={len(self.replications)}), summary=\n{self.summary}"
        )


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
    spawn), the full Phase 1 pipeline runs, and both the IPTW estimate and the
    raw (unadjusted) group mean are recorded. Aggregated bias/RMSE for both are
    the honest way to show that adjustment actually removes confounding bias.
    """
    config = get_scenario(scenario) if isinstance(scenario, str) else scenario
    dgp = SyntheticDGP(config)
    mu_true, mu_se = dgp.population_mean(n_reference=n_reference)
    n_ref = int(n_reference) if n_reference is not None else int(config.reference_size)

    kwargs = dict(estimator_kwargs or {})
    kwargs.setdefault("cross_fit", cross_fit)

    root = np.random.SeedSequence(int(random_state))
    child_seeds = root.spawn(len(list(n_values)) * int(n_replications))
    seed_iter = iter(child_seeds)

    rows: list[dict[str, Any]] = []
    for n in n_values:
        for rep in range(int(n_replications)):
            seed = int(np.asarray(next(seed_iter).generate_state(1))[0])
            df = dgp.sample(int(n), random_state=seed)
            model = CausalRank(
                treatment=TREATMENT_COL,
                outcome=OUTCOME_COL,
                confounders=list(CONFOUNDER_COLS),
                random_state=seed,
                **kwargs,
            )
            result = model.fit(df)
            est = result.effects.to_dataframe().set_index("label")
            for a, label in enumerate(config.labels):
                rows.append(
                    {
                        "scenario": config.name,
                        "n": int(n),
                        "replication": rep,
                        "seed": seed,
                        "label": label,
                        "mu_true": float(mu_true[a]),
                        "mu_true_mc_se": float(mu_se[a]),
                        "mu_hat": float(est.loc[label, "mu"]),
                        "raw_mean": float(est.loc[label, "raw_mean"]),
                        "ess": float(est.loc[label, "ess"]),
                    }
                )

    replications = pd.DataFrame(rows)
    summary = summarize_benchmark(replications)
    return BenchmarkResult(
        replications=replications,
        summary=summary,
        scenario=config.name,
        n_reference=n_ref,
        reference_seed=int(config.reference_seed),
    )


def summarize_benchmark(replications: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replication rows into bias / RMSE per (scenario, n, arm)."""
    out: list[dict[str, Any]] = []
    grouped = replications.groupby(["scenario", "n", "label"], sort=True)
    for (scenario, n, label), grp in grouped:
        mu_true = float(grp["mu_true"].iloc[0])
        err = grp["mu_hat"].to_numpy(dtype=float) - mu_true
        raw_err = grp["raw_mean"].to_numpy(dtype=float) - mu_true
        out.append(
            {
                "scenario": scenario,
                "n": int(n),
                "label": label,
                "n_replications": int(grp.shape[0]),
                "mu_true": mu_true,
                "mean_mu_hat": float(grp["mu_hat"].mean()),
                "bias": float(err.mean()),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "sd_across_replications": float(np.std(err, ddof=1)) if len(err) > 1 else np.nan,
                "bias_raw": float(raw_err.mean()),
                "rmse_raw": float(np.sqrt(np.mean(raw_err**2))),
                "mean_ess": float(grp["ess"].mean()),
            }
        )
    return pd.DataFrame(out)
