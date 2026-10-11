"""Research benchmark: repeated-experiment validation against known truth.

Run manually (not part of the fast test-suite)::

    uv run python scripts/run_simulation.py --scenario confounded --n 1500 6000 \
        --replications 20 --cross-fit 5 --out artifacts/benchmark_confounded.csv

The script reports bias / RMSE of the Hajek IPTW estimates versus the Monte
Carlo ground truth, together with the raw (unadjusted) group means for
comparison, and writes both replication-level and summary tables. Monte Carlo
error of a proportion p_hat over R replications is sqrt(p_hat(1-p_hat)/R) and
must accompany any coverage claim made from these runs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from causalrank.simulation import run_estimand_benchmark
from causalrank.simulation.dgp import scenario_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="confounded", choices=list(scenario_names()))
    parser.add_argument("--n", type=int, nargs="+", default=[1_500, 6_000])
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--cross-fit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reference-size", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("artifacts/benchmark.csv"))
    args = parser.parse_args()

    t0 = time.time()
    bench = run_estimand_benchmark(
        args.scenario,
        n_values=args.n,
        n_replications=args.replications,
        random_state=args.seed,
        cross_fit=args.cross_fit if args.cross_fit > 0 else None,
        n_reference=args.reference_size,
    )
    elapsed = time.time() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bench.replications.to_csv(args.out, index=False)
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    bench.summary.to_csv(summary_path, index=False)
    runs_path = args.out.with_name(args.out.stem + "_runs.csv")
    bench.runs.to_csv(runs_path, index=False)

    print("== arm-level bias / RMSE (successful runs only) ==")
    print(bench.summary.round(5).to_string(index=False))
    print()
    print("== run-level quality statistics ==")
    print(bench.runs_summary.round(4).to_string(index=False))
    print()
    print(f"ground truth reference: n = {bench.n_reference}, seed = {bench.reference_seed}")
    print(f"runs: {len(bench.runs)} ({int((bench.runs['status'] != 'ok').sum())} failed)")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"wrote {args.out}, {summary_path} and {runs_path}")


if __name__ == "__main__":
    main()
