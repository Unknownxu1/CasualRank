"""Research validation of the Phase 3 rank inference.

Runs the full pipeline with the joint bootstrap on DGP scenarios with known
truth and reports rank confidence-set coverage with its Monte Carlo error,
plus the false-confirmation behaviour under true ties::

    uv run python scripts/run_rank_validation.py --scenario confounded --n 1500 \
        --replications 40 --bootstrap 200 --cross-fit 5 --out artifacts/rank_validation.csv

Use ``--scenario equal_effects`` to check the red line: under true ties the
simultaneous intervals must almost never confirm a direction
(``any_confirmed_rate`` close to 0, rank sets staying ``[1, K]``).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from causalrank.simulation import run_rank_benchmark
from causalrank.simulation.dgp import scenario_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="confounded", choices=list(scenario_names()))
    parser.add_argument("--n", type=int, default=1_500)
    parser.add_argument("--replications", type=int, default=40)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--cross-fit", type=int, default=5)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="rank 1 = lowest outcome (e.g. cost) instead of the highest",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("artifacts/rank_validation.csv"))
    args = parser.parse_args()

    t0 = time.time()
    res = run_rank_benchmark(
        args.scenario,
        n=args.n,
        n_replications=args.replications,
        n_bootstrap=args.bootstrap,
        random_state=args.seed,
        cross_fit=args.cross_fit if args.cross_fit > 0 else None,
        confidence_level=args.confidence_level,
        higher_is_better=not args.lower_is_better,
        n_jobs=args.n_jobs,
    )
    elapsed = time.time() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.replications.to_csv(args.out, index=False)
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    res.summary.to_csv(summary_path, index=False)
    run_summary_path = args.out.with_name(args.out.stem + "_runs.csv")
    res.run_summary.to_csv(run_summary_path, index=False)

    print(res.summary.round(5).to_string(index=False))
    print()
    print(res.run_summary.round(5).to_string(index=False))
    print()
    print("coverage = P(true rank in [rank_low, rank_high]), tie-aware truth;")
    print("coverage_mc_se = sqrt(p(1-p)/R) is reported alongside coverage")
    print("any_confirmed_rate = share of replications with >= 1 confirmed pair")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"wrote {args.out}, {summary_path} and {run_summary_path}")


if __name__ == "__main__":
    main()
