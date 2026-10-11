"""Research validation of the Phase 2 uncertainty quantification.

Compares the joint-bootstrap SEs with the empirical sampling variability of the
estimator and reports pointwise CI coverage with its Monte Carlo error::

    uv run python scripts/run_se_validation.py --scenario confounded --n 1500 \
        --replications 40 --bootstrap 200 --cross-fit 5 --out artifacts/se_validation.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from causalrank.simulation import run_se_benchmark
from causalrank.simulation.dgp import scenario_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="confounded", choices=list(scenario_names()))
    parser.add_argument("--n", type=int, default=1_500)
    parser.add_argument("--replications", type=int, default=40)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--cross-fit", type=int, default=5)
    parser.add_argument("--ci-method", default="normal", choices=["normal", "percentile"])
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("artifacts/se_validation.csv"))
    args = parser.parse_args()

    t0 = time.time()
    res = run_se_benchmark(
        args.scenario,
        n=args.n,
        n_replications=args.replications,
        n_bootstrap=args.bootstrap,
        random_state=args.seed,
        cross_fit=args.cross_fit if args.cross_fit > 0 else None,
        ci_method=args.ci_method,
        n_jobs=args.n_jobs,
    )
    elapsed = time.time() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.replications.to_csv(args.out, index=False)
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    res.summary.to_csv(summary_path, index=False)

    print(res.summary.round(5).to_string(index=False))
    print()
    print("se_ratio = mean bootstrap SE / empirical sd of mu_hat across replications")
    print("coverage_mc_se = sqrt(p(1-p)/R) is reported alongside coverage")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"wrote {args.out} and {summary_path}")


if __name__ == "__main__":
    main()
