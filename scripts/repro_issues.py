"""Ad-hoc repro for the reported issues (kept out of the test-suite)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causalrank import CausalRank
from causalrank.config import CausalRankConfig
from causalrank.diagnostics.balance import standardized_mean_differences
from causalrank.diagnostics.overlap import overlap_diagnostics


def repro_overlap() -> None:
    print("== P1-1 overlap common support ==")
    p = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
    t = np.array([0, 1, 2])
    cfg = CausalRankConfig(treatment="t", outcome="y", confounders=("x",))
    table, warns = overlap_diagnostics(p, t, ("A", "B", "C"), cfg)
    print(table.to_string(index=False))
    print("warnings:", warns)
    print("-> expected: warnings about unsupported regions for every arm")


def repro_dtypes() -> None:
    print("== P1-2 dtypes ==")
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame(
        {
            "x_bool": rng.random(n) > 0.5,
            "x_str": pd.Series([["a", "b", "c"][i % 3] for i in range(n)]),
            "x_obj": pd.Series([("m", "n", np.nan)[i % 3] for i in range(n)], dtype=object),
            "treatment": np.array(["A", "B"] * (n // 2)),
            "outcome": rng.integers(0, 2, size=n).astype(float),
        }
    )
    print("dtypes:", dict(df.dtypes.astype(str)))
    for col in ["x_bool", "x_str", "x_obj"]:
        try:
            res = CausalRank(
                treatment="treatment",
                outcome="outcome",
                confounders=[col],
                random_state=0,
                gbm_params={"max_iter": 10, "max_depth": 2, "min_samples_leaf": 5},
            ).fit(df)
            print(f"  {col}: OK gate={res.diagnostics.gate}")
        except Exception as exc:
            print(f"  {col}: FAIL {type(exc).__name__}: {str(exc)[:120]}")
    print("-> expected: all three run without error")


def repro_balance_shift() -> None:
    print("== P1-3 SMD shift instability ==")
    cfg = CausalRankConfig(treatment="t", outcome="y", confounders=("x",))
    base = pd.DataFrame({"x": [1.0, 3.0, 2.0, 6.0]})
    shifted = base + 1e9
    for name, frame in [("base", base), ("shifted +1e9", shifted)]:
        res = standardized_mean_differences(
            frame,
            np.array([0, 0, 1, 1]),
            ("A", "B"),
            None,
            ["x"],
            [],
            cfg,
        )
        row = res.detail.iloc[0]
        print(f"  {name}: smd_unweighted={row['smd_unweighted']}, sd_ref={row['sd_ref']}")
    print("-> expected: identical SMD (-0.8944) in both cases")


if __name__ == "__main__":
    repro_overlap()
    repro_dtypes()
    repro_balance_shift()
