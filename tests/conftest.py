"""Shared fixtures and helpers for the CausalRank test-suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrank import CausalRankConfig

#: Fast GBM settings for tests where model *quality* is irrelevant.
FAST_GBM = {
    "max_iter": 30,
    "learning_rate": 0.2,
    "max_depth": 3,
    "min_samples_leaf": 10,
    "l2_regularization": 1.0,
}


def make_config(**overrides: object) -> CausalRankConfig:
    """Build a valid config on a tiny handcrafted schema."""
    params: dict[str, object] = {
        "treatment": "treatment",
        "outcome": "outcome",
        "confounders": ("x1", "x2", "x3"),
    }
    params.update(overrides)
    return CausalRankConfig(**params)  # type: ignore[arg-type]


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    """Small deterministic frame used by validation tests."""
    rng = np.random.default_rng(0)
    n = 60
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.integers(0, 2, size=n).astype(float)
    treatment = np.array(["A", "B", "C"] * (n // 3))
    outcome = ((x1 + 0.3 * x3 + (treatment == "A")) > 0).astype(float)
    return pd.DataFrame(
        {"x1": x1, "x2": x2, "x3": x3, "treatment": treatment, "outcome": outcome}
    )


@pytest.fixture()
def hand_df() -> pd.DataFrame:
    """Four-row frame used for hand-computed IPTW checks.

    Rows: two A and two B observations with known outcomes and covariates.
    """
    return pd.DataFrame(
        {
            "treatment": ["A", "A", "B", "B"],
            "outcome": [1.0, 0.0, 1.0, 0.0],
            "x1": [1.0, 3.0, 2.0, 6.0],
            "x2": [0.0, 1.0, 0.0, 0.0],
            "x3": ["a", "b", "a", "a"],
        }
    )
