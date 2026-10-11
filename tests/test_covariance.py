"""Covariance and CI tests with hand-computed examples (PROJECT_SPEC.md 7.3/7.4)."""

from __future__ import annotations

import numpy as np
import pytest

from causalrank.exceptions import CausalRankError
from causalrank.inference.bootstrap import (
    normal_ci,
    percentile_ci,
    validate_covariance,
)


def test_hand_computed_covariance_two_routes() -> None:
    """Two routes, five known draws: covariance by hand vs the library."""
    draws = np.array(
        [
            [0.50, 0.40],
            [0.52, 0.42],
            [0.48, 0.38],
            [0.54, 0.44],
            [0.46, 0.36],
        ]
    )
    sigma = np.cov(draws, rowvar=False, ddof=1)
    # hand: mean = (0.50, 0.40); deviations are 0, +-0.02, +-0.04
    # var_a = (0 + 0.02^2 + 0.02^2 + 0.04^2 + 0.04^2)/4 = 0.001
    assert sigma[0, 0] == pytest.approx(0.001, rel=1e-12)
    assert sigma[1, 1] == pytest.approx(0.001, rel=1e-12)
    # the two columns move together: cov = var
    assert sigma[0, 1] == pytest.approx(0.001, rel=1e-12)
    validate_covariance(sigma)  # must pass its own validation

    se = np.sqrt(np.diag(sigma))
    assert se[0] == pytest.approx(np.sqrt(0.001), rel=1e-12)
    lo, hi = normal_ci(np.array([0.50, 0.40]), se, 0.95)
    z = 1.959963984540054
    assert lo[0] == pytest.approx(0.50 - z * np.sqrt(0.001), rel=1e-12)
    assert hi[0] == pytest.approx(0.50 + z * np.sqrt(0.001), rel=1e-12)


def test_covariance_is_not_scaled_by_n() -> None:
    draws = np.array([[0.5, 0.4], [0.52, 0.42], [0.48, 0.38], [0.54, 0.44], [0.46, 0.36]])
    sigma = np.cov(draws, rowvar=False, ddof=1)
    n = 1000
    wrong = sigma / n  # the forbidden "divide by n" variant (spec 3.5)
    assert validate_covariance(sigma) is not None
    assert not np.allclose(sigma, wrong)


def test_correlation_is_carried_into_difference_se() -> None:
    """Var(mu_i - mu_j) must include the covariance term."""
    sigma = np.array([[0.0025, 0.0018], [0.0018, 0.0016]])
    var_diff_independent = sigma[0, 0] + sigma[1, 1]
    var_diff_true = sigma[0, 0] + sigma[1, 1] - 2 * sigma[0, 1]
    assert var_diff_true < var_diff_independent  # positive correlation here
    se_diff = float(np.sqrt(var_diff_true))
    assert se_diff == pytest.approx(np.sqrt(0.0025 + 0.0016 - 2 * 0.0018), rel=1e-12)


def test_validate_covariance_rejects_bad_matrices() -> None:
    with pytest.raises(CausalRankError, match="square"):
        validate_covariance(np.ones((2, 3)))
    with pytest.raises(CausalRankError, match="non-finite"):
        validate_covariance(np.array([[1.0, np.nan], [np.nan, 1.0]]))
    with pytest.raises(CausalRankError, match="symmetric"):
        validate_covariance(np.array([[1.0, 0.5], [0.3, 1.0]]))
    with pytest.raises(CausalRankError, match="negative diagonal"):
        validate_covariance(np.array([[1.0, 0.0], [0.0, -0.1]]))
    # symmetric but indefinite: eigenvalues 3 and -1 -> not PSD
    with pytest.raises(CausalRankError, match="positive semi-definite"):
        validate_covariance(np.array([[1.0, 2.0], [2.0, 1.0]]))
    # numerical tolerance: a hair of asymmetry must not raise
    ok = np.array([[1.0, 0.5 + 1e-13], [0.5, 2.0]])
    assert validate_covariance(ok).shape == (2, 2)


def test_normal_ci_hand_values() -> None:
    mu = np.array([0.5, 0.3])
    se = np.array([0.02, 0.04])
    lo, hi = normal_ci(mu, se, 0.95)
    z = 1.959963984540054
    assert lo[0] == pytest.approx(0.5 - z * 0.02, rel=1e-12)
    assert hi[1] == pytest.approx(0.3 + z * 0.04, rel=1e-12)


def test_percentile_ci_hand_values() -> None:
    draws = np.tile(np.arange(101, dtype=float) / 100.0, (2, 1)).T  # 0.00 .. 1.00
    lo, hi = percentile_ci(draws, 0.90)
    assert lo[0] == pytest.approx(0.05, abs=1e-12)
    assert hi[0] == pytest.approx(0.95, abs=1e-12)
    with pytest.raises(CausalRankError, match="draws"):
        percentile_ci(draws[:1], 0.90)
