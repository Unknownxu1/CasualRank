"""IPTW tests with independently hand-computed examples (PROJECT_SPEC.md 7.2)."""

from __future__ import annotations

import numpy as np
import pytest

from causalrank.estimators.iptw import (
    effective_sample_size,
    hajek_estimate,
    horvitz_thompson_estimate,
    observed_propensity,
)
from causalrank.exceptions import CausalRankError

LABELS = ("A", "B")


def _hand_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2 treatments x 4 samples with known propensities.

    t = [A, A, B, B],  y = [1, 0, 1, 0]
    e_A(x) = [0.8, 0.4] for the A rows, e_B(x) = [0.7, 0.4] for the B rows.
    """
    t_idx = np.array([0, 0, 1, 1])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    propensity = np.array(
        [
            [0.8, 0.2],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.6, 0.4],
        ]
    )
    return y, t_idx, propensity


def test_hajek_matches_hand_computation() -> None:
    y, t_idx, propensity = _hand_problem()
    est = hajek_estimate(y, t_idx, propensity, LABELS)

    # w = [1/0.8, 1/0.4, 1/0.7, 1/0.4] = [1.25, 2.5, 10/7, 2.5]
    assert est.weights == pytest.approx([1.25, 2.5, 10 / 7, 2.5], rel=1e-12)

    # mu_A = 1.25 / (1.25 + 2.5) = 1/3
    assert est.mu[0] == pytest.approx(1.0 / 3.0, rel=1e-12)
    # mu_B = (10/7) / (10/7 + 2.5) = 4/11
    assert est.mu[1] == pytest.approx(4.0 / 11.0, rel=1e-12)

    # ESS_A = (3.75)^2 / (1.25^2 + 2.5^2) = 1.8
    assert est.ess[0] == pytest.approx(1.8, rel=1e-12)
    assert est.n_treated.tolist() == [2, 2]
    assert est.raw_means == pytest.approx([0.5, 0.5], rel=1e-12)


def test_horvitz_thompson_internal_reference() -> None:
    y, t_idx, propensity = _hand_problem()
    ht = horvitz_thompson_estimate(y, t_idx, propensity, LABELS)
    # mu_A^HT = (1/4)(1/0.8) = 0.3125 ; mu_B^HT = (1/4)(1/0.7)
    assert ht[0] == pytest.approx(0.3125, rel=1e-12)
    assert ht[1] == pytest.approx(0.25 / 0.7, rel=1e-12)


def test_within_group_weight_scaling_invariance() -> None:
    """Scaling every weight of one arm by a constant must not move its mean."""
    y, t_idx, propensity = _hand_problem()
    base = hajek_estimate(y, t_idx, propensity, LABELS)

    scaled = propensity.copy()
    scaled[t_idx == 0, 0] *= 0.5  # doubles the weights of arm A rows
    alt = hajek_estimate(y, t_idx, scaled, LABELS)
    assert alt.mu[0] == pytest.approx(base.mu[0], rel=1e-12)
    assert alt.mu[1] == pytest.approx(base.mu[1], rel=1e-12)


def test_hajek_mean_in_unit_interval_for_binary_outcome() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=50).astype(float)
    t_idx = rng.integers(0, 3, size=50)
    e = rng.uniform(0.05, 1.0, size=(50, 3))
    e = e / e.sum(axis=1, keepdims=True)
    est = hajek_estimate(y, t_idx, e, ("A", "B", "C"))
    assert np.all((est.mu >= 0.0) & (est.mu <= 1.0))


def test_zero_propensity_fails_fast() -> None:
    y, t_idx, propensity = _hand_problem()
    propensity = propensity.copy()
    propensity[1, 0] = 0.0  # row treated with A has zero propensity for A
    with pytest.raises(CausalRankError, match="zero/negative propensity"):
        hajek_estimate(y, t_idx, propensity, LABELS)


def test_explicit_clipping_is_reported() -> None:
    y, t_idx, propensity = _hand_problem()
    est = hajek_estimate(y, t_idx, propensity, LABELS, clip_propensity=0.45)
    assert est.clipped_fraction == pytest.approx(0.5)  # two of four observed props below 0.45
    assert np.all(est.observed_propensity >= 0.45)
    # with clipping, weights change -> mu_A no longer equals 1/3
    assert est.mu[0] != pytest.approx(1.0 / 3.0, rel=1e-6)


def test_effective_sample_size_hand_values() -> None:
    assert effective_sample_size(np.array([1.0, 3.0])) == pytest.approx(1.6, rel=1e-12)
    assert effective_sample_size(np.array([1.0, 1.0, 1.0])) == pytest.approx(3.0, rel=1e-12)
    assert effective_sample_size(np.array([5.0, 0.0])) == pytest.approx(1.0, rel=1e-12)


def test_observed_propensity_selects_treatment_column() -> None:
    _, t_idx, propensity = _hand_problem()
    e_obs = observed_propensity(propensity, t_idx)
    assert e_obs == pytest.approx([0.8, 0.4, 0.7, 0.4], rel=1e-12)


def test_shape_mismatch_raises() -> None:
    y, t_idx, propensity = _hand_problem()
    with pytest.raises(CausalRankError, match="shape"):
        hajek_estimate(y, t_idx, propensity[:, :1], LABELS)
    with pytest.raises(CausalRankError, match="aligned"):
        hajek_estimate(y[:3], t_idx, propensity, LABELS)
