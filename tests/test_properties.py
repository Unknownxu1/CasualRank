"""Numerical invariance property tests (Hypothesis)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from causalrank.estimators.iptw import hajek_estimate
from causalrank.estimators.propensity import align_proba_columns

LABELS = ("A", "B")

_pos_floats = st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False)
_binary = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def _propensity_from_e_obs(e_obs: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
    """Build a positive (n, 2) matrix whose observed column equals ``e_obs``.

    Only the observed column enters the Hajek weights; the other column is a
    placeholder kept positive (the estimator requires non-negative entries).
    """
    p = np.ones((e_obs.size, 2), dtype=float)
    p[np.arange(e_obs.size), t_idx] = e_obs
    return p


@settings(max_examples=40, deadline=None)
@given(
    e_a=st.lists(_pos_floats, min_size=3, max_size=8),
    e_b=st.lists(_pos_floats, min_size=3, max_size=8),
    y_a=st.lists(_binary, min_size=3, max_size=8),
    y_b=st.lists(_binary, min_size=3, max_size=8),
    scale=st.floats(min_value=0.25, max_value=4.0, allow_nan=False, allow_infinity=False),
)
def test_hajek_invariant_to_within_group_weight_scaling(
    e_a: list[float], e_b: list[float], y_a: list[float], y_b: list[float], scale: float
) -> None:
    n_a = min(len(e_a), len(y_a))
    n_b = min(len(e_b), len(y_b))
    e_obs = np.array(e_a[:n_a] + e_b[:n_b], dtype=float)
    y = np.array(y_a[:n_a] + y_b[:n_b], dtype=float)
    t_idx = np.array([0] * n_a + [1] * n_b, dtype=np.int64)

    base = hajek_estimate(y, t_idx, _propensity_from_e_obs(e_obs, t_idx), LABELS)

    scaled_e = e_obs.copy()
    scaled_e[t_idx == 0] *= scale  # scaling weights of arm A by 1/scale
    alt = hajek_estimate(y, t_idx, _propensity_from_e_obs(scaled_e, t_idx), LABELS)

    assert np.allclose(alt.mu, base.mu, rtol=0, atol=1e-9)
    # arm B is untouched
    assert alt.mu[1] == base.mu[1]


@settings(max_examples=40, deadline=None)
@given(
    e=st.lists(_pos_floats, min_size=6, max_size=12),
    y=st.lists(_binary, min_size=6, max_size=12),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_hajek_invariant_to_row_order(e: list[float], y: list[float], seed: int) -> None:
    n = min(len(e), len(y))
    e_obs = np.array(e[:n], dtype=float)
    y_arr = np.array(y[:n], dtype=float)
    t_idx = np.array([i % 2 for i in range(n)], dtype=np.int64)

    base = hajek_estimate(y_arr, t_idx, _propensity_from_e_obs(e_obs, t_idx), LABELS)

    perm = np.random.default_rng(seed).permutation(n)
    alt = hajek_estimate(
        y_arr[perm], t_idx[perm], _propensity_from_e_obs(e_obs[perm], t_idx[perm]), LABELS
    )
    assert np.allclose(alt.mu, base.mu, rtol=0, atol=1e-9)


@settings(max_examples=50, deadline=None)
@given(
    rows=st.lists(
        st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=3, max_size=3),
        min_size=1,
        max_size=6,
    ),
    perm=st.permutations(["A", "B", "C"]),
)
def test_align_proba_columns_respects_model_class_order(
    rows: list[list[float]], perm: list[str]
) -> None:
    proba = np.array(rows, dtype=float)
    aligned = align_proba_columns(proba, perm, ["A", "B", "C"])
    for k, label in enumerate(["A", "B", "C"]):
        src = perm.index(label)
        assert np.allclose(aligned[:, k], proba[:, src])
