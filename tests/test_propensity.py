"""Propensity model tests: probabilities, class alignment, cross-fitting."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from causalrank.config import CausalRankConfig
from causalrank.estimators.propensity import (
    align_proba_columns,
    aligned_predict_proba,
    fit_propensity,
    make_propensity_pipeline,
)
from causalrank.exceptions import CausalRankError, CausalRankValidationError
from causalrank.simulation import make_dataset

from .conftest import FAST_GBM

LABELS = ("A", "B", "C")


def _toy_data(n: int = 240, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    ds = make_dataset("randomized", n=n, random_state=seed)
    df = ds.df
    X = df[["x1", "x2", "x3", "x4"]]
    t_idx = df["treatment"].map({lab: i for i, lab in enumerate(LABELS)}).to_numpy()
    return X, t_idx


def _config(**overrides: Any) -> CausalRankConfig:
    params: dict[str, Any] = {
        "treatment": "treatment",
        "outcome": "outcome",
        "confounders": ("x1", "x2", "x3", "x4"),
        "gbm_params": dict(FAST_GBM),
        "random_state": 0,
    }
    params.update(overrides)
    return CausalRankConfig(**params)


def test_propensity_matrix_is_valid_probability_distribution() -> None:
    X, t_idx = _toy_data()
    fit = fit_propensity(
        X, t_idx, LABELS, ["x1", "x2", "x3"], ["x4"], _config()
    )
    p = fit.propensity
    assert p.shape == (len(t_idx), 3)
    assert np.all(p >= 0.0)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(np.isfinite(p))
    assert fit.labels == LABELS
    assert fit.oof is False
    assert len(fit.models) == 1


def test_align_proba_columns_with_scrambled_classes() -> None:
    proba = np.array(
        [
            [0.1, 0.6, 0.3],
            [0.7, 0.2, 0.1],
        ]
    )
    # model column order is (B, C, A)
    aligned = align_proba_columns(proba, ["B", "C", "A"], ["A", "B", "C"])
    expected = np.array([[0.3, 0.1, 0.6], [0.1, 0.7, 0.2]])
    assert np.allclose(aligned, expected)


def test_align_proba_columns_rejects_missing_class() -> None:
    proba = np.zeros((2, 2))
    with pytest.raises(CausalRankError, match="not seen"):
        align_proba_columns(proba, ["A", "B"], ["A", "B", "C"])


class _StubModel:
    def __init__(self, classes: list[int], proba: np.ndarray) -> None:
        self.classes_ = np.array(classes)
        self._proba = proba

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._proba


def test_aligned_predict_proba_maps_model_classes_to_labels() -> None:
    """sklearn columns follow ``classes_``; the public order must not assume it."""
    proba = np.array([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]])
    # the fitted model sees codes in the order (2, 0, 1)
    stub = SimpleNamespace(named_steps={"model": _StubModel([2, 0, 1], proba)}, predict_proba=lambda X: proba)
    aligned = aligned_predict_proba(stub, pd.DataFrame({"x": [0.0, 1.0]}), ["A", "B", "C"])
    # code 0 -> A (col 1), code 1 -> B (col 2), code 2 -> C (col 0)
    expected = np.array([[0.5, 0.3, 0.2], [0.1, 0.3, 0.6]])
    assert np.allclose(aligned, expected)


def test_cross_fit_produces_out_of_fold_probabilities() -> None:
    X, t_idx = _toy_data(n=300, seed=1)
    fit = fit_propensity(X, t_idx, LABELS, ["x1", "x2", "x3"], ["x4"], _config(cross_fit=5))
    assert fit.oof is True
    assert len(fit.models) == 5
    assert fit.fold_ids is not None
    assert sorted(np.unique(fit.fold_ids).tolist()) == [0, 1, 2, 3, 4]
    assert np.allclose(fit.propensity.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(fit.propensity > 0.0)


def test_cross_fit_is_reproducible_for_fixed_seed() -> None:
    X, t_idx = _toy_data(n=200, seed=2)
    a = fit_propensity(X, t_idx, LABELS, ["x1", "x2", "x3"], ["x4"], _config(cross_fit=3))
    b = fit_propensity(X, t_idx, LABELS, ["x1", "x2", "x3"], ["x4"], _config(cross_fit=3))
    assert np.array_equal(a.propensity, b.propensity)
    assert np.array_equal(a.fold_ids, b.fold_ids)


def test_cross_fit_requires_all_classes_in_every_training_fold() -> None:
    X, t_idx = _toy_data(n=20, seed=3)
    # fabricate a class with a single member so some training fold must miss it
    t_idx = t_idx.copy()
    t_idx[:] = 0
    t_idx[0] = 1
    t_idx[1] = 2
    with pytest.raises(CausalRankValidationError, match="cross_fit"):
        fit_propensity(X, t_idx, LABELS, ["x1", "x2", "x3"], ["x4"], _config(cross_fit=3))


def test_pipeline_handles_missing_and_unknown_categories() -> None:
    X, t_idx = _toy_data(n=200, seed=4)
    X = X.copy()
    X.loc[X.index[:20], "x2"] = np.nan
    X.loc[X.index[:20], "x4"] = np.nan
    fit = fit_propensity(X, t_idx, LABELS, ["x1", "x2", "x3"], ["x4"], _config(cross_fit=4))
    assert np.all(np.isfinite(fit.propensity))
    assert np.allclose(fit.propensity.sum(axis=1), 1.0, atol=1e-8)

    # unseen category at prediction time must not break the fitted pipeline
    pipe = make_propensity_pipeline(_config(), ["x1", "x2", "x3"], ["x4"])
    pipe.fit(X, t_idx)
    X_new = X.copy()
    X_new["x4"] = X_new["x4"].astype(object)
    X_new.loc[X_new.index[:5], "x4"] = "never_seen"
    proba = pipe.predict_proba(X_new.iloc[:5])
    assert proba.shape == (5, 3)


def test_integer_labels_keep_column_order() -> None:
    """Labels 3 < 7 < 42 must map to columns in that numeric order."""
    ds = make_dataset("strong_confounding", n=400, random_state=5)
    df = ds.df
    remap = {"A": 42, "B": 3, "C": 7}
    df["treatment"] = df["treatment"].map(remap)
    labels = (3, 7, 42)
    t_idx = df["treatment"].map({lab: i for i, lab in enumerate(labels)}).to_numpy()
    fit = fit_propensity(
        df[["x1", "x2", "x3", "x4"]], t_idx, labels, ["x1", "x2", "x3"], ["x4"], _config()
    )
    assert fit.labels == labels
    # strong confounding makes the assignment nearly deterministic, so a
    # misaligned probability column order would destroy argmax agreement
    assert (fit.propensity.argmax(axis=1) == t_idx).mean() > 0.7
