"""Multi-class propensity-score estimation: preprocessing + GBM + class alignment.

Naming follows the spec: the **GBM** here is the *prediction model* used to
estimate the multi-class propensity score ``e_a(x) = P(T=a | X=x)``; it is not
itself a causal estimator. Causal interpretation comes from the IPTW step in
:mod:`causalrank.estimators.iptw` plus the identification assumptions.

Leakage rules (PROJECT_SPEC.md 3.1): imputation, encoding and the boosting fit
happen inside an sklearn :class:`~sklearn.pipeline.Pipeline`, so they are
re-fitted on the training rows only — per cross-fitting fold and (Phase 2) per
bootstrap replication. Nothing is pre-fitted on the full sample.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..config import CausalRankConfig
from ..exceptions import CausalRankError, CausalRankValidationError
from ..results import DiagnosticWarning


@dataclass
class PropensityFit:
    """Fitted propensity model(s) and the aligned propensity matrix."""

    propensity: np.ndarray  # float (n, K), columns follow `labels`, rows sum to 1
    labels: tuple[Any, ...]
    models: list[Pipeline] = field(default_factory=list)
    fold_ids: np.ndarray | None = None  # int (n,) out-of-fold assignment, None if in-sample
    oof: bool = False
    warnings: list[DiagnosticWarning] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(self.propensity.shape[0])

    @property
    def n_treatments(self) -> int:
        return len(self.labels)


def align_proba_columns(
    proba: np.ndarray,
    model_classes: Sequence[Any],
    labels: Sequence[Any],
) -> np.ndarray:
    """Re-order ``predict_proba`` columns to the public ``labels`` order.

    sklearn guarantees that column ``j`` of ``predict_proba`` belongs to
    ``model.classes_[j]``; nothing else about the order is guaranteed once
    labels leave the fitted model. This function is the *only* place where that
    mapping is materialised, and it is tested against scrambled ``classes_``.
    """
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        raise CausalRankError("predict_proba must return a 2-D array.")
    if proba.shape[1] != len(model_classes):
        raise CausalRankError(
            f"predict_proba returned {proba.shape[1]} columns but model has "
            f"{len(model_classes)} classes."
        )
    col_of = {cls: j for j, cls in enumerate(model_classes)}
    missing = [lab for lab in labels if lab not in col_of]
    if missing:
        raise CausalRankError(
            f"Treatment labels {missing} were not seen by the fitted model "
            f"(model classes: {list(model_classes)})."
        )
    idx = [col_of[lab] for lab in labels]
    return np.ascontiguousarray(proba[:, idx])


def aligned_predict_proba(pipeline: Pipeline, X: pd.DataFrame, labels: Sequence[Any]) -> np.ndarray:
    """``predict_proba`` with columns mapped back to the public label order.

    The model is fitted on integer treatment codes ``0..K-1`` (canonical label
    order), so ``model.classes_`` holds those codes; they are translated back to
    labels before the generic column alignment runs.
    """
    model = pipeline.named_steps["model"]
    proba = pipeline.predict_proba(X)
    model_labels = [labels[int(code)] for code in model.classes_]
    return align_proba_columns(proba, model_labels, labels)


def make_propensity_pipeline(
    config: CausalRankConfig,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> Pipeline:
    """Build the preprocessing + multi-class GBM pipeline (unfitted)."""
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy=config.impute_numeric))]),
                list(numeric_features),
            )
        )
    if categorical_features:
        if config.impute_categorical == "constant":
            imputer: SimpleImputer = SimpleImputer(
                strategy="constant", fill_value=config.categorical_fill_value
            )
        else:
            imputer = SimpleImputer(strategy=config.impute_categorical)
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", imputer),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(categorical_features),
            )
        )
    if not transformers:  # pragma: no cover - validation requires >=1 confounder
        raise CausalRankValidationError("No usable covariates for the propensity model.")

    preprocessor = ColumnTransformer(
        transformers=transformers, remainder="drop", sparse_threshold=0.0
    )
    model = HistGradientBoostingClassifier(**config.resolved_gbm_params)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def _check_propensity_matrix(propensity: np.ndarray, labels: Sequence[Any]) -> None:
    if propensity.ndim != 2 or propensity.shape[1] != len(labels):
        raise CausalRankError("Internal error: propensity matrix has wrong shape.")
    if not np.all(np.isfinite(propensity)):
        raise CausalRankError(
            "Propensity scores contain non-finite values; the GBM failed to produce "
            "probabilities. Check the covariates and gbm_params."
        )
    if np.any(propensity < 0.0):
        raise CausalRankError("Propensity scores must be non-negative.")
    row_sums = propensity.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise CausalRankError(
            "Propensity rows do not sum to 1 (max deviation "
            f"{float(np.max(np.abs(row_sums - 1.0))):.3e})."
        )


def fit_propensity(
    X: pd.DataFrame,
    t_idx: np.ndarray,
    labels: Sequence[Any],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    config: CausalRankConfig,
) -> PropensityFit:
    """Fit the multi-class GBM propensity model and return aligned probabilities.

    If ``config.cross_fit`` is an int ``m``, out-of-fold probabilities are
    produced with stratified ``m``-fold cross-fitting (every training fold must
    contain all K classes), which reduces the over-fitting bias of using the
    same sample to fit and evaluate the propensity model. Otherwise the model
    is fitted on the full sample and predicts that sample in-sample.
    """
    n = int(t_idx.shape[0])
    k = len(labels)
    if n != X.shape[0]:
        raise CausalRankError("X and t_idx length mismatch.")
    expected_classes = np.arange(k, dtype=int)
    observed = np.unique(t_idx)
    if not np.array_equal(observed, expected_classes):
        raise CausalRankValidationError(
            "t_idx must contain every treatment code exactly in 0..K-1; "
            f"observed {observed.tolist()}."
        )

    warnings: list[DiagnosticWarning] = []
    models: list[Pipeline] = []

    if config.cross_fit is None:
        pipeline = make_propensity_pipeline(config, numeric_features, categorical_features)
        pipeline.fit(X, t_idx)
        aligned = aligned_predict_proba(pipeline, X, labels)
        models.append(pipeline)
        fold_ids = None
        oof = False
    else:
        m = int(config.cross_fit)
        if int(np.min(np.bincount(t_idx, minlength=k))) < m:
            raise CausalRankValidationError(
                f"cross_fit={m} requires at least {m} observations per treatment class."
            )
        splitter = StratifiedKFold(n_splits=m, shuffle=True, random_state=config.random_state)
        aligned = np.empty((n, k), dtype=float)
        fold_ids = np.full(n, -1, dtype=np.int64)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(n), t_idx)):
            train_classes = np.unique(t_idx[train_idx])
            if not np.array_equal(train_classes, expected_classes):
                raise CausalRankError(
                    f"Fold {fold} training split misses treatment classes; got "
                    f"{train_classes.tolist()}."
                )
            pipeline = make_propensity_pipeline(
                config, numeric_features, categorical_features
            )
            pipeline.fit(X.iloc[train_idx], t_idx[train_idx])
            aligned[test_idx] = aligned_predict_proba(pipeline, X.iloc[test_idx], labels)
            fold_ids[test_idx] = fold
            models.append(pipeline)
        if np.any(fold_ids < 0):  # pragma: no cover - defensive
            raise CausalRankError("Cross-fitting failed to assign an out-of-fold prediction.")
        oof = True

    _check_propensity_matrix(aligned, labels)
    return PropensityFit(
        propensity=aligned,
        labels=tuple(labels),
        models=models,
        fold_ids=fold_ids,
        oof=oof,
        warnings=warnings,
    )
