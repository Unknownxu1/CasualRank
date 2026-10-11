"""End-to-end API tests on synthetic data with known truth (PROJECT_SPEC.md 7.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrank import CausalRank, CausalRankValidationError
from causalrank.simulation import make_dataset

CONF = ["x1", "x2", "x3", "x4"]


def _fit(df: pd.DataFrame, **kwargs) -> object:
    kwargs.setdefault("random_state", 2026)
    model = CausalRank(
        treatment="treatment", outcome="outcome", confounders=CONF, **kwargs
    )
    return model.fit(df)


def test_randomized_dgp_recovers_truth_and_raw_means() -> None:
    ds = make_dataset("randomized", n=3000, random_state=101)
    result = _fit(ds.df, cross_fit=5)
    est = result.effects.to_dataframe().set_index("label")
    for label, truth in ds.mu_true.items():
        assert abs(est.loc[label, "mu"] - truth) < 0.03
        # no confounding -> adjusted and raw means agree closely
        assert abs(est.loc[label, "mu"] - est.loc[label, "raw_mean"]) < 0.02
    assert result.diagnostics.gate == "pass"


def test_confounded_dgp_adjustment_beats_raw_means() -> None:
    """Repeated fixed-seed samples: IPTW removes the raw confounding bias."""
    rows = []
    for seed in (11, 22, 33):
        ds = make_dataset("confounded", n=2500, random_state=seed)
        result = _fit(ds.df, cross_fit=5)
        est = result.effects.to_dataframe().set_index("label")
        for label, truth in ds.mu_true.items():
            rows.append(
                {
                    "seed": seed,
                    "label": label,
                    "err_adjusted": abs(est.loc[label, "mu"] - truth),
                    "err_raw": abs(est.loc[label, "raw_mean"] - truth),
                }
            )
    df = pd.DataFrame(rows)
    assert df["err_adjusted"].mean() < 0.03
    assert df["err_adjusted"].mean() < 0.7 * df["err_raw"].mean()


def test_result_schema_and_phase1_flags() -> None:
    ds = make_dataset("confounded", n=800, random_state=5)
    result = _fit(ds.df)
    est = result.effects.to_dataframe()
    assert list(est.columns) == [
        "label",
        "n_treated",
        "raw_mean",
        "mu",
        "se",
        "ci_low",
        "ci_high",
        "weight_sum",
        "ess",
    ]
    # Phase 1: uncertainty fields exist but are explicitly null
    assert result.inference_available is False
    assert result.covariance is None
    assert result.pairwise is None
    assert result.ranks is None
    assert est[["se", "ci_low", "ci_high"]].isna().all().all()
    assert list(est["label"]) == sorted(ds.config.labels)
    assert np.all((est["mu"] >= 0) & (est["mu"] <= 1))


def test_summary_states_missing_inference() -> None:
    ds = make_dataset("randomized", n=600, random_state=6)
    result = _fit(ds.df)
    text = result.summary()
    assert "NOT available in Phase 1" in text
    assert "mu_a = E[Y(a)]" in text
    assert "quality gate" in text
    assert result.to_json()  # machine-readable dump works


def test_metadata_is_traceable() -> None:
    ds = make_dataset("confounded", n=400, random_state=7)
    result = _fit(ds.df, cross_fit=3, random_state=42)
    meta = result.metadata
    assert meta["package"] == "causalrank"
    assert meta["random_state"] == 42
    assert meta["cross_fit"] == 3
    assert meta["propensity_out_of_fold"] is True
    assert meta["n_rows"] == 400
    assert meta["n_treatments"] == 3
    assert meta["config"]["confounders"] == CONF
    assert meta["inference_available"] is False
    assert "causal_caveat" in meta
    assert meta["versions"]["scikit-learn"]


def test_fit_is_deterministic_for_fixed_seed() -> None:
    ds = make_dataset("confounded", n=700, random_state=8)
    a = _fit(ds.df, cross_fit=3, random_state=99)
    b = _fit(ds.df, cross_fit=3, random_state=99)
    assert np.array_equal(a.effects.mu, b.effects.mu)
    assert np.array_equal(a.effects.ess, b.effects.ess)
    c = _fit(ds.df, cross_fit=3, random_state=100)
    assert not np.array_equal(a.effects.mu, c.effects.mu)


def test_label_reordering_is_consistent() -> None:
    """Renaming arms must only permute the output rows, keyed by label."""
    ds = make_dataset("confounded", n=800, random_state=9)
    base = _fit(ds.df).effects.to_dataframe().set_index("label")

    df2 = ds.df.copy()
    swap = {"A": "Z", "B": "Y", "C": "X"}
    df2["treatment"] = df2["treatment"].map(swap)
    renamed = _fit(df2).effects.to_dataframe().set_index("label")

    for old, new in swap.items():
        assert renamed.loc[new, "mu"] == pytest.approx(base.loc[old, "mu"], rel=1e-12)
        assert renamed.loc[new, "ess"] == pytest.approx(base.loc[old, "ess"], rel=1e-12)


def test_unsupported_configuration_rejected() -> None:
    with pytest.raises(CausalRankValidationError, match="not supported"):
        CausalRank(
            treatment="treatment",
            outcome="outcome",
            confounders=CONF,
            estimator="aipw",
        )


def test_mixed_missing_scenario_runs_end_to_end() -> None:
    ds = make_dataset("mixed_missing", n=1500, random_state=12)
    result = _fit(ds.df, cross_fit=4)
    est = result.effects.to_dataframe()
    assert len(est) == 4
    assert np.all(np.isfinite(est["mu"]))
    codes = {w.code for w in result.warnings}
    assert "missing_covariates" in codes
    # imbalance of the missingness pattern is itself reported
    levels = set(result.diagnostics.balance["level"].dropna())
    assert "__missing__" in levels


def test_column_name_errors_are_clear() -> None:
    ds = make_dataset("randomized", n=100, random_state=13)
    with pytest.raises(CausalRankValidationError, match="Missing required column"):
        CausalRank(
            treatment="treatment", outcome="outcome", confounders=["nope"]
        ).fit(ds.df)


def test_real_world_dtypes_run_end_to_end() -> None:
    """bool / string / object-with-pd.NA / nullable Int64 covariates and string labels."""
    rng = np.random.default_rng(14)
    n = 600
    x_num = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x_bool": pd.Series(rng.random(n) > 0.5),
            "x_str": pd.Series([["s", "m", "l"][min(2, max(0, int(v * 2 + 1)))] for v in x_num]),
            "x_obj": pd.Series([("u", "v", pd.NA)[i % 3] for i in range(n)], dtype=object),
            "x_int": pd.Series([int(abs(v) * 3) if i % 7 else pd.NA for i, v in enumerate(x_num)],
                               dtype="Int64"),
            "x_num": x_num,
            "treatment": pd.Series(np.where(x_num > 0.5, "B", "A")),
            "outcome": pd.Series(
                ((x_num + rng.normal(scale=0.8, size=n) + 0.3 * (x_num > 0.5)) > 0).astype(float)
            ),
        }
    )
    result = CausalRank(
        treatment="treatment",
        outcome="outcome",
        confounders=["x_bool", "x_str", "x_obj", "x_int", "x_num"],
        random_state=2026,
    ).fit(df)
    est = result.effects.to_dataframe()
    assert np.all(np.isfinite(est["mu"]))
    assert list(est["label"]) == ["A", "B"]
    assert not result.diagnostics.balance.empty
