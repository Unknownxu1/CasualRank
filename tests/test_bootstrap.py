"""Joint bootstrap tests (PROJECT_SPEC.md 7.3: shared indices, re-fitting, seeds)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrank import (
    CausalRank,
    CausalRankConfig,
    CausalRankError,
    CausalRankValidationError,
)
from causalrank.inference.bootstrap import (
    derive_round_seeds,
    joint_bootstrap,
    resample_indices,
)
from causalrank.simulation import make_dataset
from causalrank.simulation.dgp import SyntheticDGP, get_scenario

from .conftest import FAST_GBM

CONF = ["x1", "x2", "x3", "x4"]


def _df(n: int = 400, seed: int = 3, scenario: str = "confounded") -> pd.DataFrame:
    return make_dataset(scenario, n=n, random_state=seed).df


def _config(**overrides: object) -> CausalRankConfig:
    params: dict[str, object] = {
        "treatment": "treatment",
        "outcome": "outcome",
        "confounders": tuple(CONF),
        "gbm_params": dict(FAST_GBM),
        "random_state": 11,
        "n_bootstrap": 20,
        "bootstrap_min_successes": 15,
    }
    params.update(overrides)
    return CausalRankConfig(**params)  # type: ignore[arg-type]


LABELS = ("A", "B", "C")


def test_resampling_uses_one_shared_index_vector_per_round() -> None:
    idx = resample_indices(50, seed=7)
    assert idx.shape == (50,)
    assert idx.min() >= 0 and idx.max() < 50
    assert np.array_equal(idx, resample_indices(50, seed=7))  # deterministic per seed
    assert not np.array_equal(idx, resample_indices(50, seed=8))


def test_round_seeds_are_deterministic_and_order_independent() -> None:
    a = derive_round_seeds(2026, 10)
    b = derive_round_seeds(2026, 10)
    assert np.array_equal(a, b)
    assert len(set(a.tolist())) == 10  # distinct per round
    # executing rounds out of order yields the same draws as in order
    df = _df()
    cfg = _config()
    forward = [joint_bootstrap_round(df, cfg, int(s)) for s in a[:4]]
    backward = [joint_bootstrap_round(df, cfg, int(s)) for s in a[:4][::-1]]
    for f, r in zip(forward, backward[::-1], strict=True):
        assert np.array_equal(f, r)


def joint_bootstrap_round(df: pd.DataFrame, cfg: CausalRankConfig, seed: int) -> np.ndarray:
    from causalrank.inference.bootstrap import _bootstrap_round

    return _bootstrap_round(df, cfg, LABELS, seed)


def test_success_returns_b_by_k_draws_and_k_by_k_sigma() -> None:
    df = _df()
    res = joint_bootstrap(df, _config(), LABELS, n_bootstrap=20)
    assert res.usable
    assert res.draws is not None
    assert res.draws.shape == (20, 3)
    assert res.covariance is not None
    assert res.covariance.shape == (3, 3)
    assert res.n_success == 20 and res.n_failed == 0
    assert res.failures == []


def test_sigma_equals_numpy_cov_of_draws() -> None:
    df = _df()
    res = joint_bootstrap(df, _config(), LABELS, n_bootstrap=20)
    assert res.draws is not None and res.covariance is not None
    expected = np.cov(res.draws, rowvar=False, ddof=1)
    assert np.allclose(res.covariance, expected, rtol=1e-12, atol=1e-15)
    # the estimator draws must NOT be divided by n again (spec 3.5)
    n = df.shape[0]
    assert not np.allclose(res.covariance, expected / n, rtol=1e-6)


def test_each_round_refits_the_full_model() -> None:
    """Every bootstrap round must re-run propensity estimation, not reuse models."""
    import causalrank.inference.bootstrap as bb

    calls: list[int] = []
    original = bb.fit_propensity

    def counting(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return original(*args, **kwargs)

    bb.fit_propensity = counting  # type: ignore[assignment]
    try:
        joint_bootstrap(_df(), _config(), LABELS, n_bootstrap=8)
    finally:
        bb.fit_propensity = original  # type: ignore[assignment]
    assert len(calls) == 8  # one full refit per round (plus none for the point estimate)


def test_keep_draws_option_and_structured_failure_report() -> None:
    df = _df()
    res = joint_bootstrap(
        df, _config(bootstrap_keep_draws=False, bootstrap_min_successes=5), LABELS, n_bootstrap=10
    )
    assert res.draws is None
    assert res.covariance is not None  # covariance is still computed

    # rare treatment at tiny n -> some resamples miss a class -> structured failures
    rare = _df(n=90, seed=5, scenario="rare_treatment")
    cfg = _config(
        random_state=4,
        n_bootstrap=10,
        bootstrap_min_successes=5,
        bootstrap_min_success_rate=0.9,
    )
    res2 = joint_bootstrap(rare, cfg, ("A", "B", "C"), n_bootstrap=10)
    if res2.n_failed > 0:
        f = res2.failures[0]
        assert set(f) == {"round", "seed", "reason"}
        assert isinstance(f["reason"], str) and f["reason"]
        warn = [w for w in res2.warnings if w.code == "bootstrap_failures"]
        assert warn and "excluded" in warn[0].message
        assert warn[0].details["n_failed"] == res2.n_failed


def test_inference_is_blocked_at_low_success_rate() -> None:
    """High failure rate must block inference and disclose the reasons."""
    rare = _df(n=70, seed=9, scenario="rare_treatment")
    cfg = _config(
        random_state=6,
        n_bootstrap=12,
        bootstrap_min_successes=5,
        bootstrap_min_success_rate=1.0,  # any failure blocks
    )
    res = joint_bootstrap(rare, cfg, ("A", "B", "C"), n_bootstrap=12)
    assert res.n_failed > 0  # with a ~4% arm at n=70 some rounds must miss it
    assert res.usable is False
    assert res.covariance is None
    codes = {w.code: w.level for w in res.warnings}
    assert codes.get("bootstrap_inference_blocked") == "critical"
    blocked = next(
        w for w in res.warnings if w.code == "bootstrap_inference_blocked"
    )
    assert "BLOCKED" in blocked.message
    assert blocked.details["n_success"] == res.n_success


def test_same_seed_and_config_reproduce_exactly() -> None:
    df = _df()
    a = joint_bootstrap(df, _config(), LABELS, n_bootstrap=12)
    b = joint_bootstrap(df, _config(), LABELS, n_bootstrap=12)
    assert a.draws is not None and b.draws is not None
    assert np.array_equal(a.draws, b.draws)
    assert np.array_equal(a.covariance, b.covariance)
    assert np.array_equal(a.seeds, b.seeds)
    assert a.base_seed == b.base_seed == 11


def test_parallel_execution_matches_sequential() -> None:
    df = _df()
    seq = joint_bootstrap(df, _config(), LABELS, n_bootstrap=10)
    par = joint_bootstrap(df, _config(n_jobs=2), LABELS, n_bootstrap=10)
    assert seq.draws is not None and par.draws is not None
    assert np.array_equal(seq.draws, par.draws)
    assert np.array_equal(seq.covariance, par.covariance)


def test_random_state_is_recorded_when_not_provided() -> None:
    res = joint_bootstrap(_df(), _config(random_state=None), LABELS, n_bootstrap=5)
    assert isinstance(res.base_seed, int)  # derived from OS entropy, then recorded
    assert res.seeds is not None and res.seeds.shape == (5,)


def test_end_to_end_fit_fills_se_ci_and_covariance() -> None:
    ds = make_dataset("confounded", n=800, random_state=21)
    result = CausalRank(
        treatment="treatment",
        outcome="outcome",
        confounders=CONF,
        random_state=2026,
        n_bootstrap=24,
        bootstrap_min_successes=20,
        gbm_params=dict(FAST_GBM),
    ).fit(ds.df)
    assert result.inference_available is True
    est = result.effects.to_dataframe()
    assert np.all(np.isfinite(est["se"])) and np.all(est["se"] > 0)
    assert np.all(est["ci_low"] < est["mu"]) and np.all(est["mu"] < est["ci_high"])
    assert result.covariance is not None
    assert np.allclose(np.diag(result.covariance), est["se"].to_numpy() ** 2, rtol=1e-10)
    meta = result.metadata["bootstrap"]
    assert meta["n_success"] == 24 and meta["usable"] is True
    assert result.metadata["inference_available"] is True
    # rank inference stays Phase 3
    assert result.ranks is None and result.pairwise is None
    text = result.summary()
    assert "NOT available" not in text
    assert "Phase 3" in text


def test_disabled_bootstrap_keeps_phase1_behaviour() -> None:
    ds = make_dataset("randomized", n=300, random_state=22)
    result = CausalRank(
        treatment="treatment",
        outcome="outcome",
        confounders=CONF,
        random_state=0,
        gbm_params=dict(FAST_GBM),
    ).fit(ds.df)
    assert result.inference_available is False
    assert result.covariance is None
    assert result.effects.to_dataframe()["se"].isna().all()
    assert "n_bootstrap=None" in result.metadata["inference_note"]


def test_config_rejects_impossible_bootstrap_gates() -> None:
    with pytest.raises(CausalRankValidationError, match="bootstrap_min_successes"):
        _config(n_bootstrap=5, bootstrap_min_successes=30)
    with pytest.raises(CausalRankValidationError, match="ci_method"):
        _config(ci_method="bca")
    with pytest.raises(CausalRankValidationError, match="percentile"):
        _config(ci_method="percentile", bootstrap_keep_draws=False)
    with pytest.raises(CausalRankValidationError, match="bootstrap_min_success_rate"):
        _config(bootstrap_min_success_rate=0.0)


def test_joint_bootstrap_requires_positive_n() -> None:
    with pytest.raises(CausalRankError, match="n_bootstrap"):
        joint_bootstrap(_df(), _config(n_bootstrap=None), LABELS, n_bootstrap=None)


def test_rare_arm_rounds_fail_without_crashing_the_batch() -> None:
    """Structural guarantee: a missing class raises inside the round, not outside."""
    from causalrank.inference.bootstrap import _bootstrap_round

    dgp = SyntheticDGP(get_scenario("rare_treatment"))
    df = dgp.sample(60, random_state=3)
    cfg = _config(random_state=2)
    with pytest.raises(CausalRankError, match="expected"):
        # force a seed whose resample can be checked for the class-coverage guard
        _bootstrap_round(df.assign(treatment=df["treatment"].replace({"C": "A"})), cfg, LABELS, 1)
