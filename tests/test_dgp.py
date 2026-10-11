"""DGP tests: known truth, reproducibility, schema and scenario behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from causalrank import CausalRankValidationError
from causalrank.simulation.dgp import (
    CONFOUNDER_COLS,
    OUTCOME_COL,
    TREATMENT_COL,
    DGPConfig,
    SyntheticDGP,
    get_scenario,
    make_dataset,
    make_oracle_frame,
    scenario_names,
)


def test_all_documented_scenarios_exist() -> None:
    expected = {
        "randomized",
        "confounded",
        "strong_confounding",
        "weak_overlap",
        "rare_treatment",
        "equal_effects",
        "close_effects",
        "mixed_missing",
    }
    assert expected <= set(scenario_names())


def test_dataset_schema() -> None:
    ds = make_dataset("confounded", n=200, random_state=0)
    assert list(ds.df.columns) == [*CONFOUNDER_COLS, TREATMENT_COL, OUTCOME_COL]
    assert ds.df[TREATMENT_COL].isin(list(ds.config.labels)).all()
    assert set(np.unique(ds.df[OUTCOME_COL])) <= {0.0, 1.0}
    assert ds.df[CONFOUNDER_COLS[3]].dtype.name == "category"


def test_mu_true_is_reproducible_and_independent_of_sample() -> None:
    a = make_dataset("confounded", n=100, random_state=1)
    b = make_dataset("confounded", n=500, random_state=999)
    assert np.array_equal(a.mu_true.to_numpy(), b.mu_true.to_numpy())
    assert a.reference_seed == b.reference_seed == 20260101
    assert a.n_reference == 200_000
    # MC reference error must be small relative to effect sizes
    assert (a.mu_true_mc_se.to_numpy() < 0.002).all()


def test_mu_true_changes_with_reference_seed_within_mc_error() -> None:
    a = make_dataset("confounded", n=50, random_state=0, reference_seed=1)
    b = make_dataset("confounded", n=50, random_state=0, reference_seed=2)
    assert not np.array_equal(a.mu_true.to_numpy(), b.mu_true.to_numpy())
    diff = np.abs(a.mu_true.to_numpy() - b.mu_true.to_numpy())
    se = a.mu_true_mc_se.to_numpy() + b.mu_true_mc_se.to_numpy()
    assert np.all(diff < 5 * se + 5e-3)


def test_population_mean_matches_empirical_potential_outcomes() -> None:
    """mu_true approximates the mean of P(Y(a)=1|X) on a fresh sample."""
    dgp = SyntheticDGP(get_scenario("confounded"))
    rng = np.random.default_rng(7)
    X = dgp.sample_covariates(20_000, rng)
    p = dgp.outcome_probability(X)
    mu_emp = p.mean(axis=0)
    mu_ref, _ = dgp.population_mean()
    assert np.allclose(mu_emp, mu_ref, atol=0.004)


def test_true_propensity_rows_sum_to_one() -> None:
    dgp = SyntheticDGP(get_scenario("weak_overlap"))
    X = dgp.sample_covariates(100, np.random.default_rng(3))
    p = dgp.true_propensity(X)
    assert p.shape == (100, 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all((p > 0) & (p < 1))


def test_outcome_probability_is_a_probability() -> None:
    dgp = SyntheticDGP(get_scenario("mixed_missing"))
    X = dgp.sample_covariates(100, np.random.default_rng(4))
    p = dgp.outcome_probability(X)
    assert p.shape == (100, 4)
    assert np.all((p > 0) & (p < 1))


def test_missingness_scenario_creates_na() -> None:
    ds = make_dataset("mixed_missing", n=500, random_state=2)
    assert ds.df["x2"].isna().any()
    assert ds.df["x4"].isna().any()
    assert not ds.df[TREATMENT_COL].isna().any()
    assert not ds.df[OUTCOME_COL].isna().any()


def test_equal_effects_have_equal_truth() -> None:
    ds = make_dataset("equal_effects", n=50, random_state=0)
    assert np.allclose(ds.mu_true.to_numpy(), ds.mu_true.to_numpy()[0], atol=1e-6)


def test_rare_treatment_is_imbalanced() -> None:
    ds = make_dataset("rare_treatment", n=2000, random_state=6)
    counts = ds.df[TREATMENT_COL].value_counts()
    assert counts.min() / counts.max() < 0.25


def test_weak_overlap_has_smaller_min_propensity_than_randomized() -> None:
    weak = SyntheticDGP(get_scenario("weak_overlap"))
    rand = SyntheticDGP(get_scenario("randomized"))
    X = weak.sample_covariates(400, np.random.default_rng(8))
    min_weak = weak.true_propensity(X).min()
    min_rand = rand.true_propensity(X).min()
    assert min_weak < min_rand
    assert min_weak < 0.01


def test_oracle_frame_exposes_true_probabilities() -> None:
    df = make_oracle_frame("confounded", 100, random_state=1)
    labels = list(get_scenario("confounded").labels)
    p_t = df[[f"p_t_{lab}" for lab in labels]].to_numpy()
    p_y = df[[f"p_y_{lab}" for lab in labels]].to_numpy()
    assert np.allclose(p_t.sum(axis=1), 1.0)
    assert np.all((p_y > 0) & (p_y < 1))


def test_dgp_config_validation() -> None:
    with pytest.raises(CausalRankValidationError):
        DGPConfig(tau=(0.0, 0.1), labels=("A",))  # length mismatch
    with pytest.raises(CausalRankValidationError):
        DGPConfig(tau=(0.0, 0.1), treatment_coef=((0.0,), (0.0,)))  # wrong width
    with pytest.raises(CausalRankValidationError):
        DGPConfig(tau=(0.0, 0.1), reference_size=100)  # reference too small
    with pytest.raises(CausalRankValidationError):
        get_scenario("does_not_exist")


def test_treatment_sampling_matches_true_propensity() -> None:
    """Empirical assignment frequencies converge to the model probabilities."""
    dgp = SyntheticDGP(get_scenario("confounded"))
    rng = np.random.default_rng(123)
    X = dgp.sample_covariates(50_000, rng)
    t_idx = dgp.sample_treatment(X, rng)
    emp = np.bincount(t_idx, minlength=3) / t_idx.size
    avg_p = dgp.true_propensity(X).mean(axis=0)
    assert np.allclose(emp, avg_p, atol=0.01)
