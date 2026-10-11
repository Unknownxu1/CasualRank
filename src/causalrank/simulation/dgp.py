"""Configurable synthetic data generator with *known* potential-outcome means.

The DGP is the project's main research asset: every scenario has an analytically
well-defined population quantity

    mu_true[a] = E_X[ P(Y(a) = 1 | X) ],

which is approximated by a large **independent** Monte Carlo reference sample
(never the estimation sample) whose size, seed and MC standard error are
recorded on the returned dataset. See docs/simulation_protocol.md.

Structure
---------
* covariates: ``x1, x2 ~ N(0,1)``, ``x3 ~ Bernoulli(p)``, ``x4 ~ Categorical(3)``;
* treatment: multi-class softmax in ``z = [x1, x2, x3, x4_code]`` plus an
  ``overlap_strength`` term on ``x1`` that can destroy common support;
* potential outcome: ``P(Y(a)=1|X) = sigmoid(b0 + beta.z + tau_a + inter*x1*tau_a)``;
* only ``Y = Y(T)`` is observed; optional MCAR missingness on ``x2`` and ``x4``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from ..exceptions import CausalRankValidationError

#: Column names used by every scenario (also the documented schema).
TREATMENT_COL = "treatment"
OUTCOME_COL = "outcome"
CONFOUNDER_COLS: tuple[str, ...] = ("x1", "x2", "x3", "x4")
CAT_LEVELS: tuple[str, ...] = ("a", "b", "c")


@dataclass(frozen=True)
class DGPConfig:
    """Fully-specified data-generating process.

    All coefficient vectors are aligned with ``z = [x1, x2, x3, x4_code]`` where
    ``x4_code`` is the integer level index of the categorical covariate.
    """

    name: str = "custom"
    labels: tuple[str, ...] = ("A", "B", "C")

    # covariates
    bernoulli_p: float = 0.5
    cat_probs: tuple[float, ...] = (0.5, 0.3, 0.2)
    missing_rate: float = 0.0

    # treatment assignment (softmax)
    treatment_intercept: tuple[float, ...] = (0.0, 0.0, 0.0)
    treatment_coef: tuple[tuple[float, ...], ...] = (
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    )
    overlap_coef: tuple[float, ...] = (1.0, -1.0, 0.0)
    overlap_strength: float = 0.0

    # outcome model
    outcome_intercept: float = -0.5
    outcome_coef: tuple[float, ...] = (0.40, -0.30, 0.25, 0.15)
    tau: tuple[float, ...] = (0.0, 0.35, -0.15)
    interaction: float = 0.0

    # truth reference
    reference_size: int = 200_000
    reference_seed: int = 20260101

    def __post_init__(self) -> None:
        k = len(self.tau)
        if k < 2:
            raise CausalRankValidationError("DGP needs at least 2 treatments.")
        if len(self.labels) != k:
            raise CausalRankValidationError("labels and tau must have the same length.")
        if len(self.treatment_intercept) != k or len(self.overlap_coef) != k:
            raise CausalRankValidationError("treatment_intercept / overlap_coef must match tau.")
        if len(self.treatment_coef) != k or any(len(row) != 4 for row in self.treatment_coef):
            raise CausalRankValidationError("treatment_coef must be K x 4.")
        if len(self.outcome_coef) != 4:
            raise CausalRankValidationError("outcome_coef must have 4 entries.")
        if len(self.cat_probs) != len(CAT_LEVELS):
            raise CausalRankValidationError(f"cat_probs must have {len(CAT_LEVELS)} entries.")
        if abs(sum(self.cat_probs) - 1.0) > 1e-9 or any(p <= 0 for p in self.cat_probs):
            raise CausalRankValidationError("cat_probs must be positive and sum to 1.")
        if not (0.0 <= self.bernoulli_p <= 1.0):
            raise CausalRankValidationError("bernoulli_p must lie in [0, 1].")
        if not (0.0 <= self.missing_rate < 1.0):
            raise CausalRankValidationError("missing_rate must lie in [0, 1).")
        if int(self.reference_size) < 10_000:
            raise CausalRankValidationError(
                "reference_size must be >= 10000 so that the MC reference error is negligible."
            )

    @property
    def n_treatments(self) -> int:
        return len(self.tau)


@dataclass
class SyntheticDataset:
    """Observed data plus the ground truth used for validation."""

    df: pd.DataFrame
    mu_true: pd.Series
    mu_true_mc_se: pd.Series
    config: DGPConfig
    scenario: str
    n_reference: int
    reference_seed: int
    notes: dict[str, Any] = field(default_factory=dict)


class SyntheticDGP:
    """Sampling and ground-truth utilities for one :class:`DGPConfig`."""

    def __init__(self, config: DGPConfig) -> None:
        self.config = config
        self._truth_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    def _z(self, X: pd.DataFrame) -> np.ndarray:
        x4_code = X["x4"].map({lvl: i for i, lvl in enumerate(CAT_LEVELS)}).to_numpy(dtype=float)
        z = np.column_stack(
            [
                X["x1"].to_numpy(dtype=float),
                X["x2"].to_numpy(dtype=float),
                X["x3"].to_numpy(dtype=float),
                x4_code,
            ]
        )
        return z

    def sample_covariates(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        x1 = rng.normal(0.0, 1.0, size=n)
        x2 = rng.normal(0.0, 1.0, size=n)
        x3 = rng.binomial(1, self.config.bernoulli_p, size=n).astype(float)
        x4_idx = rng.choice(len(CAT_LEVELS), size=n, p=list(self.config.cat_probs))
        x4 = pd.Categorical([CAT_LEVELS[i] for i in x4_idx], categories=list(CAT_LEVELS))
        return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})

    def true_propensity(self, X: pd.DataFrame) -> np.ndarray:
        """``e_a(x) = P(T=a|X=x)`` under the known softmax assignment model."""
        z = self._z(X)
        coef = np.asarray(self.config.treatment_coef, dtype=float)  # (K, 4)
        logits = (
            np.asarray(self.config.treatment_intercept, dtype=float)[None, :]
            + z @ coef.T
            + self.config.overlap_strength
            * np.asarray(self.config.overlap_coef, dtype=float)[None, :]
            * X["x1"].to_numpy(dtype=float)[:, None]
        )
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def outcome_probability(self, X: pd.DataFrame) -> np.ndarray:
        """``P(Y(a)=1|X)`` for every arm, shape (n, K)."""
        z = self._z(X)
        beta = np.asarray(self.config.outcome_coef, dtype=float)
        tau = np.asarray(self.config.tau, dtype=float)
        base = self.config.outcome_intercept + z @ beta
        eta = base[:, None] + tau[None, :] + self.config.interaction * X["x1"].to_numpy(
            dtype=float
        )[:, None] * tau[None, :]
        return expit(eta)

    def sample_treatment(self, X: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        p = self.true_propensity(X)
        cum = np.cumsum(p, axis=1)
        u = rng.random(int(p.shape[0]))
        return (u[:, None] > cum[:, :-1]).sum(axis=1).astype(np.int64)

    # ------------------------------------------------------------------
    def population_mean(
        self, n_reference: int | None = None, reference_seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """``mu_true[a] = E_X[P(Y(a)=1|X)]`` by independent Monte Carlo.

        Returns ``(mu_true, mc_se)`` where ``mc_se`` is the Monte Carlo standard
        error ``sd(P(Y(a)=1|X)) / sqrt(n_reference)`` of the reference estimate.
        The result is deterministic given ``(n_reference, reference_seed)`` and
        is cached so repeated replications of a benchmark share the same truth.
        """
        cfg = self.config
        n_ref = int(n_reference) if n_reference is not None else int(cfg.reference_size)
        seed = int(reference_seed) if reference_seed is not None else int(cfg.reference_seed)
        if n_ref < 10_000:
            raise CausalRankValidationError("n_reference must be >= 10000.")
        key = (n_ref, seed)
        if key not in self._truth_cache:
            rng = np.random.default_rng(seed)
            X_ref = self.sample_covariates(n_ref, rng)
            p = self.outcome_probability(X_ref)
            mu = p.mean(axis=0)
            mc_se = p.std(axis=0, ddof=1) / np.sqrt(n_ref)
            self._truth_cache[key] = (mu, mc_se)
        return self._truth_cache[key]

    # ------------------------------------------------------------------
    def sample(
        self,
        n: int,
        random_state: int | None = None,
        with_potential: bool = False,
    ) -> pd.DataFrame:
        """Draw one observed sample of size ``n``.

        The returned frame has columns ``x1, x2, x3, x4, treatment, outcome``
        where ``outcome`` is the realised ``Y(T)`` only. With
        ``with_potential=True`` the true assignment probabilities and all
        potential-outcome probabilities are appended (``p_t`` and ``p_y_a``,
        ``p_y_b``, ...) for oracle comparisons in tests.
        """
        rng = np.random.default_rng(random_state)
        X = self.sample_covariates(n, rng)
        t_idx = self.sample_treatment(X, rng)
        p_y = self.outcome_probability(X)
        u = rng.random(n)
        y = (u < p_y[np.arange(n), t_idx]).astype(float)

        X_obs = X.copy()
        if self.config.missing_rate > 0.0:
            miss_x2 = rng.random(n) < self.config.missing_rate
            miss_x4 = rng.random(n) < self.config.missing_rate
            X_obs.loc[miss_x2, "x2"] = np.nan
            X_obs.loc[miss_x4, "x4"] = np.nan

        df = X_obs.copy()
        df[TREATMENT_COL] = [self.config.labels[i] for i in t_idx]
        df[OUTCOME_COL] = y

        if with_potential:
            p_t = self.true_propensity(X)
            for a, label in enumerate(self.config.labels):
                df[f"p_y_{label}"] = p_y[:, a]
            for a, label in enumerate(self.config.labels):
                df[f"p_t_{label}"] = p_t[:, a]
        return df


# ---------------------------------------------------------------------------
# scenario presets
# ---------------------------------------------------------------------------
def _base_outcome() -> tuple[float, ...]:
    return (0.40, -0.30, 0.25, 0.15)


SCENARIOS: dict[str, DGPConfig] = {
    # Near-randomised assignment: raw group means are (asymptotically) unbiased.
    "randomized": DGPConfig(
        name="randomized",
        labels=("A", "B", "C"),
        treatment_intercept=(0.0, 0.0, 0.0),
        treatment_coef=((0.0, 0.0, 0.0, 0.0),) * 3,
        overlap_strength=0.0,
        outcome_intercept=-0.5,
        outcome_coef=_base_outcome(),
        tau=(0.0, 0.35, -0.15),
    ),
    # Correctly specified (from the model's viewpoint) confounded assignment:
    # moderate confounding with usable overlap, so IPTW should recover mu_true.
    "confounded": DGPConfig(
        name="confounded",
        labels=("A", "B", "C"),
        treatment_intercept=(0.10, 0.00, -0.10),
        treatment_coef=(
            (0.35, -0.25, 0.30, 0.15),
            (-0.30, 0.35, -0.20, 0.10),
            (0.35, -0.35, 0.10, 0.15),
        ),
        overlap_strength=0.0,
        outcome_intercept=-0.5,
        outcome_coef=_base_outcome(),
        tau=(0.0, 0.35, -0.15),
    ),
    # Strong confounding: raw means are badly biased; weights get more variable.
    "strong_confounding": DGPConfig(
        name="strong_confounding",
        labels=("A", "B", "C"),
        treatment_intercept=(0.3, 0.0, -0.3),
        treatment_coef=(
            (1.50, -1.00, 1.20, 0.80),
            (-1.20, 1.40, -0.90, 0.60),
            (0.80, 0.60, -1.50, -1.00),
        ),
        overlap_strength=0.0,
        outcome_intercept=-0.5,
        outcome_coef=_base_outcome(),
        tau=(0.0, 0.35, -0.15),
    ),
    # Weak common support: near-deterministic assignment on a linear score.
    "weak_overlap": DGPConfig(
        name="weak_overlap",
        labels=("A", "B", "C"),
        treatment_intercept=(0.2, 0.0, -0.2),
        treatment_coef=(
            (0.40, -0.30, 0.30, 0.20),
            (-0.30, 0.40, -0.20, 0.10),
            (0.20, 0.10, -0.40, -0.20),
        ),
        overlap_coef=(1.20, -1.20, 0.20),
        overlap_strength=1.80,
        outcome_intercept=-0.5,
        outcome_coef=_base_outcome(),
        tau=(0.0, 0.35, -0.15),
    ),
    # Rare treatment: one arm has ~3-4% of the sample.
    "rare_treatment": DGPConfig(
        name="rare_treatment",
        labels=("A", "B", "C"),
        treatment_intercept=(0.6, -0.2, -2.4),
        treatment_coef=(
            (0.50, -0.30, 0.40, 0.20),
            (-0.40, 0.50, -0.30, 0.15),
            (0.60, -0.50, 0.70, 0.30),
        ),
        overlap_strength=0.0,
        outcome_intercept=-0.5,
        outcome_coef=_base_outcome(),
        tau=(0.0, 0.35, -0.15),
    ),
    # True tie: every arm has exactly the same potential-outcome law.
    "equal_effects": DGPConfig(
        name="equal_effects",
        labels=("A", "B", "C"),
        treatment_intercept=(0.2, 0.0, -0.2),
        treatment_coef=(
            (0.60, -0.40, 0.50, 0.30),
            (-0.50, 0.60, -0.30, 0.20),
            (0.30, 0.20, -0.60, -0.40),
        ),
        overlap_strength=0.0,
        outcome_intercept=-0.5,
        outcome_coef=_base_outcome(),
        tau=(0.20, 0.20, 0.20),
    ),
    # Statistically hard: effects separated by less than sampling noise.
    "close_effects": DGPConfig(
        name="close_effects",
        labels=("A", "B", "C"),
        treatment_intercept=(0.2, 0.0, -0.2),
        treatment_coef=(
            (0.60, -0.40, 0.50, 0.30),
            (-0.50, 0.60, -0.30, 0.20),
            (0.30, 0.20, -0.60, -0.40),
        ),
        overlap_strength=0.0,
        outcome_intercept=-0.5,
        outcome_coef=_base_outcome(),
        tau=(0.00, 0.08, -0.06),
    ),
    # Mixed feature types + MCAR missingness + 4 arms (pipeline stress test).
    "mixed_missing": DGPConfig(
        name="mixed_missing",
        labels=("A", "B", "C", "D"),
        treatment_intercept=(0.10, 0.00, -0.10, -0.20),
        treatment_coef=(
            (0.30, -0.20, 0.25, 0.15),
            (-0.25, 0.30, -0.15, 0.10),
            (0.20, 0.15, -0.30, -0.20),
            (-0.15, -0.25, 0.20, 0.25),
        ),
        overlap_coef=(1.0, -1.0, 0.5, -0.5),
        overlap_strength=0.0,
        outcome_intercept=-0.5,
        outcome_coef=_base_outcome(),
        tau=(0.0, 0.30, -0.20, 0.10),
        interaction=0.15,
        missing_rate=0.10,
    ),
}


def get_scenario(name: str) -> DGPConfig:
    if name not in SCENARIOS:
        raise CausalRankValidationError(
            f"Unknown scenario {name!r}. Available: {sorted(SCENARIOS)}."
        )
    return SCENARIOS[name]


def make_dataset(
    scenario: str | DGPConfig,
    n: int,
    random_state: int | None = None,
    n_reference: int | None = None,
    reference_seed: int | None = None,
) -> SyntheticDataset:
    """Draw an observed sample plus the ground truth for one scenario.

    The Monte Carlo reference used for ``mu_true`` is independent of the
    estimation sample and deterministic in ``(n_reference, reference_seed)``
    (both recorded on the returned object).
    """
    config = get_scenario(scenario) if isinstance(scenario, str) else scenario
    dgp = SyntheticDGP(config)
    df = dgp.sample(n, random_state=random_state)
    mu, mc_se = dgp.population_mean(n_reference=n_reference, reference_seed=reference_seed)
    n_ref = int(n_reference) if n_reference is not None else int(config.reference_size)
    seed = int(reference_seed) if reference_seed is not None else int(config.reference_seed)
    return SyntheticDataset(
        df=df,
        mu_true=pd.Series(mu, index=list(config.labels), name="mu_true"),
        mu_true_mc_se=pd.Series(mc_se, index=list(config.labels), name="mu_true_mc_se"),
        config=config,
        scenario=config.name,
        n_reference=n_ref,
        reference_seed=seed,
        notes={
            "estimand": "mu_a = E_X[P(Y(a)=1|X)]",
            "mc_reference": "independent Monte Carlo covariate sample",
        },
    )


def make_oracle_frame(
    scenario: str | DGPConfig,
    n: int,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Observed frame augmented with true propensities and potential-outcome
    probabilities (used by tests to compare estimates against oracle values)."""
    config = get_scenario(scenario) if isinstance(scenario, str) else scenario
    dgp = SyntheticDGP(config)
    return dgp.sample(n, random_state=random_state, with_potential=True)


def scenario_names() -> Sequence[str]:
    return tuple(SCENARIOS.keys())
