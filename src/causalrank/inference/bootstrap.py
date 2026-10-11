"""Joint (re-fitting) bootstrap covariance for the multi-treatment means.

Protocol (PROJECT_SPEC.md 3.5): every bootstrap round ``b`` resamples the
observation rows **jointly** — one shared index vector for all treatments — and
re-runs the *entire* estimation protocol on the resampled sample: validation,
in-fold preprocessing, GBM propensity fitting (including cross-fitting folds),
weight construction and all K Hajek means. The result is one draw
``mu^(b) = (mu_1^(b), ..., mu_K^(b))`` in the canonical label order.

The covariance estimator is the sample covariance of the ``B x K`` draw matrix::

    Sigma_boot = 1/(B-1) sum_b (mu^(b) - bar_mu)(mu^(b) - bar_mu)^T
               = np.cov(draws, rowvar=False, ddof=1)

Each row of ``draws`` is a complete estimator output, **not** a single-unit
contribution, so the covariance must never be divided by ``n`` again.

Reproducibility: round ``b`` uses an explicit seed derived from
``numpy.random.SeedSequence(base_seed).spawn(B)``. The round seed drives both
the resampling and every stochastic element of the refit (fold shuffling, GBM),
so results are identical for a fixed seed and independent of execution order or
``n_jobs``.

Failure handling: a round that cannot produce all K means (missing treatment
class in the resample, numerical failure, ...) is recorded with its seed and
reason and excluded from the draw matrix — never silently dropped. Inference is
*blocked* (no covariance returned) when the success rate falls below
``bootstrap_min_success_rate`` or fewer than ``bootstrap_min_successes`` rounds
succeeded; the blocking decision and the failure reasons are surfaced as
structured warnings.

Theoretical caveat: the draws re-fit a machine-learning nuisance estimator, so
the quality of this covariance depends on algorithm stability and standard
regularity conditions. Simulation-based checks (SE vs empirical sampling
variability, CI coverage) are part of the validation protocol; this is not a
universal inference guarantee.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import CausalRankConfig
from ..estimators.iptw import hajek_estimate
from ..estimators.propensity import fit_propensity
from ..exceptions import CausalRankError
from ..results import DiagnosticWarning
from ..validation import validate_dataframe


@dataclass
class JointBootstrapResult:
    """Bootstrap draws, covariance and the structured failure report."""

    labels: tuple[Any, ...]
    n_requested: int
    n_success: int
    failures: list[dict[str, Any]] = field(default_factory=list)
    draws: np.ndarray | None = None  # (n_success, K), None when draws were not kept
    draw_mean: np.ndarray | None = None  # (K,)
    covariance: np.ndarray | None = None  # (K, K), None when inference is blocked
    seeds: np.ndarray | None = None  # (n_requested,) per-round seeds
    base_seed: int = 0
    usable: bool = False
    warnings: list[DiagnosticWarning] = field(default_factory=list)

    @property
    def n_failed(self) -> int:
        return self.n_requested - self.n_success

    @property
    def failure_rate(self) -> float:
        return self.n_failed / max(self.n_requested, 1)


# ---------------------------------------------------------------------------
# seeds and single round
# ---------------------------------------------------------------------------
def derive_round_seeds(base_seed: int, n_rounds: int) -> np.ndarray:
    """Deterministic, order-independent per-round seeds via ``SeedSequence``."""
    children = np.random.SeedSequence(int(base_seed)).spawn(int(n_rounds))
    return np.array(
        [int(np.asarray(child.generate_state(1))[0]) for child in children], dtype=np.int64
    )


def resample_indices(n: int, seed: int) -> np.ndarray:
    """One shared row-index vector (sampling with replacement) for a round."""
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n), size=int(n))


def _bootstrap_round(
    df: pd.DataFrame,
    config: CausalRankConfig,
    labels: tuple[Any, ...],
    seed: int,
) -> np.ndarray:
    """One full re-fit of the estimation protocol on a resampled row set."""
    n = int(df.shape[0])
    idx = resample_indices(n, seed)
    resampled = df.iloc[idx].reset_index(drop=True)

    validated = validate_dataframe(resampled, config)
    if validated.labels != tuple(labels):
        raise CausalRankError(
            f"resampled data covers treatments {validated.labels}, expected {tuple(labels)} "
            "(a class was missed by the resample)"
        )

    round_config = dataclasses.replace(config, random_state=int(seed))
    prop_fit = fit_propensity(
        X=validated.X,
        t_idx=validated.t_idx,
        labels=validated.labels,
        numeric_features=validated.numeric_features,
        categorical_features=validated.categorical_features,
        config=round_config,
    )
    estimate = hajek_estimate(
        y=validated.y,
        t_idx=validated.t_idx,
        propensity=prop_fit.propensity,
        labels=validated.labels,
        clip_propensity=config.clip_propensity,
    )
    return np.asarray(estimate.mu, dtype=float)


# ---------------------------------------------------------------------------
# covariance validation (no silent repair)
# ---------------------------------------------------------------------------
def validate_covariance(sigma: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Validate a covariance matrix; raise rather than repair.

    Checks: square 2-D shape, finite entries, symmetry, non-negative diagonal
    and positive semi-definiteness (up to a scale-relative numerical tolerance).
    """
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise CausalRankError(f"covariance must be square 2-D; got shape {sigma.shape}.")
    if not np.all(np.isfinite(sigma)):
        raise CausalRankError("covariance contains non-finite entries.")
    scale = float(np.max(np.abs(sigma))) if sigma.size else 0.0
    scale = max(scale, 1e-12)
    if float(np.max(np.abs(sigma - sigma.T))) > tol * scale:
        raise CausalRankError("covariance is not symmetric.")
    if float(np.min(np.diag(sigma))) < -tol * scale:
        raise CausalRankError("covariance has a negative diagonal entry.")
    eigvals = np.linalg.eigvalsh((sigma + sigma.T) / 2.0)
    if float(eigvals.min()) < -1e-8 * max(float(np.abs(eigvals).max()), 1e-12):
        raise CausalRankError(
            f"covariance is not positive semi-definite (min eigenvalue {eigvals.min():.3e}); "
            "refusing to silently repair it."
        )
    return sigma


# ---------------------------------------------------------------------------
# confidence intervals
# ---------------------------------------------------------------------------
def normal_ci(
    mu: np.ndarray, se: np.ndarray, confidence_level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise normal-approximation intervals ``mu +- z * se``."""
    from scipy.stats import norm

    z = float(norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))
    mu = np.asarray(mu, dtype=float)
    se = np.asarray(se, dtype=float)
    return mu - z * se, mu + z * se


def percentile_ci(
    draws: np.ndarray, confidence_level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise percentile intervals from the retained bootstrap draws.

    Requires draws to be kept (``bootstrap_keep_draws=True``); valid only when
    the bootstrap distribution is a reasonable approximation of the sampling
    distribution — state this condition whenever using it.
    """
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2 or draws.shape[0] < 2:
        raise CausalRankError("percentile CI requires at least 2 retained bootstrap draws.")
    alpha = 1.0 - confidence_level
    lo = np.quantile(draws, alpha / 2.0, axis=0)
    hi = np.quantile(draws, 1.0 - alpha / 2.0, axis=0)
    return lo, hi


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def joint_bootstrap(
    df: pd.DataFrame,
    config: CausalRankConfig,
    labels: tuple[Any, ...],
    n_bootstrap: int | None = None,
) -> JointBootstrapResult:
    """Run the joint re-fitting bootstrap and return draws + covariance."""
    if n_bootstrap is None:
        n_bootstrap = config.n_bootstrap
    if n_bootstrap is None or int(n_bootstrap) < 1:
        raise CausalRankError("joint_bootstrap requires n_bootstrap >= 1.")
    n_bootstrap = int(n_bootstrap)

    base_seed = (
        int(config.random_state)
        if config.random_state is not None
        else int(np.random.SeedSequence().generate_state(1)[0])
    )
    seeds = derive_round_seeds(base_seed, n_bootstrap)

    def _run(round_id: int) -> tuple[int, np.ndarray | None, str]:
        seed = int(seeds[round_id])
        try:
            mu = _bootstrap_round(df, config, tuple(labels), seed)
            return round_id, mu, ""
        except Exception as exc:  # recorded, never aborts the batch
            return round_id, None, f"{type(exc).__name__}: {exc}"

    if config.n_jobs == 1:
        outcomes = [_run(b) for b in range(n_bootstrap)]
    else:
        from joblib import Parallel, delayed

        outcomes = Parallel(n_jobs=int(config.n_jobs))(
            delayed(_run)(b) for b in range(n_bootstrap)
        )

    k = len(labels)
    draws_list: list[np.ndarray] = []
    failures: list[dict[str, Any]] = []
    draw_mean = np.full(k, np.nan)
    for round_id, mu, err in sorted(outcomes, key=lambda o: o[0]):
        if mu is None:
            failures.append({"round": round_id, "seed": int(seeds[round_id]), "reason": err})
        else:
            draws_list.append(mu)

    n_success = len(draws_list)
    warnings: list[DiagnosticWarning] = []

    if failures:
        reason_counts: dict[str, int] = {}
        for f in failures:
            key = f["reason"].split(":")[0]
            reason_counts[key] = reason_counts.get(key, 0) + 1
        warnings.append(
            DiagnosticWarning(
                code="bootstrap_failures",
                level="warning" if n_success >= config.bootstrap_min_successes else "critical",
                message=(
                    f"{len(failures)}/{n_bootstrap} bootstrap rounds failed "
                    f"(rate {len(failures) / n_bootstrap:.1%}) and were excluded from the draw "
                    f"matrix; failure reasons: {reason_counts}. The covariance is computed from "
                    f"the {n_success} successful rounds only."
                ),
                details={
                    "n_failed": len(failures),
                    "failure_rate": len(failures) / n_bootstrap,
                    "reasons": reason_counts,
                    "failed_rounds": failures[:20],
                },
            )
        )

    draws = np.vstack(draws_list) if draws_list else np.empty((0, k))
    usable = (
        n_success >= max(int(config.bootstrap_min_successes), 2)
        and n_success / n_bootstrap >= float(config.bootstrap_min_success_rate)
    )
    if not usable:
        warnings.append(
            DiagnosticWarning(
                code="bootstrap_inference_blocked",
                level="critical",
                message=(
                    "Joint bootstrap inference is BLOCKED: "
                    f"{n_success}/{n_bootstrap} rounds succeeded "
                    f"(required: success rate >= {config.bootstrap_min_success_rate:.0%} and "
                    f">= {config.bootstrap_min_successes} successes). No covariance / SE / CI is "
                    "reported; inspect the failure reasons before trusting any point estimate "
                    "for the affected arms."
                ),
                details={
                    "n_success": n_success,
                    "n_requested": n_bootstrap,
                    "min_success_rate": float(config.bootstrap_min_success_rate),
                    "min_successes": int(config.bootstrap_min_successes),
                    "failures": failures[:20],
                },
            )
        )
        return JointBootstrapResult(
            labels=tuple(labels),
            n_requested=n_bootstrap,
            n_success=n_success,
            failures=failures,
            draws=draws if config.bootstrap_keep_draws and draws.size else None,
            draw_mean=None,
            covariance=None,
            seeds=seeds,
            base_seed=base_seed,
            usable=False,
            warnings=warnings,
        )

    draw_mean = draws.mean(axis=0)
    # ddof=1 sample covariance of complete estimator draws; NEVER divide by n.
    sigma = np.cov(draws, rowvar=False, ddof=1)
    sigma = validate_covariance(sigma)

    return JointBootstrapResult(
        labels=tuple(labels),
        n_requested=n_bootstrap,
        n_success=n_success,
        failures=failures,
        draws=draws if config.bootstrap_keep_draws else None,
        draw_mean=draw_mean,
        covariance=sigma,
        seeds=seeds,
        base_seed=base_seed,
        usable=True,
        warnings=warnings,
    )
