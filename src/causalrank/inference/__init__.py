"""Inference sub-package: joint bootstrap covariance, CI helpers, rank inference.

Phase 3 adds the Bonferroni simultaneous pairwise intervals
(:mod:`causalrank.inference.pairwise`) and the conservative rank confidence
sets (:mod:`causalrank.inference.ranking`).
"""

from __future__ import annotations

from .bootstrap import (
    JointBootstrapResult,
    derive_round_seeds,
    joint_bootstrap,
    normal_ci,
    percentile_ci,
    resample_indices,
    validate_covariance,
)
from .pairwise import (
    bonferroni_critical_value,
    check_labels,
    difference_se,
    difference_variance,
    pairwise_comparisons,
)
from .ranking import rank_confidence_sets, rank_from_estimates

__all__ = [
    "JointBootstrapResult",
    "bonferroni_critical_value",
    "check_labels",
    "derive_round_seeds",
    "difference_se",
    "difference_variance",
    "joint_bootstrap",
    "normal_ci",
    "pairwise_comparisons",
    "percentile_ci",
    "rank_confidence_sets",
    "rank_from_estimates",
    "resample_indices",
    "validate_covariance",
]
