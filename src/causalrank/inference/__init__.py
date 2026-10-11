"""Inference sub-package: joint bootstrap covariance and CI helpers.

Rank inference (Bonferroni simultaneous intervals, rank confidence sets) is a
Phase 3 module and is intentionally not present yet.
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

__all__ = [
    "JointBootstrapResult",
    "derive_round_seeds",
    "joint_bootstrap",
    "normal_ci",
    "percentile_ci",
    "resample_indices",
    "validate_covariance",
]
