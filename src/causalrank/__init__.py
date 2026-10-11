"""CausalRank: multi-treatment causal estimation and ranking from observational data.

Phase 0/1 release: raw DataFrame -> multi-class GBM propensity scores ->
Hajek IPTW -> per-treatment potential-outcome means ``mu_a = E[Y(a)]`` on a
common target population, with weight / overlap / balance diagnostics and
quality-gate warnings.

Joint bootstrap covariance (Phase 2) and rank confidence sets (Phase 3) are not
implemented in this release; ``CausalRankResult.inference_available`` is
``False`` and the reserved uncertainty fields are ``None``/``NaN``.
"""

from __future__ import annotations

from ._version import __version__
from .api import CausalRank
from .config import CausalRankConfig
from .exceptions import CausalRankError, CausalRankValidationError
from .results import (
    CausalRankResult,
    DiagnosticsReport,
    DiagnosticWarning,
    EstimatesTable,
    aggregate_gate,
)

__all__ = [
    "CausalRank",
    "CausalRankConfig",
    "CausalRankError",
    "CausalRankResult",
    "CausalRankValidationError",
    "DiagnosticWarning",
    "DiagnosticsReport",
    "EstimatesTable",
    "__version__",
    "aggregate_gate",
]
