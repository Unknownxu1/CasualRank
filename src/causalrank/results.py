"""Result containers returned by :class:`causalrank.api.CausalRank`.

Design rules (PROJECT_SPEC.md sections 2.3, 3.6, 3.7, 8):

* numeric API must not depend on any plotting library;
* Phase 1 computes **estimates and diagnostics only**; uncertainty fields are
  present in the schema but explicitly null and flagged
  ``inference_available=False`` — they are filled by the Phase 2 joint
  bootstrap and Phase 3 rank inference;
* every warning is a structured record (code / level / message / details)
  instead of a bare printed string, so quality gates are machine-checkable;
* Phase 3 objects (:class:`PairwiseTable`, :class:`RankTable`) carry the
  multiplicity settings (confidence level, Bonferroni factor, tie convention)
  next to the numbers so every ranking statement is traceable to its formula.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

WarningLevel = Literal["info", "warning", "critical"]

#: Global quality gate derived from the emitted warnings.
GateLevel = Literal["pass", "caution", "fail"]

_LEVEL_ORDER: dict[str, int] = {"info": 0, "warning": 1, "critical": 2}

#: Point-estimate tie convention used by the ranking kernel (PROJECT_SPEC 3.6).
TIE_CONVENTION = (
    "exact point-estimate ties share the same (best) point rank and are flagged in "
    "'point_tie'; they are never broken by label order. When true effects tie, a strict "
    "total order is not uniquely defined and the rank confidence set widens accordingly."
)

PAIRWISE_FORMULA = (
    "d_ij = mu_i - mu_j; se(d_ij) = sqrt(Sigma_ii + Sigma_jj - 2 Sigma_ij); "
    "CI = d_ij +- z_{1-alpha/(2M)} * se(d_ij), M = K(K-1)/2 (Bonferroni)"
)

RANK_FORMULA = (
    "rank CI = [1 + B_i, K - W_i]; B_i = number of arms confirmed better than i, "
    "W_i = number of arms confirmed worse than i (from the simultaneous pairwise CIs)"
)


@dataclass(frozen=True)
class DiagnosticWarning:
    """One structured diagnostic / quality-gate finding."""

    code: str
    level: WarningLevel
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "level": self.level,
            "message": self.message,
            "details": dict(self.details),
        }

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"[{self.level.upper()}] {self.code}: {self.message}"


def aggregate_gate(warnings: list[DiagnosticWarning]) -> GateLevel:
    """Map the emitted warnings to a single quality-gate level.

    ``pass``     -> no warning-level or critical-level findings;
    ``caution``  -> at least one warning, none critical;
    ``fail``     -> at least one critical finding (positivity severely damaged,
                    degenerate weights, ...). The estimate is still returned,
    but it must not be presented as a reliable causal ranking.
    """
    worst = 0
    for w in warnings:
        worst = max(worst, _LEVEL_ORDER[w.level])
    if worst >= 2:
        return "fail"
    if worst == 1:
        return "caution"
    return "pass"


@dataclass
class EstimatesTable:
    """Per-treatment point estimates on the common target population.

    Uncertainty columns (``se``, ``ci_low``, ``ci_high``) are ``NaN`` in Phase 1
    and are produced by the joint bootstrap in Phase 2. They are kept in the
    schema so downstream code can rely on a stable column set.
    """

    labels: tuple[Any, ...]
    n_treated: np.ndarray
    raw_mean: np.ndarray
    weight_sum: np.ndarray
    ess: np.ndarray
    mu: np.ndarray
    se: np.ndarray | None = None
    ci_low: np.ndarray | None = None
    ci_high: np.ndarray | None = None

    def to_dataframe(self) -> pd.DataFrame:
        k = len(self.labels)
        se = self.se if self.se is not None else np.full(k, np.nan)
        lo = self.ci_low if self.ci_low is not None else np.full(k, np.nan)
        hi = self.ci_high if self.ci_high is not None else np.full(k, np.nan)
        return pd.DataFrame(
            {
                "label": list(self.labels),
                "n_treated": np.asarray(self.n_treated, dtype=int),
                "raw_mean": np.asarray(self.raw_mean, dtype=float),
                "mu": np.asarray(self.mu, dtype=float),
                "se": np.asarray(se, dtype=float),
                "ci_low": np.asarray(lo, dtype=float),
                "ci_high": np.asarray(hi, dtype=float),
                "weight_sum": np.asarray(self.weight_sum, dtype=float),
                "ess": np.asarray(self.ess, dtype=float),
            }
        )


@dataclass
class DiagnosticsReport:
    """Weight / overlap / balance diagnostics plus the structured warnings."""

    group_summary: pd.DataFrame
    weights: pd.DataFrame
    overlap: pd.DataFrame
    balance: pd.DataFrame  # long-form SMD table (feature x level x pair)
    balance_summary: pd.DataFrame  # per-term max |SMD| over treatment pairs
    warnings: list[DiagnosticWarning] = field(default_factory=list)

    @property
    def gate(self) -> GateLevel:
        return aggregate_gate(self.warnings)

    def summary(self) -> str:
        lines: list[str] = []
        lines.append("=== CausalRank diagnostics (Phase 1: estimation only, no inference) ===")
        lines.append("")
        lines.append("-- group summary (observed data) --")
        lines.append(self.group_summary.to_string(index=False))
        lines.append("")
        lines.append("-- weight distribution (Hajek IPTW weights w = 1/e_{T}(X)) --")
        lines.append(self.weights.to_string(index=False))
        lines.append("")
        lines.append("-- overlap: observed-treatment propensity e_{T}(X) --")
        lines.append(self.overlap.to_string(index=False))
        lines.append("")
        lines.append("-- covariate balance (max |SMD| over treatment pairs) --")
        lines.append(self.balance_summary.to_string(index=False))
        lines.append(
            "   (smd_weighted uses Hajek IPTW weights; sd_ref is the unweighted pooled "
            "within-arm SD; see docs/methodology.md)"
        )
        lines.append("")
        lines.append(f"-- quality gate: {self.gate.upper()} --")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  {w}")
        else:
            lines.append("  no warnings raised (this does NOT validate the causal assumptions)")
        lines.append(
            "Reminder: diagnostics can flag empirical problems (extreme weights, residual "
            "imbalance, weak overlap) but cannot verify no-unmeasured-confounding or consistency."
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_summary": self.group_summary.to_dict(orient="records"),
            "weights": self.weights.to_dict(orient="records"),
            "overlap": self.overlap.to_dict(orient="records"),
            "balance": self.balance.to_dict(orient="records"),
            "balance_summary": self.balance_summary.to_dict(orient="records"),
            "warnings": [w.to_dict() for w in self.warnings],
            "quality_gate": self.gate,
        }


@dataclass
class PairwiseTable:
    """Bonferroni simultaneous pairwise comparisons (Phase 3, spec 3.6).

    ``table`` has one row per unordered pair ``i < j`` in canonical label
    order; the numeric columns are fixed (see ``to_dataframe``). ``decision``
    is ``"i_better"`` / ``"j_better"`` / ``"inconclusive"`` in the *better*
sense (respecting ``higher_is_better``) — ``"inconclusive"`` is **not** an
    equality claim.
    """

    labels: tuple[Any, ...]
    table: pd.DataFrame
    confidence_level: float
    n_pairs: int
    z_crit: float
    higher_is_better: bool
    method: str = "bonferroni"

    def to_dataframe(self) -> pd.DataFrame:
        return self.table.copy()

    @property
    def confirmed(self) -> pd.DataFrame:
        """Only the pairs with a confirmed better/worse relation."""
        return self.table[self.table["confirmed"].astype(bool)].reset_index(drop=True)

    def summary(self) -> str:
        lines: list[str] = []
        lines.append("-- pairwise simultaneous comparisons (Bonferroni) --")
        lines.append(f"   formula: {PAIRWISE_FORMULA}")
        lines.append(
            f"   confidence_level = {self.confidence_level}, M = {self.n_pairs}, "
            f"z_crit = {self.z_crit:.6f}, higher_is_better = {self.higher_is_better}"
        )
        lines.append(self.table.to_string(index=False))
        lines.append(
            "   decision is in the 'better' sense; 'inconclusive' means the data did not "
            "confirm a direction — it does NOT mean the two arms are equal. Pointwise CIs "
            "are not a substitute for these simultaneous intervals."
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "method": self.method,
            "formula": PAIRWISE_FORMULA,
            "confidence_level": self.confidence_level,
            "n_pairs": self.n_pairs,
            "z_crit": self.z_crit,
            "higher_is_better": self.higher_is_better,
            "pairs": self.table.to_dict(orient="records"),
        }


@dataclass
class RankTable:
    """Point ranks and conservative rank confidence sets (Phase 3, spec 3.6).

    ``table`` columns: ``label``, ``mu``, ``point_rank``, ``rank_low``,
    ``rank_high``, ``n_confirmed_better`` (B_i), ``n_confirmed_worse`` (W_i),
    ``point_tie``. The rank interval ``[rank_low, rank_high] = [1 + B_i,
    K - W_i]`` is simultaneous-coverage based and deliberately conservative.
    """

    labels: tuple[Any, ...]
    table: pd.DataFrame
    confidence_level: float
    higher_is_better: bool
    method: str = "bonferroni"
    tie_convention: str = TIE_CONVENTION

    def to_dataframe(self) -> pd.DataFrame:
        return self.table.copy()

    def summary(self) -> str:
        lines: list[str] = []
        lines.append("-- rank point estimates and confidence sets --")
        lines.append(f"   formula: {RANK_FORMULA}")
        lines.append(
            f"   confidence_level = {self.confidence_level}, "
            f"higher_is_better = {self.higher_is_better}"
        )
        lines.append(self.table.to_string(index=False))
        lines.append(f"   tie convention: {self.tie_convention}")
        lines.append(
            "   The rank interval covers the true rank under the simultaneous pairwise "
            "intervals and the asymptotic normal approximation of the estimator. A rank of "
            "1 in the point estimate is NOT a probability of being best, and a wider "
            "interval is the honest answer when arms cannot be separated."
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "method": self.method,
            "formula": RANK_FORMULA,
            "confidence_level": self.confidence_level,
            "higher_is_better": self.higher_is_better,
            "tie_convention": self.tie_convention,
            "ranks": self.table.to_dict(orient="records"),
        }


@dataclass
class RankInference:
    """Pairwise comparisons plus the rank confidence sets derived from them."""

    pairwise: PairwiseTable
    ranks: RankTable

    @property
    def labels(self) -> tuple[Any, ...]:
        return self.ranks.labels

    def to_dataframe(self) -> pd.DataFrame:
        """The ranks table (``result.ranks.to_dataframe()`` in the user story)."""
        return self.ranks.to_dataframe()

    def summary(self) -> str:
        return self.ranks.summary() + "\n" + self.pairwise.summary()

    def to_dict(self) -> dict[str, Any]:
        return {"pairwise": self.pairwise.to_dict(), "ranks": self.ranks.to_dict()}


@dataclass
class CausalRankResult:
    """Everything returned by ``CausalRank.fit``.

    ``covariance``, ``pairwise`` and ``ranks`` are filled by the Phase 2 joint
    bootstrap and the Phase 3 rank inference; when inference is disabled or
    blocked they stay ``None`` and ``inference_available`` makes that state
    explicit instead of silently omitting uncertainty.
    """

    effects: EstimatesTable
    diagnostics: DiagnosticsReport
    metadata: dict[str, Any]
    covariance: np.ndarray | None = None
    pairwise: PairwiseTable | None = None
    ranks: RankTable | None = None
    inference_available: bool = False

    # ------------------------------------------------------------------
    @property
    def warnings(self) -> list[DiagnosticWarning]:
        return self.diagnostics.warnings

    def summary(self) -> str:
        est = self.effects.to_dataframe()
        show = est[["label", "n_treated", "raw_mean", "mu", "se", "ci_low", "ci_high", "ess"]]
        lines: list[str] = []
        lines.append("==============================================================")
        lines.append(" CausalRank result — estimates, uncertainty (if enabled) & diagnostics")
        lines.append("==============================================================")
        lines.append(
            f"estimand : mu_a = E[Y(a)] over the sample population "
            f"({self.metadata.get('estimand')})"
        )
        lines.append(
            f"estimator: Hajek IPTW with {self.metadata.get('propensity_model')} propensity scores"
        )
        lines.append(
            f"n = {self.metadata.get('n_rows')}, K = {self.metadata.get('n_treatments')}, "
            f"random_state = {self.metadata.get('random_state')}"
        )
        lines.append("")
        lines.append("-- estimates (mu_a = E[Y(a)], common target population) --")
        lines.append(show.to_string(index=False))
        lines.append("")
        if self.inference_available:
            se_note = self.metadata.get("inference_note", "")
            lines.append(f"uncertainty: {se_note}")
            if self.ranks is not None and self.pairwise is not None:
                lines.append(self.ranks.summary())
                lines.append("")
                lines.append(self.pairwise.summary())
                lines.append("")
                lines.append(
                    "!! these intervals cover sampling variability of the estimator under the\n"
                    "!! fitted model ONLY. They rely on the joint bootstrap covariance and an\n"
                    "!! asymptotic normal approximation. Do not read point-estimate ordering as\n"
                    "!! a statistically established ranking, do not treat an inconclusive pair\n"
                    "!! as equality, and do not treat the bootstrap frequency of rank 1 as the\n"
                    "!! probability of being best."
                )
            else:
                lines.append(
                    "!! rank confidence sets and multiplicity-adjusted pairwise comparisons are\n"
                    "!! NOT computed here — do not read the point-estimate ordering as a\n"
                    "!! statistically established ranking."
                )
        else:
            lines.append(
                "!! uncertainty quantification is NOT available in this result: se / ci / ranks\n"
                "!! are NaN by design (set n_bootstrap >= 1 to enable the joint re-fitting\n"
                "!! bootstrap; if inference was blocked, see diagnostics.warnings for the\n"
                "!! failure reasons). Do not read point-estimate ordering as a statistically\n"
                "!! established ranking."
            )
        lines.append("")
        lines.append(self.diagnostics.summary())
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "effects": self.effects.to_dataframe().to_dict(orient="records"),
            "covariance": None if self.covariance is None else np.asarray(self.covariance).tolist(),
            "pairwise": None if self.pairwise is None else self.pairwise.to_dict(),
            "ranks": None if self.ranks is None else self.ranks.to_dict(),
            "inference_available": bool(self.inference_available),
            "diagnostics": self.diagnostics.to_dict(),
            "metadata": dict(self.metadata),
        }

    def to_json(self, indent: int = 2) -> str:
        """Machine-readable dump; NaN fields are emitted as ``null``."""

        def _default(obj: Any) -> Any:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                val = float(obj)
                return None if not np.isfinite(val) else val
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")

        return json.dumps(self.to_dict(), indent=indent, default=_default, ensure_ascii=False)
