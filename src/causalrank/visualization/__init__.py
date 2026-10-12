"""Static reporting for CausalRank results (Phase 4).

Numeric APIs in the rest of the package never depend on this module — the
report is an optional, dependency-free view over an already-computed
``CausalRankResult``.
"""

from __future__ import annotations

from .report import PLOT_KINDS, Figure, plot_result, render_html, save_html

__all__ = [
    "PLOT_KINDS",
    "Figure",
    "plot_result",
    "render_html",
    "save_html",
]
