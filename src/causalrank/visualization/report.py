"""Static HTML report + inline SVG figures for a ``CausalRankResult`` (Phase 4).

Design rules (PROJECT_SPEC.md sections 2.1, 2.3, 6-Phase-4, 8):

* the report is **static HTML**: no server, no JavaScript, no external
  assets, and **no new runtime dependency** — charts are inline SVG built
  with the standard library;
* **traceability is a hard requirement**: the renderer only *reads* the
  result object (``effects`` / ``diagnostics`` / ``covariance`` /
  ``pairwise`` / ``ranks`` / ``metadata``). It never re-fits models or
  recomputes estimands, and every number shown is the same number the
  numeric API returns (same rounding);
* the honest form travels with the numbers: quality-gate banner,
  structured warnings, blocked/absent-inference state, the Bonferroni
  formula, the tie convention, and the causal-identification caveats are
  part of the report — a report can never look cleaner than the result
  object behind it;
* output is deterministic: no render-time clock is embedded, timestamps
  come from ``metadata["created_at"]``, and rendering the same result
  twice is byte-identical;
* every label / warning message is HTML-escaped.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..results import CausalRankResult

#: Supported chart kinds for :func:`plot_result` / ``result.plot()``.
PlotKind = Literal["estimates", "ranks", "balance", "ess", "overlap"]

PLOT_KINDS: tuple[str, ...] = ("estimates", "ranks", "balance", "ess", "overlap")

# --------------------------------------------------------------------------
# Formatting helpers (shared by tables and by the traceability tests)
# --------------------------------------------------------------------------

#: Decimal places for estimates / SMDs / propensity values.
DIGITS = 4


def _esc(value: Any) -> str:
    """HTML-escape any label, message or free text."""
    return html.escape(str(value), quote=True)


def fmt_float(value: Any, digits: int = DIGITS) -> str:
    """Format a number for the report.

    ``NaN`` (missing / not computed) renders as an em dash; ``±inf`` is a
    *real value* here — a degenerate SMD with zero within-arm variance — and
    renders as ``∞`` / ``−∞`` so it is never confused with missingness.
    """
    try:
        val = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return "&mdash;"
    if np.isnan(val):
        return "&mdash;"
    if np.isinf(val):
        return "&infin;" if val > 0 else "&minus;&infin;"
    return f"{val:.{digits}f}"


def fmt_int(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return "&mdash;"
    if not np.isfinite(val):
        return "&mdash;"
    return f"{int(val):,}"


def fmt_percent(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return "&mdash;"
    if not np.isfinite(val):
        return "&mdash;"
    return f"{val:.1%}"


#: Per-column formatters used by the table renderer.
_FORMATTERS: dict[str, Callable[[Any], str]] = {
    "n": fmt_int,
    "n_treated": fmt_int,
    "n_confirmed_better": fmt_int,
    "n_confirmed_worse": fmt_int,
    "point_rank": fmt_int,
    "rank_low": fmt_int,
    "rank_high": fmt_int,
    "ess": lambda v: fmt_float(v, 1),
    "ess_ratio": lambda v: fmt_float(v, 3),
    "mu": fmt_float,
    "raw_mean": fmt_float,
    "raw_outcome_mean": fmt_float,
    "mu_hajek_iptw": fmt_float,
    "se": fmt_float,
    "ci_low": fmt_float,
    "ci_high": fmt_float,
    "weight_sum": fmt_float,
    "max_abs_smd_unweighted": fmt_float,
    "max_abs_smd_weighted": fmt_float,
    "n_pairs_over_threshold": fmt_int,
    "n_pairs_over_severe": fmt_int,
    "share": fmt_percent,
    "max_weight_share": fmt_percent,
    "frac_arm_below_threshold": fmt_percent,
    "frac_obs_below_threshold": fmt_percent,
    "diff": fmt_float,
    "se_diff": fmt_float,
    "mu_i": fmt_float,
    "mu_j": fmt_float,
    "z_crit": lambda v: fmt_float(v, 6),
    "confidence_level": lambda v: fmt_float(v, 4),
}

#: Column display names (keeps the report readable without renaming the
#: stable machine-readable schema of ``to_dataframe()``).
_COLUMN_LABELS: dict[str, str] = {
    "mu_hajek_iptw": "mu (Hajek IPTW)",
    "raw_outcome_mean": "raw mean",
    "max_abs_smd_unweighted": "max |SMD| unweighted",
    "max_abs_smd_weighted": "max |SMD| weighted",
    "n_pairs_over_threshold": "pairs > thr",
    "n_pairs_over_severe": "pairs > severe",
    "frac_arm_below_threshold": "share target below thr",
    "frac_obs_below_threshold": "share treated below thr",
    "e_arm_min": "e_arm min",
    "e_arm_max": "e_arm max",
    "e_obs_min": "e_obs min",
    "e_obs_median": "e_obs median",
    "ess_ratio": "ESS ratio",
    "max_weight_share": "max weight share",
    "weight_sum": "weight sum",
    "se_diff": "SE(diff)",
    "mu_i": "mu_i",
    "mu_j": "mu_j",
    "point_tie": "point tie",
    "degenerate_se": "degenerate SE",
    "confirmed": "confirmed",
}

_DEFAULT_FLOAT_FMT = lambda v: fmt_float(v)  # noqa: E731 - tiny formatter table

#: Boolean columns rendered as yes/no instead of numbers.
_BOOL_COLUMNS: frozenset[str] = frozenset(
    {"point_tie", "confirmed", "degenerate_se", "usable", "inconclusive"}
)

#: Integer columns without thousands separators.
_PLAIN_INT_COLUMNS: frozenset[str] = frozenset({"n_pairs", "base_seed"})

#: Identifier columns are rendered verbatim (HTML-escaped), never through the
#: numeric formatters: treatment labels are identifiers, not numbers, so an
#: integer label 1 must stay "1" (not "1.0000") and large integers must not
#: lose precision through a float64 round-trip.
_IDENTIFIER_COLUMNS: frozenset[str] = frozenset(
    {"label", "label_i", "label_j", "feature", "level", "pair"}
)


def _cell(col: str, value: Any) -> str:
    if col in _BOOL_COLUMNS:
        return "yes" if bool(value) else "no"
    if col in _PLAIN_INT_COLUMNS:
        return fmt_int_plain(value)
    fmt = _FORMATTERS.get(col, _DEFAULT_FLOAT_FMT)
    return fmt(value)


def fmt_int_plain(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return "&mdash;"
    if not np.isfinite(val):
        return "&mdash;"
    return str(int(val))


def _df_to_html(df: pd.DataFrame) -> str:
    """Render a DataFrame as an HTML table using the shared formatters.

    Numeric columns go through the formatters above (same rounding as the
    numeric API — this is what makes the report traceable); every other
    column (labels, features, decisions, …) is HTML-escaped as text.
    """
    cols = list(df.columns)
    head = "".join(f"<th>{_esc(_COLUMN_LABELS.get(c, c))}</th>" for c in cols)
    numeric = {c: bool(pd.api.types.is_numeric_dtype(df[c])) for c in cols}
    body_rows: list[str] = []
    for row in df.itertuples(index=False):
        values = dict(zip(df.columns, row, strict=True))
        tds: list[str] = []
        for c in cols:
            v = values[c]
            if c in _IDENTIFIER_COLUMNS:
                # labels / features / pairs are identifiers: print verbatim so
                # integer labels stay exact and float formatting never applies
                if v is None or v is pd.NA or (isinstance(v, float) and not np.isfinite(v)):
                    tds.append("<td>&mdash;</td>")
                else:
                    tds.append(f"<td>{_esc(v)}</td>")
            elif numeric[c]:
                tds.append(f"<td>{_cell(c, v)}</td>")
            elif v is None or v is pd.NA or (isinstance(v, float) and not np.isfinite(v)):
                tds.append("<td>&mdash;</td>")
            else:
                tds.append(f"<td>{_esc(v)}</td>")
        body_rows.append(f"<tr>{''.join(tds)}</tr>")
    return (
        '<div class="table-wrap"><table class="data">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


# --------------------------------------------------------------------------
# Inline SVG figures (no external resources, no JS)
# --------------------------------------------------------------------------

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 28px 22px 60px;
  font: 14px/1.55 system-ui, -apple-system, "Segoe UI", "Noto Sans SC", sans-serif;
  color: #0f172a; background: #f8fafc;
}
main { max-width: 980px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 34px 0 10px; padding-bottom: 6px; border-bottom: 2px solid #e2e8f0; }
h3 { font-size: 14px; margin: 20px 0 8px; color: #334155; }
p  { margin: 8px 0; }
.meta { color: #475569; font-size: 13px; margin-bottom: 16px; }
.meta code { background: #eef2f7; padding: 1px 5px; border-radius: 4px; }
.gate { padding: 12px 14px; border-radius: 8px; margin: 14px 0; border: 1px solid; }
.gate-pass { background: #f0fdf4; border-color: #86efac; color: #14532d; }
.gate-caution { background: #fffbeb; border-color: #fcd34d; color: #78350f; }
.gate-fail { background: #fef2f2; border-color: #fca5a5; color: #7f1d1d; }
.gate .gate-title { font-weight: 700; font-size: 15px; }
.note { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;
        padding: 10px 14px; margin: 12px 0; color: #1e3a5f; }
.warn { border-radius: 8px; padding: 10px 14px; margin: 10px 0; border: 1px solid; }
.warn-info { background: #f8fafc; border-color: #cbd5e1; color: #334155; }
.warn-warning { background: #fffbeb; border-color: #fcd34d; color: #78350f; }
.warn-critical { background: #fef2f2; border-color: #fca5a5; color: #7f1d1d; }
.warn .code { font-family: ui-monospace, monospace; font-weight: 700; margin-right: 8px; }
.table-wrap { overflow-x: auto; margin: 10px 0; }
table.data { border-collapse: collapse; width: 100%; background: #fff;
             font-size: 13px; font-variant-numeric: tabular-nums; }
table.data th { background: #f1f5f9; text-align: right; padding: 7px 10px;
                border-bottom: 2px solid #cbd5e1; white-space: nowrap;
                position: sticky; top: 0; }
table.data th:first-child, table.data td:first-child { text-align: left; }
table.data td { padding: 6px 10px; border-bottom: 1px solid #e2e8f0;
                text-align: right; white-space: nowrap; }
table.data tbody tr:hover { background: #f8fafc; }
.fig { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
       padding: 12px; margin: 12px 0; overflow-x: auto; }
details { margin: 10px 0; }
summary { cursor: pointer; color: #2563eb; font-size: 13px; }
.formula { font-family: ui-monospace, monospace; font-size: 12.5px;
           background: #f1f5f9; border-radius: 6px; padding: 8px 12px;
           overflow-x: auto; color: #0f172a; }
ul.caveats li { margin: 6px 0; }
footer { margin-top: 40px; padding-top: 12px; border-top: 1px solid #e2e8f0;
         color: #64748b; font-size: 12px; }
"""


@dataclass(frozen=True)
class Figure:
    """A self-contained inline-SVG figure.

    Displays inline in Jupyter via ``_repr_html_`` and saves as standalone
    ``.svg`` or a minimal ``.html`` wrapper. Contains no scripts and no
    external references.
    """

    svg: str
    kind: str
    caption: str = ""

    def _repr_html_(self) -> str:  # pragma: no cover - notebook display
        return self.svg

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        if out.suffix.lower() == ".html":
            # the caption and quality gate are already inside the SVG footer
            body = (
                "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
                f"<title>CausalRank figure ({_esc(self.kind)})</title></head>"
                f"<body style=\"margin:16px\">{self.svg}"
                "</body></html>"
            )
            out.write_text(body, encoding="utf-8")
        else:
            out.write_text(self.svg, encoding="utf-8")
        return out


def _svg_doc(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui, sans-serif" '
        f'font-size="12" role="img">{body}</svg>'
    )


def _svg_text(x: float, y: float, content: str, *, anchor: str = "start",
              fill: str = "#334155", size: int = 12, weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" fill="{fill}" '
        f'font-size="{size}" font-weight="{weight}">{_esc(content)}</text>'
    )


def _nice_bounds(lo: float, hi: float) -> tuple[float, float]:
    """Pad a data range so the chart is not glued to its frame."""
    if not (np.isfinite(lo) and np.isfinite(hi)):  # pragma: no cover - defensive
        return 0.0, 1.0
    if hi <= lo:
        pad = max(abs(lo) * 0.1, 0.02)
        return lo - pad, hi + pad
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def _wrap_text(text: str, max_chars: int = 118) -> list[str]:
    """Greedy line wrapping for the in-figure footer (approximate width)."""
    if len(text) <= max_chars:
        return [text]
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        if current and len(current) + 1 + len(word) > max_chars:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return lines


_GATE_FOOTER_TEXT: dict[str, str] = {
    "pass": (
        "quality gate: PASS - no empirical warnings raised "
        "(this does not validate the causal assumptions)"
    ),
    "caution": "quality gate: CAUTION - read result.diagnostics.warnings before interpreting",
    "fail": "quality gate: FAIL - do not present these estimates as a reliable causal ranking",
}

_GATE_FOOTER_COLOR: dict[str, str] = {"pass": "#15803d", "caution": "#b45309", "fail": "#b91c1c"}


def _figure_footer(
    parts: list[str], width: int, height: int, caption: str, gate: str
) -> int:
    """Append caption + quality-gate lines *inside* the figure.

    A figure distributed on its own (notebook cell, ``.svg`` export) must
    carry the same context the full HTML report renders next to it: the
    caption and the result's quality gate travel with the pixels.
    """
    parts.append(
        f'<line x1="0" y1="{height}" x2="{width}" y2="{height}" stroke="#e2e8f0"/>'
    )
    y = height + 15
    for line in _wrap_text(caption):
        parts.append(_svg_text(6, y, line, fill="#64748b", size=11))
        y += 14
    parts.append(_svg_text(6, y, _GATE_FOOTER_TEXT[gate], fill=_GATE_FOOTER_COLOR[gate], size=11))
    return y + 14


def _estimates_svg(result: CausalRankResult) -> Figure:
    """Forest-style plot: point estimate and (when available) its CI."""
    est = result.effects.to_dataframe()
    labels = [str(x) for x in est["label"]]
    k = len(labels)
    has_ci = bool(result.inference_available) and est["ci_low"].notna().all()
    rows: list[float] = est["mu"].to_numpy(dtype=float)
    lo = est["ci_low"].to_numpy(dtype=float) if has_ci else rows
    hi = est["ci_high"].to_numpy(dtype=float) if has_ci else rows

    width, left, right = 760, 130, 60
    row_h, top, bottom = 36, 30, 34
    height = top + bottom + row_h * k
    x0, x1 = _nice_bounds(float(np.nanmin(lo)), float(np.nanmax(hi)))
    plot_w = width - left - right

    def sx(v: float) -> float:
        return left + (v - x0) / (x1 - x0) * plot_w

    parts: list[str] = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>'
    ]
    # axis + ticks
    axis_y = top + row_h * k + 6
    parts.append(
        f'<line x1="{left}" y1="{axis_y}" x2="{left + plot_w}" y2="{axis_y}" '
        'stroke="#94a3b8" stroke-width="1"/>'
    )
    for frac in np.linspace(0.0, 1.0, 5):
        xv = x0 + frac * (x1 - x0)
        x = sx(float(xv))
        parts.append(
            f'<line x1="{x:.1f}" y1="{axis_y}" x2="{x:.1f}" y2="{axis_y + 4}" '
            'stroke="#94a3b8"/>'
        )
        parts.append(_svg_text(x, axis_y + 17, f"{xv:.3f}", anchor="middle", fill="#64748b"))
    # rows
    for i, lab in enumerate(labels):
        y = top + i * row_h + row_h / 2
        parts.append(_svg_text(left - 10, y + 4, lab, anchor="end", fill="#0f172a"))
        if has_ci:
            parts.append(
                f'<line x1="{sx(float(lo[i])):.1f}" y1="{y:.1f}" '
                f'x2="{sx(float(hi[i])):.1f}" y2="{y:.1f}" stroke="#2563eb" '
                'stroke-width="2.5" stroke-linecap="round"/>'
            )
            for cap in (lo[i], hi[i]):
                parts.append(
                    f'<line x1="{sx(float(cap)):.1f}" y1="{y - 5:.1f}" '
                    f'x2="{sx(float(cap)):.1f}" y2="{y + 5:.1f}" stroke="#2563eb" '
                    'stroke-width="2.5"/>'
                )
        parts.append(
            f'<circle cx="{sx(float(rows[i])):.1f}" cy="{y:.1f}" r="4.5" '
            'fill="#1d4ed8" stroke="#ffffff" stroke-width="1.5"/>'
        )
    caption = (
        f"Point estimates mu_a with {float(result.metadata.get('confidence_level', 0.95)):.0%} "
        "confidence intervals (joint bootstrap)."
        if has_ci
        else "Point estimates only - inference is not available in this result."
    )
    parts.append(_svg_text(left, height - 8, "mu_a = E[Y(a)]", fill="#64748b", size=11))
    height = _figure_footer(parts, width, height, caption, result.diagnostics.gate)
    return Figure(svg=_svg_doc(width, height, "".join(parts)), kind="estimates", caption=caption)


def _ranks_svg(result: CausalRankResult) -> Figure:
    """Conservative rank confidence sets [rank_low, rank_high] per arm."""
    if result.ranks is None:
        raise ValueError("Rank figure requires rank inference (result.ranks is None).")
    ranks = result.ranks.to_dataframe()
    labels = [str(x) for x in ranks["label"]]
    k = len(labels)
    width, left, right = 760, 130, 60
    row_h, top, bottom = 34, 30, 34
    height = top + bottom + row_h * k
    plot_w = width - left - right

    def sx(rank: float) -> float:
        # rank 1 at left, K at right (ranks are integers in [1, K])
        return left + (rank - 1.0) / max(k - 1, 1) * plot_w

    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>']
    for r in range(1, k + 1):
        x = sx(float(r))
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{top + row_h * k}" '
            'stroke="#e2e8f0" stroke-dasharray="3 3"/>'
        )
        parts.append(_svg_text(x, top - 12, str(r), anchor="middle", fill="#64748b"))
    for i, lab in enumerate(labels):
        y = top + i * row_h + row_h / 2
        lo = float(ranks["rank_low"].iloc[i])
        hi = float(ranks["rank_high"].iloc[i])
        pr = float(ranks["point_rank"].iloc[i])
        parts.append(_svg_text(left - 10, y + 4, lab, anchor="end", fill="#0f172a"))
        parts.append(
            f'<line x1="{sx(lo):.1f}" y1="{y:.1f}" x2="{sx(hi):.1f}" y2="{y:.1f}" '
            'stroke="#93c5fd" stroke-width="10" stroke-linecap="round"/>'
        )
        parts.append(
            f'<circle cx="{sx(pr):.1f}" cy="{y:.1f}" r="4.5" fill="#1d4ed8" '
            'stroke="#ffffff" stroke-width="1.5"/>'
        )
        parts.append(
            _svg_text(sx(hi) + 10, y + 4, f"[{int(lo)}, {int(hi)}]", fill="#475569", size=11)
        )
    axis_y = top + row_h * k + 8
    parts.append(
        f'<line x1="{left}" y1="{axis_y}" x2="{left + plot_w}" y2="{axis_y}" '
        'stroke="#94a3b8"/>'
    )
    parts.append(_svg_text(left, height - 8, "rank (1 = best)", fill="#64748b", size=11))
    caption = "Bar: conservative rank confidence set [1 + B_i, K - W_i]; dot: point rank."
    height = _figure_footer(parts, width, height, caption, result.diagnostics.gate)
    return Figure(
        svg=_svg_doc(width, height, "".join(parts)),
        kind="ranks",
        caption=caption,
    )


def _balance_svg(result: CausalRankResult, *, top_n: int = 16) -> Figure:
    """Unweighted vs weighted max |SMD| per term (threshold reference lines)."""
    bs = result.diagnostics.balance_summary
    if bs.empty:
        raise ValueError("Balance figure requires a non-empty balance summary.")
    cfg = result.metadata.get("config", {})
    thr = float(cfg.get("smd_threshold", 0.1))
    severe = float(cfg.get("smd_severe_threshold", 0.25))
    shown = bs.head(top_n).reset_index(drop=True)
    n = len(shown)
    labels: list[str] = []
    for _, row in shown.iterrows():
        level = row["level"]
        labels.append(str(row["feature"]) if pd.isna(level) else f"{row['feature']} [{level}]")
    unweighted = shown["max_abs_smd_unweighted"].to_numpy(dtype=float)
    weighted = shown["max_abs_smd_weighted"].to_numpy(dtype=float)

    width, left, right = 760, 190, 60
    row_h, top, bottom = 26, 30, 48
    height = top + bottom + row_h * n
    plot_w = width - left - right
    # Axis bound must reflect BOTH series: weighting can make imbalance worse
    # (unweighted 0.15 -> weighted 0.8), and an unweighted-only bound would
    # squash the weighted dots against the frame and hide the deterioration.
    # Infinite |SMD| (zero within-arm variance, means differ) is off-scale by
    # definition: it is drawn at the boundary and marked, never as nan.
    finite = np.concatenate([unweighted[np.isfinite(unweighted)], weighted[np.isfinite(weighted)]])
    finite_max = float(finite.max()) if finite.size else 0.0
    xmax = max(finite_max, severe, thr) * 1.15 or 1.0

    def sx(v: float) -> float:
        if not np.isfinite(v):
            return left + plot_w  # infinite SMD: pin to the plot boundary
        return left + min(max(v, 0.0), xmax) / xmax * plot_w

    def offscale(v: float) -> bool:
        return bool(np.isinf(v))

    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>']
    for xv, color, name in ((thr, "#f59e0b", f"thr {thr:g}"),
                            (severe, "#dc2626", f"severe {severe:g}")):
        x = sx(xv)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{top + row_h * n}" '
            f'stroke="{color}" stroke-dasharray="4 3"/>'
        )
        parts.append(_svg_text(x, top - 14, name, anchor="middle", fill=color, size=11))
    for i in range(n):
        y = top + i * row_h + row_h / 2
        parts.append(_svg_text(left - 10, y + 4, labels[i], anchor="end", fill="#0f172a"))
        xu, xw = sx(float(unweighted[i])), sx(float(weighted[i]))
        parts.append(
            f'<line x1="{xu:.1f}" y1="{y:.1f}" x2="{xw:.1f}" y2="{y:.1f}" '
            'stroke="#cbd5e1" stroke-width="2"/>'
        )
        parts.append(
            f'<circle cx="{xu:.1f}" cy="{y:.1f}" r="4" fill="#ffffff" stroke="#64748b" '
            'stroke-width="2"/>'
        )
        parts.append(f'<circle cx="{xw:.1f}" cy="{y:.1f}" r="4.5" fill="#1d4ed8"/>')
        # mark values that are infinite (off-scale) instead of silently clipping
        for val, x, color in ((unweighted[i], xu, "#64748b"), (weighted[i], xw, "#1d4ed8")):
            if offscale(float(val)):
                parts.append(_svg_text(x - 6, y - 7, "∞", anchor="end", fill=color, size=12))
    axis_y = top + row_h * n + 6
    parts.append(
        f'<line x1="{left}" y1="{axis_y}" x2="{left + plot_w}" y2="{axis_y}" stroke="#94a3b8"/>'
    )
    for frac in np.linspace(0.0, 1.0, 5):
        xv = frac * xmax
        x = sx(float(xv))
        parts.append(_svg_text(x, axis_y + 15, f"{xv:.2f}", anchor="middle", fill="#64748b"))
    parts.append(
        _svg_text(
            left,
            height - 6,
            "hollow = unweighted   filled = Hajek IPTW weighted   ∞ = infinite |SMD| (off-scale)",
            fill="#64748b",
            size=11,
        )
    )
    caption = (
        "Per-term max |SMD| over treatment pairs, before/after weighting "
        f"(top {n} terms by weighted |SMD|)."
    )
    height = _figure_footer(parts, width, height, caption, result.diagnostics.gate)
    return Figure(
        svg=_svg_doc(width, height, "".join(parts)),
        kind="balance",
        caption=caption,
    )


def _ess_svg(result: CausalRankResult) -> Figure:
    """Raw sample size vs effective sample size per arm."""
    est = result.effects.to_dataframe()
    labels = [str(x) for x in est["label"]]
    n = est["n_treated"].to_numpy(dtype=float)
    ess = est["ess"].to_numpy(dtype=float)
    k = len(labels)
    width, left, right = 760, 130, 90
    row_h, top, bottom = 40, 22, 30
    height = top + bottom + row_h * k
    plot_w = width - left - right
    xmax = float(np.nanmax(n)) or 1.0

    def sx(v: float) -> float:
        return left + min(max(v, 0.0), xmax) / xmax * plot_w

    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>']
    for i, lab in enumerate(labels):
        y0 = top + i * row_h
        parts.append(_svg_text(left - 10, y0 + 18, lab, anchor="end", fill="#0f172a"))
        parts.append(
            f'<rect x="{left}" y="{y0 + 4}" width="{max(sx(float(n[i])) - left, 1):.1f}" '
            'height="10" rx="3" fill="#cbd5e1"/>'
        )
        parts.append(
            f'<rect x="{left}" y="{y0 + 18}" width="{max(sx(float(ess[i])) - left, 1):.1f}" '
            'height="10" rx="3" fill="#2563eb"/>'
        )
        parts.append(
            _svg_text(
                max(sx(float(n[i])), sx(float(ess[i]))) + 8,
                y0 + 22,
                f"n={int(n[i]):,}  ESS={ess[i]:,.0f}",
                fill="#475569",
                size=11,
            )
        )
    axis_y = top + row_h * k + 2
    parts.append(
        f'<line x1="{left}" y1="{axis_y}" x2="{left + plot_w}" y2="{axis_y}" stroke="#94a3b8"/>'
    )
    for frac in np.linspace(0.0, 1.0, 5):
        xv = frac * xmax
        x = sx(float(xv))
        parts.append(_svg_text(x, axis_y + 15, f"{xv:,.0f}", anchor="middle", fill="#64748b"))
    parts.append(
        _svg_text(left, height - 6, "gray = raw n   blue = ESS", fill="#64748b", size=11)
    )
    caption = "Effective sample size ESS_a = (sum w)^2 / sum(w^2) per arm."
    height = _figure_footer(parts, width, height, caption, result.diagnostics.gate)
    return Figure(
        svg=_svg_doc(width, height, "".join(parts)),
        kind="ess",
        caption=caption,
    )


def _overlap_svg(result: CausalRankResult) -> Figure:
    """Range of e_arm(X) over the target sample per arm.

    Thin line: [e_arm_min, e_arm_max]. Thick segment: the configured central
    quantile range when available (p05..p95 fallback to the full range).
    Red dot: the smallest propensity among rows actually treated with the arm
    (the denominator of that arm's IPTW weights).
    """
    ov = result.diagnostics.overlap
    labels = [str(x) for x in ov["label"]]
    k = len(labels)
    quantiles = [float(q) for q in result.metadata.get("config", {}).get("quantiles", ())]
    tag = lambda q: f"e_arm_p{round(q * 100):02d}"  # noqa: E731
    lo_col = tag(0.05) if 0.05 in quantiles and tag(0.05) in ov.columns else None
    hi_col = tag(0.95) if 0.95 in quantiles and tag(0.95) in ov.columns else None

    width, left, right = 760, 130, 60
    row_h, top, bottom = 36, 30, 34
    height = top + bottom + row_h * k
    plot_w = width - left - right
    xmax = max(float(ov["e_arm_max"].max()), 1e-9) * 1.05

    def sx(v: float) -> float:
        return left + min(max(v, 0.0), xmax) / xmax * plot_w

    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>']
    for i, lab in enumerate(labels):
        y = top + i * row_h + row_h / 2
        amin = float(ov["e_arm_min"].iloc[i])
        amax = float(ov["e_arm_max"].iloc[i])
        parts.append(_svg_text(left - 10, y + 4, lab, anchor="end", fill="#0f172a"))
        parts.append(
            f'<line x1="{sx(amin):.1f}" y1="{y:.1f}" x2="{sx(amax):.1f}" y2="{y:.1f}" '
            'stroke="#94a3b8" stroke-width="2"/>'
        )
        if lo_col and hi_col:
            parts.append(
                f'<line x1="{sx(float(ov[lo_col].iloc[i])):.1f}" y1="{y:.1f}" '
                f'x2="{sx(float(ov[hi_col].iloc[i])):.1f}" y2="{y:.1f}" '
                'stroke="#2563eb" stroke-width="7" stroke-linecap="round"/>'
            )
        parts.append(
            f'<circle cx="{sx(amin):.1f}" cy="{y:.1f}" r="3.5" fill="#dc2626" '
            'stroke="#ffffff" stroke-width="1"/>'
        )
        parts.append(
            f'<circle cx="{sx(float(ov["e_obs_min"].iloc[i])):.1f}" cy="{y:.1f}" r="3.5" '
            'fill="#f59e0b" stroke="#ffffff" stroke-width="1"/>'
        )
        parts.append(
            _svg_text(sx(amax) + 8, y + 4, f"min {amin:.4f}", fill="#475569", size=11)
        )
    axis_y = top + row_h * k + 6
    parts.append(
        f'<line x1="{left}" y1="{axis_y}" x2="{left + plot_w}" y2="{axis_y}" stroke="#94a3b8"/>'
    )
    for frac in np.linspace(0.0, 1.0, 5):
        xv = frac * xmax
        x = sx(float(xv))
        parts.append(_svg_text(x, axis_y + 15, f"{xv:.2f}", anchor="middle", fill="#64748b"))
    parts.append(
        _svg_text(
            left,
            height - 6,
            "line = [min, max] of e_arm(X) over the target sample   "
            "blue = p05-p95   red dot = min e_arm   amber dot = min e_{T}(X) among treated",
            fill="#64748b",
            size=11,
        )
    )
    caption = "Propensity reach per arm over the target sample (positivity diagnostic)."
    height = _figure_footer(parts, width, height, caption, result.diagnostics.gate)
    return Figure(
        svg=_svg_doc(width, height, "".join(parts)),
        kind="overlap",
        caption=caption,
    )


_FIGURES: dict[str, Callable[[CausalRankResult], Figure]] = {
    "estimates": _estimates_svg,
    "balance": _balance_svg,
    "ess": _ess_svg,
    "overlap": _overlap_svg,
}


def plot_result(result: CausalRankResult, kind: PlotKind = "estimates") -> Figure:
    """Build one inline-SVG figure from the result object.

    Kinds: ``estimates`` (forest plot), ``ranks`` (rank confidence sets),
    ``balance`` (max |SMD| before/after weighting), ``ess`` (n vs ESS),
    ``overlap`` (propensity reach per arm).
    """
    if kind not in PLOT_KINDS:
        raise ValueError(f"Unknown plot kind {kind!r}; expected one of {PLOT_KINDS}.")
    if kind == "ranks":
        if result.ranks is None:
            raise ValueError(
                "kind='ranks' requires rank inference. Set n_bootstrap >= 1 with a "
                "satisfied success-rate gate, or plot a kind that does not need inference."
            )
        return _ranks_svg(result)
    return _FIGURES[kind](result)


# --------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------


def _gate_banner(result: CausalRankResult) -> str:
    gate = result.diagnostics.gate
    text = {
        "pass": (
            "No empirical problems were flagged. This does <strong>not</strong> validate the "
            "causal assumptions (no unmeasured confounding, consistency, positivity, iid)."
        ),
        "caution": (
            "At least one diagnostic warning was raised — read them below before interpreting "
            "any adjusted mean or ranking."
        ),
        "fail": (
            "A critical diagnostic failed. The estimates are still returned for transparency, "
            "but they must <strong>not</strong> be presented as a reliable causal ranking."
        ),
    }[gate]
    return (
        f'<div class="gate gate-{gate}"><div class="gate-title">Quality gate: '
        f"{gate.upper()}</div><div>{text}</div></div>"
    )


def _warnings_html(result: CausalRankResult) -> str:
    warnings = result.warnings
    if not warnings:
        return (
            '<div class="note">No diagnostic warnings were raised. This is an empirical '
            "statement about this sample only — it does <strong>not</strong> validate any "
            "causal identification assumption.</div>"
        )
    items = []
    for w in warnings:
        details = (
            f'<div class="meta"><code>{_esc(w.details)}</code></div>' if w.details else ""
        )
        items.append(
            f'<div class="warn warn-{w.level}"><span class="code">[{_esc(w.level.upper())}] '
            f'{_esc(w.code)}</span>{_esc(w.message)}{details}</div>'
        )
    return "".join(items)


def _estimates_section(result: CausalRankResult) -> str:
    est = result.effects.to_dataframe()
    table = _df_to_html(est)
    fig = _estimates_svg(result)
    return (
        '<h2>1. Estimates</h2>'
        "<p>All arms are estimated on the <strong>same target population</strong>, "
        "so they are mutually comparable.</p>"
        f"{table}"
        f'<div class="fig">{fig.svg}</div>'
        f'<div class="formula">{_esc(result.metadata.get("estimator_formula", ""))}</div>'
    )


def _ranks_section(result: CausalRankResult) -> str:
    if result.ranks is None:
        return (
            '<h2>2. Rank confidence sets</h2>'
            '<div class="note"><strong>Not available in this result.</strong> '
            f"{_esc(result.metadata.get('rank_note', ''))}</div>"
        )
    ranks = result.ranks
    table = _df_to_html(ranks.to_dataframe())
    fig = _ranks_svg(result)
    meta = ranks.to_dict()
    return (
        '<h2>2. Rank confidence sets</h2>'
        f"{table}"
        f'<div class="fig">{fig.svg}</div>'
        f'<div class="formula">{_esc(meta["formula"])}</div>'
        f"<p class=\"meta\">confidence_level = {meta['confidence_level']:.4f}, "
        f"higher_is_better = {meta['higher_is_better']}</p>"
        f"<p class=\"meta\">tie convention: {_esc(meta['tie_convention'])}</p>"
        '<div class="note">A rank of 1 in the point estimate is <strong>not</strong> a '
        "probability of being best; a wide interval when arms cannot be separated is the "
        "honest answer, not a failure of the method.</div>"
    )


def _pairwise_section(result: CausalRankResult) -> str:
    if result.pairwise is None:
        return (
            '<h2>3. Pairwise comparisons</h2>'
            '<div class="note"><strong>Not available in this result.</strong> '
            f"{_esc(result.metadata.get('rank_note', ''))}</div>"
        )
    pairwise = result.pairwise
    table = _df_to_html(pairwise.to_dataframe())
    meta = pairwise.to_dict()
    confirmed = pairwise.confirmed
    n_confirmed = len(confirmed)
    return (
        '<h2>3. Pairwise comparisons (simultaneous)</h2>'
        f"{table}"
        f'<div class="formula">{_esc(meta["formula"])}</div>'
        f"<p class=\"meta\">M = {meta['n_pairs']}, z_crit = {meta['z_crit']:.6f}, "
        f"confidence_level = {meta['confidence_level']:.4f}, "
        f"higher_is_better = {meta['higher_is_better']}; "
        f"{n_confirmed} of {meta['n_pairs']} pairs confirmed.</p>"
        '<div class="note"><code>inconclusive</code> means the data did not confirm a '
        "direction — it is <strong>not</strong> evidence that the two arms are equal. "
        "Pointwise CIs are not a substitute for these simultaneous intervals.</div>"
    )


def _diagnostics_section(result: CausalRankResult) -> str:
    diag = result.diagnostics
    parts = ["<h2>4. Diagnostics</h2>"]
    parts.append("<h3>4.1 Group summary (observed data)</h3>")
    parts.append(_df_to_html(diag.group_summary))
    parts.append("<h3>4.2 Weights &amp; effective sample size</h3>")
    parts.append(_df_to_html(diag.weights))
    ess_fig = _ess_svg(result)
    parts.append(f'<div class="fig">{ess_fig.svg}</div>')
    parts.append("<h3>4.3 Overlap / common support</h3>")
    parts.append(_df_to_html(diag.overlap))
    ov_fig = _overlap_svg(result)
    parts.append(f'<div class="fig">{ov_fig.svg}</div>')
    parts.append("<h3>4.4 Covariate balance (SMD)</h3>")
    if diag.balance_summary.empty:  # pragma: no cover - validation guarantees terms
        parts.append('<div class="note">No balance terms computed.</div>')
    else:
        parts.append(_df_to_html(diag.balance_summary))
        bal_fig = _balance_svg(result)
        parts.append(f'<div class="fig">{bal_fig.svg}</div>')
        parts.append("<details><summary>Full long-form SMD table (feature x level x pair)</summary>")
        parts.append(_df_to_html(diag.balance))
        parts.append("</details>")
    parts.append("<h3>4.5 Bootstrap</h3>")
    boot = result.metadata.get("bootstrap")
    if boot is None:
        parts.append(
            '<div class="note">Joint bootstrap disabled '
            f"(<code>n_bootstrap=None</code>). {_esc(result.metadata.get('inference_note', ''))}"
            "</div>"
        )
    else:
        boot_df = pd.DataFrame(
            [
                {
                    "n_requested": boot.get("n_requested"),
                    "n_success": boot.get("n_success"),
                    "n_failed": boot.get("n_failed"),
                    "failure_rate": boot.get("failure_rate"),
                    "usable": boot.get("usable"),
                    "base_seed": boot.get("base_seed"),
                }
            ]
        )
        boot_df = boot_df.rename(
            columns={
                "n_requested": "n requested",
                "n_success": "n success",
                "n_failed": "n failed",
                "failure_rate": "failure rate",
                "usable": "usable",
                "base_seed": "base seed",
            }
        )
        formatters_backup = {
            "n requested": fmt_int,
            "n success": fmt_int,
            "n failed": fmt_int,
            "failure rate": fmt_percent,
            "usable": lambda v: "yes" if v else "NO",
            "base seed": fmt_int_plain,
        }
        head = "".join(f"<th>{_esc(c)}</th>" for c in boot_df.columns)
        tds = "".join(
            f"<td>{formatters_backup.get(c, fmt_float)(v)}</td>"
            for c, v in zip(boot_df.columns, boot_df.iloc[0], strict=True)
        )
        parts.append(
            f'<div class="table-wrap"><table class="data"><thead><tr>{head}</tr></thead>'
            f"<tbody><tr>{tds}</tr></tbody></table></div>"
        )
        parts.append(f'<p class="meta">{_esc(result.metadata.get("inference_note", ""))}</p>')
        reasons = boot.get("failure_reasons") or []
        if reasons:
            parts.append("<details><summary>Bootstrap failure reasons (first 20 rounds)</summary>")
            items = "".join(f"<li>{_esc(r)}</li>" for r in reasons)
            parts.append(f"<ul>{items}</ul></details>")
    return "".join(parts)


def _covariance_section(result: CausalRankResult) -> str:
    if result.covariance is None:
        return ""
    labels = [str(x) for x in result.effects.labels]
    sigma = np.asarray(result.covariance, dtype=float)
    head_cells = "".join(f"<th>{_esc(lab)}</th>" for lab in ["arm", *labels])
    rows = "".join(
        f"<tr><td>{_esc(lab)}</td>"
        + "".join(f"<td>{fmt_float(sigma[i, j], 6)}</td>" for j in range(len(labels)))
        + "</tr>"
        for i, lab in enumerate(labels)
    )
    return (
        "<h2>5. Covariance matrix (joint bootstrap)</h2>"
        f'<div class="table-wrap"><table class="data"><thead><tr>{head_cells}</tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
        '<p class="meta">K x K matrix in canonical label order; it is the same array as '
        "<code>result.covariance</code>. Non-diagonal entries come from the joint "
        "re-fitting bootstrap (shared row resampling), so arm correlations are carried "
        "into every difference SE.</p>"
    )


def _caveats_html(result: CausalRankResult) -> str:
    caveats = [
        result.metadata.get("causal_caveat", ""),
        result.metadata.get("inference_note", ""),
        result.metadata.get("rank_note", ""),
    ]
    items = "".join(f"<li>{_esc(c)}</li>" for c in caveats if c)
    return (
        "<h2>6. Assumptions, caveats &amp; red lines</h2>"
        f'<ul class="caveats">{items}'
        "<li>Model fit metrics (AUC / accuracy) are not evidence of causal validity; "
        "balance, overlap and stability diagnostics are the relevant screens.</li>"
        "<li>Software-only research use — not a basis for business, medical or policy "
        "decisions.</li></ul>"
    )


def _meta_table(result: CausalRankResult) -> str:
    md = result.metadata
    rows = [
        ("package", f"causalrank {md.get('version', '?')}"),
        ("created_at", md.get("created_at", "")),
        ("estimand", f"{md.get('estimand')} — {md.get('estimand_formula', '')}"),
        ("estimator", f"{md.get('estimator')} / {md.get('weighting')} / "
                      f"propensity={md.get('propensity_model')}"),
        ("data", f"n = {md.get('n_rows')}, K = {md.get('n_treatments')}, "
                 f"labels = {md.get('labels')}"),
        ("inference", f"n_bootstrap = {md.get('n_bootstrap')}, "
                      f"ci_method = {md.get('ci_method')}, "
                      f"confidence_level = {md.get('confidence_level')}, "
                      f"cross_fit = {md.get('cross_fit')}"),
        ("reproducibility", f"random_state = {md.get('random_state')}, "
                            f"python = {md.get('python')}"),
    ]
    body = "".join(
        f"<tr><td><strong>{_esc(k)}</strong></td><td>{_esc(v)}</td></tr>" for k, v in rows
    )
    return (
        '<div class="table-wrap"><table class="data"><tbody>'
        f"{body}</tbody></table></div>"
    )


def render_html(result: CausalRankResult, *, title: str = "CausalRank report") -> str:
    """Render the result as a self-contained static HTML document.

    Every displayed number comes from ``result`` (same rounding as the
    numeric API); nothing is recomputed. The output embeds no scripts, no
    external assets and no render-time clock.
    """
    sections = [
        _estimates_section(result),
        _ranks_section(result),
        _pairwise_section(result),
        _diagnostics_section(result),
        _covariance_section(result),
        _caveats_html(result),
    ]
    body = "".join(sections)
    warnings_html = _warnings_html(result)
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body><main>"
        f"<h1>{_esc(title)}</h1>"
        f'<p class="meta">Generated by <code>causalrank</code> '
        f"{_esc(result.metadata.get('version', ''))} — static report, every number is "
        "traceable to the result object.</p>"
        f"{_meta_table(result)}"
        f"{_gate_banner(result)}"
        f'<h2>0. Warnings &amp; quality gate</h2>{warnings_html}'
        f"{body}"
        f"<footer>causalrank {_esc(result.metadata.get('version', ''))} · "
        f"created_at {_esc(result.metadata.get('created_at', ''))} · "
        "self-contained static HTML (no scripts, no external assets)</footer>"
        "</main></body></html>"
    )


def save_html(result: CausalRankResult, path: str | Path, *,
              title: str = "CausalRank report") -> Path:
    """Render :func:`render_html` and write it to ``path``; returns the path."""
    out = Path(path)
    out.write_text(render_html(result, title=title), encoding="utf-8")
    return out
