"""Phase 4: static HTML report and SVG figures (traceability + honesty).

Key acceptance (PROJECT_SPEC.md phase 4):

* every displayed number is traceable to the result object (same values and
  same rounding as the numeric API);
* the honest form travels with the numbers — quality gate, structured
  warnings, blocked/absent inference state, formulas, tie convention and the
  causal caveats are part of the report;
* no scripts, no external assets, deterministic output;
* labels are HTML-escaped (user data never injects markup).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrank import CausalRank
from causalrank.results import CausalRankResult
from causalrank.simulation import make_dataset
from causalrank.visualization import PLOT_KINDS, Figure, plot_result, render_html, save_html

FAST_GBM = {
    "max_iter": 40,
    "learning_rate": 0.15,
    "max_depth": 3,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
}


def _fit(n: int = 500, *, bootstrap: int | None = 60, seed: int = 2026, **overrides: object) -> CausalRankResult:
    ds = make_dataset("confounded", n=n, random_state=seed)
    params: dict[str, object] = {
        "treatment": "treatment",
        "outcome": "outcome",
        "confounders": ["x1", "x2", "x3", "x4"],
        "cross_fit": 3,
        "n_bootstrap": bootstrap,
        "bootstrap_min_successes": 20,
        "random_state": seed,
        "gbm_params": FAST_GBM,
    }
    params.update(overrides)
    return CausalRank(**params).fit(ds.df)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def inferred_result() -> CausalRankResult:
    """Full result with working bootstrap inference (module-scoped: fit is slow)."""
    return _fit(bootstrap=60)


@pytest.fixture(scope="module")
def plain_result() -> CausalRankResult:
    """Point-estimates-only result (n_bootstrap=None)."""
    return _fit(bootstrap=None)


class TestTraceability:
    def test_every_mu_appears_in_html(self, inferred_result: CausalRankResult) -> None:
        html_text = inferred_result.to_html()
        est = inferred_result.effects.to_dataframe()
        for mu in est["mu"]:
            assert f"{mu:.4f}" in html_text

    def test_se_and_ci_appear_in_html(self, inferred_result: CausalRankResult) -> None:
        html_text = inferred_result.to_html()
        est = inferred_result.effects.to_dataframe()
        for col in ("se", "ci_low", "ci_high"):
            for v in est[col]:
                assert f"{v:.4f}" in html_text

    def test_labels_and_metadata_appear(self, inferred_result: CausalRankResult) -> None:
        html_text = inferred_result.to_html()
        for lab in inferred_result.effects.labels:
            assert f"<td>{lab}</td>" in html_text
        md = inferred_result.metadata
        assert md["version"] in html_text
        assert md["created_at"] in html_text
        assert str(md["random_state"]) in html_text

    def test_covariance_matches_result(self, inferred_result: CausalRankResult) -> None:
        assert inferred_result.covariance is not None
        html_text = inferred_result.to_html()
        sigma = inferred_result.covariance
        for i in range(sigma.shape[0]):
            for j in range(sigma.shape[1]):
                assert f"{sigma[i, j]:.6f}" in html_text

    def test_rank_and_pairwise_values_appear(self, inferred_result: CausalRankResult) -> None:
        assert inferred_result.ranks is not None
        assert inferred_result.pairwise is not None
        html_text = inferred_result.to_html()
        ranks = inferred_result.ranks.to_dataframe()
        for _, row in ranks.iterrows():
            assert f"<td>{int(row['rank_low'])}</td>" in html_text
            assert f"<td>{int(row['rank_high'])}</td>" in html_text
        pairs = inferred_result.pairwise.to_dataframe()
        for diff in pairs["diff"]:
            assert f"{diff:.4f}" in html_text
        assert f"z_crit = {inferred_result.pairwise.z_crit:.6f}" in html_text

    def test_warnings_rendered(self, inferred_result: CausalRankResult) -> None:
        html_text = inferred_result.to_html()
        for w in inferred_result.warnings:
            assert w.code in html_text
            assert w.message[:60] in html_text or w.level.upper() in html_text


class TestHonestStates:
    def test_inference_absent_is_explicit(self, plain_result: CausalRankResult) -> None:
        assert plain_result.inference_available is False
        html_text = plain_result.to_html()
        assert "Not available in this result" in html_text
        assert "n_bootstrap=None" in html_text or "n_bootstrap = None" in html_text
        # se columns are NaN -> rendered as an em dash, never as a fake number
        assert "&mdash;" in html_text

    def test_quality_gate_banner_present(self, inferred_result: CausalRankResult) -> None:
        html_text = inferred_result.to_html()
        gate = inferred_result.diagnostics.gate
        assert f"Quality gate: {gate.upper()}" in html_text
        assert 'class="gate gate-' in html_text

    def test_red_lines_present(self, inferred_result: CausalRankResult) -> None:
        html_text = inferred_result.to_html()
        assert "not</strong> evidence that the two arms are equal" in html_text
        assert "probability of being best" in html_text
        assert "AUC / accuracy) are not evidence of causal validity" in html_text

    def test_formulas_present(self, inferred_result: CausalRankResult) -> None:
        html_text = inferred_result.to_html()
        assert "Hajek-normalised IPTW" in html_text
        assert "Bonferroni" in html_text
        assert "[1 + B_i, K - W_i]" in html_text


class TestSafetyAndDeterminism:
    def test_no_scripts_or_external_assets(self, inferred_result: CausalRankResult) -> None:
        html_text = inferred_result.to_html()
        assert "<script" not in html_text.lower()
        assert "http-equiv" not in html_text.lower()
        # the SVG xmlns is a namespace identifier, not a fetched resource
        assert "<img" not in html_text.lower()
        assert "@import" not in html_text
        assert "url(" not in html_text

    def test_deterministic_rendering(self, inferred_result: CausalRankResult) -> None:
        assert inferred_result.to_html() == inferred_result.to_html()

    def test_labels_are_escaped(self) -> None:
        rng = np.random.default_rng(3)
        n = 240
        df = pd.DataFrame(
            {
                "x1": rng.normal(size=n),
                "x2": rng.normal(size=n),
                "treatment": np.array(["<b>A</b>", "B & C"] * (n // 2)),
                "outcome": (rng.uniform(size=n) < 0.5).astype(float),
            }
        )
        result = CausalRank(
            treatment="treatment",
            outcome="outcome",
            confounders=["x1", "x2"],
            gbm_params=FAST_GBM,
            random_state=1,
        ).fit(df)
        html_text = result.to_html()
        assert "<b>A</b>" not in html_text
        assert "&lt;b&gt;A&lt;/b&gt;" in html_text
        assert "B &amp; C" in html_text

    def test_title_is_escaped_and_applied(self, inferred_result: CausalRankResult) -> None:
        html_text = inferred_result.to_html(title="<script>x</script> report")
        assert "<script>x</script>" not in html_text
        assert "&lt;script&gt;x&lt;/script&gt; report" in html_text


class TestFileOutput:
    def test_to_html_writes_file(self, inferred_result: CausalRankResult, tmp_path) -> None:
        target = tmp_path / "sub" / "report.html"
        returned = inferred_result.to_html(target)
        assert target.exists()
        assert target.read_text(encoding="utf-8") == returned

    def test_save_html_helper(self, inferred_result: CausalRankResult, tmp_path) -> None:
        target = tmp_path / "via_helper.html"
        out = save_html(inferred_result, target, title="Helper report")
        assert out == target
        text = target.read_text(encoding="utf-8")
        assert "Helper report" in text
        assert text == render_html(inferred_result, title="Helper report")


class TestFigures:
    @pytest.mark.parametrize("kind", PLOT_KINDS)
    def test_each_kind_produces_svg(self, inferred_result: CausalRankResult, kind: str) -> None:
        fig = plot_result(inferred_result, kind)  # type: ignore[arg-type]
        assert isinstance(fig, Figure)
        assert fig.svg.startswith("<svg")
        assert "</svg>" in fig.svg
        assert fig.kind == kind

    def test_unknown_kind_raises(self, inferred_result: CausalRankResult) -> None:
        with pytest.raises(ValueError, match="Unknown plot kind"):
            plot_result(inferred_result, "scatter")  # type: ignore[arg-type]

    def test_ranks_figure_requires_inference(self, plain_result: CausalRankResult) -> None:
        with pytest.raises(ValueError, match="requires rank inference"):
            plain_result.plot("ranks")

    def test_result_plot_method(self, inferred_result: CausalRankResult) -> None:
        fig = inferred_result.plot("ess")
        assert isinstance(fig, Figure)
        assert "ESS" in fig.svg or fig.caption

    def test_figure_save_svg_and_html(self, inferred_result: CausalRankResult, tmp_path) -> None:
        fig: Figure = inferred_result.plot("overlap")
        svg_path = fig.save(tmp_path / "fig.svg")
        html_path = fig.save(tmp_path / "fig.html")
        assert svg_path.read_text(encoding="utf-8").startswith("<svg")
        html_text = html_path.read_text(encoding="utf-8")
        assert "<svg" in html_text and "<!DOCTYPE html>" in html_text

    def test_figures_have_no_scripts(self, inferred_result: CausalRankResult) -> None:
        for kind in PLOT_KINDS:
            fig = plot_result(inferred_result, kind)  # type: ignore[arg-type]
            assert "<script" not in fig.svg.lower()
            assert "http" not in fig.svg.replace('xmlns="http://www.w3.org/2000/svg"', "")

    def test_figures_deterministic(self, inferred_result: CausalRankResult) -> None:
        for kind in PLOT_KINDS:
            a = plot_result(inferred_result, kind)  # type: ignore[arg-type]
            b = plot_result(inferred_result, kind)  # type: ignore[arg-type]
            assert a.svg == b.svg


class TestBlockedInferenceReport:
    """A blocked-inference fit must render an honest report (not a crash)."""

    def test_blocked_bootstrap_report(self) -> None:
        ds = make_dataset("rare_treatment", n=60, random_state=5)
        result = CausalRank(
            treatment="treatment",
            outcome="outcome",
            confounders=["x1", "x2", "x3", "x4"],
            n_bootstrap=20,
            bootstrap_min_successes=20,
            bootstrap_min_success_rate=1.0,  # any failed round blocks inference
            random_state=11,
            gbm_params=FAST_GBM,
        ).fit(ds.df)
        assert result.inference_available is False
        html_text = result.to_html()
        assert "BLOCKED" in html_text or "Not available" in html_text
        boot = result.metadata["bootstrap"]
        assert str(boot["n_failed"]) in html_text
        assert boot["usable"] is False
        # ranks section must fall back to the explicit note, not render numbers
        assert "Rank confidence sets" in html_text


class TestInfiniteSmdNotMissing:
    """Review P1: +/-inf SMD is a real value, not missing data."""

    def test_fmt_float_distinguishes_inf_from_nan(self) -> None:
        from causalrank.visualization.report import fmt_float

        assert fmt_float(float("inf")) == "&infin;"
        assert fmt_float(float("-inf")) == "&minus;&infin;"
        assert fmt_float(float("nan")) == "&mdash;"  # NaN still means missing
        assert fmt_float(0.12345) == "0.1235"

    def test_degenerate_covariate_renders_infinity(self) -> None:
        # constant within each arm but different across arms -> zero
        # within-arm variance -> SMD is +/-inf by definition
        rng = np.random.default_rng(11)
        n = 300
        treatment = np.array(["A", "B", "C"] * (n // 3))
        x_deg = np.select([treatment == "A", treatment == "B"], [0.0, 1.0], default=2.0)
        df = pd.DataFrame(
            {
                "x1": rng.normal(size=n),
                "x_deg": x_deg,
                "treatment": treatment,
                "outcome": (rng.uniform(size=n) < 0.5).astype(float),
            }
        )
        result = CausalRank(
            treatment="treatment",
            outcome="outcome",
            confounders=["x1", "x_deg"],
            gbm_params=FAST_GBM,
            random_state=3,
        ).fit(df)
        degenerate = result.diagnostics.balance_summary[
            result.diagnostics.balance_summary["feature"] == "x_deg"
        ]
        assert np.isinf(degenerate["max_abs_smd_weighted"]).any()
        html_text = result.to_html()
        assert "&infin;" in html_text  # shown as infinity, never as the dash
        # the balance figure must not emit nan coordinates for infinite values
        fig = plot_result(result, "balance")
        assert "nan" not in fig.svg.lower()
        assert "∞" in fig.svg  # off-scale marker present


class TestBalanceAxisCoversWeighted:
    """Review P1: the SMD axis bound must include the weighted series."""

    def test_axis_covers_weighted_imbalance(self, inferred_result: CausalRankResult) -> None:
        import copy
        import re

        result = copy.deepcopy(inferred_result)
        bs = result.diagnostics.balance_summary.copy()
        bs["max_abs_smd_unweighted"] = 0.15
        bs["max_abs_smd_weighted"] = 0.80  # weighting made it much worse
        result.diagnostics.balance_summary = bs
        fig = plot_result(result, "balance")
        ticks = [float(m) for m in re.findall(r">(\d+\.\d\d)</text>", fig.svg)]
        assert ticks, "axis ticks missing"
        assert max(ticks) >= 0.80, f"axis max {max(ticks)} would clip a weighted SMD of 0.80"


class TestIdentifierLabels:
    """Review P1: labels are identifiers, never float-formatted numbers."""

    def test_integer_labels_not_float_formatted(self) -> None:
        rng = np.random.default_rng(5)
        n = 150
        df = pd.DataFrame(
            {
                "x1": rng.normal(size=n),
                "x2": rng.normal(size=n),
                "treatment": np.tile([1, 2, 3], n // 3),
                "outcome": (rng.uniform(size=n) < 0.5).astype(float),
            }
        )
        result = CausalRank(
            treatment="treatment",
            outcome="outcome",
            confounders=["x1", "x2"],
            gbm_params=FAST_GBM,
            random_state=5,
        ).fit(df)
        html_text = result.to_html()
        assert "<td>1</td>" in html_text
        assert "<td>1.0000</td>" not in html_text

    def test_large_integer_labels_keep_precision(self) -> None:
        # 2^53 and 2^53+1 are distinct integers that collapse onto the same
        # float64; labels must survive verbatim for traceability
        hi, lo = 2**53, 2**53 + 1
        rng = np.random.default_rng(6)
        n = 120
        df = pd.DataFrame(
            {
                "x1": rng.normal(size=n),
                "x2": rng.normal(size=n),
                "treatment": np.array([hi, lo] * (n // 2)),
                "outcome": (rng.uniform(size=n) < 0.5).astype(float),
            }
        )
        result = CausalRank(
            treatment="treatment",
            outcome="outcome",
            confounders=["x1", "x2"],
            gbm_params=FAST_GBM,
            random_state=6,
        ).fit(df)
        html_text = result.to_html()
        assert str(hi) in html_text
        assert str(lo) in html_text
        assert f"{float(lo):.4f}" not in html_text  # no float64 collapse


class TestStandaloneFigureContext:
    """Review P2: a figure distributed alone keeps caption and quality gate."""

    @pytest.mark.parametrize("kind", PLOT_KINDS)
    def test_svg_carries_caption_and_gate(
        self, inferred_result: CausalRankResult, kind: str
    ) -> None:
        fig = plot_result(inferred_result, kind)  # type: ignore[arg-type]
        assert fig.caption[:40] in fig.svg  # caption survives line wrapping
        assert "quality gate" in fig.svg.lower()
        assert inferred_result.diagnostics.gate.upper() in fig.svg

    def test_saved_svg_keeps_context(self, inferred_result: CausalRankResult, tmp_path) -> None:
        fig = plot_result(inferred_result, "estimates")
        out = fig.save(tmp_path / "estimates.svg")
        text = out.read_text(encoding="utf-8")
        assert "quality gate" in text.lower()
        assert "confidence intervals" in text

    def test_html_report_does_not_duplicate_caption(
        self, inferred_result: CausalRankResult
    ) -> None:
        html_text = inferred_result.to_html()
        assert "fig-caption" not in html_text  # external copy removed
        assert "Point estimates mu_a with" in html_text  # now inside the SVG
