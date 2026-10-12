"""Rank inference tests: pairwise simultaneous CIs and rank confidence sets.

Covers PROJECT_SPEC.md 7.4 (hand-computed examples, correlation in difference
SEs, PSD / zero-variance handling, multiplicity, tie rules, label-order
invariance) and the Phase 3 acceptance criteria (K=2 degeneracy, common-shift
and relabelling invariance, no equality claims from non-significance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrank.exceptions import CausalRankError
from causalrank.inference.pairwise import (
    DECISIONS,
    bonferroni_critical_value,
    difference_se,
    difference_variance,
    pairwise_comparisons,
)
from causalrank.inference.ranking import rank_confidence_sets, rank_from_estimates

Z_975 = 1.959963984540054
Z_BONF_3 = 2.3939797998185104  # norm.ppf(1 - 0.05/6): K=3 -> M=3 pairs
Z_950 = 1.6448536269514722  # norm.ppf(0.95)


# ---------------------------------------------------------------------------
# hand-computed examples (spec 7.4: 2 and 3 arms)
# ---------------------------------------------------------------------------
def test_hand_computed_two_routes() -> None:
    """K=2: difference, SE, simultaneous CI and rank intervals by hand."""
    mu = {"A": 0.5, "B": 0.4}
    sigma = np.array([[0.0025, 0.0018], [0.0018, 0.0016]])
    res = rank_from_estimates(mu, sigma, confidence_level=0.95)

    pairs = res.pairwise.to_dataframe()
    assert list(pairs["label_i"]) == ["A"] and list(pairs["label_j"]) == ["B"]
    # se_diff = sqrt(0.0025 + 0.0016 - 2*0.0018) = sqrt(0.0005)
    assert pairs.loc[0, "diff"] == pytest.approx(0.1, rel=1e-12)
    assert pairs.loc[0, "se_diff"] == pytest.approx(np.sqrt(0.0005), rel=1e-12)
    assert pairs.loc[0, "ci_low"] == pytest.approx(0.1 - Z_975 * np.sqrt(0.0005), rel=1e-12)
    assert pairs.loc[0, "ci_high"] == pytest.approx(0.1 + Z_975 * np.sqrt(0.0005), rel=1e-12)
    assert pairs.loc[0, "decision"] == "i_better"
    assert bool(pairs.loc[0, "confirmed"]) is True

    ranks = res.ranks.to_dataframe().set_index("label")
    assert ranks.loc["A", "point_rank"] == 1
    assert ranks.loc["B", "point_rank"] == 2
    # A confirmed better than B: W_A = 1 -> U_A = 1, B_B = 1 -> L_B = 2
    assert (int(ranks.loc["A", "rank_low"]), int(ranks.loc["A", "rank_high"])) == (1, 1)
    assert (int(ranks.loc["B", "rank_low"]), int(ranks.loc["B", "rank_high"])) == (2, 2)


def test_hand_computed_three_routes() -> None:
    """K=3: Bonferroni z with M=3 and a fully worked rank example."""
    mu = np.array([0.50, 0.47, 0.20])
    sigma = np.array(
        [
            [0.0001, 0.0000155, 0.0],
            [0.0000155, 0.0001, 0.0],
            [0.0, 0.0, 0.0001],
        ]
    )
    rows, info = pairwise_comparisons(mu, sigma, labels=("A", "B", "C"))
    assert info["n_pairs"] == 3
    assert info["z_crit"] == pytest.approx(Z_BONF_3, rel=1e-12)

    by_pair = {(r["label_i"], r["label_j"]): r for r in rows}
    ab = by_pair[("A", "B")]
    # se(A-B) = sqrt(2e-4 - 2*1.55e-5) = 0.013
    assert ab["se_diff"] == pytest.approx(0.013, rel=1e-9)
    assert ab["ci_low"] == pytest.approx(0.03 - Z_BONF_3 * 0.013, rel=1e-9)
    assert ab["ci_high"] == pytest.approx(0.03 + Z_BONF_3 * 0.013, rel=1e-9)
    # |diff|/se = 2.31: significant pointwise (z=1.96) but NOT after Bonferroni
    assert 0.03 - Z_975 * 0.013 > 0  # pointwise CI would exclude zero
    assert ab["decision"] == "inconclusive"

    assert by_pair[("A", "C")]["decision"] == "i_better"
    assert by_pair[("B", "C")]["decision"] == "i_better"

    ranks = pd.DataFrame(rank_confidence_sets(mu, rows, ("A", "B", "C")))
    ranks = ranks.set_index("label")
    assert (ranks.loc["A", "rank_low"], ranks.loc["A", "rank_high"]) == (1, 2)
    assert (ranks.loc["B", "rank_low"], ranks.loc["B", "rank_high"]) == (1, 2)
    # C is confirmed worse than both others -> pinned to rank 3
    assert (ranks.loc["C", "rank_low"], ranks.loc["C", "rank_high"]) == (3, 3)
    assert list(ranks["point_rank"]) == [1, 2, 3]


def test_bonferroni_k2_reduces_to_unadjusted_comparison() -> None:
    """Spec acceptance: with K=2 the Bonferroni factor is z_{1-alpha/2}."""
    assert bonferroni_critical_value(0.95, 1) == pytest.approx(Z_975, rel=1e-12)
    assert bonferroni_critical_value(0.90, 1) == pytest.approx(Z_950, rel=1e-12)
    mu = {"A": 0.5, "B": 0.4}
    sigma = np.array([[0.0025, 0.0018], [0.0018, 0.0016]])
    res = rank_from_estimates(mu, sigma, confidence_level=0.95)
    row = res.pairwise.to_dataframe().iloc[0]
    se = difference_se(sigma, 0, 1)
    assert row["ci_low"] == pytest.approx(0.1 - Z_975 * se, rel=1e-12)
    assert row["ci_high"] == pytest.approx(0.1 + Z_975 * se, rel=1e-12)


def test_correlation_is_carried_into_difference_se() -> None:
    """Var(mu_i - mu_j) must subtract 2 * Sigma_ij (spec 7.4)."""
    sigma = np.array([[0.0025, 0.0018], [0.0018, 0.0016]])
    se_indep = float(np.sqrt(0.0025 + 0.0016))
    se_true = difference_se(sigma, 0, 1)
    assert se_true == pytest.approx(np.sqrt(0.0025 + 0.0016 - 2 * 0.0018), rel=1e-12)
    assert se_true < se_indep  # positive correlation shrinks the difference SE
    assert difference_variance(sigma, 0, 1) == pytest.approx(0.0005, rel=1e-12)


def test_difference_se_rejects_meaningful_negative_variance() -> None:
    # symmetric but indefinite matrix -> negative difference variance
    sigma = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(CausalRankError, match="negative"):
        difference_variance(sigma, 0, 1)
    # numerical noise of a singular covariance is clipped to zero
    assert difference_variance(np.array([[1.0, 1.0], [1.0, 1.0]]), 0, 1) == 0.0


# ---------------------------------------------------------------------------
# multiplicity and decision rules
# ---------------------------------------------------------------------------
def test_simultaneous_ci_only_confirms_after_multiplicity_adjustment() -> None:
    """A pointwise-significant pair must stay inconclusive under Bonferroni."""
    mu = np.array([0.50, 0.47, 0.20])
    sigma = np.array(
        [
            [0.0001, 0.0000155, 0.0],
            [0.0000155, 0.0001, 0.0],
            [0.0, 0.0, 0.0001],
        ]
    )
    rows, _ = pairwise_comparisons(mu, sigma, labels=("A", "B", "C"))
    ab = next(r for r in rows if r["label_i"] == "A" and r["label_j"] == "B")
    assert ab["ci_low"] <= 0 <= ab["ci_high"]
    assert ab["decision"] == "inconclusive"
    assert ab["confirmed"] is False


def test_decision_vocabulary_never_claims_equality() -> None:
    """Non-significance is not evidence of equality: no 'equal' decision exists."""
    assert "equal" not in DECISIONS and "tie" not in DECISIONS
    mu = np.array([0.5, 0.5, 0.5])
    sigma = np.eye(3) * 0.01
    rows, _ = pairwise_comparisons(mu, sigma, labels=("A", "B", "C"))
    assert {r["decision"] for r in rows} == {"inconclusive"}
    assert all(r["decision"] in DECISIONS for r in rows)


def test_zero_variance_difference_is_explicit() -> None:
    """Degenerate variance: point interval, sign decides, exact tie stays open."""
    sigma = np.array([[0.001, 0.001], [0.001, 0.001]])  # singular but PSD
    rows, _ = pairwise_comparisons({"A": 0.5, "B": 0.4}, sigma)
    row = rows[0]
    assert row["degenerate_se"] is True
    assert row["se_diff"] == 0.0
    assert row["ci_low"] == row["ci_high"] == pytest.approx(0.1, rel=1e-12)
    assert row["decision"] == "i_better"

    rows, _ = pairwise_comparisons({"A": 0.5, "B": 0.5}, sigma)
    row = rows[0]
    assert row["decision"] == "inconclusive"
    assert row["point_tie"] is True


def test_point_estimate_ties_share_rank_and_ignore_label_order() -> None:
    """Tied point estimates share the best rank; labels never break ties."""
    mu = {"Z": 0.5, "A": 0.5, "B": 0.3}  # deliberately non-alphabetical labels
    sigma = np.eye(3) * 0.01
    res = rank_from_estimates(mu, sigma)
    ranks = res.ranks.to_dataframe().set_index("label")
    assert ranks.loc["Z", "point_rank"] == 1
    assert ranks.loc["A", "point_rank"] == 1  # tie shares the rank, not rank 2
    assert ranks.loc["B", "point_rank"] == 3
    assert ranks.loc["Z", "point_tie"] and ranks.loc["A", "point_tie"]
    assert not ranks.loc["B", "point_tie"]
    pairs = res.pairwise.to_dataframe()
    za = pairs[(pairs["label_i"] == "Z") & (pairs["label_j"] == "A")].iloc[0]
    assert za["decision"] == "inconclusive"
    assert bool(za["point_tie"]) is True


def test_equal_true_effects_give_full_rank_sets() -> None:
    """Red line: noise must not produce a determined ranking under true ties."""
    mu = {"A": 0.4, "B": 0.4, "C": 0.4}
    sigma = np.array(
        [
            [0.01, 0.002, 0.002],
            [0.002, 0.01, 0.002],
            [0.002, 0.002, 0.01],
        ]
    )
    res = rank_from_estimates(mu, sigma)
    ranks = res.ranks.to_dataframe()
    assert (ranks["rank_low"] == 1).all()
    assert (ranks["rank_high"] == 3).all()
    assert ranks["point_tie"].all()
    assert not res.pairwise.to_dataframe()["confirmed"].any()


# ---------------------------------------------------------------------------
# invariance and degenerate configurations
# ---------------------------------------------------------------------------
def test_ranking_invariant_to_label_reordering() -> None:
    """Renaming / reordering arms only permutes rows (spec acceptance)."""
    mu = {"A": 0.50, "B": 0.47, "C": 0.20}
    sigma = np.array(
        [
            [0.0001, 0.0000155, 0.0],
            [0.0000155, 0.0001, 0.0],
            [0.0, 0.0, 0.0001],
        ]
    )

    def normalize(res: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        ranks = res.ranks.to_dataframe().set_index("label").sort_index()
        pairs = res.pairwise.to_dataframe()
        rows = []
        for _, row in pairs.iterrows():
            i, j = row["label_i"], row["label_j"]
            if i > j:  # re-orient the comparison to the sorted pair (j, i)
                i, j = j, i
                better = (
                    {"i_better": j, "j_better": i, "inconclusive": None}[row["decision"]]
                )
            else:
                better = (
                    {"i_better": i, "j_better": j, "inconclusive": None}[row["decision"]]
                )
            rows.append(
                {
                    "pair": (i, j),
                    "se_diff": row["se_diff"],
                    "abs_diff": abs(row["diff"]),
                    "better": better,
                    "point_tie": bool(row["point_tie"]),
                }
            )
        return ranks, pd.DataFrame(rows).set_index("pair").sort_index()

    base_ranks, base_pairs = normalize(rank_from_estimates(mu, sigma))

    perm = ["C", "A", "B"]
    mu2 = {k: mu[k] for k in perm}
    idx = [("A", "B", "C").index(k) for k in perm]  # position in the original order
    sigma2 = sigma[np.ix_(idx, idx)]
    alt_ranks, alt_pairs = normalize(rank_from_estimates(mu2, sigma2))

    pd.testing.assert_frame_equal(
        base_ranks.reset_index(drop=True), alt_ranks.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        base_pairs.reset_index(drop=True), alt_pairs.reset_index(drop=True)
    )


def test_ranking_invariant_to_common_shift() -> None:
    """Adding a constant to every mu leaves differences and ranks unchanged."""
    mu = {"A": 0.50, "B": 0.47, "C": 0.20}
    sigma = np.array(
        [
            [0.0001, 0.0000155, 0.0],
            [0.0000155, 0.0001, 0.0],
            [0.0, 0.0, 0.0001],
        ]
    )
    base = rank_from_estimates(mu, sigma)
    shifted = rank_from_estimates({k: v + 5.0 for k, v in mu.items()}, sigma)

    b_pairs = base.pairwise.to_dataframe()
    s_pairs = shifted.pairwise.to_dataframe()
    assert np.allclose(b_pairs["diff"], s_pairs["diff"], rtol=0, atol=1e-12)
    assert list(b_pairs["decision"]) == list(s_pairs["decision"])
    b_ranks = base.ranks.to_dataframe()
    s_ranks = shifted.ranks.to_dataframe()
    for col in ("point_rank", "rank_low", "rank_high", "n_confirmed_better", "n_confirmed_worse"):
        assert list(b_ranks[col]) == list(s_ranks[col])


def test_higher_is_better_false_flips_decisions_and_ranks() -> None:
    mu = {"A": 0.50, "B": 0.47, "C": 0.20}
    sigma = np.array(
        [
            [0.0001, 0.0000155, 0.0],
            [0.0000155, 0.0001, 0.0],
            [0.0, 0.0, 0.0001],
        ]
    )
    res = rank_from_estimates(mu, sigma, higher_is_better=True)
    cost = rank_from_estimates(mu, sigma, higher_is_better=False)

    r = res.ranks.to_dataframe().set_index("label")
    c = cost.ranks.to_dataframe().set_index("label")
    assert r.loc["A", "point_rank"] == 1 and c.loc["A", "point_rank"] == 3
    assert r.loc["C", "point_rank"] == 3 and c.loc["C", "point_rank"] == 1
    assert int(r.loc["A", "rank_low"]) == 1 and int(c.loc["A", "rank_high"]) == 3

    pairs_res = res.pairwise.to_dataframe().set_index(["label_i", "label_j"])
    pairs_cost = cost.pairwise.to_dataframe().set_index(["label_i", "label_j"])
    flip = {"i_better": "j_better", "j_better": "i_better", "inconclusive": "inconclusive"}
    for key in pairs_res.index:
        assert pairs_cost.loc[key, "decision"] == flip[pairs_res.loc[key, "decision"]]
    assert (pairs_res["diff"] == pairs_cost["diff"]).all()


def test_rank_interval_contains_point_rank() -> None:
    """Internal consistency: 1 + B_i <= point_rank_i <= K - W_i."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        k = 4
        mu = rng.normal(size=k)
        a = rng.normal(size=(k, k)) * 0.05
        sigma = a @ a.T
        res = rank_from_estimates(dict(zip("ABCD", mu, strict=True)), sigma)
        ranks = res.ranks.to_dataframe()
        assert (ranks["rank_low"] <= ranks["point_rank"]).all()
        assert (ranks["point_rank"] <= ranks["rank_high"]).all()
        assert (ranks["rank_low"] >= 1).all()
        assert (ranks["rank_high"] <= k).all()


def test_rank_from_estimates_validates_inputs() -> None:
    sigma = np.eye(2) * 0.01
    with pytest.raises(CausalRankError, match="not supported"):
        rank_from_estimates({"A": 0.5, "B": 0.4}, sigma, method="max-t")
    with pytest.raises(CausalRankError, match="square"):
        rank_from_estimates({"A": 0.5, "B": 0.4}, np.ones((2, 3)))
    with pytest.raises(CausalRankError, match="positive semi-definite"):
        rank_from_estimates({"A": 0.5, "B": 0.4}, np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(CausalRankError, match="does not match"):
        rank_from_estimates({"A": 0.5, "B": 0.4}, np.eye(3) * 0.01)
    with pytest.raises(CausalRankError, match="finite"):
        rank_from_estimates({"A": 0.5, "B": np.nan}, sigma)
    with pytest.raises(CausalRankError, match="at least 2"):
        rank_from_estimates({"A": 0.5}, np.array([[0.01]]))


def test_mapping_with_relabelled_order_keeps_values_with_labels() -> None:
    """Regression (P1): ``labels`` reorders means *by label*, never inverts relations.

    estimates={'A': 0.5, 'B': 0.4} with labels=['B', 'A'] must still judge A
    better than B: the mean vector is looked up by label so the output order
    cannot silently re-attach values to the wrong arms.
    """
    sigma = np.array([[0.001, 0.0], [0.0, 0.001]])  # order-insensitive diagonal
    res = rank_from_estimates({"A": 0.5, "B": 0.4}, sigma, labels=["B", "A"])

    ranks = res.ranks.to_dataframe().set_index("label")
    assert ranks.loc["A", "mu"] == pytest.approx(0.5, rel=1e-12)
    assert ranks.loc["B", "mu"] == pytest.approx(0.4, rel=1e-12)
    assert (int(ranks.loc["A", "rank_low"]), int(ranks.loc["A", "rank_high"])) == (1, 1)
    assert (int(ranks.loc["B", "rank_low"]), int(ranks.loc["B", "rank_high"])) == (2, 2)

    pairs = res.pairwise.to_dataframe()
    # rows follow the requested order (B first), values follow their labels
    assert pairs.loc[0, "label_i"] == "B" and pairs.loc[0, "label_j"] == "A"
    assert pairs.loc[0, "diff"] == pytest.approx(-0.1, rel=1e-12)
    assert pairs.loc[0, "decision"] == "j_better"  # A (the j side) is confirmed better
    # identical relations regardless of the requested output order
    plain = rank_from_estimates({"A": 0.5, "B": 0.4}, sigma)
    plain_pairs = plain.pairwise.to_dataframe()
    assert plain_pairs.loc[0, "decision"] == "i_better"
    assert plain_pairs.loc[0, "diff"] == pytest.approx(0.1, rel=1e-12)


def test_duplicate_labels_are_rejected() -> None:
    """Regression (P2): duplicate labels must not silently mis-index the statistics."""
    sigma = np.eye(2) * 0.01
    with pytest.raises(CausalRankError, match="unique"):
        rank_from_estimates([0.5, 0.4], sigma, labels=["A", "A"])
    with pytest.raises(CausalRankError, match="unique"):
        pairwise_comparisons([0.5, 0.4], sigma, labels=["A", "A"])
    with pytest.raises(CausalRankError, match="unique"):
        rank_from_estimates({"A": 0.5, "B": 0.4}, sigma, labels=["A", "A"])
    with pytest.raises(CausalRankError, match="unique"):
        rank_confidence_sets(
            np.array([0.5, 0.4]),
            [
                {
                    "label_i": "A",
                    "label_j": "A",
                    "decision": "inconclusive",
                }
            ],
            ["A", "A"],
        )


def test_labels_must_match_mapping_keys() -> None:
    sigma = np.eye(2) * 0.01
    with pytest.raises(CausalRankError, match="permutation"):
        rank_from_estimates({"A": 0.5, "B": 0.4}, sigma, labels=["A", "C"])
    with pytest.raises(CausalRankError, match="permutation"):
        rank_from_estimates({"A": 0.5, "B": 0.4}, sigma, labels=["A"])


def test_rank_from_estimates_accepts_array_estimates_with_labels() -> None:
    sigma = np.eye(3) * 0.01
    res = rank_from_estimates(np.array([0.5, 0.4, 0.3]), sigma, labels=("A", "B", "C"))
    assert list(res.ranks.to_dataframe()["label"]) == ["A", "B", "C"]
    assert res.labels == ("A", "B", "C")
    assert res.to_dataframe().shape[0] == 3
    assert "tie" in res.summary().lower()


def test_ranking_metadata_is_traceable() -> None:
    """Spec acceptance: confidence level, formula and labels travel with results."""
    res = rank_from_estimates({"A": 0.5, "B": 0.4}, np.eye(2) * 0.01, confidence_level=0.9)
    d = res.to_dict()
    assert d["pairwise"]["confidence_level"] == 0.9
    assert d["pairwise"]["method"] == "bonferroni"
    assert "z_{1-alpha/(2M)}" in d["pairwise"]["formula"]
    assert d["ranks"]["labels"] == ["A", "B"]
    assert "[1 + B_i, K - W_i]" in d["ranks"]["formula"]
    assert "point rank" in d["ranks"]["tie_convention"]


# ---------------------------------------------------------------------------
# statistical red line on the DGP: no spurious confirmed ranking under ties
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_equal_effects_simulation_rarely_confirms_any_ranking() -> None:
    """Monte Carlo red line check on the ``equal_effects`` DGP.

    With truly equal potential-outcome means the simultaneous intervals must
    almost never confirm a direction. The thresholds are deliberately loose
    (the check is about catching a *broken* procedure, not about measuring the
    exact type-I error rate); the full calibration lives in
    ``scripts/run_rank_validation.py``.
    """
    from causalrank import CausalRank
    from causalrank.simulation import make_dataset

    from .conftest import FAST_GBM

    n_confirmed_reps = 0
    n_reps = 6
    for seed in range(101, 101 + n_reps):
        ds = make_dataset("equal_effects", n=800, random_state=seed)
        result = CausalRank(
            treatment="treatment",
            outcome="outcome",
            confounders=["x1", "x2", "x3", "x4"],
            gbm_params=dict(FAST_GBM),
            n_bootstrap=30,
            bootstrap_min_successes=10,
            random_state=seed,
        ).fit(ds.df)
        assert result.inference_available
        assert result.ranks is not None and result.pairwise is not None
        pairs = result.pairwise.to_dataframe()
        # the method never translates non-significance into an equality claim
        assert set(pairs["decision"]) <= set(DECISIONS)
        if bool(pairs["confirmed"].any()):
            n_confirmed_reps += 1
        else:
            ranks = result.ranks.to_dataframe()
            assert (ranks["rank_low"] == 1).all()
            assert (ranks["rank_high"] == 3).all()
    # under true equality a well-behaved procedure confirms in very few
    # replications; 3/6 would mean the simultaneous intervals are broken
    assert n_confirmed_reps <= n_reps // 2
