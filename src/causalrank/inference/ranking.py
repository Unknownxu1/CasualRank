"""Rank point estimates and conservative rank confidence sets (Phase 3).

Building blocks (PROJECT_SPEC.md 3.6):

* ``B_i`` - number of arms **confirmed better** than arm ``i`` by the
  Bonferroni simultaneous pairwise intervals;
* ``W_i`` - number of arms **confirmed worse** than arm ``i``;
* conservative rank confidence interval::

      [L_i, U_i] = [1 + B_i, K - W_i].

Rank 1 is the best arm under ``higher_is_better`` (rank 1 = highest ``mu`` by
default; flip with ``higher_is_better=False`` for lower-is-better outcomes).

Tie convention (spec 3.6 red line): exact point-estimate ties share the same
(best) point rank and are flagged — the kernel never invents a strict order
from label order. When true effects tie, the strict total order is not uniquely
defined; the interval ``[L_i, U_i]`` widens instead of picking a winner. An
inconclusive pairwise interval is never reported as "equal".

The method statement is deliberately narrow: coverage relies on the joint
bootstrap covariance and the asymptotic normal approximation of the estimator.
Validate with the simulation protocol (``run_rank_benchmark``), never present
it as a universal guarantee. In particular the bootstrap frequency with which
an arm lands on point-rank 1 is NOT a probability of being best.

Low-level entry point (spec 2.2)::

    from causalrank.inference import rank_from_estimates
    ranking = rank_from_estimates(
        estimates={"A": 0.25, "B": 0.24, "C": 0.17},
        covariance=sigma,
        confidence_level=0.95,
        method="bonferroni",
    )
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import CausalRankError
from ..results import PairwiseTable, RankInference, RankTable
from .bootstrap import validate_covariance
from .pairwise import check_labels, pairwise_comparisons

#: Only the Bonferroni baseline is implemented in v0.1; max-T / stepdown
#: procedures are research extensions and are rejected explicitly.
SUPPORTED_RANK_METHODS: tuple[str, ...] = ("bonferroni",)


def rank_confidence_sets(
    mu: np.ndarray,
    pairwise_rows: list[dict[str, Any]],
    labels: Sequence[Any],
    higher_is_better: bool = True,
) -> list[dict[str, Any]]:
    """Conservative rank records ``[1 + B_i, K - W_i]`` from confirmed pairs.

    Point ranks use competition ranking on the "better" score: each arm gets
    ``1 + #{arms strictly better}``, so exact ties share the same rank and the
    result never depends on label order.
    """
    mu = np.asarray(mu, dtype=float)
    lab = check_labels(labels)
    k = len(lab)
    if mu.size != k:
        raise CausalRankError("mu and labels must have the same length.")
    if not isinstance(higher_is_better, bool):
        raise CausalRankError("higher_is_better must be a bool.")

    score = mu if higher_is_better else -mu
    point_rank = np.array([1 + int(np.sum(score > score[i])) for i in range(k)], dtype=int)
    point_tie = np.array([int(np.sum(score == score[i])) > 1 for i in range(k)], dtype=bool)

    confirmed_better = np.zeros(k, dtype=int)  # B_i: arms confirmed better than i
    confirmed_worse = np.zeros(k, dtype=int)  # W_i: arms confirmed worse than i
    index = {lab[i]: i for i in range(k)}
    for row in pairwise_rows:
        try:
            i = index[row["label_i"]]
            j = index[row["label_j"]]
        except KeyError as exc:
            raise CausalRankError(
                f"pairwise row references unknown label {exc.args[0]!r}; "
                f"known labels are {list(lab)}."
            ) from exc
        decision = row["decision"]
        if decision == "i_better":
            confirmed_worse[i] += 1
            confirmed_better[j] += 1
        elif decision == "j_better":
            confirmed_better[i] += 1
            confirmed_worse[j] += 1

    records: list[dict[str, Any]] = []
    for i in range(k):
        rank_low = 1 + int(confirmed_better[i])
        rank_high = k - int(confirmed_worse[i])
        records.append(
            {
                "label": lab[i],
                "mu": float(mu[i]),
                "point_rank": int(point_rank[i]),
                "rank_low": int(rank_low),
                "rank_high": int(rank_high),
                "n_confirmed_better": int(confirmed_better[i]),
                "n_confirmed_worse": int(confirmed_worse[i]),
                "point_tie": bool(point_tie[i]),
            }
        )
    return records


def rank_from_estimates(
    estimates: Mapping[Any, float] | Sequence[float] | np.ndarray,
    covariance: np.ndarray,
    confidence_level: float = 0.95,
    method: str = "bonferroni",
    higher_is_better: bool = True,
    labels: Sequence[Any] | None = None,
) -> RankInference:
    """Rank inference from point estimates and their joint covariance.

    Parameters
    ----------
    estimates:
        Mapping ``label -> mu`` or a plain sequence of ``K`` means.
    covariance:
        ``K x K`` joint covariance of the estimates — typically the
        re-fitting joint bootstrap covariance. Validated (symmetric, PSD);
        never silently repaired. Its rows/columns follow the canonical order:
        the ``labels`` sequence when given, else the estimates iteration
        order.
    confidence_level:
        Simultaneous coverage level of the pairwise intervals.
    method:
        Multiplicity adjustment; only ``"bonferroni"`` is available in v0.1.
    higher_is_better:
        ``True`` (default) means rank 1 = highest ``mu``; ``False`` means
        rank 1 = lowest ``mu``.
    labels:
        Optional canonical order. For a mapping ``estimates`` this must be a
        permutation of its keys; the means and the covariance are re-indexed
        to follow it. Duplicate labels are rejected.

    Returns
    -------
    RankInference
        ``.pairwise`` (:class:`~causalrank.results.PairwiseTable`) and
        ``.ranks`` (:class:`~causalrank.results.RankTable`); ``to_dataframe()``
        returns the ranks table.
    """
    if method not in SUPPORTED_RANK_METHODS:
        raise CausalRankError(
            f"method={method!r} is not supported in v0.1. Supported: {SUPPORTED_RANK_METHODS}. "
            "max-T / stepdown procedures are research extensions (PROJECT_SPEC.md section 6)."
        )
    sigma = validate_covariance(np.asarray(covariance, dtype=float))

    rows, info = pairwise_comparisons(
        estimates,
        sigma,
        confidence_level=confidence_level,
        labels=labels,
        higher_is_better=higher_is_better,
    )
    lab = tuple(info["labels"])
    if isinstance(estimates, Mapping):
        mu = np.array([float(estimates[key]) for key in lab], dtype=float)
    else:
        mu = np.asarray(estimates, dtype=float).reshape(-1)

    rank_rows = rank_confidence_sets(mu, rows, lab, higher_is_better=higher_is_better)

    pairwise_df = pd.DataFrame(rows)
    ranks_df = pd.DataFrame(rank_rows)
    pairwise = PairwiseTable(
        labels=lab,
        table=pairwise_df,
        confidence_level=float(info["confidence_level"]),
        n_pairs=int(info["n_pairs"]),
        z_crit=float(info["z_crit"]),
        higher_is_better=bool(higher_is_better),
        method=str(method),
    )
    ranks = RankTable(
        labels=lab,
        table=ranks_df,
        confidence_level=float(info["confidence_level"]),
        higher_is_better=bool(higher_is_better),
        method=str(method),
    )
    return RankInference(pairwise=pairwise, ranks=ranks)
