"""Paired topology-level comparisons of spec C Section 12."""

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260911
MAX_EXACT_TOPOLOGIES = 20
TIE_TOLERANCE = 1e-12
STATISTICS_COLUMNS = (
    "set_id", "comparison", "mean_difference", "ci_low", "ci_high", "rank_biserial", "p_value", "p_holm", "topology_count",
)


@dataclass(frozen=True)
class Comparison:
    set_id: str
    first: str
    second: str

    @property
    def label(self) -> str:
        return f"{self.first} - {self.second}"


PRIMARY_COMPARISONS = tuple(
    Comparison(set_id, first, second)
    for set_id in ("v0", "v0-bh50")
    for first, second in (("P", "B2"), ("B2", "B1"), ("B2", "B3"), ("P", "B1"))
)


def topology_means(rows: Sequence[Mapping[str, str]], set_id: str, method: str) -> dict[str, float]:
    """Mean timely ratio per scenario over its realizations, then averaged over the scenarios of each topology."""
    per_scenario: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["set_id"] == set_id and row["method"] == method:
            per_scenario[(row["topology_id"], row["scenario_id"])].append(float(row["timely_ratio"]))
    per_topology: dict[str, list[float]] = defaultdict(list)
    for (topology, _), values in per_scenario.items():
        per_topology[topology].append(sum(values) / len(values))
    return {topology: sum(values) / len(values) for topology, values in sorted(per_topology.items())}


def sign_flip_p_value(differences: Sequence[float]) -> float:
    """Exact two-sided sign-flip test: share of all 2^n sign assignments whose |mean| reaches the observed |mean|."""
    values = np.asarray(differences, dtype=float)
    count = len(values)
    if not 1 <= count <= MAX_EXACT_TOPOLOGIES:
        raise ValueError(f"the exact test needs between 1 and {MAX_EXACT_TOPOLOGIES} paired values, got {count}")
    signs = ((np.arange(2**count)[:, None] >> np.arange(count)) & 1) * 2 - 1
    flipped = np.abs(signs @ values) / count
    return float(np.mean(flipped >= abs(values.sum()) / count - TIE_TOLERANCE))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    count = len(p_values)
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(sorted(range(count), key=lambda item: p_values[item])):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def rank_biserial(differences: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation (W+ - W-)/(W+ + W-) without zero differences; nan when all are zero."""
    values = np.asarray([value for value in differences if value != 0.0], dtype=float)
    if len(values) == 0:
        return math.nan
    ranks = rankdata(np.abs(values))
    positive, negative = ranks[values > 0].sum(), ranks[values < 0].sum()
    return float((positive - negative) / (positive + negative))


def bootstrap_mean_ci(differences: Sequence[float], resamples: int, seed: int, confidence: float = 0.95) -> tuple[float, float]:
    """Percentile interval of the mean over topologies resampled with replacement."""
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def compare_methods(
    rows: Sequence[Mapping[str, str]], comparisons: Sequence[Comparison], resamples: int, seed: int
) -> list[dict[str, object]]:
    """One row per comparison with Holm adjustment across all given comparisons."""
    results = []
    for comparison in comparisons:
        first = topology_means(rows, comparison.set_id, comparison.first)
        second = topology_means(rows, comparison.set_id, comparison.second)
        if not first or first.keys() != second.keys():
            raise ValueError(f"{comparison.set_id} {comparison.label}: methods must cover the same non-empty topologies")
        differences = np.array([first[topology] - second[topology] for topology in first])
        low, high = bootstrap_mean_ci(differences, resamples, seed)
        results.append(
            {
                "set_id": comparison.set_id,
                "comparison": comparison.label,
                "mean_difference": float(differences.mean()),
                "ci_low": low,
                "ci_high": high,
                "rank_biserial": rank_biserial(differences),
                "p_value": sign_flip_p_value(differences),
                "p_holm": 0.0,
                "topology_count": len(differences),
            }
        )
    for row, adjusted in zip(results, holm_adjust([row["p_value"] for row in results])):
        row["p_holm"] = adjusted
    return results
