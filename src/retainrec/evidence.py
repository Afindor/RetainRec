"""Evidence aggregation and post-transition validity labels."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from .models import Evidence


@dataclass(frozen=True)
class EvidenceSummary:
    supporting: float
    contradicting: float
    distance_since_support: int
    count: int


def aggregate_evidence(
    evidence: Iterable[Evidence], current_position: int, decay: float
) -> EvidenceSummary:
    if decay < 0:
        raise ValueError("Decay must be nonnegative")
    rows = tuple(evidence)
    supporting = 0.0
    contradicting = 0.0
    support_positions: list[int] = []
    for row in rows:
        if row.position > current_position:
            raise ValueError("Selector evidence cannot occur after the current transition")
        weight = row.confidence * math.exp(-decay * (current_position - row.position))
        if row.polarity > 0:
            supporting += weight
            support_positions.append(row.position)
        else:
            contradicting += weight
    latest_support = max(support_positions) if support_positions else 0
    return EvidenceSummary(
        supporting=supporting,
        contradicting=contradicting,
        distance_since_support=max(0, current_position - latest_support),
        count=len(rows),
    )


def future_validity_label(
    evidence: Iterable[Evidence], threshold: float, minimum_count: int
) -> int | None:
    rows = tuple(evidence)
    if len(rows) < minimum_count:
        return None
    supporting = sum(row.confidence for row in rows if row.polarity > 0)
    contradicting = sum(row.confidence for row in rows if row.polarity < 0)
    denominator = supporting + contradicting
    if denominator <= 0:
        return None
    return int(supporting / denominator >= threshold)
