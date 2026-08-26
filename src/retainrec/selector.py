"""Validation-fitted validity estimation and cost-sensitive selection."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Callable

from .evidence import EvidenceSummary
from .models import Relation


@dataclass(frozen=True)
class SelectorSample:
    summary: EvidenceSummary
    label: int


@dataclass(frozen=True)
class ValidityEstimator:
    alpha: float
    beta: float
    gamma: float
    platt_scale: float
    platt_bias: float

    def raw_score(self, summary: EvidenceSummary) -> float:
        return (
            self.alpha * summary.supporting
            - self.beta * summary.contradicting
            - self.gamma * summary.distance_since_support
        )

    def predict(self, summary: EvidenceSummary) -> float:
        return _sigmoid(self.platt_scale * self.raw_score(summary) + self.platt_bias)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, float]) -> ValidityEstimator:
        return cls(**{key: float(item) for key, item in value.items()})

    @classmethod
    def fit(
        cls,
        samples: Iterable[SelectorSample],
        *,
        learning_rate: float = 0.02,
        iterations: int = 1200,
        regularization: float = 1e-3,
    ) -> ValidityEstimator:
        rows = tuple(samples)
        if not rows or {row.label for row in rows} != {0, 1}:
            raise ValueError("Selector fitting requires both validation labels")
        alpha = beta = gamma = 0.1
        for _ in range(iterations):
            gradients = [0.0, 0.0, 0.0]
            for row in rows:
                features = (
                    row.summary.supporting,
                    -row.summary.contradicting,
                    -float(row.summary.distance_since_support),
                )
                prediction = _sigmoid(
                    alpha * features[0] + beta * features[1] + gamma * features[2]
                )
                error = prediction - row.label
                for index, feature in enumerate(features):
                    gradients[index] += error * feature
            scale = 1.0 / len(rows)
            alpha = max(
                0.0,
                alpha - learning_rate * (gradients[0] * scale + regularization * alpha),
            )
            beta = max(0.0, beta - learning_rate * (gradients[1] * scale + regularization * beta))
            gamma = max(
                0.0,
                gamma - learning_rate * (gradients[2] * scale + regularization * gamma),
            )

        raw_scores = [
            alpha * row.summary.supporting
            - beta * row.summary.contradicting
            - gamma * row.summary.distance_since_support
            for row in rows
        ]
        scale_value, bias = _fit_platt(raw_scores, [row.label for row in rows])
        return cls(alpha, beta, gamma, scale_value, bias)


@dataclass(frozen=True)
class SelectionDecision:
    relation: Relation
    probability: float
    net_value: float
    selected: bool
    reason: str


def select_relations(
    relations: Iterable[Relation],
    summaries: dict[str, EvidenceSummary],
    estimator: ValidityEstimator,
    context: dict[str, object],
    guard_pass: Callable[[Relation], bool],
    stale_cost: float,
    miss_cost: float = 1.0,
) -> tuple[SelectionDecision, ...]:
    if stale_cost <= 0 or miss_cost <= 0:
        raise ValueError("Selection costs must be positive")
    decisions = []
    for relation in relations:
        summary = summaries[relation.relation_id]
        probability = estimator.predict(summary)
        net_value = probability * miss_cost - (1.0 - probability) * stale_cost
        if net_value <= 0:
            reason = "nonpositive_value"
            selected = False
        elif not relation.applies(context):
            reason = "inactive_context"
            selected = False
        elif not guard_pass(relation):
            reason = "preupdate_guard_failure"
            selected = False
        else:
            reason = "selected"
            selected = True
        decisions.append(SelectionDecision(relation, probability, net_value, selected, reason))
    return tuple(decisions)


def choose_cost_ratio(
    probabilities: Iterable[float], labels: Iterable[int], grid: Iterable[float]
) -> float:
    probability_values = tuple(probabilities)
    label_values = tuple(labels)
    if len(probability_values) != len(label_values):
        raise ValueError("Probabilities and labels must have the same length")
    pairs = tuple(zip(probability_values, label_values))
    candidates = tuple(float(value) for value in grid)
    if not pairs or not candidates:
        raise ValueError("Cost selection requires validation predictions and a nonempty grid")

    def validation_loss(stale_cost: float) -> tuple[float, float]:
        threshold = stale_cost / (1.0 + stale_cost)
        total = 0.0
        for probability, label in pairs:
            selected = probability > threshold
            if label == 1 and not selected:
                total += 1.0
            elif label == 0 and selected:
                total += stale_cost
        return total / len(pairs), stale_cost

    return min(candidates, key=validation_loss)


def _fit_platt(scores: list[float], labels: list[int]) -> tuple[float, float]:
    scale = 1.0
    bias = 0.0
    for _ in range(800):
        grad_scale = 0.0
        grad_bias = 0.0
        for score, label in zip(scores, labels):
            error = _sigmoid(scale * score + bias) - label
            grad_scale += error * score
            grad_bias += error
        factor = 0.02 / len(scores)
        scale -= factor * grad_scale
        bias -= factor * grad_bias
    return scale, bias


def _sigmoid(value: float) -> float:
    if value >= 0:
        term = math.exp(-value)
        return 1.0 / (1.0 + term)
    term = math.exp(value)
    return term / (1.0 + term)
