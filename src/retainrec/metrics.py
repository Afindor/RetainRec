"""Cross-version audit metrics and user-macro aggregation."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from statistics import mean, pstdev

from .models import PairCase, PolicyState, Requirement
from .policy import policy_edits
from .ranking import Ranker, pair_pass


@dataclass(frozen=True)
class RelationOutcome:
    relation_id: str
    pre_passes: int
    post_passes: int
    case_count: int
    regression: float | None


@dataclass(frozen=True)
class TransitionMetrics:
    acquisition_rate: float
    update_regression_rate: float | None
    obligation_generalization_rate: float | None
    policy_edit_ratio: float
    relation_outcomes: tuple[RelationOutcome, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_transition(
    *,
    before: PolicyState,
    after: PolicyState,
    audit_cases: dict[str, tuple[PairCase, ...]],
    requirements: tuple[Requirement, ...],
    selected_probabilities: dict[str, float],
    ranker: Ranker,
) -> TransitionMetrics:
    outcomes = []
    eligible_regressions = []
    obligation_score = 0.0
    obligation_weight = 0.0
    for relation_id, cases in sorted(audit_cases.items()):
        pre = [pair_pass(ranker, before, case) for case in cases]
        post = [pair_pass(ranker, after, case) for case in cases]
        pre_passes = sum(pre)
        post_passes = sum(post)
        regression = None
        if pre_passes > 0:
            regression = sum(left * (1 - right) for left, right in zip(pre, post)) / pre_passes
            eligible_regressions.append(regression)
        if relation_id in selected_probabilities and cases:
            weight = selected_probabilities[relation_id]
            obligation_score += weight * post_passes / len(cases)
            obligation_weight += weight
        outcomes.append(
            RelationOutcome(relation_id, pre_passes, post_passes, len(cases), regression)
        )
    requirement_cases = [case for requirement in requirements for case in requirement.audit_cases]
    acquisition = (
        sum(pair_pass(ranker, after, case) for case in requirement_cases) / len(requirement_cases)
        if requirement_cases
        else 0.0
    )
    return TransitionMetrics(
        acquisition_rate=acquisition,
        update_regression_rate=(mean(eligible_regressions) if eligible_regressions else None),
        obligation_generalization_rate=(
            obligation_score / obligation_weight if obligation_weight > 0 else None
        ),
        policy_edit_ratio=len(policy_edits(before, after)) / max(1, len(before.clauses)),
        relation_outcomes=tuple(outcomes),
    )


def common_pass_urr(
    left: TransitionMetrics, right: TransitionMetrics
) -> tuple[float | None, float | None, float]:
    left_rows = {row.relation_id: row for row in left.relation_outcomes}
    right_rows = {row.relation_id: row for row in right.relation_outcomes}
    shared = set(left_rows) & set(right_rows)
    union_eligible = {
        key for key in shared if left_rows[key].pre_passes > 0 or right_rows[key].pre_passes > 0
    }
    common = {
        key for key in shared if left_rows[key].pre_passes > 0 and right_rows[key].pre_passes > 0
    }
    coverage = len(common) / len(union_eligible) if union_eligible else 0.0
    left_values = [left_rows[key].regression for key in common]
    right_values = [right_rows[key].regression for key in common]
    return (
        mean(value for value in left_values if value is not None) if common else None,
        mean(value for value in right_values if value is not None) if common else None,
        coverage,
    )


def ranking_metrics(
    ranking: tuple[str, ...], relevant: set[str], cutoffs: Iterable[int]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for cutoff in cutoffs:
        top = ranking[:cutoff]
        hits = [int(item in relevant) for item in top]
        result[f"HR@{cutoff}"] = float(any(hits))
        dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
        ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(relevant), cutoff)))
        result[f"NDCG@{cutoff}"] = dcg / ideal if ideal else 0.0
    return result


def user_macro_summary(rows: Iterable[dict[str, object]], metric: str) -> dict[str, float]:
    by_user_seed: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        by_user_seed[(str(row["user_id"]), int(row["seed"]))].append(float(value))
    by_user: dict[str, list[float]] = defaultdict(list)
    for (user_id, _), values in by_user_seed.items():
        by_user[user_id].append(mean(values))
    user_values = [mean(seed_values) for seed_values in by_user.values()]
    if not user_values:
        raise ValueError(f"No defined values for metric {metric}")
    return {
        "mean": mean(user_values),
        "std": pstdev(user_values) if len(user_values) > 1 else 0.0,
        "users": float(len(user_values)),
    }


def clustered_bootstrap_difference(
    left: dict[str, float],
    right: dict[str, float],
    *,
    replicates: int = 5000,
    seed: int = 2024001,
) -> tuple[float, float, float]:
    common = sorted(set(left) & set(right))
    if not common:
        raise ValueError("Paired bootstrap requires common users")
    differences = [left[user] - right[user] for user in common]
    generator = random.Random(seed)
    samples = []
    for _ in range(replicates):
        samples.append(mean(generator.choice(differences) for _ in differences))
    samples.sort()
    lower = samples[int(0.025 * (replicates - 1))]
    upper = samples[int(0.975 * (replicates - 1))]
    return mean(differences), lower, upper


def auc_over_trajectory(values: Iterable[float]) -> float:
    points = tuple(values)
    if not points:
        raise ValueError("AUC requires at least one point")
    if len(points) == 1:
        return points[0]
    return sum((left + right) / 2 for left, right in zip(points, points[1:])) / (len(points) - 1)
