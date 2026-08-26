"""Canonical policy manipulation and minimum-change distance."""

from __future__ import annotations

from dataclasses import dataclass

from .models import PolicyClause, PolicyState, Relation, Requirement, canonical_clause_text


@dataclass(frozen=True)
class PolicyEdit:
    operation: str
    before_relation_id: str | None
    after_relation_id: str | None


def initial_policy(user_id: str, relations: tuple[Relation, ...], clause_limit: int) -> PolicyState:
    unique = {relation.relation_id: relation for relation in relations}
    clauses = tuple(
        PolicyClause(
            unique[key],
            source="support",
            text=canonical_clause_text(unique[key]),
        )
        for key in sorted(unique)[:clause_limit]
    )
    return PolicyState(user_id=user_id, clauses=clauses, version=0)


def apply_requirements(
    state: PolicyState,
    requirements: tuple[Requirement, ...],
    *,
    clause_limit: int,
    replace_conflicts: bool,
) -> PolicyState:
    clauses = list(state.clauses)
    by_relation = {clause.relation.relation_id: clause for clause in clauses}
    for requirement in requirements:
        relation = requirement.relation
        if relation.relation_id in by_relation:
            continue
        if replace_conflicts:
            clauses = [
                clause
                for clause in clauses
                if not (
                    clause.relation.attribute == relation.attribute
                    and clause.relation.preferred_value == relation.dispreferred_value
                    and clause.relation.dispreferred_value == relation.preferred_value
                    and clause.relation.context_attribute == relation.context_attribute
                    and clause.relation.context_value == relation.context_value
                )
            ]
        clauses.append(
            PolicyClause(
                relation=relation,
                source="requirement",
                text=canonical_clause_text(relation),
            )
        )
    deduplicated = {clause.relation.relation_id: clause for clause in clauses}
    ordered = tuple(deduplicated[key] for key in sorted(deduplicated))
    if len(ordered) > clause_limit:
        ordered = ordered[-clause_limit:]
    return PolicyState(user_id=state.user_id, clauses=ordered, version=state.version + 1)


def policy_edits(before: PolicyState, after: PolicyState) -> tuple[PolicyEdit, ...]:
    before_map = {clause.relation.relation_id: clause.relation for clause in before.clauses}
    after_map = {clause.relation.relation_id: clause.relation for clause in after.clauses}
    common = set(before_map) & set(after_map)
    before_unmatched = {key: value for key, value in before_map.items() if key not in common}
    after_unmatched = {key: value for key, value in after_map.items() if key not in common}
    edits: list[PolicyEdit] = []

    for before_id, left in tuple(before_unmatched.items()):
        replacement_id = next(
            (
                after_id
                for after_id, right in after_unmatched.items()
                if left.attribute == right.attribute
                and left.context_attribute == right.context_attribute
                and left.context_value == right.context_value
            ),
            None,
        )
        if replacement_id is not None:
            edits.append(PolicyEdit("replace", before_id, replacement_id))
            del before_unmatched[before_id]
            del after_unmatched[replacement_id]

    for before_id, left in tuple(before_unmatched.items()):
        scoped_id = next(
            (
                after_id
                for after_id, right in after_unmatched.items()
                if left.attribute == right.attribute
                and left.preferred_value == right.preferred_value
                and left.dispreferred_value == right.dispreferred_value
            ),
            None,
        )
        if scoped_id is not None:
            edits.append(PolicyEdit("scope", before_id, scoped_id))
            del before_unmatched[before_id]
            del after_unmatched[scoped_id]

    edits.extend(
        PolicyEdit("delete", relation_id, None) for relation_id in sorted(before_unmatched)
    )
    edits.extend(PolicyEdit("add", None, relation_id) for relation_id in sorted(after_unmatched))
    return tuple(edits)


def policy_distance(
    before: PolicyState,
    after: PolicyState,
    operation_costs: dict[str, float],
    operation_weight: float,
    token_weight: float,
) -> float:
    edits = policy_edits(before, after)
    operation_cost = sum(operation_costs[edit.operation] for edit in edits)
    before_tokens = _tokens(before)
    after_tokens = _tokens(after)
    token_distance = _levenshtein(before_tokens, after_tokens)
    return operation_weight * operation_cost / max(
        1, len(before.clauses)
    ) + token_weight * token_distance / max(1, len(before_tokens))


def _tokens(state: PolicyState) -> tuple[str, ...]:
    return tuple(state.policy_text.lower().split())


def _levenshtein(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_value != right_value),
                )
            )
        previous = current
    return previous[-1]
