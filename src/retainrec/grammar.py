"""Frozen dataset-native relation grammar and canonical pairing."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .hashing import content_hash
from .models import Evidence, Item, PairCase, Relation


@dataclass(frozen=True)
class GrammarConfig:
    allowed_attributes: tuple[str, ...]
    context_attributes: tuple[str, ...]
    grammar_version: str = "grammar-v1"


class RelationGrammar:
    """Enumerate only the two relation families declared in grammar-v1."""

    def __init__(self, config: GrammarConfig) -> None:
        if config.grammar_version != "grammar-v1":
            raise ValueError(f"Unsupported grammar version: {config.grammar_version}")
        self.config = config

    def enumerate(
        self,
        preferred: Item,
        dispreferred: Item,
        context: dict[str, Any],
    ) -> tuple[Relation, ...]:
        relations: dict[str, Relation] = {}
        for attribute in self.config.allowed_attributes:
            left_values = _values(preferred.attributes.get(attribute))
            right_values = _values(dispreferred.attributes.get(attribute))
            for left in sorted(left_values - right_values):
                for right in sorted(right_values - left_values):
                    base = Relation(
                        attribute=attribute,
                        preferred_value=left,
                        dispreferred_value=right,
                        grammar_version=self.config.grammar_version,
                    )
                    relations[base.relation_id] = base
                    for context_attribute in self.config.context_attributes:
                        if context_attribute not in context:
                            continue
                        contextual = Relation(
                            attribute=attribute,
                            preferred_value=left,
                            dispreferred_value=right,
                            context_attribute=context_attribute,
                            context_value=str(context[context_attribute]),
                            grammar_version=self.config.grammar_version,
                        )
                        relations[contextual.relation_id] = contextual
        return tuple(relations[key] for key in sorted(relations))

    def ground_event(
        self,
        *,
        user_id: str,
        event_id: str,
        position: int,
        preferred: Item,
        dispreferred: Item,
        context: dict[str, Any],
        confidence: float,
    ) -> tuple[Evidence, ...]:
        rows = []
        for relation in self.enumerate(preferred, dispreferred, context):
            rows.append(
                Evidence(
                    evidence_id=f"{event_id}:{relation.relation_id}",
                    user_id=user_id,
                    position=position,
                    relation=relation,
                    polarity=1,
                    confidence=confidence,
                    preferred_item_id=preferred.item_id,
                    dispreferred_item_id=dispreferred.item_id,
                    context=context,
                )
            )
            inverse = Relation(
                attribute=relation.attribute,
                preferred_value=relation.dispreferred_value,
                dispreferred_value=relation.preferred_value,
                context_attribute=relation.context_attribute,
                context_value=relation.context_value,
                grammar_version=relation.grammar_version,
            )
            rows.append(
                Evidence(
                    evidence_id=f"{event_id}:{inverse.relation_id}:contradiction",
                    user_id=user_id,
                    position=position,
                    relation=inverse,
                    polarity=-1,
                    confidence=confidence,
                    preferred_item_id=preferred.item_id,
                    dispreferred_item_id=dispreferred.item_id,
                    context=context,
                )
            )
        return tuple(rows)


def canonical_pair_cases(
    evidence: Iterable[Evidence],
    items: dict[str, Item],
    candidate_item_ids: Iterable[str],
    limit: int,
    salt: str,
) -> dict[str, tuple[PairCase, ...]]:
    """Pair without replacement under a canonical, metadata-independent order."""
    grouped: dict[str, list[Evidence]] = defaultdict(list)
    for row in evidence:
        grouped[row.relation.relation_id].append(row)
    candidates = tuple(dict.fromkeys(str(item) for item in candidate_item_ids))
    result: dict[str, tuple[PairCase, ...]] = {}
    for relation_id, rows in grouped.items():
        ordered = sorted(
            rows,
            key=lambda row: content_hash(
                {
                    "salt": salt,
                    "relation_id": relation_id,
                    "preferred": row.preferred_item_id,
                    "dispreferred": row.dispreferred_item_id,
                }
            ),
        )
        used_items: set[str] = set()
        cases: list[PairCase] = []
        for row in ordered:
            preferred_item_id = (
                row.preferred_item_id if row.polarity > 0 else row.dispreferred_item_id
            )
            dispreferred_item_id = (
                row.dispreferred_item_id if row.polarity > 0 else row.preferred_item_id
            )
            pair_items = {preferred_item_id, dispreferred_item_id}
            if used_items & pair_items:
                continue
            if not pair_items.issubset(items):
                continue
            case_id = content_hash(
                {
                    "salt": salt,
                    "evidence_id": row.evidence_id,
                    "relation_id": relation_id,
                }
            )[:24]
            case_candidates = tuple(dict.fromkeys((*candidates, *pair_items)))
            cases.append(
                PairCase(
                    case_id=case_id,
                    relation=row.relation,
                    preferred_item=items[preferred_item_id],
                    dispreferred_item=items[dispreferred_item_id],
                    context=row.context,
                    candidate_item_ids=case_candidates,
                )
            )
            used_items.update(pair_items)
            if len(cases) >= limit:
                break
        result[relation_id] = tuple(cases)
    return result


def _values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    return {str(value)}
