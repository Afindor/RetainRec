"""Leakage-separated rolling transition construction."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .grammar import GrammarConfig, RelationGrammar, canonical_pair_cases
from .hashing import content_hash, file_hash
from .io import read_json, read_jsonl, write_json, write_jsonl
from .models import Evidence, Item, Relation, Requirement, TransitionManifest


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    seed: int
    validation_users: int
    initial_support: int
    trigger_support: int
    guard_pairs_per_relation: int
    audit_pairs_per_relation: int
    audit_horizon: int
    future_window: int
    minimum_future_evidence: int
    candidate_count: int
    pair_pass_threshold: float
    future_validity_threshold: float
    allowed_attributes: tuple[str, ...]
    context_attributes: tuple[str, ...]
    grammar_version: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DatasetConfig:
        converted = dict(value)
        converted["allowed_attributes"] = tuple(converted["allowed_attributes"])
        converted["context_attributes"] = tuple(converted["context_attributes"])
        return cls(**converted)

    @classmethod
    def load(cls, path: str | Path) -> DatasetConfig:
        return cls.from_dict(read_json(path))


@dataclass(frozen=True)
class PreferenceEvent:
    event_id: str
    user_id: str
    position: int
    preferred_item_id: str
    dispreferred_item_id: str
    confidence: float
    context: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PreferenceEvent:
        required = {
            "event_id",
            "user_id",
            "position",
            "preferred_item_id",
            "dispreferred_item_id",
            "confidence",
        }
        missing = required - set(value)
        if missing:
            raise ValueError(f"Missing event fields: {sorted(missing)}")
        event = cls(
            event_id=str(value["event_id"]),
            user_id=str(value["user_id"]),
            position=int(value["position"]),
            preferred_item_id=str(value["preferred_item_id"]),
            dispreferred_item_id=str(value["dispreferred_item_id"]),
            confidence=float(value["confidence"]),
            context=dict(value.get("context", {})),
        )
        if event.preferred_item_id == event.dispreferred_item_id:
            raise ValueError("A preference event requires two distinct items")
        if not 0 <= event.confidence <= 1:
            raise ValueError("Event confidence must lie in [0, 1]")
        return event


def load_normalized_data(
    data_dir: str | Path,
) -> tuple[dict[str, Item], tuple[PreferenceEvent, ...]]:
    root = Path(data_dir)
    items = {}
    for row in read_jsonl(root / "items.jsonl"):
        item = Item.from_dict(row)
        if item.item_id in items:
            raise ValueError(f"Duplicate item identifier: {item.item_id}")
        items[item.item_id] = item
    events = tuple(PreferenceEvent.from_dict(row) for row in read_jsonl(root / "events.jsonl"))
    event_ids = [event.event_id for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Event identifiers must be globally unique")
    for event in events:
        if event.preferred_item_id not in items or event.dispreferred_item_id not in items:
            raise ValueError(f"Event {event.event_id} refers to an unknown item")
    by_user: dict[str, list[int]] = defaultdict(list)
    for event in events:
        by_user[event.user_id].append(event.position)
    if any(positions != sorted(positions) for positions in by_user.values()):
        raise ValueError("Input events must be chronological within each user")
    return items, events


def build_manifests(
    config: DatasetConfig,
    items: dict[str, Item],
    events: Iterable[PreferenceEvent],
) -> tuple[TransitionManifest, ...]:
    grammar = RelationGrammar(
        GrammarConfig(config.allowed_attributes, config.context_attributes, config.grammar_version)
    )
    by_user: dict[str, list[PreferenceEvent]] = defaultdict(list)
    for event in events:
        by_user[event.user_id].append(event)
    for rows in by_user.values():
        rows.sort(key=lambda event: (event.position, event.event_id))
    user_ids = sorted(by_user)
    generator = random.Random(config.seed)
    generator.shuffle(user_ids)
    validation_count = min(config.validation_users, max(1, len(user_ids) - 1))
    validation_users = set(user_ids[:validation_count])
    manifests = []
    for user_id in sorted(by_user):
        split = "validation" if user_id in validation_users else "test"
        manifests.extend(_build_user_manifests(config, grammar, items, by_user[user_id], split))
    return tuple(sorted(manifests, key=lambda row: (row.user_id, row.position, row.transition_id)))


def write_manifest_package(
    output_dir: str | Path,
    config_path: str | Path,
    manifests: Iterable[TransitionManifest],
) -> None:
    root = Path(output_dir)
    rows = tuple(manifests)
    for row in rows:
        row.validate_boundaries()
    transition_path = root / "transitions.jsonl"
    write_jsonl(transition_path, (row.to_dict() for row in rows))
    write_json(
        root / "manifest.json",
        {
            "dataset_config_sha256": file_hash(config_path),
            "transition_count": len(rows),
            "validation_transitions": sum(row.split == "validation" for row in rows),
            "test_transitions": sum(row.split == "test" for row in rows),
            "transitions_sha256": file_hash(transition_path),
        },
    )


def normalized_data_summary(
    items: dict[str, Item], events: Iterable[PreferenceEvent]
) -> dict[str, int]:
    """Return deterministic counts after the full normalized-data validation."""
    rows = tuple(events)
    return {
        "items": len(items),
        "events": len(rows),
        "users": len({row.user_id for row in rows}),
        "minimum_position": min((row.position for row in rows), default=0),
        "maximum_position": max((row.position for row in rows), default=0),
    }


def read_manifests(path: str | Path) -> tuple[TransitionManifest, ...]:
    rows = tuple(TransitionManifest.from_dict(value) for value in read_jsonl(path))
    for row in rows:
        row.validate_boundaries()
    return rows


def _build_user_manifests(
    config: DatasetConfig,
    grammar: RelationGrammar,
    items: dict[str, Item],
    events: list[PreferenceEvent],
    split: str,
) -> list[TransitionManifest]:
    if len(events) <= config.initial_support + config.audit_horizon + config.future_window:
        return []
    grounded = [_ground_event(grammar, items, event) for event in events]
    initial_positive = {
        row.relation.relation_id
        for event_rows in grounded[: config.initial_support]
        for row in event_rows
        if row.polarity > 0
    }
    new_counts: Counter[str] = Counter()
    triggers: dict[int, list[Relation]] = defaultdict(list)
    for event_index in range(config.initial_support, len(events)):
        for row in grounded[event_index]:
            if row.polarity < 0 or row.relation.relation_id in initial_positive:
                continue
            new_counts[row.relation.relation_id] += 1
            if new_counts[row.relation.relation_id] == config.trigger_support:
                triggers[event_index].append(row.relation)

    result = []
    all_item_ids = tuple(sorted(items))
    for event_index, triggered_relations in sorted(triggers.items()):
        audit_start = event_index + 1
        audit_end = audit_start + config.audit_horizon
        future_end = audit_end + config.future_window
        if future_end > len(events):
            continue
        pre_rows = [row for batch in grounded[: event_index + 1] for row in batch]
        audit_rows = [row for batch in grounded[audit_start:audit_end] for row in batch]
        future_rows = [row for batch in grounded[audit_end:future_end] for row in batch]
        transition_id = content_hash(
            {
                "dataset": config.name,
                "user": events[event_index].user_id,
                "position": events[event_index].position,
                "grammar": config.grammar_version,
            }
        )[:24]
        candidates = tuple(
            sorted(
                all_item_ids,
                key=lambda item_id: content_hash({"transition": transition_id, "item": item_id}),
            )[: config.candidate_count]
        )
        positive_relation_ids = {row.relation.relation_id for row in pre_rows if row.polarity > 0}
        relation_map = {
            row.relation.relation_id: row.relation
            for row in pre_rows
            if row.relation.relation_id in positive_relation_ids
        }
        by_relation: dict[str, list[Evidence]] = defaultdict(list)
        for row in pre_rows:
            if row.relation.relation_id in relation_map:
                by_relation[row.relation.relation_id].append(row)
        selected_guard_ids: set[str] = set()
        for rows in by_relation.values():
            ordered = sorted(rows, key=lambda row: (row.position, row.evidence_id))
            guard_count = min(config.guard_pairs_per_relation, max(1, len(ordered) - 1))
            selected_guard_ids.update(row.evidence_id for row in ordered[-guard_count:])
        guard_rows = [row for row in pre_rows if row.evidence_id in selected_guard_ids]
        guard_item_ids = {
            item_id
            for row in guard_rows
            for item_id in (row.preferred_item_id, row.dispreferred_item_id)
        }
        support_rows = [
            row
            for row in pre_rows
            if row.evidence_id not in selected_guard_ids
            and row.preferred_item_id not in guard_item_ids
            and row.dispreferred_item_id not in guard_item_ids
        ]
        guard_cases_all = canonical_pair_cases(
            guard_rows,
            items,
            candidates,
            config.guard_pairs_per_relation,
            f"{transition_id}:guard",
        )
        audit_cases_all = canonical_pair_cases(
            audit_rows,
            items,
            candidates,
            config.audit_pairs_per_relation,
            f"{transition_id}:audit",
        )
        eligible_ids = {
            relation_id
            for relation_id in relation_map
            if guard_cases_all.get(relation_id) and audit_cases_all.get(relation_id)
        }
        required_ids = {relation.relation_id for relation in triggered_relations}
        if not required_ids.issubset(eligible_ids):
            continue
        relations = tuple(relation_map[key] for key in sorted(eligible_ids))
        future_by_relation = {
            relation_id: tuple(
                row for row in future_rows if row.relation.relation_id == relation_id
            )
            for relation_id in sorted(eligible_ids)
        }
        requirements = tuple(
            Requirement(
                requirement_id=f"{transition_id}:{relation.relation_id}",
                relation=relation,
                guard_cases=guard_cases_all[relation.relation_id],
                audit_cases=audit_cases_all[relation.relation_id],
            )
            for relation in sorted(triggered_relations, key=lambda row: row.relation_id)
        )
        manifest = TransitionManifest(
            transition_id=transition_id,
            user_id=events[event_index].user_id,
            position=events[event_index].position,
            split=split,
            support_evidence=tuple(
                row for row in support_rows if row.relation.relation_id in eligible_ids
            ),
            relations=relations,
            guard_cases={key: guard_cases_all[key] for key in sorted(eligible_ids)},
            audit_cases={key: audit_cases_all[key] for key in sorted(eligible_ids)},
            future_evidence=future_by_relation,
            requirements=requirements,
            context=dict(events[event_index].context),
            grammar_version=config.grammar_version,
        )
        try:
            manifest.validate_boundaries()
        except ValueError:
            continue
        result.append(manifest)
    return result


def _ground_event(
    grammar: RelationGrammar, items: dict[str, Item], event: PreferenceEvent
) -> tuple[Evidence, ...]:
    return grammar.ground_event(
        user_id=event.user_id,
        event_id=event.event_id,
        position=event.position,
        preferred=items[event.preferred_item_id],
        dispreferred=items[event.dispreferred_item_id],
        context=event.context,
        confidence=event.confidence,
    )
