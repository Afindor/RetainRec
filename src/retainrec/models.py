"""Typed records shared by construction, updating, and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .hashing import content_hash


@dataclass(frozen=True)
class Item:
    item_id: str
    attributes: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Item:
        return cls(item_id=str(value["item_id"]), attributes=dict(value["attributes"]))


@dataclass(frozen=True)
class Relation:
    attribute: str
    preferred_value: str
    dispreferred_value: str
    context_attribute: str | None = None
    context_value: str | None = None
    grammar_version: str = "grammar-v1"

    @property
    def relation_id(self) -> str:
        return content_hash(asdict(self))[:20]

    @property
    def family(self) -> str:
        return "contextual_override" if self.context_attribute else "pairwise_preference"

    def applies(self, context: dict[str, Any]) -> bool:
        if self.context_attribute is None:
            return True
        return str(context.get(self.context_attribute)) == self.context_value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Relation:
        return cls(
            attribute=str(value["attribute"]),
            preferred_value=str(value["preferred_value"]),
            dispreferred_value=str(value["dispreferred_value"]),
            context_attribute=value.get("context_attribute"),
            context_value=(
                str(value["context_value"]) if value.get("context_value") is not None else None
            ),
            grammar_version=str(value.get("grammar_version", "grammar-v1")),
        )


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    user_id: str
    position: int
    relation: Relation
    polarity: int
    confidence: float
    preferred_item_id: str
    dispreferred_item_id: str
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.polarity not in (-1, 1):
            raise ValueError("Evidence polarity must be -1 or 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Evidence confidence must lie in [0, 1]")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Evidence:
        return cls(
            evidence_id=str(value["evidence_id"]),
            user_id=str(value["user_id"]),
            position=int(value["position"]),
            relation=Relation.from_dict(dict(value["relation"])),
            polarity=int(value["polarity"]),
            confidence=float(value["confidence"]),
            preferred_item_id=str(value["preferred_item_id"]),
            dispreferred_item_id=str(value["dispreferred_item_id"]),
            context=dict(value.get("context", {})),
        )


@dataclass(frozen=True)
class PairCase:
    case_id: str
    relation: Relation
    preferred_item: Item
    dispreferred_item: Item
    context: dict[str, Any]
    candidate_item_ids: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PairCase:
        return cls(
            case_id=str(value["case_id"]),
            relation=Relation.from_dict(dict(value["relation"])),
            preferred_item=Item.from_dict(dict(value["preferred_item"])),
            dispreferred_item=Item.from_dict(dict(value["dispreferred_item"])),
            context=dict(value.get("context", {})),
            candidate_item_ids=tuple(str(item) for item in value["candidate_item_ids"]),
        )


@dataclass(frozen=True)
class Requirement:
    requirement_id: str
    relation: Relation
    guard_cases: tuple[PairCase, ...]
    audit_cases: tuple[PairCase, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Requirement:
        return cls(
            requirement_id=str(value["requirement_id"]),
            relation=Relation.from_dict(dict(value["relation"])),
            guard_cases=tuple(PairCase.from_dict(case) for case in value["guard_cases"]),
            audit_cases=tuple(PairCase.from_dict(case) for case in value["audit_cases"]),
        )


@dataclass(frozen=True)
class TransitionManifest:
    transition_id: str
    user_id: str
    position: int
    split: str
    support_evidence: tuple[Evidence, ...]
    relations: tuple[Relation, ...]
    guard_cases: dict[str, tuple[PairCase, ...]]
    audit_cases: dict[str, tuple[PairCase, ...]]
    future_evidence: dict[str, tuple[Evidence, ...]]
    requirements: tuple[Requirement, ...]
    context: dict[str, Any]
    grammar_version: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TransitionManifest:
        return cls(
            transition_id=str(value["transition_id"]),
            user_id=str(value["user_id"]),
            position=int(value["position"]),
            split=str(value["split"]),
            support_evidence=tuple(Evidence.from_dict(row) for row in value["support_evidence"]),
            relations=tuple(Relation.from_dict(row) for row in value["relations"]),
            guard_cases={
                key: tuple(PairCase.from_dict(case) for case in cases)
                for key, cases in value["guard_cases"].items()
            },
            audit_cases={
                key: tuple(PairCase.from_dict(case) for case in cases)
                for key, cases in value["audit_cases"].items()
            },
            future_evidence={
                key: tuple(Evidence.from_dict(row) for row in rows)
                for key, rows in value["future_evidence"].items()
            },
            requirements=tuple(Requirement.from_dict(row) for row in value["requirements"]),
            context=dict(value.get("context", {})),
            grammar_version=str(value["grammar_version"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate_boundaries(self) -> None:
        relation_ids = {relation.relation_id for relation in self.relations}
        if (
            relation_ids != set(self.guard_cases)
            or relation_ids != set(self.audit_cases)
            or relation_ids != set(self.future_evidence)
        ):
            raise ValueError("Every relation must have guard, audit, and future entries")
        requirement_ids = {requirement.relation.relation_id for requirement in self.requirements}
        if not requirement_ids.issubset(relation_ids):
            raise ValueError("Every requirement must belong to the frozen relation universe")
        support_ids = {row.evidence_id for row in self.support_evidence}
        guard_ids = {case.case_id for cases in self.guard_cases.values() for case in cases}
        audit_ids = {case.case_id for cases in self.audit_cases.values() for case in cases}
        future_ids = {row.evidence_id for rows in self.future_evidence.values() for row in rows}
        groups = [support_ids, guard_ids, audit_ids, future_ids]
        for index, left in enumerate(groups):
            for right in groups[index + 1 :]:
                if left & right:
                    raise ValueError(
                        "Support, guard, audit, and future identifiers must be disjoint"
                    )
        support_items = {
            item_id
            for row in self.support_evidence
            for item_id in (row.preferred_item_id, row.dispreferred_item_id)
        }
        guard_items = {
            item.item_id
            for cases in self.guard_cases.values()
            for case in cases
            for item in (case.preferred_item, case.dispreferred_item)
        }
        audit_items = {
            item.item_id
            for cases in self.audit_cases.values()
            for case in cases
            for item in (case.preferred_item, case.dispreferred_item)
        }
        future_items = {
            item_id
            for rows in self.future_evidence.values()
            for row in rows
            for item_id in (row.preferred_item_id, row.dispreferred_item_id)
        }
        item_groups = [support_items, guard_items, audit_items, future_items]
        for index, left in enumerate(item_groups):
            for right in item_groups[index + 1 :]:
                if left & right:
                    raise ValueError("Support, guard, audit, and future items must be disjoint")
        if any(row.position > self.position for row in self.support_evidence):
            raise ValueError("Support evidence cannot occur after the transition")
        if any(
            row.position <= self.position for rows in self.future_evidence.values() for row in rows
        ):
            raise ValueError("Future evidence must occur after the transition")
        if any(row.user_id != self.user_id for row in self.support_evidence):
            raise ValueError("Support evidence must belong to the manifest user")
        if any(
            row.user_id != self.user_id for rows in self.future_evidence.values() for row in rows
        ):
            raise ValueError("Future evidence must belong to the manifest user")
        if self.split not in {"validation", "test"}:
            raise ValueError("Manifest split must be validation or test")


@dataclass(frozen=True)
class PolicyClause:
    relation: Relation
    source: str
    active: bool = True
    text: str = ""

    @property
    def clause_id(self) -> str:
        return content_hash(asdict(self))[:20]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PolicyClause:
        return cls(
            relation=Relation.from_dict(dict(value["relation"])),
            source=str(value["source"]),
            active=bool(value.get("active", True)),
            text=str(value.get("text", "")),
        )


@dataclass(frozen=True)
class PolicyState:
    user_id: str
    clauses: tuple[PolicyClause, ...]
    version: int = 0

    @property
    def state_hash(self) -> str:
        return content_hash(asdict(self))

    @property
    def policy_text(self) -> str:
        return "\n".join(
            clause.text or canonical_clause_text(clause.relation)
            for clause in self.clauses
            if clause.active
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PolicyState:
        return cls(
            user_id=str(value["user_id"]),
            clauses=tuple(PolicyClause.from_dict(row) for row in value.get("clauses", [])),
            version=int(value.get("version", 0)),
        )


def canonical_clause_text(relation: Relation) -> str:
    base = (
        f"Prefer {relation.preferred_value} over {relation.dispreferred_value} "
        f"for attribute {relation.attribute}."
    )
    if relation.context_attribute is None:
        return base
    return base[:-1] + f" when {relation.context_attribute} is {relation.context_value}."
