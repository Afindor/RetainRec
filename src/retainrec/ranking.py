"""Serving-interface adapters and pairwise pass evaluation."""

from __future__ import annotations

import json
import subprocess
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .hashing import content_hash
from .models import Item, PairCase, PolicyState


class Ranker(ABC):
    @abstractmethod
    def rank(self, state: PolicyState, case: PairCase) -> tuple[str, ...]:
        """Return every candidate identifier in descending preference order."""


class DeterministicRanker(Ranker):
    """Reference backend for tests and contract validation."""

    def __init__(self, items: dict[str, Item]) -> None:
        self.items = items

    def rank(self, state: PolicyState, case: PairCase) -> tuple[str, ...]:
        def score(item_id: str) -> tuple[float, str]:
            item = self.items[item_id]
            value = int(content_hash({"item_id": item_id})[:8], 16) / 0xFFFFFFFF
            for clause in state.clauses:
                relation = clause.relation
                if not clause.active or not relation.applies(case.context):
                    continue
                attributes = _values(item.attributes.get(relation.attribute))
                if relation.preferred_value in attributes:
                    value += 2.0
                if relation.dispreferred_value in attributes:
                    value -= 2.0
            return value, item_id

        candidates = tuple(dict.fromkeys(case.candidate_item_ids))
        if not candidates or any(candidate not in self.items for candidate in candidates):
            raise ValueError("Every candidate must exist in the item manifest")
        return tuple(sorted(candidates, key=score, reverse=True))


@dataclass(frozen=True)
class CommandRanker(Ranker):
    """JSON-line adapter for an external local serving process."""

    command: tuple[str, ...]

    def rank(self, state: PolicyState, case: PairCase) -> tuple[str, ...]:
        payload = json.dumps({"state": state.to_dict(), "case": _case_payload(case)})
        completed = subprocess.run(
            self.command,
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        return _validate_ranking(json.loads(completed.stdout), case.candidate_item_ids)


@dataclass(frozen=True)
class VLLMRanker(Ranker):
    """OpenAI-compatible adapter for a frozen local vLLM endpoint."""

    endpoint: str
    model: str
    items: dict[str, Item]
    seed: int
    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int = 384
    timeout_seconds: float = 120.0

    def rank(self, state: PolicyState, case: PairCase) -> tuple[str, ...]:
        prompt = (
            "Rank all candidate IDs for the supplied user policy and request. "
            "Return a JSON array containing each ID exactly once.\n"
            + json.dumps(
                {
                    "state": state.to_dict(),
                    "case": _case_payload(case, self.items),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        request = urllib.request.Request(
            self.endpoint.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        return _validate_ranking(json.loads(content), case.candidate_item_ids)


def pair_pass(ranker: Ranker, state: PolicyState, case: PairCase) -> int:
    try:
        order = ranker.rank(state, case)
        preferred_rank = order.index(case.preferred_item.item_id)
        dispreferred_rank = order.index(case.dispreferred_item.item_id)
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return 0
    return int(preferred_rank < dispreferred_rank)


def relation_pass_rate(
    ranker: Ranker, state: PolicyState, cases: tuple[PairCase, ...]
) -> float | None:
    if not cases:
        return None
    return sum(pair_pass(ranker, state, case) for case in cases) / len(cases)


def relation_pass(
    ranker: Ranker, state: PolicyState, cases: tuple[PairCase, ...], threshold: float
) -> bool:
    rate = relation_pass_rate(ranker, state, cases)
    return rate is not None and rate >= threshold


def _validate_ranking(value: Any, candidates: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("Ranker output must be a JSON array")
    ranking = tuple(str(item) for item in value)
    if len(ranking) != len(candidates) or set(ranking) != set(candidates):
        raise ValueError("Ranker output must contain every candidate exactly once")
    return ranking


def _case_payload(case: PairCase, items: dict[str, Item] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "case_id": case.case_id,
        "context": case.context,
        "candidates": list(case.candidate_item_ids),
    }
    if items is not None:
        unknown = set(case.candidate_item_ids) - set(items)
        if unknown:
            raise ValueError(f"Candidate metadata is missing for: {sorted(unknown)}")
        payload["candidate_metadata"] = [
            {
                "item_id": item_id,
                "attributes": items[item_id].attributes,
            }
            for item_id in case.candidate_item_ids
        ]
    return payload


def _values(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    if value is None:
        return set()
    return {str(value)}
