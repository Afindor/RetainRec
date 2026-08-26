"""Controlled diagnostics and external published-method adapters."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass

from .models import PolicyState, Relation, Requirement, TransitionManifest
from .policy import apply_requirements
from .ranking import Ranker, relation_pass
from .selector import SelectionDecision
from .updater import (
    DeterministicProposer,
    PolicyProposer,
    UpdateConfig,
    UpdateResult,
    update_policy,
)


def no_update(state: PolicyState) -> PolicyState:
    return state


def direct_policy_rewrite(
    state: PolicyState, manifest: TransitionManifest, clause_limit: int
) -> PolicyState:
    return apply_requirements(
        state,
        manifest.requirements,
        clause_limit=clause_limit,
        replace_conflicts=True,
    )


def uniform_guard_update(
    *,
    state: PolicyState,
    manifest: TransitionManifest,
    ranker: Ranker,
    config: UpdateConfig,
    relation_cap: int,
    proposer: PolicyProposer | None = None,
) -> UpdateResult:
    ordered = sorted(
        manifest.relations,
        key=lambda relation: relation.relation_id,
    )
    protected = [
        relation
        for relation in ordered
        if relation_pass(
            ranker,
            state,
            manifest.guard_cases[relation.relation_id],
            0.60,
        )
    ][:relation_cap]
    protected_ids = {relation.relation_id for relation in protected}
    decisions = tuple(
        SelectionDecision(
            relation=relation,
            probability=1.0,
            net_value=1.0,
            selected=relation.relation_id in protected_ids,
            reason=("selected" if relation.relation_id in protected_ids else "relation_cap"),
        )
        for relation in ordered
    )
    return update_policy(
        state=state,
        requirements=manifest.requirements,
        decisions=decisions,
        guard_cases=manifest.guard_cases,
        ranker=ranker,
        proposer=proposer or DeterministicProposer(),
        config=config,
    )


def regression_replay_update(
    *,
    state: PolicyState,
    manifest: TransitionManifest,
    replay_relations: tuple[Relation, ...],
    clause_limit: int,
) -> PolicyState:
    replay_requirements = tuple(
        Requirement(
            requirement_id=f"replay:{relation.relation_id}",
            relation=relation,
            guard_cases=(),
            audit_cases=(),
        )
        for relation in replay_relations
    )
    combined_map = {
        requirement.requirement_id: requirement
        for requirement in (*manifest.requirements, *replay_requirements)
    }
    combined = tuple(combined_map[key] for key in sorted(combined_map))
    return apply_requirements(
        state,
        combined,
        clause_limit=clause_limit,
        replace_conflicts=True,
    )


@dataclass(frozen=True)
class ExternalUpdateResult:
    state: PolicyState
    latency_ms: float
    input_tokens: int
    output_tokens: int
    native_metadata: dict[str, object]


class ExternalMethodAdapter:
    """Run an upstream method while withholding audit and future records."""

    def __init__(self, method_name: str, command: tuple[str, ...]) -> None:
        self.method_name = method_name
        self.command = command

    def update(self, state: PolicyState, manifest: TransitionManifest) -> ExternalUpdateResult:
        payload = {
            "method": self.method_name,
            "state": state.to_dict(),
            "transition_id": manifest.transition_id,
            "user_id": manifest.user_id,
            "position": manifest.position,
            "context": manifest.context,
            "support_evidence": [asdict(row) for row in manifest.support_evidence],
            "requirements": [
                {
                    "requirement_id": requirement.requirement_id,
                    "relation": asdict(requirement.relation),
                }
                for requirement in manifest.requirements
            ],
        }
        started = time.perf_counter()
        completed = subprocess.run(
            self.command,
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            check=True,
        )
        measured_latency = (time.perf_counter() - started) * 1000.0
        result = json.loads(completed.stdout)
        proposed = PolicyState.from_dict(result["state"])
        if proposed.user_id != state.user_id or proposed.version != state.version + 1:
            raise ValueError("External method returned an invalid state transition")
        usage = result.get("usage", {})
        return ExternalUpdateResult(
            state=proposed,
            latency_ms=float(usage.get("latency_ms", measured_latency)),
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            native_metadata=dict(result.get("native_metadata", {})),
        )
