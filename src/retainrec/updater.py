"""Bounded proposal, guard evaluation, and minimum-change selection."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

from .models import PairCase, PolicyState, Relation, Requirement
from .policy import apply_requirements, policy_distance, policy_edits
from .ranking import Ranker, pair_pass, relation_pass_rate
from .selector import SelectionDecision


class PolicyProposer(ABC):
    @abstractmethod
    def propose(
        self,
        state: PolicyState,
        previous_candidate: PolicyState | None,
        requirements: tuple[Requirement, ...],
        selected_relations: tuple[Relation, ...],
        failed_scopes: tuple[str, ...],
        proposal_index: int,
        clause_limit: int,
    ) -> PolicyState:
        """Produce one bounded candidate without audit information."""

    def usage(self) -> BackendUsage:
        return BackendUsage()


@dataclass(frozen=True)
class BackendUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reported_latency_ms: float = 0.0


class DeterministicProposer(PolicyProposer):
    def propose(
        self,
        state: PolicyState,
        previous_candidate: PolicyState | None,
        requirements: tuple[Requirement, ...],
        selected_relations: tuple[Relation, ...],
        failed_scopes: tuple[str, ...],
        proposal_index: int,
        clause_limit: int,
    ) -> PolicyState:
        base = previous_candidate or state
        candidate = apply_requirements(
            base,
            requirements,
            clause_limit=clause_limit,
            replace_conflicts=proposal_index > 0,
        )
        if proposal_index == 0:
            return candidate
        protected = {relation.relation_id: relation for relation in selected_relations}
        clauses = list(candidate.clauses)
        existing = {clause.relation.relation_id for clause in clauses}
        from .models import PolicyClause, canonical_clause_text

        for relation_id in failed_scopes:
            if relation_id in protected and relation_id not in existing:
                relation = protected[relation_id]
                clauses.append(
                    PolicyClause(
                        relation,
                        source="guard_feedback",
                        text=canonical_clause_text(relation),
                    )
                )
        return PolicyState(
            user_id=state.user_id,
            clauses=tuple(clauses[-clause_limit:]),
            version=state.version + 1,
        )


class CommandPolicyProposer(PolicyProposer):
    """Adapter for a local process that implements the policy proposal contract."""

    def __init__(self, command: tuple[str, ...]) -> None:
        self.command = command
        self._usage = BackendUsage()

    def propose(
        self,
        state: PolicyState,
        previous_candidate: PolicyState | None,
        requirements: tuple[Requirement, ...],
        selected_relations: tuple[Relation, ...],
        failed_scopes: tuple[str, ...],
        proposal_index: int,
        clause_limit: int,
    ) -> PolicyState:
        payload = _proposal_payload(
            state,
            previous_candidate,
            requirements,
            selected_relations,
            failed_scopes,
            proposal_index,
            clause_limit,
        )
        completed = subprocess.run(
            self.command,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
        )
        result = json.loads(completed.stdout)
        usage = result.get("usage", {})
        self._usage = BackendUsage(
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            reported_latency_ms=float(usage.get("latency_ms", 0.0)),
        )
        allowed = _allowed_relation_ids(state, previous_candidate, requirements, selected_relations)
        return _validate_proposed_state(
            state,
            PolicyState.from_dict(result["state"]),
            clause_limit,
            allowed,
        )

    def usage(self) -> BackendUsage:
        return self._usage


class VLLMPolicyProposer(PolicyProposer):
    """Policy proposer backed by a frozen OpenAI-compatible vLLM endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        temperature: float = 0.1,
        top_p: float = 0.95,
        seed: int = 2024001,
        max_tokens: int = 384,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self._usage = BackendUsage()

    def propose(
        self,
        state: PolicyState,
        previous_candidate: PolicyState | None,
        requirements: tuple[Requirement, ...],
        selected_relations: tuple[Relation, ...],
        failed_scopes: tuple[str, ...],
        proposal_index: int,
        clause_limit: int,
    ) -> PolicyState:
        payload = _proposal_payload(
            state,
            previous_candidate,
            requirements,
            selected_relations,
            failed_scopes,
            proposal_index,
            clause_limit,
        )
        prompt = (
            "Revise the policy using only add, delete, replace, or scope operations. "
            "Use only relations present in the request, preserve selected obligations, "
            "and return one JSON object with user_id, version, and clauses.\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.endpoint + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        usage = result.get("usage", {})
        self._usage = BackendUsage(
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )
        content = result["choices"][0]["message"]["content"]
        allowed = _allowed_relation_ids(state, previous_candidate, requirements, selected_relations)
        return _validate_proposed_state(
            state,
            PolicyState.from_dict(json.loads(content)),
            clause_limit,
            allowed,
        )

    def usage(self) -> BackendUsage:
        return self._usage


@dataclass(frozen=True)
class UpdateConfig:
    proposal_revisions: int
    edit_budget: int
    guard_acquisition_threshold: float
    guard_regression_tolerance: float
    clause_limit: int
    operation_weight: float
    token_weight: float
    operation_costs: dict[str, float]


@dataclass(frozen=True)
class ProposalRecord:
    proposal_index: int
    parsed: bool
    edit_count: int
    acquisition_rate: float
    guard_regression_rate: float | None
    distance: float
    feasible: bool
    failed_requirement_ids: tuple[str, ...]
    failed_scopes: tuple[str, ...]
    state_hash: str
    latency_ms: float
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class UpdateResult:
    state: PolicyState
    rejected: bool
    selected_relation_ids: tuple[str, ...]
    proposals: tuple[ProposalRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def update_policy(
    *,
    state: PolicyState,
    requirements: tuple[Requirement, ...],
    decisions: tuple[SelectionDecision, ...],
    guard_cases: dict[str, tuple[PairCase, ...]],
    ranker: Ranker,
    proposer: PolicyProposer,
    config: UpdateConfig,
) -> UpdateResult:
    selected = tuple(decision.relation for decision in decisions if decision.selected)
    probabilities = {
        decision.relation.relation_id: decision.probability
        for decision in decisions
        if decision.selected
    }
    failed_scopes: tuple[str, ...] = ()
    active_requirements = requirements
    previous_candidate: PolicyState | None = None
    records: list[ProposalRecord] = []
    feasible: list[tuple[float, float, float, int, PolicyState]] = []

    for proposal_index in range(config.proposal_revisions + 1):
        started = time.perf_counter()
        try:
            candidate = proposer.propose(
                state,
                previous_candidate,
                active_requirements,
                selected,
                failed_scopes,
                proposal_index,
                config.clause_limit,
            )
            measured_latency_ms = (time.perf_counter() - started) * 1000.0
            usage = proposer.usage()
            previous_candidate = candidate
            latency_ms = usage.reported_latency_ms or measured_latency_ms
            edit_count = len(policy_edits(state, candidate))
            acquisition = _acquisition_rate(candidate, requirements, ranker)
            active_requirements = _unmet_requirements(candidate, requirements, ranker)
            guard_regression, failed_scopes = _guard_regression(
                state, candidate, selected, probabilities, guard_cases, ranker
            )
            distance = policy_distance(
                state,
                candidate,
                config.operation_costs,
                config.operation_weight,
                config.token_weight,
            )
            within_regression = (
                guard_regression is None or guard_regression <= config.guard_regression_tolerance
            )
            is_feasible = (
                edit_count <= config.edit_budget
                and acquisition >= config.guard_acquisition_threshold
                and within_regression
            )
            if is_feasible:
                regression_tie_value = guard_regression if guard_regression is not None else 0.0
                feasible.append(
                    (
                        distance,
                        -acquisition,
                        regression_tie_value,
                        proposal_index,
                        candidate,
                    )
                )
            records.append(
                ProposalRecord(
                    proposal_index=proposal_index,
                    parsed=True,
                    edit_count=edit_count,
                    acquisition_rate=acquisition,
                    guard_regression_rate=guard_regression,
                    distance=distance,
                    feasible=is_feasible,
                    failed_requirement_ids=tuple(row.requirement_id for row in active_requirements),
                    failed_scopes=failed_scopes,
                    state_hash=candidate.state_hash,
                    latency_ms=latency_ms,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
            )
        except (KeyError, TypeError, ValueError):
            records.append(
                ProposalRecord(
                    proposal_index=proposal_index,
                    parsed=False,
                    edit_count=0,
                    acquisition_rate=0.0,
                    guard_regression_rate=None,
                    distance=float("inf"),
                    feasible=False,
                    failed_requirement_ids=tuple(row.requirement_id for row in active_requirements),
                    failed_scopes=failed_scopes,
                    state_hash=state.state_hash,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    input_tokens=0,
                    output_tokens=0,
                )
            )

    if not feasible:
        return UpdateResult(
            state=state,
            rejected=True,
            selected_relation_ids=tuple(relation.relation_id for relation in selected),
            proposals=tuple(records),
        )
    _, _, _, _, committed = min(feasible)
    return UpdateResult(
        state=committed,
        rejected=False,
        selected_relation_ids=tuple(relation.relation_id for relation in selected),
        proposals=tuple(records),
    )


def _acquisition_rate(
    state: PolicyState, requirements: tuple[Requirement, ...], ranker: Ranker
) -> float:
    cases = [case for requirement in requirements for case in requirement.guard_cases]
    if not cases:
        return 0.0
    return sum(pair_pass(ranker, state, case) for case in cases) / len(cases)


def _unmet_requirements(
    state: PolicyState,
    requirements: tuple[Requirement, ...],
    ranker: Ranker,
) -> tuple[Requirement, ...]:
    return tuple(
        requirement
        for requirement in requirements
        if relation_pass_rate(ranker, state, requirement.guard_cases) != 1.0
    )


def _guard_regression(
    before: PolicyState,
    after: PolicyState,
    selected: tuple[Relation, ...],
    probabilities: dict[str, float],
    guard_cases: dict[str, tuple[PairCase, ...]],
    ranker: Ranker,
) -> tuple[float | None, tuple[str, ...]]:
    weighted_regression = 0.0
    total_weight = 0.0
    failed = []
    for relation in selected:
        cases = guard_cases.get(relation.relation_id, ())
        pre = [pair_pass(ranker, before, case) for case in cases]
        post = [pair_pass(ranker, after, case) for case in cases]
        eligible = sum(pre)
        if eligible == 0:
            continue
        regression = sum(left * (1 - right) for left, right in zip(pre, post)) / eligible
        weight = probabilities[relation.relation_id]
        weighted_regression += weight * regression
        total_weight += weight
        if relation_pass_rate(ranker, after, cases) != 1.0:
            failed.append(relation.relation_id)
    if total_weight == 0:
        return None, tuple(failed)
    return weighted_regression / total_weight, tuple(failed)


def _proposal_payload(
    state: PolicyState,
    previous_candidate: PolicyState | None,
    requirements: tuple[Requirement, ...],
    selected_relations: tuple[Relation, ...],
    failed_scopes: tuple[str, ...],
    proposal_index: int,
    clause_limit: int,
) -> dict[str, object]:
    return {
        "state": state.to_dict(),
        "previous_candidate": (
            previous_candidate.to_dict() if previous_candidate is not None else None
        ),
        "requirements": [
            {
                "requirement_id": requirement.requirement_id,
                "relation": asdict(requirement.relation),
            }
            for requirement in requirements
        ],
        "selected_relations": [asdict(relation) for relation in selected_relations],
        "failed_scopes": list(failed_scopes),
        "proposal_index": proposal_index,
        "clause_limit": clause_limit,
    }


def _validate_proposed_state(
    before: PolicyState,
    proposed: PolicyState,
    clause_limit: int,
    allowed_relation_ids: set[str],
) -> PolicyState:
    if proposed.user_id != before.user_id:
        raise ValueError("A proposer cannot change the user identifier")
    if proposed.version != before.version + 1:
        raise ValueError("A proposed state must increment the version exactly once")
    if len(proposed.clauses) > clause_limit:
        raise ValueError("A proposed state exceeds the clause limit")
    relation_ids = [clause.relation.relation_id for clause in proposed.clauses]
    if len(relation_ids) != len(set(relation_ids)):
        raise ValueError("A proposed state contains duplicate relations")
    if not set(relation_ids).issubset(allowed_relation_ids):
        raise ValueError(
            "A proposed state contains a relation outside the frozen grammar instance set"
        )
    return proposed


def _allowed_relation_ids(
    state: PolicyState,
    previous_candidate: PolicyState | None,
    requirements: tuple[Requirement, ...],
    selected_relations: tuple[Relation, ...],
) -> set[str]:
    return {
        *(clause.relation.relation_id for clause in state.clauses),
        *(
            clause.relation.relation_id
            for clause in (() if previous_candidate is None else previous_candidate.clauses)
        ),
        *(requirement.relation.relation_id for requirement in requirements),
        *(relation.relation_id for relation in selected_relations),
    }
