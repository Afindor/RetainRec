"""End-to-end experiment stages used by the command-line interface."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .construction import DatasetConfig, load_normalized_data, read_manifests
from .evidence import aggregate_evidence, future_validity_label
from .hashing import content_hash, file_hash
from .io import read_json, write_json
from .ledger import ExperimentLedger
from .metrics import audit_transition, user_macro_summary
from .models import PolicyState, TransitionManifest
from .policy import initial_policy
from .ranking import DeterministicRanker, VLLMRanker, relation_pass
from .selector import (
    SelectorSample,
    ValidityEstimator,
    choose_cost_ratio,
    select_relations,
)
from .updater import DeterministicProposer, UpdateConfig, VLLMPolicyProposer, update_policy


def fit_selector_stage(
    *,
    dataset_config_path: str | Path,
    transition_path: str | Path,
    method_config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    dataset_config = DatasetConfig.load(dataset_config_path)
    method_config = read_json(method_config_path)
    validation = [row for row in read_manifests(transition_path) if row.split == "validation"]
    if not validation:
        raise ValueError("No validation transitions are available for selector fitting")
    decay_grid = (0.0, 0.001, 0.01, 0.05, 0.1)
    best: tuple[float, float, ValidityEstimator, tuple[SelectorSample, ...]] | None = None
    for decay in decay_grid:
        samples = _selector_samples(validation, dataset_config, decay)
        if {sample.label for sample in samples} != {0, 1}:
            continue
        estimator = ValidityEstimator.fit(samples)
        loss = _log_loss(estimator, samples)
        candidate = (loss, decay, estimator, samples)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise ValueError("Validation transitions do not contain both defined future labels")
    loss, decay, estimator, samples = best
    probabilities = [estimator.predict(sample.summary) for sample in samples]
    labels = [sample.label for sample in samples]
    cost_ratio = choose_cost_ratio(
        probabilities,
        labels,
        method_config["stale_to_miss_cost_grid"],
    )
    artifact = {
        "dataset_config_hash": file_hash(dataset_config_path),
        "method_config_hash": file_hash(method_config_path),
        "transition_manifest_hash": file_hash(transition_path),
        "fit_split": "validation",
        "training_samples": len(samples),
        "decay": decay,
        "validation_log_loss": loss,
        "stale_to_miss_cost_ratio": cost_ratio,
        "estimator": estimator.to_dict(),
    }
    write_json(output_path, artifact)
    return artifact


def run_test_stage(
    *,
    dataset_config_path: str | Path,
    method_config_path: str | Path,
    selector_path: str | Path,
    transition_path: str | Path,
    data_dir: str | Path,
    ledger_path: str | Path,
    seeds: Iterable[int],
    backend: str = "deterministic",
    endpoint: str | None = None,
    model: str | None = None,
) -> int:
    dataset_config = DatasetConfig.load(dataset_config_path)
    method = read_json(method_config_path)
    selector_artifact = read_json(selector_path)
    _verify_frozen_inputs(
        selector_artifact,
        dataset_config_path,
        method_config_path,
        transition_path,
    )
    estimator = ValidityEstimator.from_dict(selector_artifact["estimator"])
    decay = float(selector_artifact["decay"])
    stale_cost = float(selector_artifact["stale_to_miss_cost_ratio"])
    items, _ = load_normalized_data(data_dir)
    if backend == "vllm":
        if not endpoint or not model:
            raise ValueError("The vLLM backend requires an endpoint and model name")
    elif backend != "deterministic":
        raise ValueError(f"Unsupported backend: {backend}")
    update_config = UpdateConfig(
        proposal_revisions=int(method["proposal_revisions"]),
        edit_budget=int(method["edit_budget"]),
        guard_acquisition_threshold=float(method["guard_acquisition_threshold"]),
        guard_regression_tolerance=float(method["guard_regression_tolerance"]),
        clause_limit=int(method["clause_limit"]),
        operation_weight=float(method["operation_weight"]),
        token_weight=float(method["token_weight"]),
        operation_costs={key: float(value) for key, value in method["operation_costs"].items()},
    )
    test_rows = [row for row in read_manifests(transition_path) if row.split == "test"]
    if not test_rows:
        raise ValueError("No test transitions are available")
    ledger_target = Path(ledger_path)
    if ledger_target.exists():
        raise FileExistsError(f"Refusing to append to an existing test ledger: {ledger_target}")
    ledger = ExperimentLedger(ledger_target)
    count = 0
    for seed in seeds:
        if backend == "deterministic":
            ranker = DeterministicRanker(items)
            proposer = DeterministicProposer()
        else:
            ranker = VLLMRanker(
                endpoint=endpoint or "",
                model=model or "",
                items=items,
                seed=int(seed),
                max_tokens=int(method["output_token_limit"]),
            )
            proposer = VLLMPolicyProposer(
                endpoint=endpoint or "",
                model=model or "",
                seed=int(seed),
                max_tokens=int(method["output_token_limit"]),
            )
        states: dict[str, PolicyState] = {}
        for manifest in test_rows:
            requirement_ids = {
                requirement.relation.relation_id for requirement in manifest.requirements
            }
            if manifest.user_id not in states:
                seed_relations = tuple(
                    relation
                    for relation in manifest.relations
                    if relation.relation_id not in requirement_ids
                )
                states[manifest.user_id] = initial_policy(
                    manifest.user_id, seed_relations, update_config.clause_limit
                )
            before = states[manifest.user_id]
            summaries = {
                relation.relation_id: aggregate_evidence(
                    (
                        row
                        for row in manifest.support_evidence
                        if row.relation.relation_id == relation.relation_id
                    ),
                    manifest.position,
                    decay,
                )
                for relation in manifest.relations
            }
            decisions = select_relations(
                manifest.relations,
                summaries,
                estimator,
                manifest.context,
                lambda relation,
                frozen_ranker=ranker,
                frozen_state=before,
                frozen_cases=manifest.guard_cases,
                frozen_threshold=dataset_config.pair_pass_threshold: relation_pass(
                    frozen_ranker,
                    frozen_state,
                    frozen_cases[relation.relation_id],
                    frozen_threshold,
                ),
                stale_cost,
            )
            update = update_policy(
                state=before,
                requirements=manifest.requirements,
                decisions=decisions,
                guard_cases=manifest.guard_cases,
                ranker=ranker,
                proposer=proposer,
                config=update_config,
            )
            after = update.state
            probabilities = {
                decision.relation.relation_id: decision.probability
                for decision in decisions
                if decision.selected
            }
            metrics = audit_transition(
                before=before,
                after=after,
                audit_cases=manifest.audit_cases,
                requirements=manifest.requirements,
                selected_probabilities=probabilities,
                ranker=ranker,
            )
            payload = {
                "dataset": dataset_config.name,
                "method": "RetainRec",
                "backend": backend,
                "model": model,
                "seed": int(seed),
                "user_id": manifest.user_id,
                "transition_id": manifest.transition_id,
                "position": manifest.position,
                "manifest_hash": content_hash(manifest.to_dict()),
                "dataset_config_hash": file_hash(dataset_config_path),
                "method_config_hash": file_hash(method_config_path),
                "selector_hash": file_hash(selector_path),
                "pre_state_hash": before.state_hash,
                "post_state_hash": after.state_hash,
                "pre_state": before.to_dict(),
                "post_state": after.to_dict(),
                "rejected": update.rejected,
                "selected_relation_ids": list(update.selected_relation_ids),
                "selector_decisions": [
                    {
                        "relation_id": decision.relation.relation_id,
                        "probability": decision.probability,
                        "net_value": decision.net_value,
                        "selected": decision.selected,
                        "reason": decision.reason,
                    }
                    for decision in decisions
                ],
                "proposals": [
                    {
                        **record.__dict__,
                        "distance": None if math.isinf(record.distance) else record.distance,
                    }
                    for record in update.proposals
                ],
                **metrics.to_dict(),
            }
            ledger.append(payload)
            states[manifest.user_id] = after
            count += 1
    ledger.verify()
    return count


def aggregate_stage(ledger_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    ledger = ExperimentLedger(ledger_path)
    ledger.verify()
    payloads = [record["payload"] for record in ledger.read()]
    if not payloads:
        raise ValueError("Cannot aggregate an empty ledger")
    identity_fields = ("dataset", "method", "backend", "model")
    identities = {key: {row.get(key) for row in payloads} for key in identity_fields}
    for key, values in identities.items():
        if len(values) != 1:
            raise ValueError(f"A ledger must contain one {key}; found {sorted(map(str, values))}")
    result = {
        **{key: next(iter(values)) for key, values in identities.items()},
        "ledger_hash": file_hash(ledger_path),
        "records": len(payloads),
        "users": len({str(row["user_id"]) for row in payloads}),
        "seeds": len({int(row["seed"]) for row in payloads}),
        "acquisition_rate": user_macro_summary(payloads, "acquisition_rate"),
        "update_regression_rate": user_macro_summary(payloads, "update_regression_rate"),
        "policy_edit_ratio": user_macro_summary(payloads, "policy_edit_ratio"),
        "rejection_rate": sum(bool(row["rejected"]) for row in payloads) / len(payloads),
    }
    ogr_rows = [row for row in payloads if row.get("obligation_generalization_rate") is not None]
    if ogr_rows:
        result["obligation_generalization_rate"] = user_macro_summary(
            ogr_rows, "obligation_generalization_rate"
        )
    write_json(output_path, result)
    return result


def _selector_samples(
    manifests: Iterable[TransitionManifest], config: DatasetConfig, decay: float
) -> tuple[SelectorSample, ...]:
    samples = []
    for manifest in manifests:
        for relation in manifest.relations:
            label = future_validity_label(
                manifest.future_evidence.get(relation.relation_id, ()),
                config.future_validity_threshold,
                config.minimum_future_evidence,
            )
            if label is None:
                continue
            summary = aggregate_evidence(
                (
                    row
                    for row in manifest.support_evidence
                    if row.relation.relation_id == relation.relation_id
                ),
                manifest.position,
                decay,
            )
            samples.append(SelectorSample(summary, label))
    return tuple(samples)


def _log_loss(estimator: ValidityEstimator, samples: tuple[SelectorSample, ...]) -> float:
    losses = []
    for sample in samples:
        probability = min(max(estimator.predict(sample.summary), 1e-9), 1 - 1e-9)
        losses.append(
            -sample.label * math.log(probability) - (1 - sample.label) * math.log(1 - probability)
        )
    return sum(losses) / len(losses)


def _verify_frozen_inputs(
    selector_artifact: dict[str, Any],
    dataset_config_path: str | Path,
    method_config_path: str | Path,
    transition_path: str | Path,
) -> None:
    expected = {
        "dataset_config_hash": file_hash(dataset_config_path),
        "method_config_hash": file_hash(method_config_path),
        "transition_manifest_hash": file_hash(transition_path),
    }
    for key, value in expected.items():
        if selector_artifact.get(key) != value:
            raise ValueError(f"Frozen selector input mismatch: {key}")
