"""Paired primary and common-pass comparisons from sealed ledgers."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .io import write_json
from .ledger import ExperimentLedger
from .metrics import clustered_bootstrap_difference


def compare_ledgers(
    left_path: str | Path,
    right_path: str | Path,
    output_path: str | Path,
    *,
    bootstrap_seed: int = 2024001,
) -> dict[str, Any]:
    """Compare two methods on identical seed-user-transition records."""
    left_rows = _payload_index(left_path)
    right_rows = _payload_index(right_path)
    shared = sorted(set(left_rows) & set(right_rows))
    if not shared:
        raise ValueError("The ledgers do not share any seed-user-transition records")
    datasets = {left_rows[key]["dataset"] for key in shared} | {
        right_rows[key]["dataset"] for key in shared
    }
    if len(datasets) != 1:
        raise ValueError("Paired ledgers must use the same dataset")

    primary_left: dict[tuple[str, int], list[float]] = defaultdict(list)
    primary_right: dict[tuple[str, int], list[float]] = defaultdict(list)
    common_left: dict[tuple[str, int], list[float]] = defaultdict(list)
    common_right: dict[tuple[str, int], list[float]] = defaultdict(list)
    common_units = 0
    union_units = 0
    for key in shared:
        left = left_rows[key]
        right = right_rows[key]
        user_seed = (str(left["user_id"]), int(left["seed"]))
        if (
            left.get("update_regression_rate") is not None
            and right.get("update_regression_rate") is not None
        ):
            primary_left[user_seed].append(float(left["update_regression_rate"]))
            primary_right[user_seed].append(float(right["update_regression_rate"]))
        left_relations = {row["relation_id"]: row for row in left["relation_outcomes"]}
        right_relations = {row["relation_id"]: row for row in right["relation_outcomes"]}
        relation_ids = set(left_relations) & set(right_relations)
        common = {
            relation_id
            for relation_id in relation_ids
            if left_relations[relation_id]["pre_passes"] > 0
            and right_relations[relation_id]["pre_passes"] > 0
        }
        union = {
            relation_id
            for relation_id in relation_ids
            if left_relations[relation_id]["pre_passes"] > 0
            or right_relations[relation_id]["pre_passes"] > 0
        }
        common_units += len(common)
        union_units += len(union)
        if common:
            common_left[user_seed].append(
                mean(float(left_relations[relation_id]["regression"]) for relation_id in common)
            )
            common_right[user_seed].append(
                mean(float(right_relations[relation_id]["regression"]) for relation_id in common)
            )

    result = {
        "dataset": next(iter(datasets)),
        "left_method": next(iter({left_rows[key]["method"] for key in shared})),
        "right_method": next(iter({right_rows[key]["method"] for key in shared})),
        "matched_records": len(shared),
        "primary_urr": _paired_result(primary_left, primary_right, bootstrap_seed),
        "common_pass_urr": {
            **_paired_result(common_left, common_right, bootstrap_seed),
            "coverage": common_units / union_units if union_units else 0.0,
            "common_relation_transition_units": common_units,
            "union_relation_transition_units": union_units,
        },
    }
    write_json(output_path, result)
    return result


def _payload_index(path: str | Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    ledger = ExperimentLedger(path)
    ledger.verify()
    result = {}
    for record in ledger.read():
        payload = record["payload"]
        key = (
            int(payload["seed"]),
            str(payload["user_id"]),
            str(payload["transition_id"]),
        )
        if key in result:
            raise ValueError(f"Duplicate ledger coordinate: {key}")
        result[key] = payload
    return result


def _paired_result(
    left: dict[tuple[str, int], list[float]],
    right: dict[tuple[str, int], list[float]],
    seed: int,
) -> dict[str, float | None]:
    left_users = _user_values(left)
    right_users = _user_values(right)
    common_users = set(left_users) & set(right_users)
    left_users = {user: left_users[user] for user in common_users}
    right_users = {user: right_users[user] for user in common_users}
    difference, lower, upper = clustered_bootstrap_difference(
        left_users,
        right_users,
        seed=seed,
    )
    differences = [left_users[user] - right_users[user] for user in sorted(left_users)]
    scale = stdev(differences) if len(differences) > 1 else 0.0
    effect = difference / scale if scale > 0 else (0.0 if difference == 0 else None)
    return {
        "left_mean": mean(left_users.values()),
        "right_mean": mean(right_users.values()),
        "paired_difference": difference,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "cohens_d": effect,
        "users": float(len(left_users)),
    }


def _user_values(rows: dict[tuple[str, int], list[float]]) -> dict[str, float]:
    by_user: dict[str, list[float]] = defaultdict(list)
    for (user_id, _seed), values in rows.items():
        if values:
            by_user[user_id].append(mean(values))
    return {user_id: mean(values) for user_id, values in by_user.items() if values}
