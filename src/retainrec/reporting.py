"""Tabular exports for aggregate experiment results."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .io import read_json

SUMMARY_COLUMNS = (
    "dataset",
    "method",
    "backend",
    "model",
    "records",
    "users",
    "seeds",
    "acquisition_rate_mean",
    "acquisition_rate_std",
    "update_regression_rate_mean",
    "update_regression_rate_std",
    "obligation_generalization_rate_mean",
    "obligation_generalization_rate_std",
    "policy_edit_ratio_mean",
    "policy_edit_ratio_std",
    "rejection_rate",
)


def export_summary_csv(inputs: Iterable[str | Path], output: str | Path) -> int:
    """Write one summary row for each aggregate result file."""
    rows = [_flatten_summary(read_json(path)) for path in inputs]
    if not rows:
        raise ValueError("At least one aggregate result is required")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _flatten_summary(summary: dict[str, Any]) -> dict[str, object]:
    row: dict[str, object] = {
        key: summary.get(key, "")
        for key in ("dataset", "method", "backend", "model", "records", "users", "seeds")
    }
    for metric in (
        "acquisition_rate",
        "update_regression_rate",
        "obligation_generalization_rate",
        "policy_edit_ratio",
    ):
        values = summary.get(metric)
        row[f"{metric}_mean"] = "" if values is None else values["mean"]
        row[f"{metric}_std"] = "" if values is None else values["std"]
    row["rejection_rate"] = summary["rejection_rate"]
    return row
