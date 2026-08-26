"""Command-line interface for construction, fitting, execution, and audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .comparison import compare_ledgers
from .construction import (
    DatasetConfig,
    build_manifests,
    load_normalized_data,
    normalized_data_summary,
    write_manifest_package,
)
from .io import read_json
from .ledger import ExperimentLedger
from .pipeline import aggregate_stage, fit_selector_stage, run_test_stage
from .reporting import export_summary_csv


def main() -> None:
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.command == "build-manifest":
        config = DatasetConfig.load(arguments.dataset_config)
        items, events = load_normalized_data(arguments.data_dir)
        manifests = build_manifests(config, items, events)
        write_manifest_package(
            arguments.output_dir,
            arguments.dataset_config,
            manifests,
        )
        print(json.dumps({"transitions": len(manifests)}, sort_keys=True))
    elif arguments.command == "validate-data":
        items, events = load_normalized_data(arguments.data_dir)
        print(json.dumps(normalized_data_summary(items, events), sort_keys=True))
    elif arguments.command == "fit-selector":
        result = fit_selector_stage(
            dataset_config_path=arguments.dataset_config,
            transition_path=arguments.transitions,
            method_config_path=arguments.method_config,
            output_path=arguments.output,
        )
        print(json.dumps(result, sort_keys=True))
    elif arguments.command == "run-test":
        count = run_test_stage(
            dataset_config_path=arguments.dataset_config,
            method_config_path=arguments.method_config,
            selector_path=arguments.selector,
            transition_path=arguments.transitions,
            data_dir=arguments.data_dir,
            ledger_path=arguments.ledger,
            seeds=arguments.seeds,
            backend=arguments.backend,
            endpoint=arguments.endpoint,
            model=arguments.model,
        )
        print(json.dumps({"ledger_records": count}, sort_keys=True))
    elif arguments.command == "aggregate":
        result = aggregate_stage(arguments.ledger, arguments.output)
        print(json.dumps(result, sort_keys=True))
    elif arguments.command == "verify-ledger":
        count = ExperimentLedger(arguments.ledger).verify()
        print(json.dumps({"verified_records": count}, sort_keys=True))
    elif arguments.command == "export-summary":
        rows = export_summary_csv(arguments.inputs, arguments.output)
        print(json.dumps({"exported_rows": rows}, sort_keys=True))
    elif arguments.command == "compare-ledgers":
        result = compare_ledgers(
            arguments.left,
            arguments.right,
            arguments.output,
            bootstrap_seed=arguments.bootstrap_seed,
        )
        print(json.dumps(result, sort_keys=True))
    elif arguments.command == "smoke":
        _smoke(Path(arguments.config))
    else:
        parser.error(f"Unknown command: {arguments.command}")


def _smoke(config_path: Path) -> None:
    experiment = read_json(config_path)
    root = config_path.resolve().parents[2]
    dataset_config = root / experiment["dataset_config"]
    method_config = root / experiment["method_config"]
    data_dir = root / experiment["data_dir"]
    artifact_dir = root / experiment["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    transitions = artifact_dir / "transitions.jsonl"
    selector = artifact_dir / "selector.json"
    ledger = artifact_dir / "ledger.jsonl"
    aggregate = artifact_dir / "aggregate.json"
    for path in (transitions, selector, ledger, aggregate, artifact_dir / "manifest.json"):
        if path.exists():
            path.unlink()
    config = DatasetConfig.load(dataset_config)
    items, events = load_normalized_data(data_dir)
    manifests = build_manifests(config, items, events)
    write_manifest_package(artifact_dir, dataset_config, manifests)
    fit_selector_stage(
        dataset_config_path=dataset_config,
        transition_path=transitions,
        method_config_path=method_config,
        output_path=selector,
    )
    run_test_stage(
        dataset_config_path=dataset_config,
        method_config_path=method_config,
        selector_path=selector,
        transition_path=transitions,
        data_dir=data_dir,
        ledger_path=ledger,
        seeds=experiment["seeds"],
        backend="deterministic",
    )
    result = aggregate_stage(ledger, aggregate)
    print(json.dumps(result, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retainrec")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-manifest")
    build.add_argument("--dataset-config", required=True)
    build.add_argument("--data-dir", required=True)
    build.add_argument("--output-dir", required=True)

    validate = subparsers.add_parser("validate-data")
    validate.add_argument("--data-dir", required=True)

    fit = subparsers.add_parser("fit-selector")
    fit.add_argument("--dataset-config", required=True)
    fit.add_argument("--method-config", required=True)
    fit.add_argument("--transitions", required=True)
    fit.add_argument("--output", required=True)

    run = subparsers.add_parser("run-test")
    run.add_argument("--dataset-config", required=True)
    run.add_argument("--method-config", required=True)
    run.add_argument("--selector", required=True)
    run.add_argument("--transitions", required=True)
    run.add_argument("--data-dir", required=True)
    run.add_argument("--ledger", required=True)
    run.add_argument("--seeds", nargs="+", type=int, required=True)
    run.add_argument("--backend", choices=("deterministic", "vllm"), default="deterministic")
    run.add_argument("--endpoint")
    run.add_argument("--model")

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--ledger", required=True)
    aggregate.add_argument("--output", required=True)

    verify = subparsers.add_parser("verify-ledger")
    verify.add_argument("--ledger", required=True)

    export = subparsers.add_parser("export-summary")
    export.add_argument("--inputs", nargs="+", required=True)
    export.add_argument("--output", required=True)

    compare = subparsers.add_parser("compare-ledgers")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--bootstrap-seed", type=int, default=2024001)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--config", default="configs/experiments/smoke.json")
    return parser


if __name__ == "__main__":
    main()
