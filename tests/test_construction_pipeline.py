import tempfile
import unittest
from pathlib import Path

from retainrec.comparison import compare_ledgers
from retainrec.construction import (
    DatasetConfig,
    build_manifests,
    load_normalized_data,
    normalized_data_summary,
    write_manifest_package,
)
from retainrec.ledger import ExperimentLedger
from retainrec.pipeline import aggregate_stage, fit_selector_stage, run_test_stage
from retainrec.reporting import export_summary_csv


class ConstructionPipelineTest(unittest.TestCase):
    def test_complete_reference_run(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dataset_config_path = root / "configs/datasets/smoke.json"
        method_config_path = root / "configs/methods/retainrec.json"
        data_dir = root / "data/examples/smoke"
        config = DatasetConfig.load(dataset_config_path)
        items, events = load_normalized_data(data_dir)
        self.assertEqual(normalized_data_summary(items, events)["users"], 4)
        manifests = build_manifests(config, items, events)
        self.assertEqual(len(manifests), 4)
        self.assertEqual(sum(row.split == "validation" for row in manifests), 2)
        self.assertEqual(sum(row.split == "test" for row in manifests), 2)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            write_manifest_package(target, dataset_config_path, manifests)
            transitions = target / "transitions.jsonl"
            selector = target / "selector.json"
            ledger = target / "ledger.jsonl"
            aggregate = target / "aggregate.json"
            fitted = fit_selector_stage(
                dataset_config_path=dataset_config_path,
                transition_path=transitions,
                method_config_path=method_config_path,
                output_path=selector,
            )
            self.assertEqual(fitted["fit_split"], "validation")
            records = run_test_stage(
                dataset_config_path=dataset_config_path,
                method_config_path=method_config_path,
                selector_path=selector,
                transition_path=transitions,
                data_dir=data_dir,
                ledger_path=ledger,
                seeds=(2024001, 2024002),
            )
            self.assertEqual(records, 4)
            self.assertEqual(ExperimentLedger(ledger).verify(), 4)
            summary = aggregate_stage(ledger, aggregate)
            self.assertEqual(summary["records"], 4)
            self.assertEqual(summary["dataset"], "smoke")
            self.assertEqual(summary["method"], "RetainRec")
            table = target / "summary.csv"
            self.assertEqual(export_summary_csv((aggregate,), table), 1)
            self.assertIn("update_regression_rate_mean", table.read_text(encoding="utf-8"))
            comparison = compare_ledgers(ledger, ledger, target / "comparison.json")
            self.assertEqual(comparison["primary_urr"]["paired_difference"], 0.0)
            self.assertEqual(comparison["common_pass_urr"]["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
