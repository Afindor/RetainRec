import sys
import unittest
from pathlib import Path

from retainrec.construction import DatasetConfig, build_manifests, load_normalized_data
from retainrec.controls import ExternalMethodAdapter
from retainrec.models import PolicyState


class ExternalAdapterTest(unittest.TestCase):
    def test_external_method_cannot_receive_sealed_records(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = DatasetConfig.load(root / "configs/datasets/smoke.json")
        items, events = load_normalized_data(root / "data/examples/smoke")
        manifest = build_manifests(config, items, events)[0]
        program = (
            "import json,sys;"
            "p=json.load(sys.stdin);"
            "assert 'audit_cases' not in p and 'future_evidence' not in p;"
            "s=p['state'];s['version']+=1;"
            "json.dump({'state':s,'native_metadata':{'sealed':True}},sys.stdout)"
        )
        adapter = ExternalMethodAdapter("reference", (sys.executable, "-c", program))
        result = adapter.update(PolicyState(manifest.user_id, ()), manifest)
        self.assertEqual(result.state.version, 1)
        self.assertEqual(result.native_metadata, {"sealed": True})


if __name__ == "__main__":
    unittest.main()
