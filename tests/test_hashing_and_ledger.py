import json
import tempfile
import unittest
from pathlib import Path

from retainrec.hashing import canonical_json, content_hash
from retainrec.ledger import ExperimentLedger


class HashingAndLedgerTest(unittest.TestCase):
    def test_canonical_hash_ignores_dictionary_order(self) -> None:
        left = {"b": 2, "a": {"y": 1, "x": 0}}
        right = {"a": {"x": 0, "y": 1}, "b": 2}
        self.assertEqual(canonical_json(left), canonical_json(right))
        self.assertEqual(content_hash(left), content_hash(right))

    def test_hash_chain_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            ledger = ExperimentLedger(path)
            ledger.append({"value": 1})
            ledger.append({"value": 2})
            self.assertEqual(ledger.verify(), 2)
            rows = path.read_text(encoding="utf-8").splitlines()
            value = json.loads(rows[1])
            value["payload"]["value"] = 3
            rows[1] = json.dumps(value)
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                ledger.verify()


if __name__ == "__main__":
    unittest.main()
