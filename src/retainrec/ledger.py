"""Append-only experiment ledger with a verifiable hash chain."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .hashing import content_hash

GENESIS_HASH = "0" * 64


class ExperimentLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        previous_hash = GENESIS_HASH
        if self.path.exists():
            rows = list(self.read())
            if rows:
                previous_hash = str(rows[-1]["record_hash"])
        body = {"previous_hash": previous_hash, "payload": payload}
        record = {**body, "record_hash": content_hash(body)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        return record

    def read(self) -> Iterable[dict[str, Any]]:
        if not self.path.exists():
            return ()
        rows = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid ledger JSON at line {line_number}") from exc
                rows.append(value)
        return tuple(rows)

    def verify(self) -> int:
        expected_previous = GENESIS_HASH
        count = 0
        for count, record in enumerate(self.read(), start=1):
            if record.get("previous_hash") != expected_previous:
                raise ValueError(f"Broken ledger predecessor at row {count}")
            body = {"previous_hash": record["previous_hash"], "payload": record["payload"]}
            expected_hash = content_hash(body)
            if record.get("record_hash") != expected_hash:
                raise ValueError(f"Invalid ledger hash at row {count}")
            expected_previous = expected_hash
        return count
