from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from .models import SourceRecord


def write_source_ledger(path: Path, records: Sequence[SourceRecord]) -> None:
    payload = [asdict(record) for record in sorted(records, key=lambda item: item.source_id)]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
