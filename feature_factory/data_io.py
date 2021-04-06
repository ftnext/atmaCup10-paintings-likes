from __future__ import annotations

import csv
from pathlib import Path


def load_data(file_path: Path) -> list[dict[str, str]]:
    with file_path.open() as fin:
        reader = csv.DictReader(fin)
        return list(reader)
