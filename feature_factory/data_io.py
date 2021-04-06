from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Union


def load_data(file_path: Path) -> list[dict[str, str]]:
    with file_path.open() as fin:
        reader = csv.DictReader(fin)
        return list(reader)


def dump_data(
    file_path: Path, rows: Iterable[dict[str, Union[str, int, float]]]
) -> None:
    field_names = rows[0].keys()
    with file_path.open("w") as fout:
        writer = csv.DictWriter(fout, field_names)
        writer.writeheader()
        writer.writerows(rows)
