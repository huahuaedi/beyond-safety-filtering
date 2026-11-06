"""Inspect available numeric scene metadata before defining physics mappings."""

from __future__ import annotations

import argparse
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np


PATTERN = re.compile(
    r"speed|limit|curvature|friction|weather|rain|surface|road|map|location",
    re.IGNORECASE,
)


def walk(value, prefix="", depth=0):
    if depth > 4:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, child
            yield from walk(child, path, depth + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()
    with args.metadata.open("rb") as file:
        payload = pickle.load(file)
    records = payload if isinstance(payload, (list, tuple)) else [payload]
    if args.max_records > 0:
        records = records[: args.max_records]
    presence, numeric, examples = Counter(), Counter(), {}
    for record in records:
        seen = set()
        for path, value in walk(record):
            if not PATTERN.search(path):
                continue
            seen.add(path)
            array = np.asarray(value) if not isinstance(value, (dict, str)) else None
            if array is not None and np.issubdtype(array.dtype, np.number):
                numeric[path] += 1
            examples.setdefault(path, repr(value)[:160])
        presence.update(seen)
    print(f"records={len(records)}")
    for path, count in sorted(presence.items(), key=lambda item: (-item[1], item[0])):
        print(
            f"{path}\tpresent={count}/{len(records)}\tnumeric={numeric[path]}"
            f"\texample={examples[path]}"
        )


if __name__ == "__main__":
    main()
