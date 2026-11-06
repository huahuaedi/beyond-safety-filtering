#!/usr/bin/env python3
"""Map selector scene tokens to source-data log IDs and audit split isolation.

The original log pickle can have been written by NumPy 2 while the DriveVLA
environment uses NumPy 1. The script intentionally fails closed if log
identity cannot be established for every scene.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


class NumpyCoreCompatUnpickler(pickle.Unpickler):
    """Resolve NumPy 2's renamed private core modules under NumPy 1."""

    def find_class(self, module: str, name: str):
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)




def read_tokens(path: Path) -> set[str]:
    obj = json.loads(path.read_text())
    if isinstance(obj, dict):
        for key in ("scene_tokens", "tokens", "train_tokens"):
            if key in obj:
                obj = obj[key]
                break
    if not isinstance(obj, list):
        raise ValueError(f"Expected a JSON list of scene tokens in {path}")
    return {str(x) for x in obj}


def collection_splits(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            token, split = str(z["scene_token"]), str(z["split"])
        if token in result:
            raise ValueError(f"Duplicate collection scene token: {token}")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unexpected split {split!r} for {token}")
        result[token] = split
    if not result:
        raise ValueError(f"No collection NPZ files under {root}")
    return result


def raw_log_map(root: Path, needed_tokens: set[str]) -> dict[str, str]:
    """Match NAVSIM anchor/frame tokens to explicit log_name in raw logs.

    Matches the token and log_name fields used by this project's
    navsim.common.dataloader.filter_scenes/get_tokens_list_per_log.
    Never infer log identity from filenames or processed image paths.
    """
    files = sorted(root.glob("*.pkl"))
    if not files:
        raise ValueError(f"No raw NAVSIM log pickle files in {root}")
    found = {}
    print(f"Reading {len(files)} raw logs from {root}; requested tokens={len(needed_tokens)}", flush=True)
    for index, path in enumerate(files, 1):
        with path.open("rb") as handle:
            frames = NumpyCoreCompatUnpickler(handle).load()
        if not isinstance(frames, (list, tuple)):
            raise ValueError(f"Expected raw frame list in {path}; got {type(frames).__name__}")
        for frame in frames:
            if not isinstance(frame, dict):
                raise ValueError(f"Non-dictionary raw frame in {path}")
            token = str(frame.get("token", ""))
            if token not in needed_tokens:
                continue
            log_name = frame.get("log_name")
            if not isinstance(log_name, str) or not log_name.strip():
                raise ValueError(f"Raw frame {token} in {path} lacks explicit log_name; keys={sorted(frame)}")
            log_id = "log_name:" + log_name
            if token in found and found[token] != log_id:
                raise ValueError(f"Conflicting log identities for {token}: {found[token]} vs {log_id}")
            found[token] = log_id
        del frames
        if index == 1 or index % 100 == 0 or index == len(files):
            print(f"Scanned {index}/{len(files)} logs; mapped {len(found)}/{len(needed_tokens)} tokens", flush=True)
    missing = sorted(needed_tokens - found.keys())
    if missing:
        raise ValueError(f"{len(missing)} requested tokens absent from raw logs; examples={missing[:10]}")
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", type=Path, required=True)
    ap.add_argument("--navsim-log-dir", type=Path, required=True,
                    help="Original NAVSIM navsim_logs/trainval directory, not processed VLA records.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--policy-training-tokens", type=Path,
                    help="Optional JSON list of scenes used to fit the evaluated policy checkpoint.")
    ap.add_argument("--allow-overlap", action="store_true",
                    help="Write the audit despite overlap; intended for debugging only.")
    args = ap.parse_args()

    splits = collection_splits(args.collection)
    policy_tokens = read_tokens(args.policy_training_tokens) if args.policy_training_tokens else set()
    tokens = set(splits) | policy_tokens
    token_to_log = raw_log_map(args.navsim_log_dir, tokens)
    rows = []
    for i, token in enumerate(sorted(tokens), 1):
        log_id = token_to_log[token]
        rows.append({"scene_token": token, "selector_split": splits.get(token, ""),
                     "policy_training_scene": int(token in policy_tokens), "log_id": log_id})
        if i % 250 == 0:
            print(f"mapped {i}/{len(tokens)} scenes", flush=True)

    log_sets = {s: {token_to_log[t] for t, split in splits.items() if split == s}
                for s in ("train", "val", "test")}
    overlaps = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        common = sorted(log_sets[a] & log_sets[b])
        overlaps[f"{a}__{b}"] = {"count": len(common), "examples": common[:20]}
    policy_logs = {token_to_log[t] for t in policy_tokens if t in token_to_log}
    overlaps["policy_train__val"] = {
        "count": len(policy_logs & log_sets["val"]),
        "examples": sorted(policy_logs & log_sets["val"])[:20],
    }
    overlaps["policy_train__test"] = {
        "count": len(policy_logs & log_sets["test"]),
        "examples": sorted(policy_logs & log_sets["test"])[:20],
    }
    failed = any(v["count"] for v in overlaps.values())
    report = {
        "mapping_source": str(args.navsim_log_dir),
        "mapping_rule": "raw frame token -> explicit log_name; full directory scanned for conflicting mappings",
        "collection_scene_counts": {s: sum(x == s for x in splits.values()) for s in ("train", "val", "test")},
        "mapped_scene_count_including_policy_tokens": len(token_to_log),
        "policy_training_token_count": len(policy_tokens),
        "policy_training_manifest_provided": bool(args.policy_training_tokens),
        "log_overlap": overlaps,
        "status": ("FAIL_OVERLAP" if failed else
                   "PASS_LOG_DISJOINT" if args.policy_training_tokens else
                   "SPLITS_DISJOINT_POLICY_USAGE_UNVERIFIED"),
        "interpretation": "PASS requires all selector splits to be log-disjoint and evaluated-policy training logs to be disjoint from selector val/test logs.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scene_log_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("scene_token", "selector_split", "policy_training_scene", "log_id"))
        writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "log_isolation_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failed and not args.allow_overlap:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
