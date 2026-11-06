"""CPU-only manifest with driving-log isolation between training and evaluation.

world_action_full_v1/{train,val,test} is split by a hash of the scene token, so
neighbouring frames of one log appear on both sides. This builder pools all
three directories and re-splits by log: every log that contains a final-eval
token, a legacy validation token, or a hash-selected extra log is held out.
Training scenes come only from the remaining logs. The log of a scene is read
from the NAVSIM metric-cache layout  <cache>/<log>/<type>/<token>/metric_cache.pkl.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from offline_verified_data import (read_scene, verified_labels, topk_safe_labels, POSITIVE)

SPLITS = ("train", "val", "test")


def token_to_log(cache_dirs):
    mapping = {}
    for cache in cache_dirs:
        for pkl in Path(cache).glob("*/*/*/metric_cache.pkl"):
            token, log = pkl.parent.name, pkl.parents[2].name
            if mapping.setdefault(token, log) != log:
                raise ValueError(f"token {token} appears under two logs")
    if not mapping:
        raise ValueError("no metric cache entries found")
    return mapping


def log_bucket(log):
    return int.from_bytes(hashlib.sha256(("log:" + log).encode()).digest()[:8], "little") % 10000


def choose_heldout_logs(token_log, forced_tokens, extra_fraction):
    missing = [t for t in forced_tokens if t not in token_log]
    if missing:
        raise ValueError(f"{len(missing)} forced held-out tokens have no metric cache, e.g. {missing[0]}")
    held = {token_log[t] for t in forced_tokens}
    if extra_fraction > 0:
        for log in set(token_log.values()):
            if log_bucket(log) < extra_fraction * 10000:
                held.add(log)
    return held


def label_scene(path, mode, margin, topk):
    _, scores, ref = read_scene(path)
    if mode == "margin":
        return verified_labels(scores, ref, margin)["labels"]
    return topk_safe_labels(scores, ref, k=topk, margin=margin)["labels"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--metric-cache-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--final-eval-tokens", type=Path, required=True)
    p.add_argument("--val-tokens", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label-mode", choices=["margin", "topk_safe"], default="topk_safe")
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--extra-heldout-log-fraction", type=float, default=0.05)
    p.add_argument("--max-heldout-eval", type=int, default=600)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 <= args.extra_heldout_log_fraction < 1:
        raise ValueError("extra fraction must be in [0, 1)")

    final_eval = set(json.loads(args.final_eval_tokens.read_text()))
    val_tokens = set(json.loads(args.val_tokens.read_text()))
    token_log = token_to_log(args.metric_cache_dirs)
    held_logs = choose_heldout_logs(token_log, final_eval | val_tokens, args.extra_heldout_log_fraction)

    scenes = {}
    for split in SPLITS:
        for path in sorted((args.root / split).glob("*.npz")):
            if path.stem in scenes:
                raise ValueError(f"scene {path.stem} appears in two split directories")
            scenes[path.stem] = (split, path)
    unmapped = [t for t in scenes if t not in token_log]
    if unmapped:
        raise ValueError(f"{len(unmapped)} candidate scenes lack a log mapping, e.g. {unmapped[0]}")

    train, heldout = [], []
    stats = {"candidates": 0, "positive": 0, "scenes_with_positive": 0}
    for i, (token, (split, path)) in enumerate(sorted(scenes.items())):
        if token_log[token] in held_logs:
            heldout.append(token)
            continue
        labels = label_scene(path, args.label_mode, args.margin, args.topk)
        n = int((labels == POSITIVE).sum())
        stats["candidates"] += len(labels)
        stats["positive"] += n
        stats["scenes_with_positive"] += int(n > 0)
        train.append({"token": token, "split": split, "candidates": len(labels), "positive": n,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if (i + 1) % 1000 == 0:
            print("labelled", i + 1, flush=True)
    if not train or not heldout:
        raise ValueError("empty training or held-out set")
    train_logs = {token_log[r["token"]] for r in train}
    if train_logs & held_logs:
        raise AssertionError("log leak between training and held-out")
    heldout.sort(key=lambda t: hashlib.sha256(("eval:" + t).encode()).hexdigest())
    heldout_eval = heldout[: args.max_heldout_eval] if args.max_heldout_eval else heldout

    report = {
        "schema_version": 2, "margin": args.margin, "label_mode": args.label_mode, "topk": args.topk,
        "source_root": str(args.root.resolve()), "excluded_training_tokens": [],
        "splits": {"train": {"scenes": len(train), "logs": len(train_logs), **stats},
                   "heldout": {"scenes": len(heldout), "logs": len(held_logs)}},
        "train": train, "validation_tokens": sorted(val_tokens),
        "heldout_eval_tokens": heldout_eval,
        "final_eval_count": len(final_eval),
        "extra_heldout_log_fraction": args.extra_heldout_log_fraction,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("train", "validation_tokens", "heldout_eval_tokens")}, indent=2))
    print(f"heldout eval scenes used: {len(heldout_eval)} of {len(heldout)}")


if __name__ == "__main__":
    main()
