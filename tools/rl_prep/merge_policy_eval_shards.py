"""Merge the shards written by eval_action_policy.py (--num-shards N) into one run directory that summarize_policy_eval.py reads.

Usage: merge_policy_eval_shards.py <root> <label> <out_dir> [--expect-scenes N]
Reads <root>/<label>__shard*/validation/scores.json, requires that every shard is complete and that no scene appears twice, and
writes <out_dir>/validation/scores.json (all rows) plus <out_dir>/scores.csv in the layout of NAVSIM's per-scene score file.
"""
import argparse
import csv
import json
from pathlib import Path

COLUMNS = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound",
           "comfort", "driving_direction_compliance", "score")


def merge(root, label, out_dir, expect=None):
    shards = sorted(Path(root).glob(f"{label}__shard*"))
    if not shards:
        raise FileNotFoundError(f"no shards named {label}__shard* under {root}")
    rows, seen = [], set()
    for d in shards:
        summary = json.loads((d / "summary.json").read_text())
        if not summary.get("complete"):
            raise ValueError(f"{d} is not complete")
        for r in json.loads((d / "validation" / "scores.json").read_text())["rows"]:
            if r["token"] in seen:
                raise ValueError(f"scene {r['token']} appears in two shards")
            seen.add(r["token"])
            rows.append(r)
    n_shards = json.loads((shards[0] / "summary.json").read_text())["shard"][1]
    if len(shards) != n_shards:
        raise ValueError(f"expected {n_shards} shards, found {len(shards)}")
    if expect is not None and len(rows) != expect:
        raise ValueError(f"expected {expect} scenes, merged {len(rows)}")
    out = Path(out_dir)
    (out / "validation").mkdir(parents=True)
    metrics = {k: sum(r[k] for r in rows) / len(rows) for k in COLUMNS if k in rows[0]}
    (out / "validation" / "scores.json").write_text(json.dumps({"split": "merged", "mode": "unconditional", "num_scenes": len(rows),
                                                                 "metrics": metrics, "rows": rows}, indent=2) + "\n")
    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "token", "valid", *COLUMNS])
        for i, r in enumerate(sorted(rows, key=lambda r: r["token"])):
            w.writerow([i, r["token"], True, *[r[k] for k in COLUMNS]])
        w.writerow(["", "average", True, *[metrics[k] for k in COLUMNS]])
    return len(rows), metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("label")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--expect-scenes", type=int, default=None)
    a = ap.parse_args()
    n, m = merge(a.root, a.label, a.out_dir, a.expect_scenes)
    print(f"merged {n} scenes; score {m['score'] * 100:.3f}")


if __name__ == "__main__":
    main()
