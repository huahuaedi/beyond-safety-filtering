"""Where does the navtest degradation of RL vs SFT come from? (per-scene, CPU)

Pairs each RL seed with the SFT run on the same scenes and reports: how
concentrated the loss is, which safety metric flips, whether seeds fail on the
same scenes, and how many driving logs carry the damage. Counts first; no
trajectory cherry-picking.
"""
import argparse
import csv
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np

SAFE = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")


def read_csv(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["token"] == "average" or r["token"] == "":
                continue
            if r["valid"] != "True":
                continue
            rows[r["token"]] = {k: float(r[k]) for k in (*SAFE, "ego_progress", "driving_direction_compliance", "score")}
    return rows


def find_arm(root, suffix):
    hits = sorted(h for h in glob.glob(str(root / "metrics" / f"*_{suffix}" / "*" / "*.csv"))
                  if "smoke" not in h)
    if len(hits) != 1:
        raise ValueError(f"{suffix}: expected one csv, found {len(hits)}")
    return read_csv(hits[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--rl", default="baseline", help="arm name prefix of the RL runs (baseline = original RL)")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    logmap = json.loads((args.root / "token_log_map.json").read_text())
    sft = find_arm(args.root, "sft")
    runs = [find_arm(args.root, f"{args.rl}_seed{s}") for s in range(args.seeds)]
    tokens = sorted(set(sft).intersection(*[set(r) for r in runs]))
    n = len(tokens)
    print(f"scenes common to SFT and all {args.seeds} seeds: {n}")
    out = {"scenes": n, "per_seed": [], }
    S = np.array([sft[t]["score"] for t in tokens])
    D = np.array([[r[t]["score"] for t in tokens] for r in runs]) - S          # [seeds, scenes]
    fail = {k: np.array([[r[t][k] < 1 - 1e-6 for t in tokens] for r in runs]) for k in SAFE}
    sft_fail = {k: np.array([sft[t][k] < 1 - 1e-6 for t in tokens]) for k in SAFE}
    prog = np.array([[r[t]["ego_progress"] for t in tokens] for r in runs]) - \
        np.array([sft[t]["ego_progress"] for t in tokens])
    for s in range(args.seeds):
        d = D[s]
        neg = d[d < -1e-9]
        order = np.sort(d)
        k1, k5 = max(1, n // 100), max(1, n // 20)
        big = np.abs(d) > 0.3
        row = {
            "mean_delta_x100": float(d.mean() * 100),
            "scenes_worse": int((d < -1e-6).sum()), "scenes_better": int((d > 1e-6).sum()),
            "scenes_big_change_gt0.3": int(big.sum()),
            "share_of_mean_from_big_changes": float(d[big].sum() / d.sum()) if d.sum() != 0 else None,
            "mean_delta_from_small_changes_x100": float(d[~big].sum() / n * 100),
            "mean_delta_from_big_changes_x100": float(d[big].sum() / n * 100),
            "worst_1pct_share_of_total_loss": float(order[:k1].sum() / neg.sum()) if len(neg) else None,
            "worst_5pct_share_of_total_loss": float(order[:k5].sum() / neg.sum()) if len(neg) else None,
            "mean_progress_delta_x100": float(prog[s].mean() * 100),
        }
        for k in SAFE:
            row[k] = {"new_failures": int((fail[k][s] & ~sft_fail[k]).sum()),
                      "fixed": int((~fail[k][s] & sft_fail[k]).sum())}
        out["per_seed"].append(row)
    # seed agreement on new failures
    out["new_failure_seed_overlap"] = {}
    for k in SAFE:
        new = fail[k] & ~sft_fail[k][None]
        cnt = new.sum(axis=0)
        out["new_failure_seed_overlap"][k] = {f"in_{m}_of_{args.seeds}_seeds": int((cnt == m).sum()) for m in range(1, args.seeds + 1)}
    anyfail = np.zeros(n, bool)
    for k in SAFE:
        anyfail |= (fail[k] & ~sft_fail[k][None]).any(axis=0)
    # logs carrying the damage: scenes that are worse by > 0.05 in at least 2 seeds
    worse2 = ((D < -0.05).sum(axis=0) >= 2)
    logs = Counter(logmap[t] for t, w in zip(tokens, worse2) if w)
    out["scenes_worse_by_gt0.05_in_ge2_seeds"] = int(worse2.sum())
    out["logs_with_such_scenes"] = len(logs)
    out["top5_logs_share"] = float(sum(c for _, c in logs.most_common(5)) / max(1, worse2.sum()))
    out["total_logs"] = len(set(logmap[t] for t in tokens))
    out["scenes_with_new_safety_failure_in_any_seed"] = int(anyfail.sum())
    out["mean_delta_by_seed_averaged_x100"] = float(D.mean() * 100)
    print(json.dumps(out, indent=2))
    (args.root / f"degradation_{args.rl}.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
