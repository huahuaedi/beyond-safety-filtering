"""Paired score difference between two per-scene NAVSIM score tables with a whole-log bootstrap interval.

Usage: paired_scores_by_log.py BASE.csv ARM.csv[,ARM2.csv,...] token_log_map.json [--n 4000]
Prints, for the score and each metric column, the mean difference (x100), its log-cluster 95% interval, and wins/ties/losses; also the mean size
of the gains and of the losses, because a positive mean can come from few large gains against many small losses.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

METRICS = ("score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort",
           # NAVSIM v2 columns; skipped when the table has none of them
           "driving_direction_compliance", "traffic_light_compliance", "lane_keeping", "history_comfort", "two_frame_extended_comfort")


def read(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["token"] == "" or r["token"].startswith("average") or r.get("valid", "True") != "True":  # v1 "average", v2 "average_all_frames"
                continue
            out[r["token"]] = {k: float(r[k]) for k in METRICS if k in r and r[k] != ""}
    return out


def read_mean(paths):
    """Average the per-scene metrics of several score tables (e.g. the seeds/arms of one family); the scene sets must be identical."""
    tables = [read(p) for p in paths]
    if any(set(t) != set(tables[0]) for t in tables):
        raise ValueError("score tables cover different scenes")
    return {tok: {k: float(np.mean([t[tok][k] for t in tables])) for k in tables[0][tok]} for tok in tables[0]}


def boot_by_log(d, logs, rng, n):
    uniq = sorted(set(logs))
    idx_of = {l: i for i, l in enumerate(uniq)}
    lid = np.array([idx_of[l] for l in logs])
    sums = np.bincount(lid, weights=d, minlength=len(uniq))
    cnt = np.bincount(lid, minlength=len(uniq)).astype(float)
    pick = rng.integers(0, len(uniq), size=(n, len(uniq)))
    m = sums[pick].sum(axis=1) / cnt[pick].sum(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def compare(base, arm, mapping, n=4000, seed=0):
    tokens = sorted(set(base) & set(arm))
    if len(tokens) != len(base) or len(tokens) != len(arm):
        raise ValueError(f"scene sets differ: base {len(base)}, arm {len(arm)}, common {len(tokens)}")
    logs = [mapping[t].split("log_name:")[-1] for t in tokens]
    rng = np.random.default_rng(seed)
    out = {"scenes": len(tokens), "logs": len(set(logs)), "metrics": {}}
    for k in METRICS:
        if k not in base[tokens[0]] or k not in arm[tokens[0]]:
            continue
        d = np.array([arm[t][k] - base[t][k] for t in tokens]) * 100
        up, down = d[d > 1e-9], d[d < -1e-9]
        out["metrics"][k] = {"mean_x100": float(d.mean()), "log_ci95_x100": boot_by_log(d, logs, rng, n),
                             "wins": int(len(up)), "ties": int(len(d) - len(up) - len(down)), "losses": int(len(down)),
                             "mean_win_x100": float(up.mean()) if len(up) else 0.0, "mean_loss_x100": float(down.mean()) if len(down) else 0.0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=str, help="score csv, or several separated by commas (per-scene mean)")
    ap.add_argument("arm", type=str, help="score csv, or several separated by commas (their per-scene mean is compared with the base)")
    ap.add_argument("token_log_map", type=Path)
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()
    arms = [Path(p) for p in str(a.arm).split(",")]
    bases = [Path(p) for p in str(a.base).split(",")]
    r = compare(read_mean(bases) if len(bases) > 1 else read(bases[0]),
                read_mean(arms) if len(arms) > 1 else read(arms[0]), json.loads(a.token_log_map.read_text()), a.n)
    if len(bases) > 1 or len(arms) > 1:
        print(f"base: mean of {len(bases)} score tables; arm: mean of {len(arms)} score tables")
    print(f"{r['scenes']} scenes, {r['logs']} logs")
    for k, v in r["metrics"].items():
        lo, hi = v["log_ci95_x100"]
        print(f"{k:34s} {v['mean_x100']:+.3f} [{lo:+.3f}, {hi:+.3f}]  W/T/L {v['wins']}/{v['ties']}/{v['losses']}  "
              f"mean win {v['mean_win_x100']:+.2f}, mean loss {v['mean_loss_x100']:+.2f}")


if __name__ == "__main__":
    main()
