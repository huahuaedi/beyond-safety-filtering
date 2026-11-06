"""Paired analysis of the original five-seed result (SFT vs direct_rl seeds 20-24).

Inputs are the per-scene NAVSIM CSVs of the SAME scene set. Reports, seed-averaged:
paired PDMS and sub-metric deltas with scene and log-cluster bootstrap intervals,
failure conversions (new failures vs fixed), how much of the PDMS gain the progress
and TTC components explain, and optionally the repeatability of a second evaluation
of the same checkpoints (e.g. the *_rawonly reruns).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

COLS = ("score", "ego_progress", "time_to_collision_within_bound", "no_at_fault_collisions",
        "drivable_area_compliance", "comfort", "driving_direction_compliance")
FAIL = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")


def read(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["token"] in ("", "average") or r.get("valid", "True") != "True":
                continue
            rows[r["token"]] = {c: float(r[c]) for c in COLS}
    return rows


def token_to_log(cache_dir):
    out = {}
    for pkl in Path(cache_dir).glob("*/*/*/metric_cache.pkl"):
        out[pkl.parent.name] = pkl.parents[2].name
    return out


def boot_scene(d, rng, n=5000):
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def boot_log(d, logs, rng, n=5000):
    uniq = sorted(set(logs))
    lid = np.array([uniq.index(l) for l in logs])
    sums = np.array([d[lid == i].sum() for i in range(len(uniq))])
    cnt = np.array([(lid == i).sum() for i in range(len(uniq))])
    idx = rng.integers(0, len(uniq), size=(n, len(uniq)))
    m = sums[idx].sum(axis=1) / cnt[idx].sum(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def matrix(runs, tokens, col):
    return np.array([[r[t][col] for t in tokens] for r in runs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", type=Path, required=True)
    ap.add_argument("--rl", type=Path, nargs="+", required=True, help="one CSV per seed (selected checkpoint)")
    ap.add_argument("--repeat", type=Path, nargs="*", default=[], help="second evaluation of the same checkpoints, same order")
    ap.add_argument("--cache-dir", type=Path, default=None, help="metric cache dir of the eval scenes, for log clustering")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    sft = read(args.sft)
    rls = [read(p) for p in args.rl]
    tokens = sorted(set(sft).intersection(*[set(r) for r in rls]))
    n = len(tokens)
    out = {"scenes": n, "sft_only_scenes": len(sft) - n, "seeds": len(rls)}
    logs = None
    if args.cache_dir is not None:
        t2l = token_to_log(args.cache_dir)
        missing = [t for t in tokens if t not in t2l]
        if missing:
            raise ValueError(f"{len(missing)} scenes have no log mapping, e.g. {missing[0]}")
        logs = [t2l[t] for t in tokens]
        out["logs"] = len(set(logs))
    S = {c: np.array([sft[t][c] for t in tokens]) for c in COLS}
    R = {c: matrix(rls, tokens, c) for c in COLS}           # [seeds, scenes]
    out["means_x100"] = {c: {"sft": float(S[c].mean() * 100), "rl_seed_avg": float(R[c].mean() * 100),
                             "rl_per_seed": [float(x * 100) for x in R[c].mean(axis=1)]} for c in COLS}
    out["paired"] = {}
    for c in COLS:
        d = R[c].mean(axis=0) - S[c]
        entry = {"delta_x100": float(d.mean() * 100), "scene_ci95_x100": [x * 100 for x in boot_scene(d, rng)]}
        if logs is not None:
            entry["log_cluster_ci95_x100"] = [x * 100 for x in boot_log(d, logs, rng)]
        out["paired"][c] = entry
    d = R["score"].mean(axis=0) - S["score"]
    out["per_seed_pdms_delta_x100"] = [float(x * 100) for x in (R["score"] - S["score"]).mean(axis=1)]
    out["improve_rate"] = float((d > 1e-9).mean())
    out["regression_rate"] = float((d < -1e-9).mean())
    out["scenes_with_any_change"] = int((np.abs(R["score"] - S["score"]) > 1e-9).any(axis=0).sum())
    conv = {}
    for c in FAIL:
        sf = S[c] < 1 - 1e-6
        rf = R[c] < 1 - 1e-6
        conv[c] = {"sft_failures": int(sf.sum()), "rl_failures_per_seed": [int(x) for x in rf.sum(axis=1)],
                   "new_failures_per_seed": [int((rf[i] & ~sf).sum()) for i in range(len(rls))],
                   "fixed_per_seed": [int((~rf[i] & sf).sum()) for i in range(len(rls))],
                   "new_failure_scenes_any_seed": int((rf & ~sf).any(axis=0).sum()),
                   "new_failure_scenes_all_seeds": int((rf & ~sf).all(axis=0).sum())}
    out["failure_conversions"] = conv
    # explained share: PDMS = NC*DAC*(5*EP+5*TTC+2*C)/12 -> additive parts when NC, DAC unchanged
    wsum = (5 * (R["ego_progress"].mean() - S["ego_progress"].mean()) +
            5 * (R["time_to_collision_within_bound"].mean() - S["time_to_collision_within_bound"].mean()) +
            2 * (R["comfort"].mean() - S["comfort"].mean())) / 12
    out["formula_check_x100"] = {
        "progress_part": float(5 / 12 * (R["ego_progress"].mean() - S["ego_progress"].mean()) * 100),
        "ttc_part": float(5 / 12 * (R["time_to_collision_within_bound"].mean() - S["time_to_collision_within_bound"].mean()) * 100),
        "weighted_sum_ignoring_gates": float(wsum * 100), "actual_pdms_delta": float(d.mean() * 100),
        "note": "gates NC/DAC multiply the weighted sum, so the identity is approximate when they change"}
    order = np.sort(d)
    neg = order[order < 0]
    out["concentration"] = {"worst_5_scenes_sum_x100_of_mean": float(order[:5].sum() / n * 100),
                            "total_negative_mass_x100_of_mean": float(neg.sum() / n * 100),
                            "total_positive_mass_x100_of_mean": float(d[d > 0].sum() / n * 100)}
    if args.repeat:
        reps = [read(p) for p in args.repeat]
        if len(reps) != len(rls):
            raise ValueError("--repeat needs one CSV per --rl CSV, same order")
        diffs = []
        for a, b in zip(rls, reps):
            common = sorted(set(a) & set(b))
            diffs.append(np.array([b[t]["score"] - a[t]["score"] for t in common]))
        out["repeatability"] = {"scenes_compared": [len(x) for x in diffs],
                                "mean_abs_score_diff_x100": [float(np.abs(x).mean() * 100) for x in diffs],
                                "scenes_with_different_score": [int((np.abs(x) > 1e-9).sum()) for x in diffs],
                                "mean_score_diff_x100": [float(x.mean() * 100) for x in diffs]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
