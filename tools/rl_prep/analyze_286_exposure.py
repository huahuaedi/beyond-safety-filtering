"""Did the original gain on the 286-scene cohort depend on the RL run having trained on the same driving log?

For every seed we read from its training log the scenes actually trained on up to the selected step,
map them to driving logs, and call an evaluation scene 'exposed' for that seed when its log contains a
trained scene (the scene itself is excluded by construction). We then compare the per-scene gain of that
seed's selected checkpoint over SFT between exposed and unexposed scenes, with a whole-log bootstrap.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

COLS = ("score", "ego_progress", "time_to_collision_within_bound", "drivable_area_compliance")


def read(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["token"] in ("", "average") or r.get("valid", "True") != "True":
                continue
            rows[r["token"]] = {c: float(r[c]) for c in COLS}
    return rows


def token_logs(*cache_dirs):
    out = {}
    for cache in cache_dirs:
        for pkl in Path(cache).glob("*/*/*/metric_cache.pkl"):
            out[pkl.parent.name] = pkl.parents[2].name
    return out


def trained_tokens(log_path, n_steps):
    tokens = []
    for line in open(log_path, errors="replace"):
        m = re.match(r"step (\d+) scene=([0-9a-f]+) ", line)
        if m and int(m.group(1)) < n_steps:
            tokens.append(m.group(2))
    return tokens


def boot_diff(delta, exposed, logs, rng, n=4000):
    """Whole-log bootstrap of mean(delta | exposed) - mean(delta | unexposed) over (seed, scene) pairs."""
    uniq = sorted(set(logs))
    idx = {l: i for i, l in enumerate(uniq)}
    L = np.array([idx[l] for l in logs])
    s1 = np.zeros(len(uniq)); c1 = np.zeros(len(uniq)); s0 = np.zeros(len(uniq)); c0 = np.zeros(len(uniq))
    for k in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            if exposed[k, j]:
                s1[L[j]] += delta[k, j]; c1[L[j]] += 1
            else:
                s0[L[j]] += delta[k, j]; c0[L[j]] += 1
    draws = rng.integers(0, len(uniq), size=(n, len(uniq)))
    a = s1[draws].sum(1) / np.maximum(c1[draws].sum(1), 1)
    b = s0[draws].sum(1) / np.maximum(c0[draws].sum(1), 1)
    ok = (c1[draws].sum(1) > 0) & (c0[draws].sum(1) > 0)
    d = (a - b)[ok]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", type=Path, required=True)
    ap.add_argument("--rl", type=Path, nargs="+", required=True)
    ap.add_argument("--train-logs", type=Path, nargs="+", required=True)
    ap.add_argument("--steps", type=int, nargs="+", required=True, help="selected step of each seed (number of updates trained)")
    ap.add_argument("--pool-cache", type=Path, required=True)
    ap.add_argument("--eval-cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not (len(args.rl) == len(args.train_logs) == len(args.steps)):
        raise ValueError("--rl, --train-logs and --steps need one entry per seed")
    rng = np.random.default_rng(args.seed)
    t2l = token_logs(args.pool_cache, args.eval_cache)
    sft = read(args.sft)
    rls = [read(p) for p in args.rl]
    tokens = sorted(set(sft).intersection(*[set(r) for r in rls]))
    missing = [t for t in tokens if t not in t2l]
    if missing:
        raise ValueError(f"{len(missing)} eval scenes lack a log mapping")
    logs = [t2l[t] for t in tokens]
    exposed = np.zeros((len(rls), len(tokens)), bool)
    trained_info = []
    for k, (log_path, steps) in enumerate(zip(args.train_logs, args.steps)):
        trained = trained_tokens(log_path, steps)
        tl = {t2l[t] for t in trained if t in t2l}
        if set(trained) & set(tokens):
            raise ValueError("an evaluation scene was itself trained on")
        exposed[k] = np.array([l in tl for l in logs])
        trained_info.append({"trained_scenes": len(trained), "trained_logs": len(tl), "exposed_eval_scenes": int(exposed[k].sum())})
    out = {"scenes": len(tokens), "logs": len(set(logs)), "per_seed": trained_info, "metrics": {}}
    for c in COLS:
        S = np.array([sft[t][c] for t in tokens])
        delta = np.array([[r[t][c] for t in tokens] for r in rls]) - S[None]
        ex, un = delta[exposed], delta[~exposed]
        ci, ok = boot_diff(delta, exposed, logs, rng) if exposed.any() and (~exposed).any() else (None, 0)
        out["metrics"][c] = {"pairs_exposed": int(exposed.sum()), "pairs_unexposed": int((~exposed).sum()),
                             "mean_delta_exposed_x100": float(ex.mean() * 100) if ex.size else None,
                             "mean_delta_unexposed_x100": float(un.mean() * 100) if un.size else None,
                             "exposed_minus_unexposed_x100": float((ex.mean() - un.mean()) * 100) if ex.size and un.size else None,
                             "log_bootstrap_ci95_x100": [x * 100 for x in ci] if ci else None}
    # dose: number of seeds that touched the scene's log
    dose = exposed.sum(0)
    S = np.array([sft[t]["score"] for t in tokens])
    dscore = np.array([[r[t]["score"] for t in tokens] for r in rls]).mean(0) - S
    out["by_number_of_exposing_seeds"] = {str(k): {"scenes": int((dose == k).sum()),
                                                  "mean_seed_avg_pdms_delta_x100": float(dscore[dose == k].mean() * 100)}
                                          for k in sorted(set(dose.tolist()))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
