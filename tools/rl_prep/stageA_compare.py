"""Paired comparison of Stage A runs on one evaluation set (whole-log bootstrap), following docs/stageA_protocol.md.

Runs are directories <root>/<variant>_seed<k>/<set>.npz written by stageA_run.py. For each variant the per-scene selected PDMS is averaged over its
seeds. Primary comparisons c-b and c-d use 97.5% intervals (Bonferroni over the two). NC, DAC and TTC are reported separately: scenes the selected
trajectory repairs and newly fails relative to the reference probes of the same scene.
Usage: stageA_compare.py ROOT SET [--variants a b c d e] [--seeds 0 1 2]
"""
import argparse
import json
from pathlib import Path

import numpy as np

GATES = ("NC", "DAC", "TTC")


def load(root, variant, name, seeds):
    runs = []
    for s in seeds:
        p = Path(root) / f"{variant}_seed{s}" / f"{name}.npz"
        if not p.is_file():
            raise FileNotFoundError(p)
        with np.load(p, allow_pickle=False) as z:
            runs.append({k: z[k] for k in z.files})
    tokens = runs[0]["tokens"]
    if any(not np.array_equal(r["tokens"], tokens) for r in runs):
        raise ValueError(f"{variant}: seeds were evaluated on different scenes")
    mean = lambda k: np.mean([r[k].astype(float) for r in runs], axis=0)
    return {"tokens": tokens, "logs": runs[0]["logs"], "pdms": mean("selected_pdms"), "gates": mean("selected_gates"),
            "ref_pdms": runs[0]["ref_pdms"], "ref_fail": runs[0]["ref_gate_fail"], "oracle": runs[0]["oracle_pdms"],
            "per_seed_pdms": [float(r["selected_pdms"].mean() * 100) for r in runs], "runs": runs}


def boot_by_log(d, logs, rng, n=4000, level=95.0):
    uniq = sorted(set(logs.tolist()))
    pos = {l: i for i, l in enumerate(uniq)}
    lid = np.array([pos[l] for l in logs.tolist()])
    sums = np.bincount(lid, weights=d, minlength=len(uniq)); cnt = np.bincount(lid, minlength=len(uniq)).astype(float)
    pick = rng.integers(0, len(uniq), size=(n, len(uniq)))
    m = sums[pick].sum(axis=1) / cnt[pick].sum(axis=1)
    lo = (100 - level) / 2
    return [float(np.percentile(m, lo)), float(np.percentile(m, 100 - lo))]


def transitions(run):
    """Per gate, over the seeds: mean number of scenes repaired / newly failed relative to the reference probes."""
    out = []
    for g in range(3):
        fixed = np.mean([(r["ref_gate_fail"][:, g] > 0.5) & (r["selected_gates"][:, g] > 0.5) for r in run["runs"]], axis=0).sum()
        new = np.mean([(r["ref_gate_fail"][:, g] < 0.5) & (r["selected_gates"][:, g] < 0.5) for r in run["runs"]], axis=0).sum()
        out.append((float(fixed), float(new)))
    return out


def compare(a, b, rng, level):
    d = (a["pdms"] - b["pdms"]) * 100
    seed_diffs = [x - y for x, y in zip(a["per_seed_pdms"], b["per_seed_pdms"])]
    ta, tb = transitions(a), transitions(b)
    return {"mean": float(d.mean()), "ci": boot_by_log(d, a["logs"], rng, level=level), "per_seed_diff": seed_diffs,
            "net_a": [f - n for f, n in ta], "net_b": [f - n for f, n in tb]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("set")
    ap.add_argument("--variants", nargs="+", default=list("abcde"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    runs = {v: load(a.root, v, a.set, a.seeds) for v in a.variants if (a.root / f"{v}_seed{a.seeds[0]}" / f"{a.set}.npz").is_file()}
    rng = np.random.default_rng(0)
    n = len(next(iter(runs.values()))["tokens"])
    print(f"set {a.set}: {n} scenes, {len(set(next(iter(runs.values()))['logs'].tolist()))} logs")
    ref = next(iter(runs.values()))
    print(f"reference probes {ref['ref_pdms'].mean() * 100:.3f}   oracle {ref['oracle'].mean() * 100:.3f}")
    for v, r in runs.items():
        tr = transitions(r)
        print(f"  {v}: selected PDMS {r['pdms'].mean() * 100:.3f}  per seed {[round(x, 3) for x in r['per_seed_pdms']]}  "
              + "  ".join(f"{g} +{f:.0f}/-{nn:.0f}" for g, (f, nn) in zip(GATES, tr)))
    for v, r in runs.items():
        d = (r["pdms"] - r["ref_pdms"]) * 100
        print(f"{v} - reference probes: {d.mean():+.3f} {boot_by_log(d, r['logs'], rng)} (95% log interval)")
    for x, y in (("c", "b"), ("c", "d"), ("c", "a"), ("e", "c")):
        if x in runs and y in runs:
            level = 97.5 if (x, y) in (("c", "b"), ("c", "d")) else 95.0
            r = compare(runs[x], runs[y], rng, level)
            print(f"{x} - {y}: {r['mean']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] ({level}% log interval)  per-seed {[round(s, 3) for s in r['per_seed_diff']]}  "
                  f"net passes (NC, DAC, TTC) {x}: {[round(v) for v in r['net_a']]} {y}: {[round(v) for v in r['net_b']]}")


if __name__ == "__main__":
    main()
