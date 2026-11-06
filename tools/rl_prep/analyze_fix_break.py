"""What does post-training fix, and what does it break? Per-scene transitions of each safety metric.

For every trained arm and metric (NC, DAC, TTC; a scene fails when the score is below 1) against the SFT run on the
same scenes:  fixed = SFT fails and the arm passes;  broken = SFT passes and the arm fails.
  P(fix)   = fixed  / (scenes where SFT fails)
  P(break) = broken / (scenes where SFT passes)
and the net change in the number of passing scenes = fixed - broken.
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

METRICS = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "TTC": "time_to_collision_within_bound"}


def read(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["token"] in ("", "average") or r.get("valid", "True") != "True":
                continue
            rows[r["token"]] = {k: float(r[v]) for k, v in METRICS.items()}
    return rows


def arm_csv(root, suffix):
    hits = sorted(h for h in glob.glob(str(Path(root) / "metrics" / f"*_{suffix}" / "*" / "*.csv")) if "smoke" not in h)
    if len(hits) != 1:
        raise ValueError(f"{suffix}: expected one csv, found {len(hits)}")
    return read(hits[0])


def transitions(sft, arm):
    tokens = sorted(set(sft) & set(arm))
    out = {}
    for m in METRICS:
        s = np.array([sft[t][m] < 1 - 1e-6 for t in tokens])
        a = np.array([arm[t][m] < 1 - 1e-6 for t in tokens])
        out[m] = {"sft_fail": int(s.sum()), "sft_pass": int((~s).sum()),
                  "fixed": int((s & ~a).sum()), "broken": int((~s & a).sum())}
    return out, len(tokens)


def summarize(per_seed):
    res = {}
    for m in METRICS:
        fixed = np.array([p[m]["fixed"] for p in per_seed], float)
        broken = np.array([p[m]["broken"] for p in per_seed], float)
        sf = per_seed[0][m]["sft_fail"]; sp = per_seed[0][m]["sft_pass"]
        res[m] = {"sft_fail": sf, "sft_pass": sp,
                  "fixed_mean": float(fixed.mean()), "broken_mean": float(broken.mean()),
                  "P_fix": float(fixed.mean() / sf) if sf else None, "P_break": float(broken.mean() / sp) if sp else None,
                  "net_passing_scenes": float((fixed - broken).mean()),
                  "fixed_per_seed": fixed.astype(int).tolist(), "broken_per_seed": broken.astype(int).tolist()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="results dir with metrics/<run>_<arm>/…/*.csv")
    ap.add_argument("--groups", nargs="+", required=True, help="name:prefix where arms are <prefix>_seed0,1,2 (e.g. rl:baseline il:il)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    sft = arm_csv(args.root, "sft")
    out = {"root": str(args.root), "scenes_sft": len(sft), "groups": {}}
    for g in args.groups:
        name, prefix = g.split(":")
        per_seed, n = [], None
        for s in range(args.seeds):
            t, n = transitions(sft, arm_csv(args.root, f"{prefix}_seed{s}"))
            per_seed.append(t)
        out["groups"][name] = {"scenes": n, **{"summary": summarize(per_seed)}}
    for name, v in out["groups"].items():
        print(f"== {name} ({v['scenes']} scenes)")
        for m, r in v["summary"].items():
            print(f"   {m}: SFT fails {r['sft_fail']}, passes {r['sft_pass']} | fixed {r['fixed_mean']:.1f} (P_fix {r['P_fix']:.3f}) "
                  f"| broken {r['broken_mean']:.1f} (P_break {r['P_break']:.4f}) | net passing {r['net_passing_scenes']:+.1f}  "
                  f"[fixed/seed {r['fixed_per_seed']}, broken/seed {r['broken_per_seed']}]")
    if args.output:
        args.output.write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    main()
