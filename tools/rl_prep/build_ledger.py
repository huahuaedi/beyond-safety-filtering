"""One ledger of what each way of using the candidates fixes and breaks, on the same scenes and against the same baseline.

Rows come from three kinds of sources:
  --group  "label|baseline_csv_glob|arm_csv_glob"   per-scene NAVSIM CSVs of trained policies (mean over the matched arms/seeds)
  --selector "label=path.json:model"               fit_and_apply_selector output (selected candidate vs the independent SFT evaluation)
  --policy-eval "label=path.json:family"           summarize_policy_eval output (policy scored by the training script's evaluator)
For every row: change of PDMS where it can be computed, and for NC / DAC / TTC the scenes fixed (baseline fails, row passes) and newly
failed (baseline passes, row fails), averaged over seeds, plus the net change of passing scenes summed over the three criteria.
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

CRIT = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")
SHORT = ("NC", "DAC", "TTC")


def read(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["token"] in ("", "average") or r.get("valid", "True") != "True":
                continue
            rows[r["token"]] = {**{c: float(r[c]) for c in CRIT}, "score": float(r["score"])}
    return rows


def pick_csvs(pattern):
    return sorted(p for p in glob.glob(pattern) if "smoke" not in p)


def group_row(spec):
    label, base_glob, arm_glob = spec.split("|")
    base_paths, arm_paths = pick_csvs(base_glob), pick_csvs(arm_glob)
    if len(base_paths) != 1 or not arm_paths:
        raise SystemExit(f"{label}: expected one baseline csv and at least one arm csv, found {len(base_paths)} and {len(arm_paths)}")
    base = read(base_paths[0])
    fixed = {c: [] for c in CRIT}
    new = {c: [] for c in CRIT}
    dpdms = []
    for path in arm_paths:
        arm = read(path)
        tokens = sorted(set(base) & set(arm))
        for c in CRIT:
            b = np.array([base[t][c] < 1 - 1e-6 for t in tokens])
            a = np.array([arm[t][c] < 1 - 1e-6 for t in tokens])
            fixed[c].append(int((b & ~a).sum()))
            new[c].append(int((~b & a).sum()))
        dpdms.append(float(np.mean([arm[t]["score"] - base[t]["score"] for t in tokens]) * 100))
    return {"label": label, "runs": len(arm_paths), "scenes": len(tokens), "delta_pdms": float(np.mean(dpdms)),
            "fixed": {c: float(np.mean(v)) for c, v in fixed.items()}, "new": {c: float(np.mean(v)) for c, v in new.items()}}


def selector_row(spec):
    label, _, rest = spec.partition("=")
    path, _, model = rest.rpartition(":")
    m = json.loads(Path(path).read_text())["models"][model]["detail"]["vs_independent_sft"]
    fb = m["fix_break"]
    return {"label": label, "runs": 1, "scenes": m["scenes"], "delta_pdms": m["gain_x100"],
            "fixed": {c: float(fb[c]["fixed"]) for c in CRIT}, "new": {c: float(fb[c]["broken"]) for c in CRIT}}


def policy_row(spec):
    label, _, rest = spec.partition("=")
    path, _, family = rest.rpartition(":")
    r = json.loads(Path(path).read_text())
    f = r["families"][family]
    fb = f["fix_break_mean_over_seeds"]
    return {"label": label, "runs": len(f["seeds"]), "scenes": r["scenes"], "delta_pdms": f["vs_baseline"]["score"]["delta_x100"],
            "fixed": {c: float(fb[c]["fixed"]) for c in CRIT}, "new": {c: float(fb[c]["new"]) for c in CRIT}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", default=[])
    ap.add_argument("--selector", action="append", default=[])
    ap.add_argument("--policy-eval", action="append", default=[])
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    rows = [group_row(s) for s in args.group] + [selector_row(s) for s in args.selector] + [policy_row(s) for s in args.policy_eval]
    print(f"{'method':46s} {'runs':>4s} {'dPDMS':>7s} | " + " ".join(f"{n:>9s}" for n in SHORT) + " | net passing (NC+DAC+TTC)")
    for r in rows:
        cells = " ".join(f"{r['fixed'][c]:4.1f}/{r['new'][c]:<4.1f}" for c in CRIT)
        net = sum(r["fixed"][c] - r["new"][c] for c in CRIT)
        r["net_passing"] = net
        print(f"{r['label']:46s} {r['runs']:4d} {r['delta_pdms']:+7.2f} | {cells} | {net:+6.1f}")
    print("cells are scenes fixed / newly failed (means over runs); the baseline of each row is stated in its source")
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
