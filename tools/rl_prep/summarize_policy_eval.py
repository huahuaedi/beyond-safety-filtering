"""Paired comparison of policies scored by the same evaluator on the same scenes, with whole-log bootstrap intervals.

Each run directory holds validation/scores.json (rows with per-scene metrics). Runs are named <family>_seed<k>;
a family is the mean over its seeds. Reported: every family against the baseline run (PDMS, components, fixed/new
failures averaged over seeds) and the differences between families.
"""
import argparse
import json
from pathlib import Path

import numpy as np

SAFETY = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")
METRICS = ("score", "ego_progress", *SAFETY)


def rows_of(run):
    data = json.loads((Path(run) / "validation" / "scores.json").read_text())
    return {r["token"]: r for r in data["rows"]}


def boot_by_log(d, logs, rng, n=4000):
    uniq = sorted(set(logs))
    idx_of = {l: i for i, l in enumerate(uniq)}
    lid = np.array([idx_of[l] for l in logs])
    sums = np.bincount(lid, weights=d, minlength=len(uniq))
    cnt = np.bincount(lid, minlength=len(uniq)).astype(float)
    idx = rng.integers(0, len(uniq), size=(n, len(uniq)))
    m = sums[idx].sum(axis=1) / cnt[idx].sum(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def family_of(label):
    return label.rsplit("_seed", 1)[0] if "_seed" in label else label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=run_dir, e.g. baseline=... marginpref_seed20=...")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--token-log-map", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    runs = {}
    for spec in args.runs:
        label, _, path = spec.partition("=")
        runs[label] = rows_of(path)
    mapping = json.loads(args.token_log_map.read_text())
    tokens = sorted(set.intersection(*[set(r) for r in runs.values()]))
    missing = [t for t in tokens if t not in mapping]
    if missing:
        raise SystemExit(f"{len(missing)} scenes have no log mapping, e.g. {missing[0]}")
    logs = [mapping[t].split("log_name:")[-1] for t in tokens]
    base = runs[args.baseline]
    fam = {}
    for label in runs:
        if label != args.baseline:
            fam.setdefault(family_of(label), []).append(label)

    def mean_metric(labels, m):
        return np.array([np.mean([runs[l][t][m] for l in labels]) for t in tokens])

    base_m = {m: np.array([base[t][m] for t in tokens]) for m in METRICS}
    out = {"scenes": len(tokens), "logs": len(set(logs)), "baseline": args.baseline,
           "baseline_means_x100": {m: float(v.mean() * 100) for m, v in base_m.items()}, "families": {}, "pairs": {}}
    fam_m = {}
    for name, labels in fam.items():
        fam_m[name] = {m: mean_metric(labels, m) for m in METRICS}
        entry = {"seeds": labels, "means_x100": {m: float(v.mean() * 100) for m, v in fam_m[name].items()}, "vs_baseline": {}}
        for m in METRICS:
            d = fam_m[name][m] - base_m[m]
            entry["vs_baseline"][m] = {"delta_x100": float(d.mean() * 100), "log_ci95_x100": [x * 100 for x in boot_by_log(d, logs, rng)]}
        entry["fix_break_mean_over_seeds"] = {}
        for m in SAFETY:
            bf = base_m[m] < 1 - 1e-6
            fixed = np.mean([int((bf & ~(np.array([runs[l][t][m] for t in tokens]) < 1 - 1e-6)).sum()) for l in labels])
            broken = np.mean([int((~bf & (np.array([runs[l][t][m] for t in tokens]) < 1 - 1e-6)).sum()) for l in labels])
            entry["fix_break_mean_over_seeds"][m] = {"baseline_failures": int(bf.sum()), "fixed": float(fixed), "new": float(broken)}
        out["families"][name] = entry
    names = sorted(fam)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out["pairs"][f"{a} minus {b}"] = {m: {
                "delta_x100": float((fam_m[a][m] - fam_m[b][m]).mean() * 100),
                "log_ci95_x100": [x * 100 for x in boot_by_log(fam_m[a][m] - fam_m[b][m], logs, rng)]} for m in METRICS}
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"scenes {out['scenes']} logs {out['logs']} baseline PDMS {out['baseline_means_x100']['score']:.2f}")
    for name, e in out["families"].items():
        v = e["vs_baseline"]
        fb = e["fix_break_mean_over_seeds"]
        print(f"{name:14s} seeds {len(e['seeds'])} PDMS {v['score']['delta_x100']:+.2f} {[round(x,2) for x in v['score']['log_ci95_x100']]} "
              f"progress {v['ego_progress']['delta_x100']:+.2f} TTC {v['time_to_collision_within_bound']['delta_x100']:+.2f} "
              f"{[round(x,2) for x in v['time_to_collision_within_bound']['log_ci95_x100']]} DAC {v['drivable_area_compliance']['delta_x100']:+.2f} "
              f"| fixed/new NC {fb['no_at_fault_collisions']['fixed']:.1f}/{fb['no_at_fault_collisions']['new']:.1f} "
              f"DAC {fb['drivable_area_compliance']['fixed']:.1f}/{fb['drivable_area_compliance']['new']:.1f} "
              f"TTC {fb['time_to_collision_within_bound']['fixed']:.1f}/{fb['time_to_collision_within_bound']['new']:.1f}")
    for k, v in out["pairs"].items():
        print(f"[pair] {k}: PDMS {v['score']['delta_x100']:+.2f} {[round(x,2) for x in v['score']['log_ci95_x100']]} "
              f"progress {v['ego_progress']['delta_x100']:+.2f} {[round(x,2) for x in v['ego_progress']['log_ci95_x100']]} "
              f"TTC {v['time_to_collision_within_bound']['delta_x100']:+.2f} {[round(x,2) for x in v['time_to_collision_within_bound']['log_ci95_x100']]}")


if __name__ == "__main__":
    main()
