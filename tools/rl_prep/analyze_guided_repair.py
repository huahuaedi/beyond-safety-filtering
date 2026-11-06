"""Does guidance repair scenes that sampling alone could not? Compares guided collections (same seed, same scenes) with lambda = 0.

Per criterion: share of candidates failing, scenes with at least one passing candidate, scenes repaired / lost relative to lambda = 0,
and the change of the best-of-K PDMS and of the mean candidate PDMS.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

CRITERIA = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")


def pdms(z):
    return z["score__no_at_fault_collisions"] * z["score__drivable_area_compliance"] * (
        5 * z["score__ego_progress"] + 5 * z["score__time_to_collision_within_bound"] + 2 * z["score__comfort"]) / 12


def load(root):
    out = {}
    for f in glob.glob(str(Path(root) / "collection" / "*" / "*.npz")):
        with np.load(f, allow_pickle=False) as z:
            out[os.path.basename(f)] = {k: z[k] for k in z.files if k.startswith("score__")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=dir, the first one is the unguided control")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    runs = {}
    for spec in args.runs:
        label, _, path = spec.partition("=")
        runs[label] = load(path)
    labels = list(runs)
    base = runs[labels[0]]
    tokens = sorted(set.intersection(*[set(r) for r in runs.values()]))
    out = {"scenes": len(tokens), "control": labels[0], "runs": {}}
    for label in labels:
        r = runs[label]
        entry = {"mean_candidate_pdms_x100": float(np.mean([pdms(r[t]).mean() for t in tokens]) * 100),
                 "best_of_k_pdms_x100": float(np.mean([pdms(r[t]).max() for t in tokens]) * 100),
                 "mean_candidate_progress": float(np.mean([r[t]["score__ego_progress"].mean() for t in tokens]))}
        for c in CRITERIA:
            has = np.array([(r[t]["score__" + c] >= 1 - 1e-6).any() for t in tokens])
            has0 = np.array([(base[t]["score__" + c] >= 1 - 1e-6).any() for t in tokens])
            entry[c] = {"candidate_failure_rate": float(np.mean([(r[t]["score__" + c] < 1 - 1e-6).mean() for t in tokens])),
                        "scenes_with_a_passing_candidate": int(has.sum()),
                        "repaired_vs_control": int((has & ~has0).sum()), "lost_vs_control": int((~has & has0).sum())}
        out["runs"][label] = entry
    text = json.dumps(out, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
