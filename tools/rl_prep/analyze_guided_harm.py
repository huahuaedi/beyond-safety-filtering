"""Candidate by candidate: what does guidance change? Same scenes, same seed, so candidate i of the guided collection descends from the
same initial noise as candidate i of the unguided one. Counts pass->fail and fail->pass transitions per criterion over candidates whose
trajectory actually changed, and the change of progress, independent of any selector."""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

CRITERIA = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unguided", type=Path, required=True)
    ap.add_argument("--guided", type=Path, required=True)
    ap.add_argument("--changed-threshold", type=float, default=0.02, help="mean abs trajectory change (m) that counts as changed")
    args = ap.parse_args()
    out = {c: {"fail_to_pass": 0, "pass_to_fail": 0} for c in CRITERIA}
    n = changed = 0
    d_progress, d_pdms = [], []
    for f in sorted(glob.glob(str(args.unguided / "*" / "*.npz"))):
        g = args.guided / Path(f).parent.name / os.path.basename(f)
        if not g.exists():
            continue
        with np.load(f, allow_pickle=False) as a, np.load(g, allow_pickle=False) as b:
            move = np.abs(a["absolute_trajectories"] - b["absolute_trajectories"]).mean(axis=(1, 2))
            mask = move > args.changed_threshold
            n += len(mask)
            changed += int(mask.sum())
            for c in CRITERIA:
                pa, pb = a["score__" + c] >= 1 - 1e-6, b["score__" + c] >= 1 - 1e-6
                out[c]["fail_to_pass"] += int((~pa & pb & mask).sum())
                out[c]["pass_to_fail"] += int((pa & ~pb & mask).sum())
            pd = lambda z: z["score__no_at_fault_collisions"] * z["score__drivable_area_compliance"] * (
                5 * z["score__ego_progress"] + 5 * z["score__time_to_collision_within_bound"] + 2 * z["score__comfort"]) / 12
            d_progress += list((b["score__ego_progress"] - a["score__ego_progress"])[mask])
            d_pdms += list((pd(b) - pd(a))[mask])
    out["candidates"] = n
    out["changed_candidates"] = changed
    out["mean_progress_change_of_changed"] = float(np.mean(d_progress)) if d_progress else None
    out["mean_pdms_change_of_changed_x100"] = float(np.mean(d_pdms) * 100) if d_pdms else None
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
