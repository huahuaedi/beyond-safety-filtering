"""How much room is there for guidance beyond selection? Uses stored, simulator-scored candidate collections only.

For each criterion (no-at-fault collision, drivable area, time to collision) over the scenes of a collection:
  - share of candidates that fail,
  - scenes whose deployed reference fails the criterion, split into those where at least one candidate passes (selection can repair)
    and those where no candidate passes (only a change of the candidates, e.g. guidance, could repair),
  - scenes where a reference passes but every candidate fails (nothing to select).
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

CRITERIA = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    files = sorted(glob.glob(str(args.collection / "*" / "*.npz")))
    if not files:
        raise SystemExit("no scenes")
    out = {"scenes": len(files)}
    stats = {c: {"cand_fail": [], "ref_fail": 0, "ref_fail_repairable": 0, "ref_fail_unrepairable": 0,
                 "ref_ok_all_cand_fail": 0, "scenes_all_cand_fail": 0} for c in CRITERIA}
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            for c in CRITERIA:
                cand = z["score__" + c] < 1 - 1e-6
                ref = bool((z["reference_probe__" + c] < 1 - 1e-6).any())
                s = stats[c]
                s["cand_fail"].append(float(cand.mean()))
                s["scenes_all_cand_fail"] += int(cand.all())
                if ref:
                    s["ref_fail"] += 1
                    s["ref_fail_repairable" if (~cand).any() else "ref_fail_unrepairable"] += 1
                elif cand.all():
                    s["ref_ok_all_cand_fail"] += 1
    for c, s in stats.items():
        s["candidate_failure_rate"] = float(np.mean(s.pop("cand_fail")))
        out[c] = s
    text = json.dumps(out, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
