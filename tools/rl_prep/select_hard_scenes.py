"""Scenes whose failures selection can hardly repair: the deployed reference fails a criterion and at most ``--max-pass`` of the
stored candidates pass it. Used as a targeted test set for guidance (no learning is involved)."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

CRITERIA = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="collection root with train/val/test npz files")
    ap.add_argument("--max-pass", type=int, default=2)
    ap.add_argument("--exclude-tokens-json", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    excluded = set(json.loads(args.exclude_tokens_json.read_text())) if args.exclude_tokens_json else set()
    hard, counts = {}, {c: 0 for c in CRITERIA}
    files = sorted(glob.glob(str(args.root / "*" / "*.npz")))
    for f in files:
        token = Path(f).stem
        if token in excluded:
            continue
        with np.load(f, allow_pickle=False) as z:
            for c in CRITERIA:
                n_pass = int((z["score__" + c] >= 1 - 1e-6).sum())
                if bool((z["reference_probe__" + c] < 1 - 1e-6).any()) and n_pass <= args.max_pass:
                    hard.setdefault(token, []).append(c)
                    counts[c] += 1
    args.output.write_text(json.dumps(sorted(hard)))
    print(json.dumps({"scenes_scanned": len(files), "hard_scenes": len(hard), "per_criterion": counts, "max_pass": args.max_pass}, indent=1))


if __name__ == "__main__":
    main()
