"""Exploratory: use the guided candidate set only as a fallback.

For each scene, take the selector's pick from the unguided collection; if that pick violates the geometric checks (negative clearance or any
footprint point off the drivable area, computed from its geometric features), take the guided collection's pick instead. Both collections must
share scenes and sampling seed. The rule is fixed in advance and nothing is tuned.
Inputs: the two fit_and_apply_selector outputs (per-scene picks) and the geometric-feature roots of the two collections.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def boot_by_log(d, logs, rng, n=4000):
    uniq = sorted(set(logs))
    idx_of = {l: i for i, l in enumerate(uniq)}
    lid = np.array([idx_of[l] for l in logs])
    sums = np.bincount(lid, weights=d, minlength=len(uniq))
    cnt = np.bincount(lid, minlength=len(uniq)).astype(float)
    pick = rng.integers(0, len(uniq), size=(n, len(uniq)))
    m = sums[pick].sum(axis=1) / cnt[pick].sum(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def violates(geom_root, token, pick):
    for split in ("train", "val", "test"):
        f = Path(geom_root) / split / f"{token}.npz"
        if f.exists():
            g = np.load(f, allow_pickle=False)["geom_features"][pick]
            return bool(g[8] > 0 or g[0] < 0)
    raise FileNotFoundError(f"no geometric features for {token} under {geom_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unguided-result", type=Path, required=True)
    ap.add_argument("--guided-result", type=Path, required=True)
    ap.add_argument("--unguided-geom", type=Path, required=True)
    ap.add_argument("--model", default="hgb")
    args = ap.parse_args()
    a = {r["token"]: r for r in json.loads(args.unguided_result.read_text())["models"][args.model]["per_scene"]}
    b = {r["token"]: r for r in json.loads(args.guided_result.read_text())["models"][args.model]["per_scene"]}
    tokens = sorted(set(a) & set(b))
    delta, switched, logs = [], 0, []
    for t in tokens:
        use_guided = violates(args.unguided_geom, t, a[t]["pick"])
        switched += int(use_guided)
        delta.append((b[t]["pdms_x100"] - a[t]["pdms_x100"]) if use_guided else 0.0)
        logs.append(a[t]["log"])
    d = np.array(delta)
    lo, hi = boot_by_log(d, logs, np.random.default_rng(0))
    print(f"{args.model}: fallback - unguided = {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] over {len(tokens)} scenes, {len(set(logs))} logs; "
          f"switched {switched} scenes; better in {int((d > 1e-9).sum())}, worse in {int((d < -1e-9).sum())}")


if __name__ == "__main__":
    main()
