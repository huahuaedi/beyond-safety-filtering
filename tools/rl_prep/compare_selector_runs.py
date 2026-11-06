"""Paired comparison of two fit_and_apply_selector outputs on the same scenes (whole-log bootstrap).

per_scene entries carry token, log and the selected PDMS (x100) of each model; the difference of run B minus run A is taken per scene.
"""
import json
import sys
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


def main():
    a, b = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
    rng = np.random.default_rng(0)
    for model in ("ridge", "hgb"):
        pa = {r["token"]: r for r in a["models"][model]["per_scene"]}
        pb = {r["token"]: r for r in b["models"][model]["per_scene"]}
        tokens = sorted(set(pa) & set(pb))
        d = np.array([pb[t]["pdms_x100"] - pa[t]["pdms_x100"] for t in tokens])
        logs = [pa[t]["log"] for t in tokens]
        lo, hi = boot_by_log(d, logs, rng)
        print(f"{model}: B - A = {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] over {len(tokens)} scenes, {len(set(logs))} logs; "
              f"B better in {int((d > 1e-9).sum())}, A better in {int((d < -1e-9).sum())}")


if __name__ == "__main__":
    main()
