"""Paired comparison of two choices stored in ONE fit_and_apply_selector output (models, ablations or baselines).

Names are written section.name, for example models.hgb, ablations.hgb_geom_only, baselines.geometric_veto_then_closest.
Prints B - A on the selected PDMS (x100) with a whole-log bootstrap interval.
"""
import json
import sys
from pathlib import Path

import numpy as np

from compare_selector_runs import boot_by_log


def entry(report, name):
    section, _, key = name.partition(".")
    return report[section][key]


def main():
    report = json.loads(Path(sys.argv[1]).read_text())
    rng = np.random.default_rng(0)
    for spec in sys.argv[2:]:
        a_name, _, b_name = spec.partition(":")
        pa = {r["token"]: r for r in (entry(report, a_name).get("per_scene") or [])}
        pb = {r["token"]: r for r in (entry(report, b_name).get("per_scene") or [])}
        if not pa or not pb:
            print(f"{spec}: per-scene picks missing (rerun fit_and_apply_selector.py with the current version)")
            continue
        tokens = sorted(set(pa) & set(pb))
        d = np.array([pb[t]["pdms_x100"] - pa[t]["pdms_x100"] for t in tokens])
        logs = [pa[t]["log"] for t in tokens]
        lo, hi = boot_by_log(d, logs, rng)
        print(f"{b_name} - {a_name} = {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  ({len(tokens)} scenes, {len(set(logs))} logs; B better {int((d > 1e-9).sum())}, A better {int((d < -1e-9).sum())})")


if __name__ == "__main__":
    main()
