"""Print the winner-rule comparison from the report.json files written by build_margin_preferences.py."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "DriveVLA-Safety-RL-offline-20260907/output"


def fmt(x, key="mean_x100"):
    return "n/a" if not x else f"{x[key]:+.2f}"


for rule in ("margin", "pdms", "nearest"):
    path = root / f"prefs_v3_{rule}" / "report.json"
    if not path.is_file():
        print(f"===== {rule}: missing {path}")
        continue
    r = json.loads(path.read_text())
    print(f"===== winner rule: {rule} | scenes with a pair: {r['scenes_with_margin_pair']}")
    for g, v in r["stratified_by_reference_safety"].items():
        dev = v["winner_deviation_from_reference_m"]
        devtxt = "n/a" if not dev else f"{dev['mean_x100']:.2f} m (median {dev['median_x100']:.2f})"
        print(f"  {g:10s} pairs {v['scenes_with_margin_pair']:4d} | winner progress {fmt(v['margin_winner_progress'])} "
              f"(median {fmt(v['margin_winner_progress'], 'median_x100')}) | winner PDMS {fmt(v['margin_winner_pdms'])} | deviation from reference output {devtxt}")
base = root / "prefs_v3_margin" / "report.json"
if base.is_file():
    r = json.loads(base.read_text())
    print("===== PDMS top-2 labels (used earlier)")
    for g, v in r["stratified_by_reference_safety"].items():
        dev = v["pdms_top2_deviation_from_reference_m"]
        devtxt = "n/a" if not dev else f"{dev['mean_x100']:.2f} m (median {dev['median_x100']:.2f})"
        print(f"  {g:10s} scenes {v['scenes_with_pdms_top2_positive']:4d} | progress {fmt(v['pdms_top2_progress'])} | PDMS {fmt(v['pdms_top2_pdms'])} | deviation {devtxt}")
