"""Pick, for every Stage A variant, the (lr, dropout) with the highest mean selected PDMS on the `tune` subset, from the runs written by
submit_stageA.slurm in tune mode (<root>/<variant>_lr<lr>_dr<dropout>/tune.npz). The `decision` subset is never read here.
Usage: stageA_pick_config.py TUNE_ROOT OUTPUT_JSON
"""
import json
import re
import sys
from pathlib import Path

import numpy as np


def pick(root):
    table = {}
    for d in sorted(Path(root).glob("*_lr*_dr*")):
        m = re.fullmatch(r"([a-e])_lr([0-9.e-]+)_dr([0-9.]+)", d.name)
        f = d / "tune.npz"
        if not m or not f.is_file():
            continue
        with np.load(f, allow_pickle=False) as z:
            score = float(z["selected_pdms"].mean() * 100)
        table.setdefault(m.group(1), []).append({"lr": float(m.group(2)), "dropout": float(m.group(3)), "tune_pdms": score, "dir": d.name})
    chosen = {v: max(rows, key=lambda r: (r["tune_pdms"], -r["lr"])) for v, rows in table.items()}
    return table, chosen


if __name__ == "__main__":
    table, chosen = pick(sys.argv[1])
    for v, rows in sorted(table.items()):
        print(v, [(r["lr"], r["dropout"], round(r["tune_pdms"], 3)) for r in rows], "->", (chosen[v]["lr"], chosen[v]["dropout"]))
    Path(sys.argv[2]).write_text(json.dumps({v: {"lr": r["lr"], "dropout": r["dropout"], "tune_pdms": r["tune_pdms"]} for v, r in chosen.items()}, indent=1) + "\n")
