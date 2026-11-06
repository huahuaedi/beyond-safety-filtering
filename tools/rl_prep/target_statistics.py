"""Target-side statistics of a preference pair file (CPU only): constraint margin and displacement of the winners.

For each pairs.jsonl reports, over the winner trajectories: mean/median min_clearance, the fraction with clearance <= gamma
(gamma in --gammas), mean xy displacement from the reference output (the mean of the stored reference probes), and mean progress
minus the reference's. It does NOT measure the fitting error eps_w (that needs the fitted policy's outputs on the training scenes).
Usage: python target_statistics.py --root R --geom-root G --pairs name=pairs.jsonl [name=pairs.jsonl ...] --output out.json
"""
import argparse, json
from pathlib import Path
import numpy as np
from build_margin_preferences import load_geom, read_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--geom-root", type=Path, required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="label=path/to/pairs.jsonl")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = {}
    for item in args.pairs:
        label, path = item.split("=", 1)
        clear, disp, dprog = [], [], []
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            token, split, w = r["token"], r["split"], r["winner"]
            geom = load_geom(args.geom_root / split / f"{token}.npz")
            with np.load(args.root / split / f"{token}.npz", allow_pickle=False) as z:
                absolute = z["absolute_trajectories"].astype(np.float64)
                ref_abs = z["reference_probe_absolute_trajectories"].astype(np.float64).mean(axis=0)
            _, scores, reference = read_scene(args.root / split / f"{token}.npz")
            clear.append(float(geom["min_clearance"][w]))
            disp.append(float(np.linalg.norm(absolute[w, :, :2] - ref_abs[:, :2], axis=-1).mean()))
            dprog.append(float(scores["ego_progress"][w]) - float(np.mean(reference["ego_progress"])))
        c = np.array(clear)
        out[label] = {"pairs": len(c), "clearance_mean": float(c.mean()), "clearance_median": float(np.median(c)),
                      "frac_clearance_le_gamma": {str(g): float((c <= g).mean()) for g in args.gammas},
                      "displacement_from_reference_mean_m": float(np.mean(disp)),
                      "progress_minus_reference_mean": float(np.mean(dprog)) if dprog else None}
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
