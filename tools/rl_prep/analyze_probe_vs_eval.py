"""Where does the gap between the collection's reference probes and an independent evaluation come from?

For every scene of a collection, each reference probe has its own simulator scores. Compare their mean
with the per-scene scores of an independent evaluation CSV of the same policy: overall and per metric,
how many scenes are affected, and whether the gap sits in a few scenes (sampling luck, gate failures)
or in a uniform shift (a systematic difference between the two pipelines).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

METRICS = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound",
           "ego_progress", "comfort")


def pdms(m):
    return m["no_at_fault_collisions"] * m["drivable_area_compliance"] * (
        5 * m["ego_progress"] + 5 * m["time_to_collision_within_bound"] + 2 * m["comfort"]) / 12


def read_csv(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["token"] in ("", "average") or r.get("valid", "True") != "True":
                continue
            out[r["token"]] = {**{k: float(r[k]) for k in METRICS}, "score": float(r["score"])}
    return out


def read_probes(root):
    out = {}
    for p in sorted(Path(root).glob("*/*.npz")):
        with np.load(p, allow_pickle=False) as z:
            n = len(z["reference_probe__ego_progress"])
            rows = [{k: float(z["reference_probe__" + k][j]) for k in METRICS} for j in range(n)]
        out[p.stem] = rows
    return out


def boot(d, rng, n=3000):
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", type=Path, required=True)
    ap.add_argument("--eval-csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    probes, ev = read_probes(args.collection), read_csv(args.eval_csv)
    tokens = sorted(set(probes) & set(ev))
    if not tokens:
        raise SystemExit("no common scenes")
    out = {"scenes": len(tokens), "probes_per_scene": len(probes[tokens[0]])}
    probe_pdms = np.array([[pdms(r) for r in probes[t]] for t in tokens])       # [S, P]
    eval_pdms = np.array([ev[t]["score"] for t in tokens])
    mean_probe = probe_pdms.mean(axis=1)
    d = eval_pdms - mean_probe
    out["pdms_x100"] = {"probe_mean": float(mean_probe.mean() * 100), "eval": float(eval_pdms.mean() * 100),
                        "eval_minus_probe": float(d.mean() * 100), "scene_ci95": [x * 100 for x in boot(d, rng)]}
    if probe_pdms.shape[1] > 1:
        out["pdms_x100"]["probe_vs_probe_mean_abs_diff"] = float(np.abs(probe_pdms[:, 0] - probe_pdms[:, 1]).mean() * 100)
        out["pdms_x100"]["eval_minus_probe0"] = float((eval_pdms - probe_pdms[:, 0]).mean() * 100)
        out["pdms_x100"]["eval_minus_probe1"] = float((eval_pdms - probe_pdms[:, 1]).mean() * 100)
    out["per_metric_eval_minus_probe_x100"] = {}
    for k in METRICS:
        pm = np.array([np.mean([r[k] for r in probes[t]]) for t in tokens])
        em = np.array([ev[t][k] for t in tokens])
        out["per_metric_eval_minus_probe_x100"][k] = {
            "probe": float(pm.mean() * 100), "eval": float(em.mean() * 100), "diff": float((em - pm).mean() * 100),
            "scene_ci95": [x * 100 for x in boot(em - pm, rng)],
            "scenes_eval_gate_pass_probe_fail" if k in METRICS[:3] else "scenes_eval_higher": int(
                ((em >= 1 - 1e-6) & (pm < 1 - 1e-6)).sum() if k in METRICS[:3] else (em > pm + 1e-6).sum()),
            "scenes_probe_gate_pass_eval_fail" if k in METRICS[:3] else "scenes_probe_higher": int(
                ((pm >= 1 - 1e-6) & (em < 1 - 1e-6)).sum() if k in METRICS[:3] else (pm > em + 1e-6).sum())}
    order = np.sort(d)
    out["concentration"] = {"share_scenes_diff_gt_1pt": float((np.abs(d) > 0.01).mean()),
                            "share_scenes_identical": float((np.abs(d) < 1e-9).mean()),
                            "quantiles_x100": {q: float(np.quantile(d, q) * 100) for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
                            "gap_from_top_20_scenes_x100": float(np.sort(d)[-20:].sum() / len(d) * 100),
                            "gap_from_bottom_20_scenes_x100": float(order[:20].sum() / len(d) * 100),
                            "corr_probe_eval": float(np.corrcoef(mean_probe, eval_pdms)[0, 1])}
    text = json.dumps(out, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
