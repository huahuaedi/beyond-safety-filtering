"""Headroom of the scored-candidate dataset (read-only, CPU, numpy only).

Question: if a perfect selector picked among the 64 verified candidates per
scene, how much PDMS / safety could it gain over the frozen-reference probe?
This bounds what any post-training that learns from these candidates can
reach, before spending GPU time. It says nothing about whether a learnable
selector exists.

PDMS follows the NAVSIM v1 scorer in this repo: NC * DAC * (5*EP + 5*TTC + 2*C) / 12.
The reference value for a scene is the mean of its reference probes.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

SAFETY = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")
EXTRA = ("ego_progress", "comfort", "driving_direction_compliance")
KS = (1, 2, 4, 8, 16, 32, 64)


def pdms(nc, dac, ep, ttc, comfort):
    return nc * dac * (5.0 * ep + 5.0 * ttc + 2.0 * comfort) / 12.0


def load_scene(path):
    with np.load(path, allow_pickle=False) as z:
        cand = {k: z["score__" + k].astype(np.float64) for k in SAFETY + EXTRA}
        ref = {k: z["reference_probe__" + k].astype(np.float64) for k in SAFETY + EXTRA}
    n = len(cand["ego_progress"])
    for k, v in {**cand, **{"ref_" + k: v for k, v in ref.items()}}.items():
        if not np.isfinite(v).all():
            raise ValueError(f"non-finite {k}: {path}")
    return cand, ref, n


def expected_best_of_k(values, k):
    """E[max of k draws without replacement] from the candidate pool."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = len(v)
    if k > n:
        k = n
    w = np.array([comb(i, k - 1) / comb(n, k) for i in range(n)], dtype=np.float64)
    return float((v * w).sum())


def scene_stats(cand, ref):
    c_pdms = pdms(cand["no_at_fault_collisions"], cand["drivable_area_compliance"],
                  cand["ego_progress"], cand["time_to_collision_within_bound"], cand["comfort"])
    ref_mean = {k: float(ref[k].mean()) for k in ref}
    r_pdms = float(pdms(ref_mean["no_at_fault_collisions"], ref_mean["drivable_area_compliance"],
                        ref_mean["ego_progress"], ref_mean["time_to_collision_within_bound"],
                        ref_mean["comfort"]))
    # lenient: not worse than the worse reference probe; strict: not worse than the better one
    lenient = np.ones(len(c_pdms), bool)
    strict = np.ones(len(c_pdms), bool)
    for k in SAFETY + ("driving_direction_compliance",):
        lenient &= cand[k] >= ref[k].min() - 1e-6
        strict &= cand[k] >= ref[k].max() - 1e-6
    fully_safe = np.ones(len(c_pdms), bool)
    for k in SAFETY + ("driving_direction_compliance",):
        fully_safe &= cand[k] >= 1 - 1e-6
    ref_failing = {k: bool(ref[k].min() < 1 - 1e-6) for k in SAFETY}
    row = {
        "ref_pdms": r_pdms,
        "cand_mean_pdms": float(c_pdms.mean()),
        "cand_max_pdms": float(c_pdms.max()),
        "best_of_k": {k: expected_best_of_k(c_pdms, k) for k in KS},
        "oracle_lenient": float(c_pdms[lenient].max()) if lenient.any() else r_pdms,
        "oracle_strict": float(c_pdms[strict].max()) if strict.any() else r_pdms,
        "any_better_than_ref": bool((c_pdms > r_pdms + 1e-6).any()),
        "ref_unsafe": bool(any(ref_failing.values())),
        "ref_failing": ref_failing,
        "fully_safe_exists": bool(fully_safe.any()),
        "frac_cand_fully_safe": float(fully_safe.mean()),
        "frac_cand_fail": {k: float((cand[k] < 1 - 1e-6).mean()) for k in SAFETY},
        "cand_spread": float(c_pdms.max() - c_pdms.min()),
    }
    return row


def bootstrap_mean(x, rng, n=2000):
    x = np.asarray(x, dtype=np.float64)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarize(rows, rng):
    n = len(rows)
    ref = np.array([r["ref_pdms"] for r in rows])
    out = {"scenes": n, "ref_pdms_x100": float(ref.mean() * 100)}
    out["cand_mean_pdms_x100"] = float(np.mean([r["cand_mean_pdms"] for r in rows]) * 100)
    out["best_of_k_gain_x100"] = {}
    for k in KS:
        d = np.array([r["best_of_k"][k] for r in rows]) - ref
        out["best_of_k_gain_x100"][str(k)] = {
            "mean": float(d.mean() * 100),
            "ci95": [x * 100 for x in bootstrap_mean(d, rng)],
        }
    for key in ("oracle_lenient", "oracle_strict"):
        d = np.array([r[key] for r in rows]) - ref
        out[key + "_gain_x100"] = {"mean": float(d.mean() * 100),
                                   "ci95": [x * 100 for x in bootstrap_mean(d, rng)]}
    # where does the oracle gain come from: scenes whose reference already fails
    # a safety metric (repair) versus scenes where it is clean (progress only)
    gain = np.array([r["best_of_k"][64] for r in rows]) - ref
    unsafe_mask = np.array([r["ref_unsafe"] for r in rows])
    out["oracle_gain_from_ref_unsafe_scenes_x100"] = float(gain[unsafe_mask].sum() / n * 100)
    out["oracle_gain_from_ref_clean_scenes_x100"] = float(gain[~unsafe_mask].sum() / n * 100)
    out["frac_scenes_any_candidate_better"] = float(np.mean([r["any_better_than_ref"] for r in rows]))
    out["frac_candidates_fully_safe"] = float(np.mean([r["frac_cand_fully_safe"] for r in rows]))
    unsafe = [r for r in rows if r["ref_unsafe"]]
    out["reference_unsafe_scenes"] = len(unsafe)
    out["reference_unsafe_frac"] = len(unsafe) / n
    out["repairable_given_unsafe"] = (
        float(np.mean([r["fully_safe_exists"] for r in unsafe])) if unsafe else None
    )
    out["ref_failure_rate"] = {
        k: float(np.mean([r["ref_failing"][k] for r in rows])) for k in SAFETY
    }
    out["cand_failure_rate"] = {
        k: float(np.mean([r["frac_cand_fail"][k] for r in rows])) for k in SAFETY
    }
    out["mean_candidate_spread_x100"] = float(np.mean([r["cand_spread"] for r in rows]) * 100)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True,
                   help="dir with train/ val/ test/ of scene .npz files")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--max-scenes", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    report = {"formula": "NC*DAC*(5*EP+5*TTC+2*C)/12", "splits": {}}
    for split in args.splits:
        paths = sorted((args.root / split).glob("*.npz"))
        if args.max_scenes and len(paths) > args.max_scenes:
            # random, not first-N: file names are hashed tokens but may not be unbiased
            keep = sorted(rng.choice(len(paths), args.max_scenes, replace=False))
            paths = [paths[i] for i in keep]
        rows = []
        for path in paths:
            cand, ref, _ = load_scene(path)
            rows.append(scene_stats(cand, ref))
        if rows:
            report["splits"][split] = summarize(rows, rng)
            print(split, json.dumps(report["splits"][split], indent=1), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
