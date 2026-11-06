"""Safety-margin preference pairs from the simulator-scored candidate set, and a drift check.

Construction follows the recipe reported for DriveReferee (arXiv:2609.22762): within a scene the winner is the
candidate that passes all safety checks with the largest safety margin, the loser is the violating candidate closest
to the expert action. Here "pass" is decided by the simulator scores (NC, DAC, TTC all 1) or, with --pass-rule
geometric, by the non-learned geometric features alone (no off-area departure, positive clearance); the margin is the
minimum clearance to observed agents.

The point of this script is a data-level check before any training: do margin-selected winners move the policy toward
higher progress than the reference output (as the PDMS-ranked labels used earlier do)? It reports progress and PDMS of
the winners relative to the deterministic reference probes, next to the earlier top-2 PDMS labels, on the same scenes.
"""
import hashlib
import argparse
import json
from pathlib import Path

import numpy as np

from offline_verified_data import read_scene, normalize_action, topk_safe_labels, POSITIVE, SAFETY, pdms

EPS = 1e-6


def load_geom(path):
    with np.load(path, allow_pickle=False) as z:
        names = [str(n) for n in z["feature_names"]]
        return {n: z["geom_features"][:, i].astype(np.float64) for i, n in enumerate(names)}


def passing_mask(scores, geom, rule):
    if rule == "simulator":
        ok = np.ones(len(scores["ego_progress"]), bool)
        for k in SAFETY[:3]:
            ok &= np.asarray(scores[k]) >= 1 - EPS
        return ok
    return (geom["offroad_frac_max"] <= 0) & (geom["t_clear_lt_0m"] <= 0)


def scene_pair(relative, scores, reference, geom, expert_norm, low, high, rule, margin="min_clearance", winner_rule="margin",
               absolute=None, ref_abs=None, random_seed=0):
    """Return (winner, loser) candidate indices or None.

    winner_rule="margin": passing candidate with the largest safety margin (DriveReferee-style).
    winner_rule="pdms":   passing candidate with the highest PDMS. The loser is chosen identically in all cases, so
                          the pair files differ only in the winner, which isolates the effect of the label type.
    winner_rule="nearest": passing candidate closest (mean xy distance) to the deterministic reference output, i.e.
                          the minimal change from the policy's own output that passes. Needs absolute and ref_abs.
    """
    ok = passing_mask(scores, geom, rule)
    if not ok.any() or ok.all():
        return None
    if winner_rule == "margin":
        m = geom[margin]
    elif winner_rule == "pdms":
        m = np.array([cand_pdms(scores, i) for i in range(len(ok))])
    elif winner_rule == "nearest":
        if absolute is None or ref_abs is None:
            raise ValueError("nearest rule needs absolute trajectories and the reference output")
        m = -np.linalg.norm(absolute[..., :2] - ref_abs[None, :, :2], axis=-1).mean(axis=1)
    elif winner_rule == "random":
        m = np.zeros(len(ok))
        win = int(np.random.default_rng(random_seed).choice(np.flatnonzero(ok)))
        m[win] = 1.0
    elif winner_rule == "pdms_capped":
        # Mechanism ablation: highest PDMS among passing candidates whose progress does not exceed the reference's mean progress.
        # Removes the progress preference that Proposition 1 attributes to the PDMS rule; scenes without such a candidate are skipped.
        ref_ep = float(np.mean(reference["ego_progress"]))
        capped = np.array([float(scores["ego_progress"][i]) <= ref_ep for i in range(len(ok))])
        ok_w = ok & capped
        if not ok_w.any():
            return None
        m = np.array([cand_pdms(scores, i) for i in range(len(ok))])
        win = int(np.flatnonzero(ok_w)[np.argmax(m[ok_w])])
        cand_norm = normalize_action(relative, low, high)
        dist = np.linalg.norm((cand_norm - expert_norm[None]).reshape(len(relative), -1), axis=1)
        bad = np.flatnonzero(~ok)
        return win, int(bad[np.argmin(dist[bad])])
    else:
        raise ValueError(f"unknown winner rule {winner_rule}")
    win = int(np.flatnonzero(ok)[np.argmax(m[ok])])
    cand_norm = normalize_action(relative, low, high)
    dist = np.linalg.norm((cand_norm - expert_norm[None]).reshape(len(relative), -1), axis=1)
    bad = np.flatnonzero(~ok)
    lose = int(bad[np.argmin(dist[bad])])
    return win, lose


def ref_pdms(reference):
    r = {k: float(np.mean(v)) for k, v in reference.items()}
    return pdms(r["no_at_fault_collisions"], r["drivable_area_compliance"], r["ego_progress"],
                r["time_to_collision_within_bound"], r["comfort"])


def cand_pdms(scores, i):
    return float(pdms(scores["no_at_fault_collisions"][i], scores["drivable_area_compliance"][i],
                      scores["ego_progress"][i], scores["time_to_collision_within_bound"][i], scores["comfort"][i]))


def summarize(x, scale=100.0):
    """Summary statistics; keys keep the historical *_x100 names (scale 100 for scores, 1 for metres)."""
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return None
    return {"n": int(len(x)), "mean_x100": float(x.mean() * scale), "median_x100": float(np.median(x) * scale),
            "frac_positive": float((x > 1e-9).mean()), "p10_x100": float(np.percentile(x, 10) * scale),
            "p90_x100": float(np.percentile(x, 90) * scale)}


def deviation_m(absolute, ref_abs, idx):
    """Mean xy distance in metres between candidate(s) idx and the reference output."""
    idx = np.atleast_1d(idx)
    return float(np.mean(np.linalg.norm(absolute[idx][..., :2] - ref_abs[None, :, :2], axis=-1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--geom-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True, help="log-disjoint schema-2 manifest")
    ap.add_argument("--norm-stats", type=Path, required=True, help="normalizer json with norm_stats.libero.q01/q99")
    ap.add_argument("--pass-rule", choices=["simulator", "geometric"], default="simulator")
    ap.add_argument("--winner-rule", choices=["margin", "pdms", "nearest", "pdms_capped", "random"], default="margin")
    ap.add_argument("--selection-seed", type=int, default=0, help="Fixed random-winner seed, independent of training seeds")
    ap.add_argument("--restrict-to", type=Path, default=None, help="pairs.jsonl; only scenes (tokens) present there are used")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("needs a log-disjoint (schema 2) manifest")
    stats = json.loads(args.norm_stats.read_text())["norm_stats"]["libero"]
    low, high = np.asarray(stats["q01"], np.float32), np.asarray(stats["q99"], np.float32)

    keep = None
    if args.restrict_to is not None:
        keep = {json.loads(l)["token"] for l in args.restrict_to.read_text().splitlines() if l.strip()}
    pairs = []
    rows = {"margin": {"progress": [], "pdms": [], "deviation": []},
            "pdms_top2": {"progress": [], "pdms": [], "deviation": []},
            "loser": {"progress": [], "pdms": []}}
    strat = {g: {"margin_progress": [], "margin_pdms": [], "top2_progress": [], "top2_pdms": [], "scenes": 0, "pairs": 0, "top2_scenes": 0,
                 "margin_dev": [], "top2_dev": []} for g in ("ref_safe", "ref_unsafe")}
    scenes = with_pair = with_pdms_label = 0
    for rec in manifest["train"]:
        token, split = rec["token"], rec.get("split", "train")
        if keep is not None and token not in keep:
            continue
        relative, scores, reference = read_scene(args.root / split / f"{token}.npz")
        with np.load(args.root / split / f"{token}.npz", allow_pickle=False) as z:
            expert = z["expert_action_normalized"].astype(np.float32)
            absolute = z["absolute_trajectories"].astype(np.float64)
            ref_abs = z["reference_probe_absolute_trajectories"].astype(np.float64).mean(axis=0)
        geom = load_geom(args.geom_root / split / f"{token}.npz")
        scenes += 1
        ref_prog = float(np.mean(reference["ego_progress"]))
        r_pdms = ref_pdms(reference)
        group = "ref_unsafe" if any(np.min(reference[k]) < 1 - EPS for k in SAFETY[:3] if k in reference) else "ref_safe"
        strat[group]["scenes"] += 1
        pair = scene_pair(relative, scores, reference, geom, expert, low, high, args.pass_rule, winner_rule=args.winner_rule,
                          absolute=absolute, ref_abs=ref_abs,
                          random_seed=int.from_bytes(hashlib.sha256(f"random-winner:{args.selection_seed}:{token}".encode()).digest()[:8], "little"))
        if pair is not None:
            w, l = pair
            with_pair += 1
            pairs.append({"token": token, "split": split, "winner": w, "loser": l})
            rows["margin"]["progress"].append(float(scores["ego_progress"][w]) - ref_prog)
            rows["margin"]["pdms"].append(cand_pdms(scores, w) - r_pdms)
            strat[group]["pairs"] += 1
            strat[group]["margin_progress"].append(float(scores["ego_progress"][w]) - ref_prog)
            strat[group]["margin_pdms"].append(cand_pdms(scores, w) - r_pdms)
            strat[group]["margin_dev"].append(deviation_m(absolute, ref_abs, w))
            rows["loser"]["progress"].append(float(scores["ego_progress"][l]) - ref_prog)
            rows["loser"]["pdms"].append(cand_pdms(scores, l) - r_pdms)
        labels = topk_safe_labels(scores, reference, k=args.topk)["labels"]
        pos = np.flatnonzero(labels == POSITIVE)
        if len(pos):
            with_pdms_label += 1
            strat[group]["top2_scenes"] += 1
            strat[group]["top2_progress"].append(float(np.mean([scores["ego_progress"][i] for i in pos])) - ref_prog)
            strat[group]["top2_pdms"].append(float(np.mean([cand_pdms(scores, i) for i in pos])) - r_pdms)
            strat[group]["top2_dev"].append(deviation_m(absolute, ref_abs, pos))
            rows["pdms_top2"]["progress"].append(float(np.mean([scores["ego_progress"][i] for i in pos])) - ref_prog)
            rows["pdms_top2"]["pdms"].append(float(np.mean([cand_pdms(scores, i) for i in pos])) - r_pdms)

    report = {"selection_seed": args.selection_seed, "pass_rule": args.pass_rule, "winner_rule": args.winner_rule, "scenes": scenes, "scenes_with_margin_pair": with_pair,
              "scenes_with_pdms_top2_positive": with_pdms_label,
              "winner_minus_reference": {"progress": summarize(rows["margin"]["progress"]), "pdms": summarize(rows["margin"]["pdms"])},
              "loser_minus_reference": {"progress": summarize(rows["loser"]["progress"]), "pdms": summarize(rows["loser"]["pdms"])},
              "pdms_top2_positives_minus_reference": {"progress": summarize(rows["pdms_top2"]["progress"]),
                                                      "pdms": summarize(rows["pdms_top2"]["pdms"])},
              "stratified_by_reference_safety": {
                  g: {"scenes": v["scenes"], "scenes_with_margin_pair": v["pairs"], "scenes_with_pdms_top2_positive": v["top2_scenes"],
                      "margin_winner_progress": summarize(v["margin_progress"]), "margin_winner_pdms": summarize(v["margin_pdms"]),
                      "pdms_top2_progress": summarize(v["top2_progress"]), "pdms_top2_pdms": summarize(v["top2_pdms"]),
                      "winner_deviation_from_reference_m": summarize(v["margin_dev"], scale=1.0),
                      "pdms_top2_deviation_from_reference_m": summarize(v["top2_dev"], scale=1.0)}
                  for g, v in strat.items()},
              "reading": ("If the margin winners' progress difference is near zero or negative while the PDMS top-2 positives "
                          "are clearly positive, margin labels do not push toward faster driving at the data level.")}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
