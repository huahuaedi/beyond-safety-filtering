"""Fit the trajectory+geometry selector on all simulator-scored pool scenes and apply it to a new cohort.

Typical use: the frozen navtest cohort, whose logs do not occur in the RL training pool. The cohort's
candidate collection (same sampler as the pool, scored by the same simulator) provides true scores for
every candidate, so the selection can be evaluated exactly. Reports the gain over the deterministic
reference probes of the collection and, when given, over the scenes of an independent SFT evaluation.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

import analyze_trajectory_selector as A


def scene_files(root):
    return [p for split in ("train", "val", "test") for p in sorted((Path(root) / split).glob("*.npz"))]


def boot_by_log(values, logs, rng, n=4000):
    uniq = sorted(set(logs))
    lid = np.array([uniq.index(l) for l in logs])
    sums = np.array([values[lid == i].sum() for i in range(len(uniq))])
    cnt = np.array([(lid == i).sum() for i in range(len(uniq))])
    idx = rng.integers(0, len(uniq), size=(n, len(uniq)))
    m = sums[idx].sum(axis=1) / cnt[idx].sum(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def read_sft_csv(path):
    """token -> PDMS of an independent evaluation (per-metric columns are optional extras, see read_sft_metrics)."""
    import csv
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["token"] not in ("", "average") and r.get("valid", "True") == "True":
                out[r["token"]] = float(r["score"])
    return out


def read_sft_metrics(path):
    """token -> {metric: value} for the safety metrics, when the CSV carries them."""
    import csv
    out = {}
    with open(path) as f:
        rows = csv.DictReader(f)
        for r in rows:
            if r["token"] in ("", "average") or r.get("valid", "True") != "True":
                continue
            if all(k in r and r[k] != "" for k in A.SAFETY):
                out[r["token"]] = {k: float(r[k]) for k in A.SAFETY}
    return out


def _transitions(ref_fail, pick_fail):
    ref_fail, pick_fail = np.asarray(ref_fail, bool), np.asarray(pick_fail, bool)
    fixed, broken = int((ref_fail & ~pick_fail).sum()), int((~ref_fail & pick_fail).sum())
    return {"ref_failures": int(ref_fail.sum()), "fixed": fixed, "broken": broken,
            "p_fix": fixed / max(int(ref_fail.sum()), 1), "p_break": broken / max(int((~ref_fail).sum()), 1),
            "net_passing_scenes": fixed - broken}


def pick_report(meta, tops, tokens, logs, rng, sft=None, sft_metrics=None):
    """Everything about one choice of candidate per scene: gain, component changes with log-cluster
    intervals, fix/break counts against the collection's reference probes and, when available,
    against an independent SFT evaluation of the same scenes."""
    tops = np.asarray(tops)
    gains = np.array([m["pdms"][t] - m["ref"] for m, t in zip(meta, tops)])
    out = {"gain_vs_reference_probes_x100": float(gains.mean() * 100),
           "gain_log_ci95_x100": [x * 100 for x in boot_by_log(gains, logs, rng)],
           "selected_pdms_x100": float(np.mean([m["pdms"][t] for m, t in zip(meta, tops)]) * 100)}
    comp = {}
    for k in (*A.SAFETY, "ego_progress"):
        d = np.array([m["metrics"][k][t] - m["ref_metrics"][k] for m, t in zip(meta, tops)])
        comp[k] = {"delta_x100": float(d.mean() * 100), "log_ci95_x100": [x * 100 for x in boot_by_log(d, logs, rng)]}
    out["component_vs_reference_probes"] = comp
    out["fix_break_vs_reference_probes"] = {
        k: _transitions([m["ref_fail"][k] for m in meta], [m["fail"][k][t] for m, t in zip(meta, tops)]) for k in A.SAFETY}
    if sft is not None:
        keep = [i for i, t in enumerate(tokens) if t in sft]
        if keep:
            d = np.array([meta[i]["pdms"][tops[i]] - sft[tokens[i]] for i in keep])
            out["vs_independent_sft"] = {"scenes": len(keep), "gain_x100": float(d.mean() * 100),
                                         "log_ci95_x100": [x * 100 for x in boot_by_log(d, [logs[i] for i in keep], rng)]}
            if sft_metrics:
                both = [i for i in keep if tokens[i] in sft_metrics]
                if both:
                    out["vs_independent_sft"]["fix_break"] = {
                        k: _transitions([sft_metrics[tokens[i]][k] < 1 - 1e-6 for i in both],
                                        [meta[i]["fail"][k][tops[i]] for i in both]) for k in A.SAFETY}
                    out["vs_independent_sft"]["fix_break_scenes"] = len(both)
    out["per_scene"] = [{"token": t, "log": l, "pick": int(p), "pdms_x100": float(m["pdms"][p] * 100)}
                        for t, l, p, m in zip(tokens, logs, tops, meta)]
    return out


def baseline_tops(meta):
    """Non-learned choices among the same candidates. Column 8 of the trajectory features is the mean
    deviation from the reference probes, column 0 the path-length difference."""
    tops = {"closest_to_reference": [], "longest_path": [], "geometric_veto_then_closest": [], "oracle": []}
    for m in meta:
        f, veto = m["feat"], m["veto"]
        tops["closest_to_reference"].append(int(np.argmin(f[:, 8])))
        tops["longest_path"].append(int(np.argmax(f[:, 0])))
        tops["geometric_veto_then_closest"].append(int(np.argmin(f[:, 8])) if veto.all() else int(np.argmin(np.where(veto, np.inf, f[:, 8]))))
        tops["oracle"].append(int(np.argmax(m["pdms"])))
    return tops


def apply_model(name, model, meta, tokens, logs, cols, sft, rng, sft_metrics=None):
    gains, picks, refs, to_sft = [], [], [], []
    comps = {k: [] for k in (*A.SAFETY, "ego_progress")}
    for m, t in zip(meta, tokens):
        pred = model.predict_scene(m["feat"], cols)
        top = int(np.argmax(pred))
        gains.append(m["pdms"][top] - m["ref"]); picks.append(m["pdms"][top]); refs.append(m["ref"])
        for k in comps:
            comps[k].append(m["metrics"][k][top] - m["ref_metrics"][k])
        if sft is not None and t in sft:
            to_sft.append(m["pdms"][top] - sft[t])
    gains = np.array(gains)
    res = A.evaluate(model, meta, rng, cols)
    out = {
        "scenes": len(meta),
        "reference_probe_pdms_x100": float(np.mean(refs) * 100),
        "selected_pdms_x100": float(np.mean(picks) * 100),
        "gain_vs_reference_probes_x100": float(gains.mean() * 100),
        "gain_scene_ci95_x100": [x * 100 for x in A.boot(gains, rng)],
        "gain_log_ci95_x100": [x * 100 for x in boot_by_log(gains, logs, rng)],
        "component_delta_vs_reference_x100": {k: float(np.mean(v) * 100) for k, v in comps.items()},
        "oracle_gain_x100": res["oracle"]["gain_vs_ref_x100"],
        "random_gain_x100": res["random"]["gain_vs_ref_x100"],
        "failure_rate_of_pick": res["failure_rate_of_pick"],
        "failure_rate_random": res["failure_rate_random"],
        "k_curve": res["k_curve"],
    }
    if to_sft:
        d = np.array(to_sft)
        sub_logs = [l for l, t in zip(logs, tokens) if t in sft]
        out["gain_vs_independent_sft_eval_x100"] = float(d.mean() * 100)
        out["gain_vs_independent_sft_eval_log_ci95_x100"] = [x * 100 for x in boot_by_log(d, sub_logs, rng)]
        out["independent_sft_eval_scenes"] = len(d)
    tops = [int(np.argmax(model.predict_scene(m["feat"], cols))) for m in meta]
    out["detail"] = pick_report(meta, tops, tokens, logs, rng, sft, sft_metrics)
    out["per_scene"] = [{"token": t, "log": l, "pick": p, "pdms_x100": float(m["pdms"][p] * 100),
                         "reference_probe_pdms_x100": float(m["ref"] * 100)}
                        for t, l, p, m in zip(tokens, logs, tops, meta)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", type=Path, required=True)
    ap.add_argument("--train-geom-root", type=Path, required=True)
    ap.add_argument("--apply-root", type=Path, required=True)
    ap.add_argument("--apply-geom-root", type=Path, required=True)
    ap.add_argument("--token-log-map", type=Path, required=True)
    ap.add_argument("--sft-csv", type=Path, default=None, help="per-scene scores of an independent SFT evaluation of the same cohort")
    ap.add_argument("--train-manifest", type=Path, default=None,
                    help="log-disjoint manifest: fit only on its training scenes (needed when the applied scenes come from the pool)")
    ap.add_argument("--train-extra-root", action="append", default=[], metavar="NAME=DIR",
                    help="extra per-candidate feature block of the training scenes (npz key 'features'); repeatable")
    ap.add_argument("--apply-extra-root", action="append", default=[], metavar="NAME=DIR",
                    help="the same blocks for the applied scenes; names must match --train-extra-root")
    ap.add_argument("--split-geometry", action="store_true",
                    help="also fit ablations that separate the map-derived geometry (drivable-area features) from the agent-derived "
                         "geometry (clearance, headway): traj_map, traj_agent, map, agent")
    ap.add_argument("--split-extras", action="store_true",
                    help="also fit ablations with each extra block alone / with the trajectory features (and the map features with --split-geometry)")
    ap.add_argument("--ridge-lambda", type=float, default=0.1)
    ap.add_argument("--hgb-iters", type=int, default=100)
    ap.add_argument("--model-out", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    mapping = json.loads(args.token_log_map.read_text())

    train_paths = scene_files(args.train_root)
    if args.train_manifest is not None:
        keep = {r["token"] for r in json.loads(args.train_manifest.read_text())["train"]}
        train_paths = [p for p in train_paths if p.stem in keep]
        print(f"restricted to {len(train_paths)} manifest training scenes", flush=True)
    apply_paths = scene_files(args.apply_root)
    if not train_paths or not apply_paths:
        raise ValueError("empty training or application set")
    tokens = [p.stem for p in apply_paths]
    if set(tokens) - set(mapping):
        raise ValueError(f"{len(set(tokens) - set(mapping))} applied scenes lack a log mapping")
    logs = [mapping[t].split("log_name:")[-1] for t in tokens]
    print(f"fit on {len(train_paths)} scenes, apply to {len(apply_paths)} scenes in {len(set(logs))} logs", flush=True)

    def extras_of(specs):
        out = []
        for spec in specs:
            name, _, root = spec.partition("=")
            if not root:
                raise SystemExit(f"expected NAME=DIR, got {spec!r}")
            out.append(A.Extra(name, root))
        return out
    train_extras, apply_extras = extras_of(args.train_extra_root), extras_of(args.apply_extra_root)
    if [e.name for e in train_extras] != [e.name for e in apply_extras]:
        raise SystemExit("--train-extra-root and --apply-extra-root must name the same blocks in the same order")
    x, y, g, tmeta = A.build(train_paths, args.train_geom_root, train_extras)
    cols = list(range(x.shape[1]))
    ridge = A.Ridge(args.ridge_lambda).fit(x, y)
    hgb = A.HGB(args.hgb_iters).fit_scenes(tmeta, cols)
    _, _, _, hmeta = A.build(apply_paths, args.apply_geom_root, apply_extras)
    sft = read_sft_csv(args.sft_csv) if args.sft_csv else None
    sft_metrics = read_sft_metrics(args.sft_csv) if args.sft_csv else None

    report = {"train_scenes": len(train_paths), "apply_scenes": len(apply_paths), "apply_logs": len(set(logs)),
              "candidates_per_scene": int(len(hmeta[0]["pdms"])), "models": {}}
    spans = tmeta[0]["spans"]
    sub = {"traj": list(range(*spans["traj"]))}
    if "geom" in spans:
        sub["geom"] = list(range(*spans["geom"]))
    ablations = {}
    if args.split_geometry:
        if "geom" not in spans:
            raise SystemExit("--split-geometry needs --train-geom-root features")
        g0 = spans["geom"][0]
        # geometry block layout (geometric_features.FEATURE_NAMES + 3 indicators): columns 0-7 clearance / headway to agents,
        # 8-11 off-road fractions and time of first departure from the drivable area (map only), 12-13 off-road indicators
        # (map), 14 clearance indicator (agents)
        map_cols = [g0 + i for i in (8, 9, 10, 11, 12, 13)]
        agent_cols = [g0 + i for i in (0, 1, 2, 3, 4, 5, 6, 7, 14)]
        sub.update({"map": map_cols, "agent": agent_cols,
                    "traj_map": sub["traj"] + map_cols, "traj_agent": sub["traj"] + agent_cols})
    if args.split_extras:
        # for every extra block: the block alone, trajectory + block and, with --split-geometry, trajectory + map + block
        for ex in train_extras:
            block = list(range(*spans[ex.name]))
            sub[ex.name] = block
            sub[f"traj_{ex.name}"] = sub["traj"] + block
            if args.split_geometry:
                sub[f"traj_map_{ex.name}"] = sub["traj"] + map_cols + block
    for tag, c in sub.items():                       # feature-set ablations of the same two model families
        ablations[f"ridge_{tag}_only"] = (A.Ridge(args.ridge_lambda).fit(x[:, c], y), c)
        ablations[f"hgb_{tag}_only"] = (A.HGB(args.hgb_iters).fit_scenes(tmeta, c), c)
    for name, model in (("ridge", ridge), ("hgb", hgb)):
        report["models"][name] = apply_model(name, model, hmeta, tokens, logs, cols, sft, rng, sft_metrics)
        r = report["models"][name]
        print(f"[{name}] gain vs reference probes {r['gain_vs_reference_probes_x100']:+.3f} "
              f"log CI {[round(v, 3) for v in r['gain_log_ci95_x100']]} (reference {r['reference_probe_pdms_x100']:.3f}, "
              f"selected {r['selected_pdms_x100']:.3f})", flush=True)
    report["ablations"] = {name: pick_report(hmeta, [int(np.argmax(mdl.predict_scene(m["feat"], c))) for m in hmeta],
                                             tokens, logs, rng, sft, sft_metrics)
                           for name, (mdl, c) in ablations.items()}
    rr = np.random.default_rng(args.seed + 1)
    rand = [int(rr.integers(len(m["pdms"]))) for m in hmeta]       # one uniformly random candidate per scene
    report["baselines"] = {"random_candidate": pick_report(hmeta, rand, tokens, logs, rng, sft, sft_metrics)}
    for name, t in baseline_tops(hmeta).items():
        report["baselines"][name] = pick_report(hmeta, t, tokens, logs, rng, sft, sft_metrics)
    print("[baselines] " + ", ".join(f"{k} {v['gain_vs_reference_probes_x100']:+.2f}" for k, v in report["baselines"].items()), flush=True)
    print("[ablations] " + ", ".join(f"{k} {v['gain_vs_reference_probes_x100']:+.2f}" for k, v in report["ablations"].items()), flush=True)
    report["note"] = ("Selector fitted on pool scenes only; the application cohort has no log in common with them. "
                      "Candidates are SFT samples scored by the simulator, so selection is evaluated exactly; "
                      "no new policy rollout is involved.")
    if args.model_out:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.model_out, "wb") as f:
            pickle.dump({"ridge": ridge, "hgb": hgb, "n_columns": len(cols)}, f)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
