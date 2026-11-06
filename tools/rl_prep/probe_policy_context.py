"""Scene-level linear probe of the policy's pooled VLM context (extract_policy_vlm_context.py): can it predict how hard a scene is?

Targets are scene constants computed from the simulator-scored candidates:
  offroad_share   fraction of the scene's candidates that fail DAC            (the quantity the map features decide)
  ref_dac_fail    1 if either deterministic reference probe fails DAC         (classification, AUC)
  mean_progress   mean ego_progress of the candidates                          (positive control: visual input should predict it)
A ridge model is fitted on the pool scenes (penalty chosen on a held-out part of the pool) and evaluated on the application cohort; the
interval resamples whole logs. A weak result for offroad_share next to a clear one for mean_progress says the pooled context carries
scene-speed information but little about drivable-area difficulty; it does not say the information is absent from finer representations.
Usage: probe_policy_context.py --pool-contexts D --pool-root R --apply-contexts D --apply-root R --token-log-map JSON --output JSON
"""
import argparse
import json
from pathlib import Path

import numpy as np

from make_policy_visual_blocks import load_contexts, scene_files


def targets(path):
    with np.load(path, allow_pickle=False) as z:
        dac = z["score__drivable_area_compliance"].astype(float)
        prog = z["score__ego_progress"].astype(float)
        ref = z["reference_probe__drivable_area_compliance"].astype(float)
    return {"offroad_share": float((dac < 1 - 1e-6).mean()), "ref_dac_fail": float((ref < 1 - 1e-6).any()),
            "mean_progress": float(prog.mean())}


def collect(root, tokens_with_x):
    out = {}
    for p in scene_files(root):
        if p.stem in tokens_with_x:
            out[p.stem] = targets(p)
    return out


def ridge_fit(x, y, lam):
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xs = (x - mu) / sd
    ym = y.mean()
    a = xs.T @ xs + lam * np.eye(xs.shape[1])
    w = np.linalg.solve(a, xs.T @ (y - ym))
    return lambda z: ((z - mu) / sd) @ w + ym


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def boot_by_log(stat, y, p, logs, rng, n=1000):
    logs = np.asarray(logs)
    uniq = sorted(set(logs))
    idx = {l: np.flatnonzero(logs == l) for l in uniq}
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[uniq[i]] for i in pick])
        vals.append(stat(y[sel], p[sel]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-contexts", type=Path, required=True)
    ap.add_argument("--pool-root", type=Path, required=True)
    ap.add_argument("--apply-contexts", type=Path, required=True)
    ap.add_argument("--apply-root", type=Path, required=True)
    ap.add_argument("--token-log-map", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    ptok, px = load_contexts(args.pool_contexts)
    atok, ax = load_contexts(args.apply_contexts)
    pt, at = collect(args.pool_root, set(ptok)), collect(args.apply_root, set(atok))
    pi = [i for i, t in enumerate(ptok) if t in pt]
    ai = [i for i, t in enumerate(atok) if t in at]
    px, ax = px[pi], ax[ai]
    ptok, atok = [ptok[i] for i in pi], [atok[i] for i in ai]
    mapping = json.loads(args.token_log_map.read_text())
    logs = [mapping[t].split("log_name:")[-1] for t in atok]
    rng = np.random.default_rng(0)
    cut = int(0.8 * len(px))
    res = {"pool_scenes": len(px), "apply_scenes": len(ax), "apply_logs": len(set(logs)), "targets": {}}
    for name in ("offroad_share", "ref_dac_fail", "mean_progress"):
        y = np.array([pt[t][name] for t in ptok]); ya = np.array([at[t][name] for t in atok])
        best = None
        for lam in (1e1, 1e2, 1e3, 1e4, 1e5):
            f = ridge_fit(px[:cut], y[:cut], lam)
            score = r2(y[cut:], f(px[cut:]))
            if best is None or score > best[0]:
                best = (score, lam)
        f = ridge_fit(px, y, best[1])
        p = f(ax)
        if name == "ref_dac_fail":
            stat, label = (lambda a, b: roc_auc_score(a, b)), "auc"
            if len(set(ya)) < 2:
                continue
        else:
            stat, label = r2, "r2"
        res["targets"][name] = {"metric": label, "value": float(stat(ya, p)), "log_ci95": boot_by_log(stat, ya, p, logs, rng),
                                "penalty": best[1], "pool_val_r2": best[0], "apply_base_rate": float(ya.mean())}
        print(name, res["targets"][name], flush=True)
    args.output.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
