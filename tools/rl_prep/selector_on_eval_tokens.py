"""Selector gain restricted to the exact scenes used to evaluate the trained policies.

The policy arms (offline distillation, preference distillation) are evaluated on the first N entries of
manifest['heldout_eval_tokens']. Those scenes belong to the log-disjoint held-out set of the candidate data, so the
selector fitted on the training scenes can be evaluated on exactly the same scenes, and the two kinds of gain can be
put side by side on one cohort. Selector gains are relative to the deterministic reference probes of the stored
collection; the policy arms are relative to the separately evaluated SFT policy, so the two baselines differ slightly.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import analyze_trajectory_selector as A


def find(root, token):
    for split in ("train", "val", "test"):
        p = Path(root) / split / f"{token}.npz"
        if p.is_file():
            return p
    raise FileNotFoundError(token)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--geom-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--n-eval", type=int, default=600)
    ap.add_argument("--ridge-lambda", type=float, default=0.1)
    ap.add_argument("--hgb-iters", type=int, default=100)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema_version") != 2 or "heldout_eval_tokens" not in manifest:
        raise ValueError("needs a schema-2 manifest with heldout_eval_tokens")
    tokens = manifest["heldout_eval_tokens"][:args.n_eval]
    train_paths = [Path(args.root) / r.get("split", "train") / f"{r['token']}.npz" for r in manifest["train"]]
    eval_paths = [find(args.root, t) for t in tokens]
    if {p.stem for p in train_paths} & {p.stem for p in eval_paths}:
        raise ValueError("training and evaluation scenes overlap")
    x, y, g, tmeta = A.build(train_paths, args.geom_root)
    cols = list(range(x.shape[1]))
    models = {"ridge": A.Ridge(args.ridge_lambda).fit(x, y), "hgb": A.HGB(args.hgb_iters).fit_scenes(tmeta, cols)}
    _, _, _, hmeta = A.build(eval_paths, args.geom_root)
    out = {"eval_scenes": len(eval_paths), "train_scenes": len(train_paths),
           "reference_probe_pdms_x100": float(np.mean([m["ref"] for m in hmeta]) * 100), "models": {}}
    for name, model in models.items():
        r = A.evaluate(model, hmeta, rng, cols)
        out["models"][name] = {
            "gain_vs_reference_probes_x100": r["selector"]["gain_vs_ref_x100"],
            "ci95_x100": r["selector"]["ci95_x100"],
            "random_gain_x100": r["random"]["gain_vs_ref_x100"], "oracle_gain_x100": r["oracle"]["gain_vs_ref_x100"],
            "component_delta_vs_reference_x100": r["pick_metric_delta_vs_ref_x100"],
            "k_curve": {K: round(c["selector_gain_x100"], 3) for K, c in r["k_curve"].items()}}
        m = out["models"][name]
        print(f"[{name}] on {len(eval_paths)} policy-evaluation scenes: gain {m['gain_vs_reference_probes_x100']:+.3f} "
              f"{[round(v, 3) for v in m['ci95_x100']]} (oracle {m['oracle_gain_x100']:+.2f}, random {m['random_gain_x100']:+.2f}) "
              f"components {({k: round(v, 3) for k, v in m['component_delta_vs_reference_x100'].items()})}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print("reference-probe PDMS on these scenes (x100):", round(out["reference_probe_pdms_x100"], 3), "| wrote", args.output)


if __name__ == "__main__":
    main()
