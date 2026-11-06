"""Descriptive repair/regression stratification by empirical candidate pass rate.

Target reweighting gains and single-output policy changes use different sampling
protocols. Do not divide them or interpret bin associations as causal effects.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import budgeted_awfm as B
from awfm_statistics import conversions, log_labels, bootstrap
from build_ledger import CRIT, read, pick_csvs
from offline_verified_data import read_scene


def find_scene(root, token):
    for split in ("train", "val", "test"):
        path = Path(root) / split / f"{token}.npz"
        if path.exists():
            return path
    raise FileNotFoundError(f"no candidate pool for {token} under {root}")


def flips(base, arm, tokens):
    return conversions(base, arm, tokens)


def analyse(root, base_csv, arm_csvs, eps_list, token_log_map=None, metric_cache_dir=None):
    base = read(base_csv)
    tokens = sorted(base)
    if not tokens: raise ValueError("empty baseline")
    logs = log_labels(tokens, token_log_map, metric_cache_dir)
    rng = np.random.default_rng(0)
    pool = {}
    for t in tokens:
        _, scores, _ = read_scene(find_scene(root, t))
        pool[t] = scores
    p0 = np.array([B.gate_pass(pool[t]).mean() for t in tokens])
    out = {"scenes": len(tokens), "share_with_failing_candidate": float((p0 < 1).mean()),
           "share_without_any_passer": float((p0 == 0).mean()),
           "share_partial": float(((p0 > 0) & (p0 < 1)).mean()), "target_pool_gain": {}, "arms": {},
           "schema_version": 2, "bootstrap_unit": "log" if logs is not None else "scene",
           "interpretation": "Descriptive associations; no transfer-efficiency or causal claim. Candidate reweighting is not policy KL."}
    gains = {}
    for eps in eps_list:
        gains[eps] = np.array([B.promised_pass(pool[t], eps)[1] - B.promised_pass(pool[t], eps)[0] for t in tokens])
        out["target_pool_gain"][str(eps)] = float(gains[eps].mean())
    for label, path in arm_csvs.items():
        arm = read(path)
        if set(arm) != set(base):
            raise ValueError(f"{label}: baseline and arm scene sets differ")
        common = tokens
        idx = np.arange(len(tokens))
        fix, new = flips(base, arm, common)
        out["arms"][label] = {"scenes": len(common), "fixed": float(fix.sum()), "new": float(new.sum()),
                              "joint_pass_delta": float((fix-new).mean()),
                              "joint_pass_delta_ci95": bootstrap(fix-new, rng, logs),
                              "component_conversions": {c: {"fixed": float(conversions(base,arm,common,(c,))[0].sum()),
                                  "new": float(conversions(base,arm,common,(c,))[1].sum())} for c in CRIT},
                              "by_pool_pass_rate": {str(eps): B.transfer_table(p0[idx], gains[eps][idx], fix, new)
                                                    for eps in eps_list}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-root", type=Path, required=True)
    ap.add_argument("--baseline-csv", required=True, help="glob of the independent SFT evaluation csv (exactly one)")
    ap.add_argument("--arm", action="append", default=[], help="label=glob of a trained policy's per-scene csv")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    ap.add_argument("--token-log-map", type=Path, help="JSON token -> full log ID")
    ap.add_argument("--metric-cache-dir", type=Path, help="derive log IDs from cache layout")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    base = pick_csvs(args.baseline_csv)
    if len(base) != 1:
        raise SystemExit(f"expected one baseline csv, found {len(base)}")
    arms = {}
    for spec in args.arm:
        label, _, pattern = spec.partition("=")
        found = pick_csvs(pattern)
        if len(found) != 1:
            raise SystemExit(f"{label}: expected one csv, found {len(found)}")
        arms[label] = found[0]
    result = analyse(args.candidate_root, base[0], arms, args.eps, args.token_log_map, args.metric_cache_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}, indent=2))
    for label, arm in result["arms"].items():
        print(f"\n{label}: fixed={arm['fixed']:.0f} new={arm['new']:.0f}")
        for row in arm["by_pool_pass_rate"][str(args.eps[0])]:
            if row["n"]:
                print(f"  p in [{row['p_lo']:.2f},{row['p_hi']:.2f}) n={row['n']:4d} fixed={row['fixed']:.3f} new={row['new']:.3f}")


if __name__ == "__main__":
    main()
