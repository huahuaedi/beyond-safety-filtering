"""Token pools for RL derived from the simulator-scored candidate set.

Online RL gets almost no safety signal from a group whose 16 candidates are all
reference-feasible. A candidate is infeasible when any of NC/DAC/TTC is below the
componentwise maximum of the reference probes (the same rule the RL advantage uses).
For a scene whose infeasible-candidate rate is p, a random group of G candidates
contains at least one infeasible candidate with probability 1-(1-p)^G. This tool keeps
the log-isolated training scenes whose probability reaches a threshold and reports
how much the expected share of uninformative (all-feasible) groups drops. It only
selects scenes; it does not change any reward.
"""
import argparse
import json
from pathlib import Path

import numpy as np

SAFETY = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")


def scene_rate(path):
    with np.load(path, allow_pickle=False) as z:
        cand = {k: z["score__" + k].astype(np.float64) for k in SAFETY}
        ref = {k: z["reference_probe__" + k].astype(np.float64) for k in SAFETY}
    bad = np.zeros(len(cand[SAFETY[0]]), bool)
    for k in SAFETY:
        bad |= cand[k] < ref[k].max() - 1e-8
    ref_unsafe = any(ref[k].min() < 1 - 1e-6 for k in SAFETY)
    return float(bad.mean()), bool(ref_unsafe)


def group_hit(p, g):
    return 1.0 - (1.0 - p) ** g


def pools(rates, ref_unsafe, group_size, min_hit):
    tokens = sorted(rates)
    hit = {t: group_hit(rates[t], group_size) for t in tokens}
    chosen = [t for t in tokens if hit[t] >= min_hit or ref_unsafe[t]]
    return tokens, chosen, hit


def summarize(name, tokens, hit, ref_unsafe):
    if not tokens:
        return {"name": name, "scenes": 0}
    h = np.array([hit[t] for t in tokens])
    return {"name": name, "scenes": len(tokens),
            "expected_share_of_all_feasible_groups": float(1.0 - h.mean()),
            "scenes_ref_unsafe": int(sum(ref_unsafe[t] for t in tokens))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="candidate data root with train/val/test")
    ap.add_argument("--manifest", type=Path, required=True, help="schema-2 log-disjoint manifest")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--min-group-hit", type=float, default=0.75)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("needs a log-disjoint (schema 2) manifest")
    rates, unsafe = {}, {}
    for r in manifest["train"]:
        rates[r["token"]], unsafe[r["token"]] = scene_rate(args.root / r.get("split", "train") / f"{r['token']}.npz")
    all_tokens, chosen, hit = pools(rates, unsafe, args.group_size, args.min_group_hit)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "pool_all.json").write_text(json.dumps(all_tokens) + "\n")
    (args.output_dir / "pool_failure_enriched.json").write_text(json.dumps(chosen) + "\n")
    stats = {"group_size": args.group_size, "min_group_hit": args.min_group_hit,
             "all": summarize("all", all_tokens, hit, unsafe),
             "failure_enriched": summarize("failure_enriched", chosen, hit, unsafe),
             "note": "expected_share_of_all_feasible_groups assumes candidates in a group are drawn like the stored ones"}
    (args.output_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
