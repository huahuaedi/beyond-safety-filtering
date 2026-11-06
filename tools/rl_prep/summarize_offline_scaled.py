"""Paired comparison of scaled offline-training arms on one log-disjoint eval set.

Reads <root>/<arm>/validation/scores.json for every arm directory, requires the
same scene set across arms, and reports each arm minus the pretrained baseline
(and conditioned minus filtered/unconditional). Seeds are averaged per scene
before the scene bootstrap, so the interval covers scene sampling only, not
training-seed variance; per-seed means are listed separately.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from awfm_statistics import bootstrap, log_labels, conversions, pass_status

METRICS = ("score", "no_at_fault_collisions", "drivable_area_compliance",
           "time_to_collision_within_bound", "ego_progress", "driving_direction_compliance")
FAIL = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")


def load_arm(path):
    payload = json.loads((path / "validation" / "scores.json").read_text())
    rows = {r["token"]: r for r in payload["rows"]}
    if len(rows) != len(payload["rows"]):
        raise ValueError(f"duplicate scenes in {path}")
    return rows


def arm_key(name):
    m = re.match(r"(.+)_seed(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, 0)


def mean_over_seeds(arms, tokens, metric):
    return np.mean([[a[t][metric] for t in tokens] for a in arms], axis=0)


def boot(delta, rng, n=4000, labels=None):
    return bootstrap(delta, rng, labels, n)


def compare(a_arms, b_arms, tokens, rng, logs=None):
    out = {}
    for metric in METRICS:
        d = mean_over_seeds(a_arms, tokens, metric) - mean_over_seeds(b_arms, tokens, metric)
        out[metric] = {"mean_x100": float(d.mean() * 100),
                       "ci95_x100": [x * 100 for x in boot(d, rng, labels=logs)]}
    # Compare matched seeds, or each arm seed against one frozen baseline.
    if len(b_arms) == 1:
        pairs = [(a, b_arms[0]) for a in a_arms]
    elif len(a_arms) == 1:
        pairs = [(a_arms[0], b) for b in b_arms]
    elif len(a_arms) == len(b_arms):
        pairs = list(zip(a_arms, b_arms))
    else:
        raise ValueError('multi-seed comparisons need matched seeds or one fixed baseline')
    conv = {}
    for metric in FAIL:
        counts = [conversions(b, a, tokens, (metric,)) for a, b in pairs]
        conv[metric] = {'fixed': float(np.mean([f.sum() for f, n in counts])),
                        'new_failures': float(np.mean([n.sum() for f, n in counts]))}
    joint = [conversions(b, a, tokens) for a, b in pairs]
    jd = np.mean([f-n for f,n in joint], axis=0)
    out['joint_safety'] = {'fixed': float(np.mean([f.sum() for f,n in joint])),
                           'new_failures': float(np.mean([n.sum() for f,n in joint])),
                           'delta_x100': float(jd.mean()*100),
                           'ci95_x100': [x*100 for x in boot(jd,rng,labels=logs)]}
    d = mean_over_seeds(a_arms, tokens, "score") - mean_over_seeds(b_arms, tokens, "score")
    out["failure_conversions"] = conv
    out["improve_rate"] = float((d > 1e-8).mean())
    out["regression_rate"] = float((d < -1e-8).mean())
    out["worst_10pct_mean_x100"] = float(np.sort(d)[: max(1, int(np.ceil(.1 * len(d))))].mean() * 100)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", default="scaled_v1_")
    p.add_argument("--baseline-prefix", default="scaled_v1_")
    p.add_argument("--require-arm", action="append", default=[], help="arm:seed,seed; fail if incomplete")
    p.add_argument("--require-complete", action="store_true")
    p.add_argument("--token-log-map", type=Path)
    p.add_argument("--metric-cache-dir", type=Path)
    p.add_argument("--out-name", default="scaled_comparison.json")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    groups = {}
    prefix = args.prefix
    for d in sorted(args.root.iterdir()):
        # smoke_* directories hold two scenes and must never enter the comparison
        use_prefix = args.baseline_prefix if d.name.startswith(args.baseline_prefix+'baseline_seed') else prefix
        if d.name.startswith(use_prefix) and (d / "validation" / "scores.json").is_file():
            name, seed = arm_key(d.name[len(use_prefix):])
            if args.require_complete:
                summary = json.loads((d/'summary.json').read_text())
                if summary.get('complete') is not True: raise ValueError(f'incomplete arm {d}')
            groups.setdefault(name, {})[seed] = load_arm(d)
    if "baseline" not in groups:
        raise ValueError("no baseline arm found")
    for spec in args.require_arm:
        name, seeds = spec.split(':')
        if set(groups.get(name, {})) != {int(s) for s in seeds.split(',')}:
            raise ValueError(f'missing/extra seeds for required arm {spec}')
    token_sets = [set(a) for arms in groups.values() for a in arms.values()]
    if any(t != token_sets[0] for t in token_sets):
        raise ValueError("arms were evaluated on different scenes")
    tokens = sorted(token_sets[0])
    logs = log_labels(tokens, args.token_log_map, args.metric_cache_dir)
    report = {"schema_version":2, "bootstrap_unit":"log" if logs is not None else "scene",
              "logs":len(set(logs)) if logs is not None else None,
              "failure_count_semantics":"mean of per-seed conversions; not thresholded seed-average scores",
              "scenes": len(tokens), "arms": {}, "vs_baseline": {}, "pairs": {},
              "caveat": ("Log-cluster" if logs is not None else "Scene-level") + " bootstrap on seed-averaged scores; training-seed variance is not covered. "
                        "Eval scenes are log-disjoint from training but come from the same dataset."}
    for name, seeds in groups.items():
        report["arms"][name] = {
            "seeds": sorted(seeds),
            "per_seed_pdms_x100": {str(s): float(np.mean([a[t]["score"] for t in tokens]) * 100)
                                   for s, a in sorted(seeds.items())},
            **{m: float(mean_over_seeds(list(seeds.values()), tokens, m).mean() * 100) for m in METRICS}}
    base = [groups["baseline"][s] for s in sorted(groups["baseline"])]
    for name, seeds in groups.items():
        if name != "baseline":
            if len(base)>1 and set(seeds)!=set(groups["baseline"]):
                raise ValueError("baseline and arm seed sets differ")
            report["vs_baseline"][name] = compare([seeds[s] for s in sorted(seeds)], base, tokens, rng, logs)
    for a, b in (("conditioned", "filtered"), ("conditioned", "unconditional"), ("filtered", "unconditional"),
                 ("marginpref", "pdmspref"), ("marginpref", "filtered"), ("marginpref", "conditioned"),
                 ("nearestpref", "pdmspref"), ("nearestpref", "marginpref"),
                 ("awfme0p3", "awfme0p0"), ("awfmgloe0p3", "awfme0p3"), ("awfmtgte0p3", "awfme0p3"), ("awfmtgte0p3", "awfmgloe0p3"),
                 ("awfme0p3", "awfme0p1"), ("awfme1p0", "awfme0p3"), ("awfme1p0", "awfme0p0"), ("awfme2p0", "awfme0p0"), ("awfme2p0", "awfme1p0"), ("awfme0p3", "filtered"), ("awfme0p3", "nearestpref")):
        if a in groups and b in groups:
            if set(groups[a]) != set(groups[b]):
                raise ValueError(f'{a}-{b}: seed sets differ')
            order = sorted(groups[a])
            report["pairs"][f"{a}-{b}"] = compare([groups[a][s] for s in order], [groups[b][s] for s in order], tokens, rng, logs)
    out = args.root / args.out_name
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
