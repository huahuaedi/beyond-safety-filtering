"""Summarise the WM-vs-baseline risk-mining comparison with paired scene bootstrap.

Reads ``oof_predictions.npz`` (outer folds) and, when present,
``fresh_predictions.npz`` (disjoint fresh scenes) for each mode.  For every risk
head it reports the budgeted Top-K recall (share of a scene's risky candidates
found among its K highest-ranked candidates), its lift over random selection,
pooled AP / AUROC, and the paired-scene-bootstrap difference against a
reference mode.  The verdict lines only restate what the intervals show.

Also audits every geometric feature on its own (within-scene AUROC per label),
which tells whether the geometric baseline is informative at all.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

LABELS = ("unsafe", "ttc_failure", "dac_failure", "nc_failure")
KS = (8, 16)


def average_precision(y, score):
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    ranked = y[np.argsort(-score, kind="stable")]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / positives)


def auroc(y, score):
    positives, negatives = int(y.sum()), int((1 - y).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    sorted_scores = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float((ranks[y.astype(bool)].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def captured_at_k(labels, probability, k):
    """[S, L] number of risky candidates inside each scene's Top-K for each head."""
    top = np.argsort(-probability, axis=1, kind="stable")[:, :k]  # [S, k, L]
    return np.take_along_axis(labels, top, axis=1).sum(axis=1)


def load_units(root, mode, seeds, folds, kind):
    """List of (scene_names, probabilities, labels) evaluation units for one mode.

    ``oof``: one unit per seed (folds hold disjoint scenes and are concatenated).
    ``fresh``: one unit per (seed, fold) model, all scoring the same fresh scenes.
    """
    units = []
    for seed in seeds:
        base_dir = Path(root) / mode / f"seed_{seed}"
        if kind == "oof":
            names, probs, labels = [], [], []
            for fold in range(folds):
                with np.load(base_dir / f"fold_{fold}" / "oof_predictions.npz", allow_pickle=False) as d:
                    if str(d["selection_protocol"].item()) != "inner_scene_holdout_v1":
                        raise ValueError(f"{mode} seed {seed} fold {fold}: legacy selection protocol")
                    names += [str(n) for n in d["scene_names"]]
                    probs.append(d["probabilities"])
                    labels.append(d["labels"])
            units.append((names, np.concatenate(probs), np.concatenate(labels)))
        else:
            for fold in range(folds):
                path = base_dir / f"fold_{fold}" / "fresh_predictions.npz"
                if not path.is_file():
                    return []
                with np.load(path, allow_pickle=False) as d:
                    units.append(([str(n) for n in d["scene_names"]], d["probabilities"], d["labels"]))
    return units


def align(units_by_mode):
    """Reorder every unit to one sorted scene list; require identical scene sets."""
    reference = sorted(units_by_mode[next(iter(units_by_mode))][0][0])
    aligned = {}
    for mode, units in units_by_mode.items():
        aligned[mode] = []
        for names, probs, labels in units:
            if sorted(names) != reference:
                raise ValueError(f"{mode}: scene set differs from the other modes; cannot pair")
            order = np.argsort(np.asarray(names))
            aligned[mode].append((probs[order], labels[order]))
    return reference, aligned


def summarise_mode(units, ks=KS):
    """Per-scene captured counts (mean over units) and pooled ranking metrics."""
    labels = units[0][1]
    group = labels.shape[1]
    captured = {k: np.mean([captured_at_k(lab, prob, k) for prob, lab in units], axis=0) for k in ks}
    ap = np.mean([[average_precision(lab[..., h].ravel(), prob[..., h].ravel()) for h in range(labels.shape[-1])]
                  for prob, lab in units], axis=0)
    au = np.mean([[auroc(lab[..., h].ravel(), prob[..., h].ravel()) for h in range(labels.shape[-1])]
                  for prob, lab in units], axis=0)
    return {"captured": captured, "ap": ap, "auroc": au, "group": group, "positives": labels.sum(axis=1)}


def recall_from(captured, positives, index=None):
    if index is not None:
        captured, positives = captured[index], positives[index]
    return captured.sum(axis=0) / np.maximum(positives.sum(axis=0), 1)


def paired_bootstrap(a, b, positives, k_captured, resamples, seed):
    """CI of recall(a) - recall(b) per head, resampling whole scenes."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(positives), (resamples, len(positives)))
    ra = a[k_captured][idx].sum(axis=1) / np.maximum(positives[idx].sum(axis=1), 1)
    rb = b[k_captured][idx].sum(axis=1) / np.maximum(positives[idx].sum(axis=1), 1)
    delta = ra - rb
    return np.quantile(delta, [0.025, 0.975], axis=0)


def compare(units_by_mode, reference_mode, resamples=5000, seed=0):
    scene_names, aligned = align(units_by_mode)
    stats = {mode: summarise_mode(units) for mode, units in aligned.items()}
    positives = next(iter(stats.values()))["positives"]
    group = next(iter(stats.values()))["group"]
    report = {"scenes": len(scene_names), "reference": reference_mode, "modes": {}}
    for mode, s in stats.items():
        entry = {"average_precision": dict(zip(LABELS, map(float, s["ap"]))),
                 "auroc": dict(zip(LABELS, map(float, s["auroc"])))}
        for k in KS:
            recall = recall_from(s["captured"][k], positives)
            entry[f"recall_at_{k}"] = dict(zip(LABELS, map(float, recall)))
            entry[f"lift_at_{k}"] = dict(zip(LABELS, map(float, recall / (k / group))))
            if mode != reference_mode:
                delta = recall - recall_from(stats[reference_mode]["captured"][k], positives)
                lo, hi = paired_bootstrap(s["captured"], stats[reference_mode]["captured"], positives,
                                          k, resamples, seed)
                entry[f"recall_at_{k}_minus_reference"] = {
                    h: {"delta": float(delta[i]), "ci95": [float(lo[i]), float(hi[i])],
                        "excludes_zero": bool(lo[i] > 0 or hi[i] < 0)}
                    for i, h in enumerate(LABELS)
                }
        report["modes"][mode] = entry
    report["total_positives"] = dict(zip(LABELS, map(float, positives.sum(axis=0))))
    return report


def feature_audit(data_root, geom_root, splits=("train", "val", "test")):
    """Within-scene AUROC of each geometric feature (mean over scenes with both classes)."""
    import geometric_features as gf
    from train_world_action_risk_ranker import label_matrix

    per_feature = [[[] for _ in LABELS] for _ in gf.FEATURE_NAMES]
    scenes = 0
    for split in splits:
        for path in sorted((Path(data_root) / split).glob("*.npz")):
            geom_path = Path(geom_root) / split / path.name
            if not geom_path.is_file():
                continue
            with np.load(path, allow_pickle=False) as d:
                labels = label_matrix(d)
            with np.load(geom_path, allow_pickle=False) as g:
                features = g["geom_features"]
            scenes += 1
            for f in range(features.shape[1]):
                for h in range(len(LABELS)):
                    value = auroc(labels[:, h], features[:, f].astype(np.float64))
                    if not np.isnan(value):
                        per_feature[f][h].append(value)
    audit = {}
    for f, name in enumerate(gf.FEATURE_NAMES):
        audit[name] = {}
        for h, label in enumerate(LABELS):
            values = per_feature[f][h]
            mean = float(np.mean(values)) if values else float("nan")
            audit[name][label] = {"within_scene_auroc": mean, "scenes_with_both_classes": len(values)}
    return {"scenes": scenes, "features": audit}


def render_audit(audit):
    rows = ["### Geometric features, within-scene AUROC (0.5 = uninformative; <0.5 means higher value = safer)", "",
            f"scenes audited: {audit['scenes']}", "",
            "| feature | " + " | ".join(LABELS) + " |", "|---|" + "---|" * len(LABELS)]
    for name, per_label in audit["features"].items():
        rows.append(f"| {name} | " + " | ".join(f"{per_label[h]['within_scene_auroc']:.3f}" for h in LABELS) + " |")
    return "\n".join(rows) + "\n"


def render_markdown(report, title):
    lines = [f"### {title} ({report['scenes']} scenes; reference = {report['reference']})", ""]
    for k in KS:
        lines += [f"Recall@{k} of risky candidates (random = {k / 64:.3f}); lift in parentheses; "
                  f"delta vs reference with 95% CI, `*` = CI excludes 0", "",
                  "| mode | " + " | ".join(LABELS) + " |", "|---|" + "---|" * len(LABELS)]
        for mode, e in report["modes"].items():
            cells = []
            for h in LABELS:
                cell = f"{e[f'recall_at_{k}'][h]:.3f} ({e[f'lift_at_{k}'][h]:.2f}x)"
                d = e.get(f"recall_at_{k}_minus_reference", {}).get(h)
                if d:
                    cell += f" {d['delta']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]{'*' if d['excludes_zero'] else ''}"
                cells.append(cell)
            lines.append(f"| {mode} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["Pooled AUROC / AP:", "", "| mode | " + " | ".join(LABELS) + " |", "|---|" + "---|" * len(LABELS)]
    for mode, e in report["modes"].items():
        lines.append(f"| {mode} | " + " | ".join(
            f"{e['auroc'][h]:.3f} / {e['average_precision'][h]:.3f}" for h in LABELS) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-root", type=Path)
    ap.add_argument("--modes", nargs="+")
    ap.add_argument("--reference")
    ap.add_argument("--audit-only", action="store_true", help="only print the geometric-feature audit")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--resamples", type=int, default=5000)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--geom-root", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    if args.audit_only:
        if not (args.data_root and args.geom_root):
            raise SystemExit("--audit-only needs --data-root and --geom-root")
        audit = feature_audit(args.data_root, args.geom_root)
        if audit["scenes"] == 0:
            raise SystemExit("no scenes with geometric features found")
        print(render_audit(audit))
        return
    if not (args.oof_root and args.modes and args.reference and args.output):
        raise SystemExit("--oof-root, --modes, --reference and --output are required")
    if args.reference not in args.modes:
        raise SystemExit("--reference must be one of --modes")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    output = {"seeds": args.seeds, "folds": args.folds}
    text = []
    for kind, title in (("oof", "Cross-fitted development scenes"), ("fresh", "Fresh held-out scenes")):
        units = {m: load_units(args.oof_root, m, args.seeds, args.folds, kind) for m in args.modes}
        if any(not u for u in units.values()):
            if kind == "oof":
                raise SystemExit(f"missing OOF predictions for modes {[m for m, u in units.items() if not u]}")
            continue
        output[kind] = compare(units, args.reference, args.resamples)
        text.append(render_markdown(output[kind], title))
    if args.data_root and args.geom_root:
        output["geometric_feature_audit"] = feature_audit(args.data_root, args.geom_root)
        text.append(render_audit(output["geometric_feature_audit"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=True))
    args.output.with_suffix(".md").write_text("\n".join(text))
    print("\n".join(text))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
