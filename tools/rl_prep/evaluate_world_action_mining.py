"""Evaluate held-out Top-K risk mining from cross-fitted ranker predictions."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


LABEL_NAMES = ("unsafe", "ttc_failure", "dac_failure", "nc_failure")


def random_scene_coverage(target, k):
    values = []
    group_size = target.shape[1]
    for scene in target:
        positives = int(scene.sum())
        if positives:
            miss = math.comb(group_size - positives, k) / math.comb(group_size, k)
            values.append(1.0 - miss)
    return float(np.mean(values)) if values else float("nan")


def mining_metrics(target, score, k):
    selected_positive = 0
    covered_scenes = 0
    positive_scenes = int((target.sum(axis=1) > 0).sum())
    total_positive = int(target.sum())
    for scene_i in range(len(target)):
        chosen = np.argsort(-score[scene_i], kind="stable")[:k]
        found = int(target[scene_i, chosen].sum())
        selected_positive += found
        covered_scenes += int(found > 0)
    random_recall = k / target.shape[1]
    recall = selected_positive / total_positive if total_positive else float("nan")
    precision = selected_positive / (len(target) * k)
    coverage = covered_scenes / positive_scenes if positive_scenes else float("nan")
    oracle_found = sum(min(int(scene.sum()), k) for scene in target)
    return {
        "k": k,
        "navsim_fraction": k / target.shape[1],
        "selected_positive": selected_positive,
        "total_positive": total_positive,
        "recall": recall,
        "random_recall": random_recall,
        "recall_lift": recall / random_recall if random_recall else float("nan"),
        "precision": precision,
        "scene_coverage": coverage,
        "random_scene_coverage": random_scene_coverage(target, k),
        "oracle_recall": oracle_found / total_positive if total_positive else float("nan"),
    }


def load_oof(root, mode, folds, seed=0):
    names, probabilities, labels = [], [], []
    for fold in range(folds):
        path = root / mode / f"seed_{seed}" / f"fold_{fold}" / "oof_predictions.npz"
        with np.load(path, allow_pickle=False) as data:
            if ("selection_protocol" not in data or
                    str(data["selection_protocol"].item()) != "inner_scene_holdout_v1"):
                raise ValueError(f"{path}: legacy/unknown OOF selection protocol; regenerate CV predictions")
            outer_names = set(data["scene_names"].tolist())
            train_names = set(data["training_scene_names"].tolist())
            selection_names = set(data["selection_scene_names"].tolist())
            if not train_names or not selection_names or not outer_names:
                raise ValueError(f"{path}: empty split provenance")
            if outer_names & (train_names | selection_names) or train_names & selection_names:
                raise ValueError(f"{path}: overlapping training, selection, or outer scenes")
            names.extend(data["scene_names"].tolist())
            probabilities.append(data["probabilities"])
            labels.append(data["labels"])
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate OOF scene names for {mode}")
    return names, np.concatenate(probabilities), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--modes", nargs="+", default=["action", "layer1", "layer2", "fusion"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = {}
    reference_names = reference_labels = None
    for mode in args.modes:
        names, probability, labels = load_oof(args.root, mode, args.folds, seed=args.seed)
        if reference_names is None:
            reference_names, reference_labels = names, labels
        elif names != reference_names or not np.array_equal(labels, reference_labels):
            raise ValueError(f"OOF scenes/labels for {mode} do not align with the first mode")
        results[mode] = {}
        print(f"\n===== {mode} ({len(names)} OOF scenes) =====")
        for label_i, label_name in enumerate(LABEL_NAMES):
            target, score = labels[:, :, label_i], probability[:, :, label_i]
            rows = [mining_metrics(target, score, k) for k in args.top_k]
            results[mode][label_name] = rows
            print(f"{label_name}:")
            for row in rows:
                print(
                    f"  K={row['k']:2d} calls={row['navsim_fraction']:.3f} "
                    f"recall={row['recall']:.4f} lift={row['recall_lift']:.3f} "
                    f"coverage={row['scene_coverage']:.4f} "
                    f"random_coverage={row['random_scene_coverage']:.4f}"
                )
    output = args.output or args.root / "mining_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as file:
        json.dump(results, file, indent=2, allow_nan=True)
    print(f"\nsaved={output}")


if __name__ == "__main__":
    main()
