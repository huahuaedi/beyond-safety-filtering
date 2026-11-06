"""Build equal-budget, NAVSIM-verified risk-memory manifests from OOF scores.

Selectors may rank every candidate, but verification labels are consulted only
for the selected Top-K.  The resulting manifest contains scene-level memories
only when at least one selected candidate is verified reference-relative
unsafe; unselected labels never affect inclusion or priority.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_world_action_mining import load_oof
from adaptive_reward import estimate_tail_risk_baselines, scene_constraint_weights


def append_ranked(selected, ranking, limit, budget):
    if len(selected) >= budget or limit <= 0:
        return
    added = 0
    for candidate in ranking:
        candidate = int(candidate)
        if candidate not in selected:
            selected.append(candidate)
            added += 1
            if added >= limit or len(selected) >= budget:
                break


def deterministic_random_ranking(scene_token, group_size, seed):
    digest = hashlib.sha256(f"{seed}:{scene_token}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return rng.permutation(group_size)


def select_candidates(selector, scene_token, scores, budget, seed):
    group_size = next(iter(scores.values())).shape[0]
    selected = []
    if selector == "random_topk":
        append_ranked(
            selected,
            deterministic_random_ranking(scene_token, group_size, seed),
            budget,
            budget,
        )
    elif selector == "action_topk":
        append_ranked(selected, np.argsort(-scores["action"][:, 0]), budget, budget)
    elif selector == "world_model_routed_topk":
        unsafe_budget = budget // 2
        ttc_budget = budget - unsafe_budget
        append_ranked(selected, np.argsort(-scores["layer1"][:, 0]), unsafe_budget, budget)
        append_ranked(selected, np.argsort(-scores["fusion"][:, 1]), ttc_budget, budget)
        # Cross-head selections can overlap. Fill the remaining fixed budget
        # with the fusion unsafe ranking without inspecting verification labels.
        append_ranked(selected, np.argsort(-scores["fusion"][:, 0]), budget, budget)
    else:
        raise ValueError(f"unknown selector {selector!r}")
    if len(selected) != budget:
        raise RuntimeError(f"selector {selector} returned {len(selected)} candidates, expected {budget}")
    return selected


def original_scene_files(data_root):
    files = {}
    for split in ("train", "val"):
        for path in (data_root / split).glob("*.npz"):
            if path.stem in files:
                raise ValueError(f"duplicate scene token {path.stem}")
            files[path.stem] = path
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("oof_root", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ranker-seed", type=int, default=0)
    parser.add_argument("--dynamic-weight-topk", type=int, default=8)
    parser.add_argument("--dynamic-weight-strength", type=float, default=0.5)
    parser.add_argument("--dynamic-weight-min", type=float, default=0.5)
    parser.add_argument("--dynamic-weight-max", type=float, default=2.0)
    args = parser.parse_args()
    if not 1 <= args.budget <= 64:
        raise ValueError("budget must be in [1, 64]")

    modes = ("action", "layer1", "fusion")
    names_by_mode, probabilities = {}, {}
    reference_labels = None
    for mode in modes:
        names, probability, labels = load_oof(args.oof_root, mode, args.folds, seed=args.ranker_seed)
        names_by_mode[mode] = names
        probabilities[mode] = probability
        if reference_labels is None:
            reference_labels = labels
        elif names != names_by_mode["action"] or not np.array_equal(labels, reference_labels):
            raise ValueError(f"OOF data for {mode} do not align")
    scene_names = names_by_mode["action"]
    dynamic_risk_baselines = estimate_tail_risk_baselines(
        probabilities["fusion"], topk=args.dynamic_weight_topk
    )
    scene_files = original_scene_files(args.data_root)
    missing = sorted(set(scene_names) - set(scene_files))
    if missing:
        raise FileNotFoundError(f"missing source files for {len(missing)} scenes, first={missing[0]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selectors = ("random_topk", "action_topk", "world_model_routed_topk")
    for selector in selectors:
        entries = []
        verified_scene_count = verified_candidate_count = 0
        for scene_i, scene_token in enumerate(scene_names):
            scores = {mode: probabilities[mode][scene_i] for mode in modes}
            dynamic_weights, wm_tail_risk = scene_constraint_weights(
                scores["fusion"],
                topk=args.dynamic_weight_topk,
                strength=args.dynamic_weight_strength,
                minimum=args.dynamic_weight_min,
                maximum=args.dynamic_weight_max,
                risk_baselines=dynamic_risk_baselines,
            )
            selected = select_candidates(selector, scene_token, scores, args.budget, args.seed)
            selected_array = np.asarray(selected, dtype=np.int64)
            selected_labels = reference_labels[scene_i, selected_array]
            verified_unsafe = selected_labels[:, 0].astype(bool)
            verified_indices = selected_array[verified_unsafe].tolist()
            with np.load(scene_files[scene_token], allow_pickle=False) as source:
                violation = source["reference_violation"].astype(np.float32)[selected_array]
            verified_violation = violation[verified_unsafe]
            unsafe_fraction = float(verified_unsafe.mean())
            priority = max(
                unsafe_fraction,
                float(verified_violation.max()) if len(verified_violation) else 0.0,
            )
            label_counts = {
                "unsafe": int(selected_labels[:, 0].sum()),
                "ttc_failure": int(selected_labels[:, 1].sum()),
                "dac_failure": int(selected_labels[:, 2].sum()),
                "nc_failure": int(selected_labels[:, 3].sum()),
            }
            verified_scene_count += int(bool(verified_indices))
            verified_candidate_count += len(verified_indices)
            entries.append(
                {
                    "scene_token": scene_token,
                    "selected_indices": selected,
                    "verified_unsafe_indices": verified_indices,
                    "verified_label_counts": label_counts,
                    "verified_unsafe_fraction": unsafe_fraction,
                    "risk_priority": priority,
                    "wm_tail_risk": wm_tail_risk,
                    "dynamic_constraint_weights": dynamic_weights,
                }
            )
        payload = {
            "schema_version": 1,
            "oof_selection_protocol": "inner_scene_holdout_v1",
            "oof_root": str(args.oof_root.resolve()),
            "ranker_seed": args.ranker_seed,
            "selector": selector,
            "candidate_group_size": 64,
            "verification_budget_per_scene": args.budget,
            "development_scenes": len(entries),
            "verification_calls": len(entries) * args.budget,
            "verified_risk_scenes": verified_scene_count,
            "verified_unsafe_candidates": verified_candidate_count,
            "seed": args.seed,
            "dynamic_weight_config": {
                "feature_mode": "fusion",
                "topk": args.dynamic_weight_topk,
                "strength": args.dynamic_weight_strength,
                "minimum": args.dynamic_weight_min,
                "maximum": args.dynamic_weight_max,
                "risk_baselines": dynamic_risk_baselines,
            },
            "entries": entries,
        }
        output = args.output_dir / f"{selector}_budget{args.budget}.json"
        with open(output, "w") as file:
            json.dump(payload, file, indent=2)
        print(
            f"selector={selector} budget={args.budget} scenes={len(entries)} "
            f"verified_risk_scenes={verified_scene_count} "
            f"verified_unsafe_candidates={verified_candidate_count} "
            f"calls={payload['verification_calls']} path={output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
