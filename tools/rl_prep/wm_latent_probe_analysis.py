#!/usr/bin/env python3
"""World-model latent representation analysis (offline, CPU-only, no
simulator needed): does each WM layer's cached latent
(world_action_latents_full_v1, already extracted -- see the (64,4,4096)
eoa_latents schema) carry per-metric risk information beyond the raw
action itself?

For each feature in {Action-only, WM layer -1/-9/-17/-25, Fusion (all 4
concatenated)} and each label in {NC, TTC, DAC, DDC, Unsafe (union)}, fits a
simple linear probe (logistic regression -- deliberately weak, so a high
score reflects information already linearly present in the feature, not
probe capacity) under 5-fold SCENE-DISJOINT cross-validation (a fold never
sees a scene's OTHER 63 candidates during that fold's training), and
reports AP (primary -- these labels are rare positives, accuracy would be
misleading), AUROC, Recall@8, Recall@16 (fixed per-scene candidate-review
budget), and scene-level coverage@8 (fraction of scenes with >=1 true
positive whose top-8-by-probe-score ranking catches at least one).

Memory design: processes ONE FEATURE AT A TIME across all ~7171 scenes
(load -> run all 5 label probes for that feature -> discard -> next
feature), rather than materializing all 6 features simultaneously (the
naive version of this OOM'd at ~60-90GB on a login node). Peak resident
set is one feature's data (~15GB for a single WM layer in float16, briefly
~30GB float32 for Fusion during its own fit) instead of the sum of all six.
Still needs a real memory allocation (submit as an sbatch job, not a
login-node background process) -- see the accompanying submit script.

Usage:
    python3 wm_latent_probe_analysis.py --out-dir results_summary/wm_probe
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
LATENT_ROOT = REPO_ROOT / "navsim_data/exp/world_action_latents_full_v1"
LABEL_ROOT = REPO_ROOT / "navsim_data/exp/world_action_full_v1"
N_FOLDS = 5
CV_SEED = 0
TOP_K = (8, 16)
LABEL_NAMES = ["NC", "TTC", "DAC", "DDC", "Unsafe"]


def list_scenes():
    scene_tokens = []
    for split in ("train", "val", "test"):
        for p in sorted((LATENT_ROOT / split).glob("*.npz")):
            scene_tokens.append((p.stem, split))
    return scene_tokens


def load_labels(token, split):
    with np.load(LABEL_ROOT / split / f"{token}.npz", allow_pickle=False) as d:
        nc_fail = (d["score__no_at_fault_collisions"] < 1.0 - 1e-8)
        ttc_fail = (d["score__time_to_collision_within_bound"] < 1.0 - 1e-8)
        dac_fail = (d["score__drivable_area_compliance"] < 1.0 - 1e-8)
        ddc_fail = (d["score__driving_direction_compliance"] < 1.0 - 1e-8)
    return {
        "NC": nc_fail.astype(np.float32),
        "TTC": ttc_fail.astype(np.float32),
        "DAC": dac_fail.astype(np.float32),
        "DDC": ddc_fail.astype(np.float32),
        "Unsafe": (nc_fail | ttc_fail | dac_fail | ddc_fail).astype(np.float32),
    }


def resolve_feature_names(scene_tokens):
    token, split = scene_tokens[0]
    with np.load(LATENT_ROOT / split / f"{token}.npz", allow_pickle=False) as d:
        resolved_layers = d["resolved_layers"].tolist()
    return ["Action only"] + [f"WM layer {l}" for l in resolved_layers] + ["Fusion"]


def gather_labels_and_scene_ids(scene_tokens):
    per_label = {name: [] for name in LABEL_NAMES}
    scene_id_per_candidate = []
    for scene_idx, (token, split) in enumerate(scene_tokens):
        labels = load_labels(token, split)
        for name in LABEL_NAMES:
            per_label[name].append(labels[name])
        scene_id_per_candidate.append(np.full(64, scene_idx, dtype=np.int64))
    per_label = {k: np.concatenate(v, axis=0) for k, v in per_label.items()}
    scene_id_per_candidate = np.concatenate(scene_id_per_candidate, axis=0)
    return per_label, scene_id_per_candidate


def gather_one_feature(feature_name, scene_tokens, resolved_layers):
    """Loads only what's needed for a single feature across every scene,
    as float32 (sklearn requires it), so peak memory is this one array,
    not all six at once.
    """
    chunks = []
    for token, split in scene_tokens:
        if feature_name == "Action only":
            with np.load(LABEL_ROOT / split / f"{token}.npz", allow_pickle=False) as d:
                chunks.append(d["relative_trajectories"].astype(np.float32).reshape(64, -1))
        else:
            with np.load(LATENT_ROOT / split / f"{token}.npz", allow_pickle=False) as d:
                eoa = d["eoa_latents"]  # [64, 4, 4096], float16
                if feature_name == "Fusion":
                    chunks.append(eoa.reshape(64, -1).astype(np.float32))
                else:
                    layer_idx = [f"WM layer {l}" for l in resolved_layers].index(feature_name)
                    chunks.append(eoa[:, layer_idx, :].astype(np.float32))
    return np.concatenate(chunks, axis=0)


def scene_disjoint_folds(n_scenes, n_folds, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_scenes)
    return np.array_split(order, n_folds)


def recall_at_k_and_coverage(oof_score, label, scene_id, n_scenes, ks):
    results = {k: {"recall": None, "coverage": None} for k in ks}
    by_scene_scores = [[] for _ in range(n_scenes)]
    by_scene_labels = [[] for _ in range(n_scenes)]
    for score, lab, sid in zip(oof_score, label, scene_id):
        by_scene_scores[sid].append(score)
        by_scene_labels[sid].append(lab)
    total_positive = float(label.sum())
    for k in ks:
        caught = 0.0
        scenes_with_positive = 0
        scenes_covered = 0
        for scores, labs in zip(by_scene_scores, by_scene_labels):
            scores = np.asarray(scores)
            labs = np.asarray(labs)
            if labs.sum() == 0:
                continue
            scenes_with_positive += 1
            top = np.argsort(-scores)[:k]
            hits = labs[top].sum()
            caught += hits
            if hits > 0:
                scenes_covered += 1
        results[k]["recall"] = caught / total_positive if total_positive > 0 else float("nan")
        results[k]["coverage"] = (
            scenes_covered / scenes_with_positive if scenes_with_positive > 0 else float("nan")
        )
    return results


def run_probe_cv(X, y, scene_id, n_scenes, n_folds, seed):
    """SGD-based logistic regression, not full-batch LogisticRegression:
    on ~370K training candidates x up to 4096 dims, LBFGS-based
    LogisticRegression did not converge within a 2-hour job (see
    submit_wm_latent_probe.slurm's original TIMEOUT). SGDClassifier's
    stochastic, few-epoch optimization is the standard choice at this
    scale and is still a linear probe -- the "deliberately weak" design
    intent is unchanged, only the optimizer is. Features are standardized
    per-fold (fit on the training fold only, applied to both) since SGD
    convergence is sensitive to feature scale, unlike LBFGS -- otherwise
    "Action only" (raw, unnormalized trajectory coordinates) and the WM
    latents (roughly unit-scale transformer hidden states) would not be on
    a comparable footing.
    """
    folds = scene_disjoint_folds(n_scenes, n_folds, seed)
    oof_score = np.zeros_like(y)
    for fold_scenes in folds:
        test_mask = np.isin(scene_id, fold_scenes)
        train_mask = ~test_mask
        if y[train_mask].sum() == 0 or y[train_mask].sum() == train_mask.sum():
            oof_score[test_mask] = float(y[train_mask].mean())
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])
        clf = SGDClassifier(
            loss="log_loss", class_weight="balanced", max_iter=30,
            tol=1e-3, random_state=seed,
        )
        clf.fit(X_train, y[train_mask])
        oof_score[test_mask] = clf.predict_proba(X_test)[:, 1]
    return oof_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_tokens = list_scenes()
    print(f"{len(scene_tokens)} scenes with cached latents", flush=True)
    n_scenes = len(scene_tokens)

    token0, split0 = scene_tokens[0]
    with np.load(LATENT_ROOT / split0 / f"{token0}.npz", allow_pickle=False) as d:
        resolved_layers = d["resolved_layers"].tolist()
    feature_names = resolve_feature_names(scene_tokens)
    print("features:", feature_names, flush=True)

    per_label, scene_id = gather_labels_and_scene_ids(scene_tokens)
    print("prevalence:", {k: float(v.mean()) for k, v in per_label.items()}, flush=True)

    # Resume support: a prior run may have TIMEOUT'd partway through (this
    # analysis has already needed it once) -- table1/table3 are persisted
    # incrementally after every (feature, label) pair specifically so a
    # rerun can skip everything already computed rather than repeating a
    # multi-hour job for one missing cell.
    table1_path = out_dir / "table1_representation_quality.json"
    table3_path = out_dir / "table3_budget_discovery.json"
    table1 = json.loads(table1_path.read_text()) if table1_path.exists() else {}
    table3 = json.loads(table3_path.read_text()) if table3_path.exists() else {}
    table1.setdefault("Random", {
        name: {"AP": float(per_label[name].mean()), "AUROC": 0.5} for name in LABEL_NAMES
    })
    for feature_name in feature_names:
        if all(label_name in table1.get(feature_name, {}) for label_name in LABEL_NAMES):
            print(f"=== feature already complete, skipping: {feature_name} ===", flush=True)
            continue
        print(f"=== loading feature: {feature_name} ===", flush=True)
        X = gather_one_feature(feature_name, scene_tokens, resolved_layers)
        print(f"  shape={X.shape} dtype={X.dtype} "
              f"~{X.nbytes / 1e9:.2f} GB", flush=True)
        table1.setdefault(feature_name, {})
        table3.setdefault(feature_name, {})
        for label_name in LABEL_NAMES:
            if label_name in table1[feature_name]:
                print(f"  {feature_name} {label_name} already computed, skipping", flush=True)
                continue
            y = per_label[label_name]
            oof = run_probe_cv(X, y, scene_id, n_scenes, N_FOLDS, CV_SEED)
            ap = average_precision_score(y, oof)
            try:
                auroc = roc_auc_score(y, oof)
            except ValueError:
                auroc = float("nan")
            rk = recall_at_k_and_coverage(oof, y, scene_id, n_scenes, TOP_K)
            table1[feature_name][label_name] = {"AP": ap, "AUROC": auroc}
            table3[feature_name][label_name] = rk
            print(f"{feature_name:16s} {label_name:8s} AP={ap:.4f} AUROC={auroc:.4f} "
                  f"R@8={rk[8]['recall']:.4f} R@16={rk[16]['recall']:.4f} "
                  f"cov@8={rk[8]['coverage']:.4f}", flush=True)
            # persist incrementally so a later OOM/timeout doesn't lose
            # already-computed features
            (out_dir / "table1_representation_quality.json").write_text(json.dumps(table1, indent=2))
            (out_dir / "table3_budget_discovery.json").write_text(json.dumps(table3, indent=2))
        del X
        gc.collect()

    print(f"wrote results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
