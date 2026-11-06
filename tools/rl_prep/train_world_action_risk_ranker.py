"""Train scene-aware risk rankers on verified world-action representations.

V1 established that different world-model depths encode complementary risks,
but its global BCE objective did not consistently improve within-scene Top-K
recall.  V2 keeps complete 64-candidate scene groups, combines classification
with an all-positive/all-negative pairwise ranking loss, and supports the
critical action-only and single-layer controls alongside action+layer1+layer2
fusion.
"""
import json
import os
import random
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from risk_ranker_splits import inner_scene_split


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("WM_ACTION_DATA_ROOT", REPO_ROOT / "navsim_data/exp/world_action_v1"))
LATENT_ROOT = Path(os.environ.get("WM_LATENT_ROOT", REPO_ROOT / "navsim_data/exp/world_action_latents_v1"))
OUTPUT_ROOT = Path(os.environ.get("WM_RISK_RANKER_OUTPUT", REPO_ROOT / "output/world_action_risk_ranker_v2"))
FEATURE_MODE = os.environ.get("WM_RISK_FEATURE_MODE", "fusion")
SEED = int(os.environ.get("WM_RISK_SEED", 0))
EPOCHS = int(os.environ.get("WM_RISK_EPOCHS", 40))
SCENES_PER_BATCH = int(os.environ.get("WM_RISK_SCENES_PER_BATCH", 4))
EVAL_BATCH_SIZE = int(os.environ.get("WM_RISK_EVAL_BATCH_SIZE", 512))
LR = float(os.environ.get("WM_RISK_LR", 3e-4))
RANKING_COEF = float(os.environ.get("WM_RISK_RANKING_COEF", 1.0))
RANKING_MARGIN = float(os.environ.get("WM_RISK_RANKING_MARGIN", 0.25))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_NAMES = ("unsafe", "ttc_failure", "dac_failure", "nc_failure")
SELECTION_LABELS = ("unsafe", "ttc_failure", "dac_failure")
VALID_FEATURE_MODES = ("action", "layer1", "layer2", "fusion")
CV_FOLD = int(os.environ.get("WM_RISK_CV_FOLD", -1))
CV_FOLDS = int(os.environ.get("WM_RISK_CV_FOLDS", 5))


def label_matrix(label_data):
    """[C, 4] float labels in LABEL_NAMES order from one scene's label npz."""
    feasible = label_data["reference_feasible"].astype(bool)
    return np.stack(
        [
            ~feasible,
            label_data["score__time_to_collision_within_bound"] < 1.0 - 1e-8,
            label_data["score__drivable_area_compliance"] < 1.0 - 1e-8,
            label_data["score__no_at_fault_collisions"] < 1.0 - 1e-8,
        ],
        axis=1,
    ).astype(np.float32)


def load_paths(label_files):
    actions, layer1, layer2, labels, names = [], [], [], [], []
    for label_path in sorted(label_files):
        split = label_path.parent.name
        latent_path = LATENT_ROOT / split / label_path.name
        if not latent_path.is_file():
            raise FileNotFoundError(latent_path)
        with np.load(label_path, allow_pickle=False) as label_data, np.load(
            latent_path, allow_pickle=False
        ) as latent_data:
            action = label_data["relative_trajectories"].reshape(-1, 24).astype(np.float32)
            latent = latent_data["eoa_latents"].astype(np.float32)
            y = label_matrix(label_data)
        if len(action) != 64 or latent.shape != (64, 4, 4096):
            raise ValueError(f"unexpected candidate shapes in {label_path}: {action.shape}, {latent.shape}")
        actions.append(action)
        # Copy the selected depths so the temporary four-layer array can be
        # released immediately instead of being retained through NumPy views.
        layer1.append(latent[:, 1].copy())
        layer2.append(latent[:, 2].copy())
        labels.append(y)
        names.append(label_path.stem)
    if not names:
        raise ValueError("no scene files supplied")
    return tuple(np.stack(x) for x in (actions, layer1, layer2, labels)) + (names,)


def load_split(split):
    return load_paths((DATA_ROOT / split).glob("*.npz"))


def scene_risk_counts(label_path):
    with np.load(label_path, allow_pickle=False) as data:
        feasible = data["reference_feasible"].astype(bool)
        return np.asarray(
            [
                (~feasible).sum(),
                (data["score__time_to_collision_within_bound"] < 1.0 - 1e-8).sum(),
                (data["score__drivable_area_compliance"] < 1.0 - 1e-8).sum(),
                (data["score__no_at_fault_collisions"] < 1.0 - 1e-8).sum(),
            ],
            dtype=np.float64,
        )


def stratified_development_folds():
    """Deterministically balance candidate-level multi-risk counts by scene."""
    if CV_FOLDS < 2:
        raise ValueError("WM_RISK_CV_FOLDS must be at least 2")
    paths = sorted(list((DATA_ROOT / "train").glob("*.npz")) + list((DATA_ROOT / "val").glob("*.npz")))
    if len(paths) < CV_FOLDS or len({p.stem for p in paths}) != len(paths):
        raise ValueError("CV requires enough scenes with unique tokens across train and val")
    counts = np.stack([scene_risk_counts(path) for path in paths])
    presence = (counts > 0).astype(np.float64)
    total = counts.sum(axis=0)
    target = np.maximum(total / CV_FOLDS, 1.0)
    presence_target = np.maximum(presence.sum(axis=0) / CV_FOLDS, 1.0)
    target_size = len(paths) / CV_FOLDS

    # Place scenes rich in rare risks first; SHA256 provides stable tie-breaking.
    rarity = (counts / np.maximum(total, 1.0)).sum(axis=1)
    order = sorted(
        range(len(paths)),
        key=lambda i: (-rarity[i], hashlib.sha256(paths[i].stem.encode()).hexdigest()),
    )
    fold_counts = np.zeros((CV_FOLDS, len(LABEL_NAMES)), dtype=np.float64)
    fold_presence = np.zeros((CV_FOLDS, len(LABEL_NAMES)), dtype=np.float64)
    fold_sizes = np.zeros(CV_FOLDS, dtype=np.int64)
    assignments = np.full(len(paths), -1, dtype=np.int64)
    for scene_i in order:
        costs = []
        for fold_i in range(CV_FOLDS):
            trial_counts = fold_counts.copy()
            trial_presence = fold_presence.copy()
            trial_sizes = fold_sizes.copy()
            trial_counts[fold_i] += counts[scene_i]
            trial_presence[fold_i] += presence[scene_i]
            trial_sizes[fold_i] += 1
            # Minimize imbalance across all folds. Balancing scene presence in
            # addition to candidate counts prevents rare NC scenes from being
            # concentrated in a small number of folds.
            count_cost = np.var(trial_counts / target[None, :], axis=0).sum()
            presence_cost = np.var(trial_presence / presence_target[None, :], axis=0).sum()
            size_cost = np.var(trial_sizes / target_size)
            costs.append((count_cost + 2.0 * presence_cost + 0.5 * size_cost, fold_sizes[fold_i], fold_i))
        chosen = min(costs)[-1]
        assignments[scene_i] = chosen
        fold_counts[chosen] += counts[scene_i]
        fold_presence[chosen] += presence[scene_i]
        fold_sizes[chosen] += 1
    eligible = presence.sum(axis=0) >= CV_FOLDS
    if (fold_sizes == 0).any() or (fold_presence[:, eligible] == 0).any():
        raise RuntimeError(
            f"stratification failed: sizes={fold_sizes.tolist()} "
            f"risk_scene_counts={fold_presence.astype(int).tolist()}"
        )
    print(
        f"cv_folds={CV_FOLDS} sizes={fold_sizes.tolist()} "
        f"positive_candidates={fold_counts.astype(int).tolist()} "
        f"positive_scenes={fold_presence.astype(int).tolist()}", flush=True,
    )
    return paths, assignments


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class RiskRanker(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in VALID_FEATURE_MODES:
            raise ValueError(f"unknown feature mode {mode!r}; choose from {VALID_FEATURE_MODES}")
        self.mode = mode
        self.action_encoder = Encoder(24, 128) if mode in ("action", "fusion") else None
        self.layer1_encoder = Encoder(4096, 256) if mode in ("layer1", "fusion") else None
        self.layer2_encoder = Encoder(4096, 256) if mode in ("layer2", "fusion") else None
        hidden_dim = {"action": 128, "layer1": 256, "layer2": 256, "fusion": 640}[mode]
        self.fusion = nn.Sequential(nn.Linear(hidden_dim, 256), nn.GELU(), nn.Dropout(0.1))
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1)) for _ in LABEL_NAMES]
        )

    def forward(self, action, layer1, layer2):
        encoded = []
        if self.action_encoder is not None:
            encoded.append(self.action_encoder(action))
        if self.layer1_encoder is not None:
            encoded.append(self.layer1_encoder(layer1))
        if self.layer2_encoder is not None:
            encoded.append(self.layer2_encoder(layer2))
        hidden = self.fusion(torch.cat(encoded, dim=-1))
        return torch.cat([head(hidden) for head in self.heads], dim=-1)


def pairwise_ranking_loss(logits, targets):
    """Margin ranking over every valid positive/negative pair per scene/head."""
    losses = []
    for scene_i in range(logits.shape[0]):
        for label_i in range(logits.shape[-1]):
            positive = logits[scene_i, targets[scene_i, :, label_i] > 0.5, label_i]
            negative = logits[scene_i, targets[scene_i, :, label_i] < 0.5, label_i]
            if positive.numel() and negative.numel():
                difference = positive[:, None] - negative[None, :]
                losses.append(torch.relu(RANKING_MARGIN - difference).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


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
    ranks = np.empty(len(score), dtype=np.float64)
    sorted_scores = score[order]
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = ranks[y.astype(bool)].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def expected_calibration_error(y, probability, bins=10):
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        mask = (probability >= edges[i]) & (
            probability <= edges[i + 1] if i == bins - 1 else probability < edges[i + 1]
        )
        if mask.any():
            value += mask.mean() * abs(probability[mask].mean() - y[mask].mean())
    return float(value)


def recall_at_k(y, score, k):
    captured = total = 0
    for scene_i in range(y.shape[0]):
        positives = int(y[scene_i].sum())
        if positives == 0:
            continue
        chosen = np.argsort(-score[scene_i], kind="stable")[:k]
        captured += int(y[scene_i, chosen].sum())
        total += positives
    return float(captured / total) if total else float("nan")


def ndcg_at_k(y, score, k):
    values = []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for scene_i in range(y.shape[0]):
        positives = int(y[scene_i].sum())
        if positives == 0:
            continue
        order = np.argsort(-score[scene_i], kind="stable")[:k]
        dcg = float((y[scene_i, order] * discounts).sum())
        ideal = float(discounts[: min(positives, k)].sum())
        values.append(dcg / ideal)
    return float(np.mean(values)) if values else float("nan")


def predict(model, arrays):
    action, layer1, layer2, _ = arrays
    flat = [x.reshape(-1, x.shape[-1]) for x in (action, layer1, layer2)]
    probability = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(flat[0]), EVAL_BATCH_SIZE):
            inputs = [torch.from_numpy(x[start:start + EVAL_BATCH_SIZE]).to(DEVICE) for x in flat]
            probability.append(torch.sigmoid(model(*inputs)).cpu().numpy())
    return np.concatenate(probability).reshape(action.shape[0], action.shape[1], -1)


def metrics_from_probability(y, probability):
    """Per-label ranking/calibration metrics; y and probability are [S, C, L]."""
    metrics = {}
    for label_i, name in enumerate(LABEL_NAMES):
        target, score = y[:, :, label_i], probability[:, :, label_i]
        metrics[name] = {
            "prevalence": float(target.mean()),
            "auroc": auroc(target.reshape(-1), score.reshape(-1)),
            "average_precision": average_precision(target.reshape(-1), score.reshape(-1)),
            "ece": expected_calibration_error(target.reshape(-1), score.reshape(-1)),
            "recall_at_8": recall_at_k(target, score, 8),
            "recall_at_16": recall_at_k(target, score, 16),
            "ndcg_at_16": ndcg_at_k(target, score, 16),
            "random_recall_at_8": 0.125,
            "random_recall_at_16": 0.25,
        }
    return metrics


def evaluate(model, arrays):
    return metrics_from_probability(arrays[3], predict(model, arrays))


def selection_score(metrics):
    values = []
    for name in SELECTION_LABELS:
        values.extend([metrics[name]["average_precision"], metrics[name]["recall_at_16"]])
    return float(np.nanmean(values))


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(
        f"feature_mode={FEATURE_MODE} seed={SEED} device={DEVICE} "
        f"ranking_coef={RANKING_COEF} margin={RANKING_MARGIN}", flush=True,
    )
    if CV_FOLD >= 0:
        if CV_FOLD >= CV_FOLDS:
            raise ValueError(f"WM_RISK_CV_FOLD={CV_FOLD} outside [0, {CV_FOLDS})")
        development_paths, assignments = stratified_development_folds()
        outer_train_paths = [p for p, fold in zip(development_paths, assignments) if fold != CV_FOLD]
        outer_paths = [p for p, fold in zip(development_paths, assignments) if fold == CV_FOLD]
        train_paths, val_paths = inner_scene_split(outer_train_paths, seed=SEED)
        train = load_paths(train_paths)
        val = load_paths(val_paths)
        test = None
    else:
        train = load_split("train")
        val = load_split("val")
        test = load_split("test")
    print(
        f"train={train[0].shape}/{len(train[-1])} val={val[0].shape}/{len(val[-1])} "
        + (f"test={test[0].shape}/{len(test[-1])}" if test is not None else f"cv_fold={CV_FOLD}"),
        flush=True,
    )

    model = RiskRanker(FEATURE_MODE).to(DEVICE)
    positives = train[3].sum(axis=(0, 1))
    total = train[3].shape[0] * train[3].shape[1]
    pos_weight = (total - positives) / np.maximum(positives, 1.0)
    classification_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    tensors = [torch.from_numpy(x) for x in train[:4]]
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=SCENES_PER_BATCH,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    best_state, best_score, best_epoch = None, -float("inf"), -1
    for epoch in range(EPOCHS):
        model.train()
        epoch_total, epoch_bce, epoch_rank = [], [], []
        for action, layer1, layer2, targets in loader:
            action, layer1, layer2, targets = [x.to(DEVICE) for x in (action, layer1, layer2, targets)]
            batch_shape = action.shape[:2]
            logits = model(
                action.reshape(-1, action.shape[-1]),
                layer1.reshape(-1, layer1.shape[-1]),
                layer2.reshape(-1, layer2.shape[-1]),
            ).reshape(*batch_shape, -1)
            bce = classification_loss(logits, targets)
            rank = pairwise_ranking_loss(logits, targets)
            loss = bce + RANKING_COEF * rank
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_total.append(float(loss.item()))
            epoch_bce.append(float(bce.item()))
            epoch_rank.append(float(rank.item()))
        val_metrics = evaluate(model, val[:4])
        score = selection_score(val_metrics)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch={epoch:02d} loss={np.mean(epoch_total):.6f} bce={np.mean(epoch_bce):.6f} "
            f"rank={np.mean(epoch_rank):.6f} val_selection={score:.6f}", flush=True,
        )

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    results = {
        "feature_mode": FEATURE_MODE,
        "seed": SEED,
        "cv_fold": CV_FOLD,
        "cv_folds": CV_FOLDS,
        "selection_protocol": "inner_scene_holdout_v1" if CV_FOLD >= 0 else "development_validation",
        "training_scenes": train[-1],
        "selection_scenes": val[-1],
        "ranking_coef": RANKING_COEF,
        "ranking_margin": RANKING_MARGIN,
        "best_epoch": best_epoch,
        "best_val_selection": best_score,
        "val": evaluate(model, val[:4]),
    }
    if CV_FOLD >= 0:
        # Load the outer fold only after all checkpoint selection is complete.
        outer = load_paths(outer_paths)
        results["outer"] = evaluate(model, outer[:4])
        results["outer_scenes"] = outer[-1]
    if test is not None:
        results["test"] = evaluate(model, test[:4])
    output_dir = OUTPUT_ROOT / FEATURE_MODE / f"seed_{SEED}"
    if CV_FOLD >= 0:
        output_dir = output_dir / f"fold_{CV_FOLD}"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": best_state, "config": {k: results[k] for k in (
            "feature_mode", "seed", "ranking_coef", "ranking_margin", "best_epoch"
        )}},
        output_dir / "ranker.pt",
    )
    with open(output_dir / "metrics.json", "w") as file:
        json.dump(results, file, indent=2, allow_nan=True)
    if CV_FOLD >= 0:
        np.savez_compressed(
            output_dir / "oof_predictions.npz",
            scene_names=np.asarray(outer[-1]),
            probabilities=predict(model, outer[:4]),
            labels=outer[3],
            selection_protocol=np.asarray("inner_scene_holdout_v1"),
            training_scene_names=np.asarray(train[-1]),
            selection_scene_names=np.asarray(val[-1]),
        )
    print(json.dumps(results.get("test", results.get("outer", results["val"])), indent=2), flush=True)
    print(f"saved={output_dir}", flush=True)


if __name__ == "__main__":
    main()
