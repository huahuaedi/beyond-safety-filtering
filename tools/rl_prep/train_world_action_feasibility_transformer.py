"""Train scene-aware action/world-model safety and feasibility transformers.

The model consumes complete 64-candidate scene groups and supports four
controlled feature modes: action-only, mean-pooled WM layers, a transformer
over WM depths, and the full depth+candidate-set transformer. Quality and
advantage sidecars are supervision only: feeding their post-outcome risk bits
to the predictor would leak NAVSIM labels.
"""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("WM_ACTION_DATA_ROOT", REPO_ROOT / "navsim_data/exp/world_action_full_v1"))
LATENT_ROOT = Path(os.environ.get("WM_LATENT_ROOT", REPO_ROOT / "navsim_data/exp/world_action_latents_full_v1"))
QUALITY_ROOT = Path(os.environ.get("WM_QUALITY_ROOT", REPO_ROOT / "navsim_data/exp/world_action_quality_full_v1"))
DYNAMICS_ROOT = Path(os.environ.get("WM_DYNAMICS_ROOT", REPO_ROOT / "navsim_data/exp/world_action_dynamics_full_v2"))
OUTPUT_ROOT = Path(os.environ.get("WM_FEASIBILITY_OUTPUT", REPO_ROOT / "output/world_action_feasibility_transformer_v1"))
FEATURE_MODE = os.environ.get("WM_FEASIBILITY_MODE", "full")
SEED = int(os.environ.get("WM_FEASIBILITY_SEED", 0))
EPOCHS = int(os.environ.get("WM_FEASIBILITY_EPOCHS", 12))
BATCH_SIZE = int(os.environ.get("WM_FEASIBILITY_BATCH_SIZE", 2))
WORKERS = int(os.environ.get("WM_FEASIBILITY_WORKERS", 4))
LR = float(os.environ.get("WM_FEASIBILITY_LR", 2e-4))
RANK_COEF = float(os.environ.get("WM_FEASIBILITY_RANK_COEF", 0.5))
REGRESSION_COEF = float(os.environ.get("WM_FEASIBILITY_REGRESSION_COEF", 0.25))
MODEL_DIM = int(os.environ.get("WM_FEASIBILITY_MODEL_DIM", 256))
MAX_SCENES = int(os.environ.get("WM_FEASIBILITY_MAX_SCENES", 0))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODES = ("action", "wm_mean", "wm_layer", "full")
LABEL_NAMES = (
    "unsafe", "ttc_failure", "dac_failure", "nc_failure",
    "dynamics_infeasible", "dyn_jerk", "dyn_curvature", "dyn_curvature_rate",
)
# Checkpoint selection follows the primary deployable environmental-risk
# objective. Easy analytically-derived dynamics labels must not dominate early
# stopping, and ultra-rare NC remains a reported secondary endpoint.
SELECTION_LABELS = ("unsafe", "ttc_failure", "dac_failure")
REGRESSION_NAMES = (
    "advantage", "speed_ratio", "acceleration_ratio", "deceleration_ratio",
    "jerk_ratio", "curvature_ratio", "curvature_rate_ratio",
    "lateral_acceleration_ratio",
)


def dynamics_targets(data):
    """Return learnable binary heads and continuous limit-normalized targets."""
    components = (data["violation_components"] > 0).astype(np.float32)
    binary = np.stack(
        [
            ~data["dynamics_feasible"].astype(bool),
            components[:, 3].astype(bool),
            components[:, 4].astype(bool),
            components[:, 5].astype(bool),
        ], axis=1,
    ).astype(np.float32)
    limits = data["limits"].astype(np.float32)
    ratios = np.stack(
        [
            data["max_speed"] / limits[0],
            np.maximum(data["max_acceleration"], 0.0) / limits[1],
            np.maximum(-data["min_acceleration"], 0.0) / limits[2],
            data["max_abs_jerk"] / limits[3],
            data["max_abs_curvature"] / limits[4],
            data["max_abs_curvature_rate"] / limits[5],
            data["max_abs_lateral_acceleration"] / limits[6],
        ], axis=1,
    ).astype(np.float32)
    return binary, np.log1p(np.maximum(ratios, 0.0)).astype(np.float32)


class SceneDataset(Dataset):
    def __init__(self, split):
        self.paths = sorted((DATA_ROOT / split).glob("*.npz"))
        if MAX_SCENES > 0:
            self.paths = self.paths[:MAX_SCENES]
        if not self.paths:
            raise ValueError(f"no {split} scenes under {DATA_ROOT}")
        for source in self.paths:
            for root in (LATENT_ROOT, QUALITY_ROOT, DYNAMICS_ROOT):
                expected = root / split / source.name
                if not expected.is_file():
                    raise FileNotFoundError(expected)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        source_path = self.paths[index]
        split, name = source_path.parent.name, source_path.name
        with np.load(source_path, allow_pickle=False) as data:
            action = data["relative_trajectories"].reshape(-1, 24).astype(np.float32)
            feasible = data["reference_feasible"].astype(bool)
            risk = np.stack(
                [
                    ~feasible,
                    data["score__time_to_collision_within_bound"] < 1.0 - 1e-8,
                    data["score__drivable_area_compliance"] < 1.0 - 1e-8,
                    data["score__no_at_fault_collisions"] < 1.0 - 1e-8,
                ], axis=1,
            ).astype(np.float32)
        with np.load(LATENT_ROOT / split / name, allow_pickle=False) as data:
            latent = data["eoa_latents"].astype(np.float32)
        with np.load(QUALITY_ROOT / split / name, allow_pickle=False) as data:
            condition = data["condition_vector"].astype(np.float32)
            advantage = data["advantage"].astype(np.float32)
        with np.load(DYNAMICS_ROOT / split / name, allow_pickle=False) as data:
            dynamics_binary, dynamics_regression = dynamics_targets(data)
        labels = np.concatenate([risk, dynamics_binary], axis=1)
        expected = (64,)
        if action.shape != (64, 24) or latent.shape != (64, 4, 4096):
            raise ValueError(f"unexpected shapes in {source_path}: {action.shape}, {latent.shape}")
        if labels.shape != (64, len(LABEL_NAMES)) or advantage.shape != expected:
            raise ValueError(f"bad sidecar shapes for {source_path}")
        return tuple(torch.from_numpy(x) for x in (
            action, latent, condition, labels, advantage, dynamics_regression,
        ))


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim), nn.GELU())

    def forward(self, x):
        return self.net(x)


class FeasibilityTransformer(nn.Module):
    def __init__(self, mode, dim=256):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.action_encoder = Encoder(24, dim)
        self.wm_encoder = Encoder(4096, dim) if mode != "action" else None
        if mode in ("wm_layer", "full"):
            self.layer_embedding = nn.Parameter(torch.zeros(1, 1, 4, dim))
            layer = nn.TransformerEncoderLayer(
                dim, 8, dim_feedforward=dim * 4, dropout=0.1,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.layer_transformer = nn.TransformerEncoder(layer, num_layers=2)
        else:
            self.layer_embedding = None
            self.layer_transformer = None
        parts = 1 + int(mode != "action")
        self.fusion = nn.Sequential(nn.Linear(parts * dim, dim), nn.GELU(), nn.Dropout(0.1))
        if mode == "full":
            candidate_layer = nn.TransformerEncoderLayer(
                dim, 8, dim_feedforward=dim * 4, dropout=0.1,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.candidate_transformer = nn.TransformerEncoder(candidate_layer, num_layers=3)
        else:
            self.candidate_transformer = None
        self.classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, len(LABEL_NAMES)))
        self.regressor = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, len(REGRESSION_NAMES)))

    def forward(self, action, latent, condition):
        parts = [self.action_encoder(action)]
        if self.wm_encoder is not None:
            wm = self.wm_encoder(latent)
            if self.layer_transformer is not None:
                shape = wm.shape
                flat = (wm + self.layer_embedding).reshape(-1, 4, shape[-1])
                wm = self.layer_transformer(flat).mean(dim=1).reshape(shape[0], shape[1], -1)
            else:
                wm = wm.mean(dim=2)
            parts.append(wm)
        # `condition` is intentionally not consumed here. It contains
        # post-outcome NAVSIM-derived fields and is reserved for subsequent
        # desired-quality policy conditioning, not risk prediction.
        hidden = self.fusion(torch.cat(parts, dim=-1))
        if self.candidate_transformer is not None:
            hidden = self.candidate_transformer(hidden)
        return self.classifier(hidden), self.regressor(hidden)


def ranking_loss(logits, labels, margin=0.25):
    losses = []
    for batch_i in range(logits.shape[0]):
        for label_i in range(logits.shape[-1]):
            positive = logits[batch_i, labels[batch_i, :, label_i] > 0.5, label_i]
            negative = logits[batch_i, labels[batch_i, :, label_i] < 0.5, label_i]
            if positive.numel() and negative.numel():
                losses.append(torch.relu(margin - positive[:, None] + negative[None, :]).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def average_precision(y, score):
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    ranked = y[np.argsort(-score, kind="stable")]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / positives)


def recall_at_k(y, score, k):
    captured = total = 0
    for scene_i in range(y.shape[0]):
        positives = int(y[scene_i].sum())
        if positives:
            chosen = np.argsort(-score[scene_i], kind="stable")[:k]
            captured += int(y[scene_i, chosen].sum())
            total += positives
    return float(captured / total) if total else float("nan")


def evaluate(model, loader):
    probabilities, labels_all, regression_all, targets_all = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            action, latent, condition, labels, advantage, dynamics_regression = [x.to(DEVICE) for x in batch]
            logits, regression = model(action, latent, condition)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels_all.append(labels.cpu().numpy())
            regression_all.append(regression.cpu().numpy())
            targets_all.append(
                torch.cat([advantage.unsqueeze(-1), dynamics_regression], dim=-1).cpu().numpy()
            )
    probability = np.concatenate(probabilities)
    labels = np.concatenate(labels_all)
    regression = np.concatenate(regression_all)
    targets = np.concatenate(targets_all)
    metrics = {}
    for label_i, name in enumerate(LABEL_NAMES):
        target, score = labels[:, :, label_i], probability[:, :, label_i]
        metrics[name] = {
            "prevalence": float(target.mean()),
            "average_precision": average_precision(target.reshape(-1), score.reshape(-1)),
            "recall_at_8": recall_at_k(target, score, 8),
            "recall_at_16": recall_at_k(target, score, 16),
        }
    metrics["regression"] = {
        f"{name}_mae": float(np.abs(regression[:, :, index] - targets[:, :, index]).mean())
        for index, name in enumerate(REGRESSION_NAMES)
    }
    return metrics


def selection_score(metrics):
    values = []
    for name in SELECTION_LABELS:
        values.extend([metrics[name]["average_precision"], metrics[name]["recall_at_16"]])
    return float(np.nanmean(values))


def count_labels(dataset):
    positives = np.zeros(len(LABEL_NAMES), dtype=np.float64)
    total = 0
    for source_path in dataset.paths:
        split, name = source_path.parent.name, source_path.name
        with np.load(source_path, allow_pickle=False) as data:
            feasible = data["reference_feasible"].astype(bool)
            labels = np.stack([
                ~feasible,
                data["score__time_to_collision_within_bound"] < 1.0 - 1e-8,
                data["score__drivable_area_compliance"] < 1.0 - 1e-8,
                data["score__no_at_fault_collisions"] < 1.0 - 1e-8,
            ], axis=1).astype(np.float32)
        with np.load(DYNAMICS_ROOT / split / name, allow_pickle=False) as data:
            dynamics, _ = dynamics_targets(data)
        labels = np.concatenate([labels, dynamics], axis=1)
        positives += labels.sum(axis=0)
        total += labels.shape[0]
    return positives, total


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"mode={FEATURE_MODE} seed={SEED} device={DEVICE} data={DATA_ROOT}", flush=True)
    train, val, test = (SceneDataset(split) for split in ("train", "val", "test"))
    print(f"scenes train={len(train)} val={len(val)} test={len(test)}", flush=True)
    positives, total = count_labels(train)
    print(f"train positives={positives.astype(int).tolist()} total={total}", flush=True)

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS,
        pin_memory=True, persistent_workers=WORKERS > 0, generator=generator,
    )
    eval_kwargs = dict(batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)
    val_loader, test_loader = DataLoader(val, **eval_kwargs), DataLoader(test, **eval_kwargs)
    model = FeasibilityTransformer(FEATURE_MODE, MODEL_DIM).to(DEVICE)
    pos_weight = (total - positives) / np.maximum(positives, 1.0)
    bce_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    best_state, best_score, best_epoch = None, -float("inf"), -1
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for batch in train_loader:
            action, latent, condition, labels, advantage, dynamics_regression = [x.to(DEVICE) for x in batch]
            logits, regression = model(action, latent, condition)
            classification = bce_loss(logits, labels)
            rank = ranking_loss(logits, labels)
            target_regression = torch.cat(
                [advantage.unsqueeze(-1), dynamics_regression], dim=-1
            )
            regression_loss = nn.functional.smooth_l1_loss(regression, target_regression)
            loss = classification + RANK_COEF * rank + REGRESSION_COEF * regression_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        val_metrics = evaluate(model, val_loader)
        score = selection_score(val_metrics)
        if score > best_score:
            best_state, best_score, best_epoch = copy.deepcopy(model.state_dict()), score, epoch
        print(f"epoch={epoch:02d} loss={np.mean(losses):.6f} val_selection={score:.6f}", flush=True)

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    results = {
        "feature_mode": FEATURE_MODE,
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_val_selection": best_score,
        "label_names": LABEL_NAMES,
        "val": evaluate(model, val_loader),
        "test": evaluate(model, test_loader),
    }
    output_dir = OUTPUT_ROOT / FEATURE_MODE / f"seed_{SEED}"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "results": results, "model_dim": MODEL_DIM}, output_dir / "model.pt")
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(results, handle, indent=2, allow_nan=True)
    print(json.dumps(results["test"], indent=2), flush=True)
    print(f"saved={output_dir}", flush=True)


if __name__ == "__main__":
    main()
