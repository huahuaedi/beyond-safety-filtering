"""Train scene-disjoint risk probes on verified world-action latents.

Feature modes provide the critical controls: action-only, each individual
world-model depth, and all depths. Labels remain vector-valued so progress
cannot compensate for NC/DAC/TTC/reference-relative safety failures.
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("WM_ACTION_DATA_ROOT", REPO_ROOT / "navsim_data/exp/world_action_v1"))
LATENT_ROOT = Path(os.environ.get("WM_LATENT_ROOT", REPO_ROOT / "navsim_data/exp/world_action_latents_v1"))
OUTPUT_ROOT = Path(os.environ.get("WM_RISK_PROBE_OUTPUT", REPO_ROOT / "output/world_action_risk_probe_v1"))
FEATURE_MODE = os.environ.get("WM_RISK_FEATURE_MODE", "action")
SEED = int(os.environ.get("WM_RISK_SEED", 0))
EPOCHS = int(os.environ.get("WM_RISK_EPOCHS", 30))
BATCH_SIZE = int(os.environ.get("WM_RISK_BATCH_SIZE", 256))
LR = float(os.environ.get("WM_RISK_LR", 3e-4))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_NAMES = ("unsafe", "ttc_failure", "dac_failure", "nc_failure")


def feature_from_files(label_data, latent_data):
    if FEATURE_MODE == "action":
        return label_data["relative_trajectories"].reshape(-1, 24).astype(np.float32)
    latents = latent_data["eoa_latents"].astype(np.float32)
    if FEATURE_MODE.startswith("layer"):
        layer = int(FEATURE_MODE.removeprefix("layer"))
        if not 0 <= layer < latents.shape[1]:
            raise ValueError(f"bad layer mode {FEATURE_MODE} for {latents.shape}")
        return latents[:, layer]
    if FEATURE_MODE == "latent_all":
        return latents.reshape(latents.shape[0], -1)
    raise ValueError(f"unknown WM_RISK_FEATURE_MODE={FEATURE_MODE!r}")


def load_split(split):
    features, labels, scene_ids = [], [], []
    label_files = sorted((DATA_ROOT / split).glob("*.npz"))
    for scene_id, label_path in enumerate(label_files):
        latent_path = LATENT_ROOT / split / label_path.name
        if not latent_path.is_file():
            raise FileNotFoundError(latent_path)
        with np.load(label_path, allow_pickle=False) as label_data, np.load(
            latent_path, allow_pickle=False
        ) as latent_data:
            x = feature_from_files(label_data, latent_data)
            feasible = label_data["reference_feasible"].astype(bool)
            y = np.stack(
                [
                    ~feasible,
                    label_data["score__time_to_collision_within_bound"] < 1.0 - 1e-8,
                    label_data["score__drivable_area_compliance"] < 1.0 - 1e-8,
                    label_data["score__no_at_fault_collisions"] < 1.0 - 1e-8,
                ],
                axis=1,
            ).astype(np.float32)
        features.append(x)
        labels.append(y)
        scene_ids.extend([scene_id] * len(x))
    return (
        np.concatenate(features),
        np.concatenate(labels),
        np.asarray(scene_ids, dtype=np.int64),
        len(label_files),
    )


class RiskProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, len(LABEL_NAMES)),
        )

    def forward(self, x):
        return self.net(x)


def average_precision(y, score):
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    ranked = y[order]
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


def recall_at_k(y, score, scene_ids, k=16):
    captured = total = 0
    for scene_id in np.unique(scene_ids):
        idx = np.flatnonzero(scene_ids == scene_id)
        positives = int(y[idx].sum())
        if positives == 0:
            continue
        chosen = idx[np.argsort(-score[idx], kind="stable")[:k]]
        captured += int(y[chosen].sum())
        total += positives
    return float(captured / total) if total else float("nan")


def evaluate(model, x, y, scene_ids):
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(x), BATCH_SIZE):
            logits = model(torch.from_numpy(x[start:start + BATCH_SIZE]).to(DEVICE))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    probability = np.concatenate(probabilities)
    metrics = {}
    for label_i, name in enumerate(LABEL_NAMES):
        target, score = y[:, label_i], probability[:, label_i]
        metrics[name] = {
            "prevalence": float(target.mean()),
            "auroc": auroc(target, score),
            "average_precision": average_precision(target, score),
            "ece": expected_calibration_error(target, score),
            "recall_at_16": recall_at_k(target, score, scene_ids, 16),
            "random_recall_at_16": 0.25,
        }
    return metrics


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"feature_mode={FEATURE_MODE} seed={SEED} device={DEVICE}", flush=True)
    train_x, train_y, _, train_scenes = load_split("train")
    val_x, val_y, val_scene_ids, val_scenes = load_split("val")
    test_x, test_y, test_scene_ids, test_scenes = load_split("test")
    print(
        f"train={train_x.shape}/{train_scenes} val={val_x.shape}/{val_scenes} "
        f"test={test_x.shape}/{test_scenes}", flush=True,
    )

    model = RiskProbe(train_x.shape[1]).to(DEVICE)
    positives = train_y.sum(axis=0)
    pos_weight = (len(train_y) - positives) / np.maximum(positives, 1.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=BATCH_SIZE, shuffle=True, generator=generator,
    )

    best_state, best_val_ap = None, -float("inf")
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x.to(DEVICE)), batch_y.to(DEVICE))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_metrics = evaluate(model, val_x, val_y, val_scene_ids)
        val_ap = np.nanmean([x["average_precision"] for x in val_metrics.values()])
        if val_ap > best_val_ap:
            best_val_ap = float(val_ap)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch={epoch:02d} loss={np.mean(losses):.6f} val_mean_ap={val_ap:.6f}",
            flush=True,
        )

    model.load_state_dict(best_state)
    results = {
        "feature_mode": FEATURE_MODE,
        "seed": SEED,
        "best_val_mean_ap": best_val_ap,
        "val": evaluate(model, val_x, val_y, val_scene_ids),
        "test": evaluate(model, test_x, test_y, test_scene_ids),
    }
    output_dir = OUTPUT_ROOT / FEATURE_MODE / f"seed_{SEED}"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_dir / "probe.pt")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, allow_nan=True)
    print(json.dumps(results["test"], indent=2), flush=True)
    print(f"saved={output_dir}", flush=True)


if __name__ == "__main__":
    main()

