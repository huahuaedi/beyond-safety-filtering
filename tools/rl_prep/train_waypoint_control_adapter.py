"""Train and evaluate the standalone DriveVLA waypoint-to-control adapter."""

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

from trajectory_dynamics import analyze_pose_trajectories
from waypoint_control_adapter import WaypointControlAdapter


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("CONTROL_DATA_ROOT", REPO_ROOT / "navsim_data/exp/world_action_full_v1"))
CONTROL_ROOT = Path(os.environ.get("CONTROL_SIDECAR_ROOT", REPO_ROOT / "navsim_data/exp/world_action_controls_full_v1"))
OUTPUT_ROOT = Path(os.environ.get("CONTROL_ADAPTER_OUTPUT", REPO_ROOT / "output/waypoint_control_adapter_v1"))
SEED = int(os.environ.get("CONTROL_ADAPTER_SEED", 0))
EPOCHS = int(os.environ.get("CONTROL_ADAPTER_EPOCHS", 20))
BATCH_SIZE = int(os.environ.get("CONTROL_ADAPTER_BATCH_SIZE", 4))
WORKERS = int(os.environ.get("CONTROL_ADAPTER_WORKERS", 4))
LR = float(os.environ.get("CONTROL_ADAPTER_LR", 2e-4))
MODEL_DIM = int(os.environ.get("CONTROL_ADAPTER_DIM", 128))
MODEL_LAYERS = int(os.environ.get("CONTROL_ADAPTER_LAYERS", 2))
MAX_SCENES = int(os.environ.get("CONTROL_ADAPTER_MAX_SCENES", 0))
CONTROL_COEF = float(os.environ.get("CONTROL_ADAPTER_CONTROL_COEF", 1.0))
POSITION_COEF = float(os.environ.get("CONTROL_ADAPTER_POSITION_COEF", 1.0))
HEADING_COEF = float(os.environ.get("CONTROL_ADAPTER_HEADING_COEF", 2.0))
SMOOTH_COEF = float(os.environ.get("CONTROL_ADAPTER_SMOOTH_COEF", 0.05))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ControlSceneDataset(Dataset):
    def __init__(self, split: str):
        self.paths = sorted((DATA_ROOT / split).glob("*.npz"))
        if MAX_SCENES > 0:
            self.paths = self.paths[:MAX_SCENES]
        if not self.paths:
            raise ValueError(f"no {split} data under {DATA_ROOT}")
        for path in self.paths:
            sidecar = CONTROL_ROOT / split / path.name
            if not sidecar.is_file():
                raise FileNotFoundError(sidecar)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        split = path.parent.name
        with np.load(path, allow_pickle=False) as data:
            relative = data["relative_trajectories"].astype(np.float32)
            absolute = data["absolute_trajectories"].astype(np.float32)
        with np.load(CONTROL_ROOT / split / path.name, allow_pickle=False) as data:
            initial = data["initial_state"].astype(np.float32)
            controls = data["pseudo_controls"].astype(np.float32)
            valid = data["control_supervision_valid"].astype(bool)
        if relative.shape != (64, 8, 3) or initial.shape != (64, 5) or controls.shape != (64, 8, 2):
            raise ValueError(f"unexpected shapes in {path}")
        return tuple(torch.from_numpy(x) for x in (relative, absolute, initial, controls, valid))


def masked_candidates(batch):
    relative, absolute, initial, controls, valid = batch
    valid = valid.bool()
    if not valid.any():
        return None
    return relative[valid], absolute[valid], initial[valid], controls[valid]


def control_scale(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_tensor([6.0, 0.20])


def training_loss(model, selected):
    relative, absolute, initial, target_controls = selected
    output = model(relative, initial)
    predicted_controls = output["controls"]
    predicted_absolute = output["absolute_trajectory"]
    normalized_control = nn.functional.smooth_l1_loss(
        predicted_controls / control_scale(predicted_controls),
        target_controls / control_scale(target_controls),
    )
    position = nn.functional.smooth_l1_loss(
        predicted_absolute[..., :2], absolute[..., :2]
    )
    heading_delta = predicted_absolute[..., 2] - absolute[..., 2]
    heading = (1.0 - heading_delta.cos()).mean()
    normalized = predicted_controls / control_scale(predicted_controls)
    smooth = torch.diff(normalized, dim=-2).square().mean()
    total = (
        CONTROL_COEF * normalized_control
        + POSITION_COEF * position
        + HEADING_COEF * heading
        + SMOOTH_COEF * smooth
    )
    return total, {
        "loss": float(total.detach()),
        "control": float(normalized_control.detach()),
        "position": float(position.detach()),
        "heading": float(heading.detach()),
        "smooth": float(smooth.detach()),
    }


def evaluate(model, loader):
    position_errors, heading_errors = [], []
    control_errors, predicted_trajectories = [], []
    candidates = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            selected = masked_candidates([x.to(DEVICE) for x in batch])
            if selected is None:
                continue
            relative, absolute, initial, target_controls = selected
            output = model(relative, initial)
            predicted = output["absolute_trajectory"]
            position_errors.append(
                torch.linalg.vector_norm(predicted[..., :2] - absolute[..., :2], dim=-1).cpu().numpy()
            )
            heading_delta = predicted[..., 2] - absolute[..., 2]
            heading_errors.append(
                torch.atan2(heading_delta.sin(), heading_delta.cos()).abs().cpu().numpy()
            )
            control_errors.append((output["controls"] - target_controls).abs().cpu().numpy())
            predicted_trajectories.append(predicted.cpu().numpy())
            candidates += len(relative)
    if candidates == 0:
        raise RuntimeError("evaluation split contained no valid candidates")
    position = np.concatenate(position_errors)
    heading = np.concatenate(heading_errors)
    control = np.concatenate(control_errors)
    trajectories = np.concatenate(predicted_trajectories)
    dynamics = analyze_pose_trajectories(trajectories, dt=0.5)
    return {
        "candidates": candidates,
        "position_rmse_mean": float(np.sqrt(np.mean(position**2, axis=1)).mean()),
        "position_rmse_p50": float(np.quantile(np.sqrt(np.mean(position**2, axis=1)), 0.5)),
        "position_rmse_p95": float(np.quantile(np.sqrt(np.mean(position**2, axis=1)), 0.95)),
        "heading_rmse_mean": float(np.sqrt(np.mean(heading**2, axis=1)).mean()),
        "heading_rmse_p50": float(np.quantile(np.sqrt(np.mean(heading**2, axis=1)), 0.5)),
        "heading_rmse_p95": float(np.quantile(np.sqrt(np.mean(heading**2, axis=1)), 0.95)),
        "acceleration_mae": float(control[..., 0].mean()),
        "curvature_rate_mae": float(control[..., 1].mean()),
        "predicted_dynamics_infeasible_fraction": float((~dynamics["dynamics_feasible"]).mean()),
        "predicted_curvature_failure_fraction": float((dynamics["violation_components"][:, 4] > 0).mean()),
        "predicted_curvature_rate_failure_fraction": float((dynamics["violation_components"][:, 5] > 0).mean()),
    }


def selection_score(metrics):
    return metrics["position_rmse_mean"] + 2.0 * metrics["heading_rmse_mean"]


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    train, val, test = (ControlSceneDataset(split) for split in ("train", "val", "test"))
    print(
        f"seed={SEED} device={DEVICE} scenes train={len(train)} val={len(val)} test={len(test)}",
        flush=True,
    )
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS,
        pin_memory=True, persistent_workers=WORKERS > 0, generator=generator,
    )
    eval_args = dict(batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)
    val_loader, test_loader = DataLoader(val, **eval_args), DataLoader(test, **eval_args)
    model = WaypointControlAdapter(dim=MODEL_DIM, layers=MODEL_LAYERS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    best_state, best_score, best_epoch = None, float("inf"), -1
    for epoch in range(EPOCHS):
        model.train()
        epoch_values = []
        for batch in train_loader:
            selected = masked_candidates([x.to(DEVICE) for x in batch])
            if selected is None:
                continue
            loss, values = training_loss(model, selected)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_values.append(values)
        val_metrics = evaluate(model, val_loader)
        score = selection_score(val_metrics)
        if score < best_score:
            best_state, best_score, best_epoch = copy.deepcopy(model.state_dict()), score, epoch
        means = {key: float(np.mean([x[key] for x in epoch_values])) for key in epoch_values[0]}
        print(
            f"epoch={epoch:02d} loss={means['loss']:.6f} control={means['control']:.6f} "
            f"position={means['position']:.6f} heading={means['heading']:.6f} "
            f"val_pos={val_metrics['position_rmse_mean']:.6f} "
            f"val_yaw={val_metrics['heading_rmse_mean']:.6f}", flush=True,
        )
    if best_state is None:
        raise RuntimeError("no checkpoint produced")
    model.load_state_dict(best_state)
    results = {
        "seed": SEED, "best_epoch": best_epoch, "best_val_score": best_score,
        "model": model.metadata(), "loss_weights": {
            "control": CONTROL_COEF, "position": POSITION_COEF,
            "heading": HEADING_COEF, "smooth": SMOOTH_COEF,
        },
        "val": evaluate(model, val_loader), "test": evaluate(model, test_loader),
    }
    output = OUTPUT_ROOT / f"seed_{SEED}"
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "results": results, **model.metadata()}, output / "model.pt")
    (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["test"], indent=2), flush=True)
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()

