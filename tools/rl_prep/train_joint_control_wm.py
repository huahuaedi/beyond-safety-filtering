"""Distill a continuous WM-risk student and jointly tune the control adapter."""

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

from continuous_scene_risk_critic import ContinuousSceneRiskCritic
from trajectory_dynamics import analyze_pose_trajectories
from waypoint_control_adapter import WaypointControlAdapter
from wm_mean_risk_critic import WMMeanRiskCritic


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("JOINT_DATA_ROOT", REPO_ROOT / "navsim_data/exp/world_action_full_v1"))
LATENT_ROOT = Path(os.environ.get("JOINT_LATENT_ROOT", REPO_ROOT / "navsim_data/exp/world_action_latents_full_v1"))
CONTROL_ROOT = Path(os.environ.get("JOINT_CONTROL_ROOT", REPO_ROOT / "navsim_data/exp/world_action_controls_full_v1"))
ADAPTER_CKPT = Path(os.environ.get("JOINT_ADAPTER_CHECKPOINT", REPO_ROOT / "output/waypoint_control_adapter_v1/seed_0/model.pt"))
TEACHER_CKPT = Path(os.environ["JOINT_WM_TEACHER_CHECKPOINT"])
OUTPUT_ROOT = Path(os.environ.get("JOINT_OUTPUT", REPO_ROOT / "output/joint_control_wm_v1"))
STUDENT_CKPT = os.environ.get("JOINT_STUDENT_CHECKPOINT", "")
SEED = int(os.environ.get("JOINT_SEED", 0))
STUDENT_EPOCHS = int(os.environ.get("JOINT_STUDENT_EPOCHS", 12))
ADAPTER_EPOCHS = int(os.environ.get("JOINT_ADAPTER_EPOCHS", 8))
BATCH_SIZE = int(os.environ.get("JOINT_BATCH_SIZE", 2))
WORKERS = int(os.environ.get("JOINT_WORKERS", 4))
MAX_SCENES = int(os.environ.get("JOINT_MAX_SCENES", 0))
RISK_LAMBDAS = [float(x) for x in os.environ.get("JOINT_RISK_LAMBDAS", "0,0.01,0.05").split(",")]
DISTILL_COEF = float(os.environ.get("JOINT_DISTILL_COEF", 0.5))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError as error:
        if "weights_only" not in str(error):
            raise
        return torch.load(path, map_location="cpu")


class JointSceneDataset(Dataset):
    def __init__(self, split):
        self.paths = sorted((DATA_ROOT / split).glob("*.npz"))
        if MAX_SCENES > 0:
            self.paths = self.paths[:MAX_SCENES]
        if not self.paths:
            raise ValueError(f"no {split} scenes")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        split = path.parent.name
        with np.load(path, allow_pickle=False) as data:
            relative = data["relative_trajectories"].astype(np.float32)
            absolute = data["absolute_trajectories"].astype(np.float32)
            labels = np.stack([
                ~data["reference_feasible"].astype(bool),
                data["score__time_to_collision_within_bound"] < 1.0 - 1e-8,
                data["score__drivable_area_compliance"] < 1.0 - 1e-8,
                data["score__no_at_fault_collisions"] < 1.0 - 1e-8,
            ], axis=-1).astype(np.float32)
        with np.load(LATENT_ROOT / split / path.name, allow_pickle=False) as data:
            latent = data["eoa_latents"].astype(np.float32)
        with np.load(CONTROL_ROOT / split / path.name, allow_pickle=False) as data:
            initial = data["initial_state"].astype(np.float32)
            controls = data["pseudo_controls"].astype(np.float32)
            valid = data["control_supervision_valid"].astype(bool)
        return tuple(torch.from_numpy(x) for x in (
            relative, absolute, labels, latent, initial, controls, valid,
        ))


def make_loaders():
    generator = torch.Generator().manual_seed(SEED)
    train, val, test = (JointSceneDataset(x) for x in ("train", "val", "test"))
    common = dict(batch_size=BATCH_SIZE, num_workers=WORKERS, pin_memory=True)
    return (
        DataLoader(train, shuffle=True, generator=generator, **common),
        DataLoader(val, shuffle=False, **common),
        DataLoader(test, shuffle=False, **common),
        (len(train), len(val), len(test)),
    )


def train_student(student, teacher, train_loader, val_loader):
    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4, weight_decay=1e-4)
    pos_weight = torch.tensor([18.0, 35.0, 22.0, 80.0], device=DEVICE)
    best_state, best_loss, best_epoch = None, float("inf"), -1
    for epoch in range(STUDENT_EPOCHS):
        student.train()
        values = []
        for batch in train_loader:
            relative, _, labels, latent, *_ = [x.to(DEVICE) for x in batch]
            action = relative.flatten(-2)
            context = student.scene_context(latent)
            logits = student(action, context)
            with torch.no_grad():
                teacher_probability = teacher.probabilities(action, latent)
            supervised = nn.functional.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight
            )
            distill = nn.functional.binary_cross_entropy_with_logits(
                logits, teacher_probability
            )
            loss = supervised + DISTILL_COEF * distill
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()
            values.append(float(loss.detach()))
        val_loss = evaluate_student(student, val_loader)
        if val_loss < best_loss:
            best_state, best_loss, best_epoch = copy.deepcopy(student.state_dict()), val_loss, epoch
        print(f"student epoch={epoch:02d} train={np.mean(values):.6f} val={val_loss:.6f}", flush=True)
    student.load_state_dict(best_state)
    return best_epoch, best_loss, best_state


def evaluate_student(student, loader):
    student.eval()
    values = []
    with torch.inference_mode():
        for batch in loader:
            relative, _, labels, latent, *_ = [x.to(DEVICE) for x in batch]
            logits = student(relative.flatten(-2), student.scene_context(latent))
            values.append(float(nn.functional.binary_cross_entropy_with_logits(logits, labels)))
    return float(np.mean(values))


def adapter_loss(adapter, student, batch, risk_lambda):
    relative, absolute, _, latent, initial, target_controls, valid = batch
    context = student.scene_context(latent)
    expanded_context = context[:, None, :].expand(-1, relative.shape[1], -1)
    valid = valid.bool()
    # A scene batch can contain no candidates that passed pseudo-control
    # supervision. Reductions over that empty selection return NaN and used to
    # silently poison the entire epoch/checkpoint.
    if not bool(valid.any()):
        return None
    relative, absolute = relative[valid], absolute[valid]
    initial, target_controls = initial[valid], target_controls[valid]
    selected_context = expanded_context[valid]
    output = adapter(relative, initial)
    predicted = output["absolute_trajectory"]
    control_scale = predicted.new_tensor([6.0, 0.20])
    control = nn.functional.smooth_l1_loss(
        output["controls"] / control_scale, target_controls / control_scale
    )
    position = nn.functional.smooth_l1_loss(predicted[..., :2], absolute[..., :2])
    yaw_delta = predicted[..., 2] - absolute[..., 2]
    heading = (1.0 - yaw_delta.cos()).mean()
    smooth = torch.diff(output["controls"] / control_scale, dim=-2).square().mean()
    risk = student.risk(output["relative_trajectory"].flatten(-2), selected_context).mean()
    total = control + position + 2.0 * heading + 0.05 * smooth + risk_lambda * risk
    return total, risk


def tune_adapter(base_payload, student, train_loader, val_loader, risk_lambda):
    adapter = WaypointControlAdapter(
        dim=int(base_payload.get("dim", 128)), layers=int(base_payload.get("layers", 2)),
        heads=int(base_payload.get("heads", 4)),
    ).to(DEVICE)
    adapter.load_state_dict(base_payload["model"])
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=5e-5, weight_decay=1e-4)
    best_state, best_score, best_epoch = copy.deepcopy(adapter.state_dict()), float("inf"), -1
    for epoch in range(ADAPTER_EPOCHS):
        adapter.train()
        losses, risks = [], []
        for batch in train_loader:
            batch = [x.to(DEVICE) for x in batch]
            result = adapter_loss(adapter, student, batch, risk_lambda)
            if result is None:
                continue
            loss, risk = result
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite adapter loss for lambda={risk_lambda:g}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                adapter.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss.detach())); risks.append(float(risk.detach()))
        if not losses:
            raise RuntimeError("adapter epoch contained no valid control supervision")
        score = evaluate_adapter_objective(adapter, student, val_loader, risk_lambda)
        if not np.isfinite(score):
            raise FloatingPointError(
                f"non-finite validation objective for lambda={risk_lambda:g}"
            )
        if score < best_score:
            best_state, best_score, best_epoch = copy.deepcopy(adapter.state_dict()), score, epoch
        print(
            f"adapter lambda={risk_lambda:g} epoch={epoch:02d} loss={np.mean(losses):.6f} "
            f"risk={np.mean(risks):.6f} val={score:.6f}", flush=True,
        )
    adapter.load_state_dict(best_state)
    return adapter, best_state, best_epoch, best_score


def evaluate_adapter_objective(adapter, student, loader, risk_lambda):
    adapter.eval(); values = []
    with torch.inference_mode():
        for batch in loader:
            result = adapter_loss(
                adapter, student, [x.to(DEVICE) for x in batch], risk_lambda
            )
            if result is None:
                continue
            loss, _ = result
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite adapter validation loss for lambda={risk_lambda:g}"
                )
            values.append(float(loss))
    if not values:
        raise RuntimeError("validation loader contained no valid control supervision")
    return float(np.mean(values))


def evaluate_adapter(adapter, student, loader):
    adapter.eval(); position, heading, risks, trajectories = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            relative, absolute, _, latent, initial, _, valid = [x.to(DEVICE) for x in batch]
            context = student.scene_context(latent)[:, None, :].expand(-1, 64, -1)
            valid = valid.bool()
            output = adapter(relative[valid], initial[valid])
            predicted = output["absolute_trajectory"]
            position.append(torch.linalg.vector_norm(predicted[..., :2] - absolute[valid][..., :2], dim=-1).cpu().numpy())
            delta = predicted[..., 2] - absolute[valid][..., 2]
            heading.append(torch.atan2(delta.sin(), delta.cos()).abs().cpu().numpy())
            risks.append(student.risk(output["relative_trajectory"].flatten(-2), context[valid]).cpu().numpy())
            trajectories.append(predicted.cpu().numpy())
    position = np.concatenate(position); heading = np.concatenate(heading)
    dynamics = analyze_pose_trajectories(np.concatenate(trajectories), dt=0.5)
    return {
        "position_rmse_mean": float(np.sqrt(np.mean(position**2, axis=1)).mean()),
        "heading_rmse_mean": float(np.sqrt(np.mean(heading**2, axis=1)).mean()),
        "student_risk_mean": float(np.concatenate(risks).mean()),
        "dynamics_infeasible_fraction": float((~dynamics["dynamics_feasible"]).mean()),
    }


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    train_loader, val_loader, test_loader, sizes = make_loaders()
    print(f"device={DEVICE} seed={SEED} scenes={sizes} lambdas={RISK_LAMBDAS}", flush=True)
    student = ContinuousSceneRiskCritic().to(DEVICE)
    if STUDENT_CKPT:
        student_payload = load_torch(Path(STUDENT_CKPT))
        student.load_state_dict(student_payload["model"])
        student_state = copy.deepcopy(student.state_dict())
        best_epoch = int(student_payload.get("best_epoch", -1))
        best_loss = float(student_payload.get("best_val_loss", float("nan")))
        print(
            f"loaded continuous risk student={STUDENT_CKPT} "
            f"best_epoch={best_epoch} best_val_loss={best_loss}",
            flush=True,
        )
    else:
        teacher = WMMeanRiskCritic.from_checkpoint(
            TEACHER_CKPT, map_location=DEVICE
        ).to(DEVICE)
        best_epoch, best_loss, student_state = train_student(
            student, teacher, train_loader, val_loader
        )
        teacher.cpu(); del teacher
    for parameter in student.parameters(): parameter.requires_grad_(False)
    student.eval()
    base_payload = load_torch(ADAPTER_CKPT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": student_state, "dim": student.dim, "latent_dim": student.latent_dim,
        "best_epoch": best_epoch, "best_val_loss": best_loss,
    }, OUTPUT_ROOT / "continuous_risk_student.pt")
    summary = {"seed": SEED, "student_best_epoch": best_epoch, "student_val_loss": best_loss, "variants": {}}
    for risk_lambda in RISK_LAMBDAS:
        adapter, state, epoch, score = tune_adapter(base_payload, student, train_loader, val_loader, risk_lambda)
        metrics = evaluate_adapter(adapter, student, test_loader)
        name = f"risk_{risk_lambda:g}".replace(".", "p")
        directory = OUTPUT_ROOT / name
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"model": state, **adapter.metadata(), "risk_lambda": risk_lambda}, directory / "model.pt")
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        summary["variants"][name] = {"best_epoch": epoch, "best_val": score, **metrics}
        print(f"saved={directory} metrics={metrics}", flush=True)
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"saved={OUTPUT_ROOT / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
