"""Train the unified WM/physics-conditioned correction policy.

Phase A learns a differentiable outcome model (risk + progress) from verified
world-action data. Phase B identity-warm-starts the residual policy and then
optimizes the same policy with a reference-anchored constrained RL objective.
No post-hoc adapter, selector, or independent correction network is involved.
"""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from continuous_scene_outcome_critic import ContinuousSceneOutcomeCritic
from unified_correction_policy import UnifiedCorrectionPolicy
from unified_correction_rl import CorrectionRLConfig, UnifiedCorrectionRL


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get(
    "UNIFIED_DATA_ROOT", REPO_ROOT / "navsim_data/exp/world_action_full_v1"
))
LATENT_ROOT = Path(os.environ.get(
    "UNIFIED_LATENT_ROOT", REPO_ROOT / "navsim_data/exp/world_action_latents_full_v1"
))
CONTROL_ROOT = Path(os.environ.get(
    "UNIFIED_CONTROL_ROOT", REPO_ROOT / "navsim_data/exp/world_action_controls_full_v1"
))
OUTPUT_ROOT = Path(os.environ.get(
    "UNIFIED_OUTPUT", REPO_ROOT / "output/unified_correction_rl_v1"
))
SEED = int(os.environ.get("UNIFIED_SEED", 0))
OUTCOME_EPOCHS = int(os.environ.get("UNIFIED_OUTCOME_EPOCHS", 12))
WARMUP_EPOCHS = int(os.environ.get("UNIFIED_WARMUP_EPOCHS", 4))
RL_EPOCHS = int(os.environ.get("UNIFIED_RL_EPOCHS", 8))
BATCH_SIZE = int(os.environ.get("UNIFIED_BATCH_SIZE", 2))
WORKERS = int(os.environ.get("UNIFIED_WORKERS", 8))
CANDIDATES_PER_SCENE = int(os.environ.get("UNIFIED_CANDIDATES_PER_SCENE", 8))
PROJECTION_ITERATIONS = int(os.environ.get("UNIFIED_PROJECTION_ITERATIONS", 30))
MAX_SCENES = int(os.environ.get("UNIFIED_MAX_SCENES", 0))
OUTCOME_LR = float(os.environ.get("UNIFIED_OUTCOME_LR", 2e-4))
POLICY_LR = float(os.environ.get("UNIFIED_POLICY_LR", 1e-4))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTCOME_ONLY = os.environ.get("UNIFIED_OUTCOME_ONLY", "0") == "1"


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


class SceneDataset(Dataset):
    def __init__(self, split: str):
        self.paths = sorted((DATA_ROOT / split).glob("*.npz"))
        if MAX_SCENES > 0:
            self.paths = self.paths[:MAX_SCENES]
        if not self.paths:
            raise ValueError(f"no scenes found in {DATA_ROOT / split}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        split = path.parent.name
        with np.load(path, allow_pickle=False) as data:
            relative = data["relative_trajectories"].astype(np.float32)
            reference_feasible = data["reference_feasible"].astype(bool)
            risk = np.stack([
                ~reference_feasible,
                data["score__time_to_collision_within_bound"] < 1.0 - 1e-8,
                data["score__drivable_area_compliance"] < 1.0 - 1e-8,
                data["score__no_at_fault_collisions"] < 1.0 - 1e-8,
            ], -1).astype(np.float32)
            progress = data["score__ego_progress"].astype(np.float32)
            direction = data["score__driving_direction_compliance"].astype(np.float32)
            comfort = data["score__comfort"].astype(np.float32)
            violation = data["reference_violation"].astype(np.float32)
        with np.load(LATENT_ROOT / split / path.name, allow_pickle=False) as data:
            latent = data["eoa_latents"].astype(np.float32)
        with np.load(CONTROL_ROOT / split / path.name, allow_pickle=False) as data:
            initial = data["initial_state"].astype(np.float32)
            controls = data["pseudo_controls"].astype(np.float32)
            valid = data["control_supervision_valid"].astype(bool)
            dynamics_feasible = data["dynamics_feasible"].astype(bool)
        values = (
            relative, risk, progress, direction, comfort, violation, latent,
            initial, controls, valid, reference_feasible, dynamics_feasible,
        )
        return tuple(torch.from_numpy(value) for value in values)


def loaders():
    generator = torch.Generator().manual_seed(SEED)
    datasets = [SceneDataset(split) for split in ("train", "val", "test")]
    common = dict(batch_size=BATCH_SIZE, num_workers=WORKERS, pin_memory=True)
    return (
        DataLoader(datasets[0], shuffle=True, generator=generator, **common),
        DataLoader(datasets[1], shuffle=False, **common),
        DataLoader(datasets[2], shuffle=False, **common),
        tuple(map(len, datasets)),
    )


def move(batch):
    return [value.to(DEVICE, non_blocking=True) for value in batch]


def outcome_loss(model, batch):
    relative, risk, progress, _, _, _, latent, *_ = batch
    context = model.scene_context(latent)
    context = context[:, None].expand(-1, relative.shape[1], -1)
    output = model(relative.flatten(-2), context)
    positive_weight = risk.new_tensor([18.0, 35.0, 22.0, 80.0])
    risk_loss = nn.functional.binary_cross_entropy_with_logits(
        output["risk_logits"], risk, pos_weight=positive_weight
    )
    progress_loss = nn.functional.smooth_l1_loss(output["progress"], progress)
    return risk_loss + progress_loss, risk_loss, progress_loss


def evaluate_outcome(model, loader):
    model.eval()
    values = []
    with torch.inference_mode():
        for batch in loader:
            loss, risk, progress = outcome_loss(model, move(batch))
            values.append((float(loss), float(risk), float(progress)))
    return np.asarray(values).mean(0).tolist()


def train_outcome(model, train_loader, val_loader):
    checkpoint = os.environ.get("UNIFIED_OUTCOME_CHECKPOINT", "")
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(payload["model"])
        print(f"loaded frozen outcome critic={checkpoint}", flush=True)
        return payload.get("best_epoch", -1), payload.get("best_val", float("nan"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=OUTCOME_LR, weight_decay=1e-4)
    best_state, best_val, best_epoch = None, float("inf"), -1
    for epoch in range(OUTCOME_EPOCHS):
        model.train(); losses = []
        for batch in train_loader:
            loss, _, _ = outcome_loss(model, move(batch))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
        validation = evaluate_outcome(model, val_loader)
        if validation[0] < best_val:
            best_val, best_epoch = validation[0], epoch
            best_state = cpu_state(model)
        print(
            f"outcome epoch={epoch:02d} train={np.mean(losses):.6f} "
            f"val={validation[0]:.6f} risk={validation[1]:.6f} "
            f"progress={validation[2]:.6f}", flush=True,
        )
    model.load_state_dict(best_state)
    return best_epoch, best_val


def select_training_candidates(batch):
    (
        relative, risk, progress, direction, comfort, violation, latent,
        initial, controls, valid, reference_feasible, dynamics_feasible,
    ) = batch
    batch_size, candidates = relative.shape[:2]
    count = min(CANDIDATES_PER_SCENE, candidates)
    sampled = torch.stack([
        torch.randperm(candidates, device=relative.device)[:count]
        for _ in range(batch_size)
    ])
    scene = torch.arange(batch_size, device=relative.device)[:, None]
    proposal = relative[scene, sampled].reshape(-1, 8, 3)
    proposal_initial = initial[scene, sampled].reshape(-1, 5)
    proposal_bad = (
        (~reference_feasible[scene, sampled])
        | (~dynamics_feasible[scene, sampled])
        | (~valid[scene, sampled])
    ).reshape(-1)

    eligible = valid & reference_feasible & dynamics_feasible
    utility = progress + 0.25 * direction + 0.10 * comfort - violation
    utility = utility.masked_fill(~eligible, -torch.inf)
    # Every generated scene normally has feasible controls. Fall back to the
    # least-violating valid candidate rather than manufacturing a target.
    no_eligible = ~eligible.any(-1)
    fallback = (-violation).masked_fill(~valid, -torch.inf)
    best = utility.argmax(-1)
    best = torch.where(no_eligible, fallback.argmax(-1), best)
    target_relative = relative[torch.arange(batch_size, device=relative.device), best]
    target_controls = controls[torch.arange(batch_size, device=relative.device), best]
    target_relative = target_relative[:, None].expand(-1, count, -1, -1).reshape(-1, 8, 3)
    target_controls = target_controls[:, None].expand(-1, count, -1, -1).reshape(-1, 8, 2)
    # Safe proposals are identity examples; only bad proposals use correction
    # targets. This preserves the strong original planner by construction.
    selected_target_relative = torch.where(
        proposal_bad[:, None, None], target_relative, proposal
    )
    selected_target_controls = torch.where(
        proposal_bad[:, None, None], target_controls,
        controls[scene, sampled].reshape(-1, 8, 2),
    )
    return {
        "proposal": proposal,
        "initial": proposal_initial,
        "bad": proposal_bad.float(),
        "target_relative": selected_target_relative,
        "target_controls": selected_target_controls,
        "latent": latent,
        "count": count,
    }


def correction_batch(batch, outcome):
    selected = select_training_candidates(batch)
    with torch.no_grad():
        context = outcome.scene_context(selected["latent"])
    context = context[:, None].expand(-1, selected["count"], -1).reshape(-1, context.shape[-1])
    selected["context"] = context
    return selected


def warmup_loss(policy, outcome, selected, projected=None):
    # tau^0 -> tau^ref via project() first (no trainable parameters), then
    # risk on tau^ref -- the trajectory correct_projected() actually
    # conditions on -- not on the raw proposal. See
    # docs/reference_anchored_correction_rl.tex Step 3 implementation note.
    #
    # ``projected`` lets a caller that already projected this exact
    # (proposal, initial_state) pair -- e.g. train_policy()'s RL loop, which
    # also needs tau^ref for the RL forward pass on the same batch -- reuse
    # it instead of paying for a second identical LBFGS solve.
    if projected is None:
        projected = policy.project(selected["proposal"], initial_state=selected["initial"])
    with torch.no_grad():
        risk = outcome.outcomes(
            projected["reference_relative"].flatten(-2), selected["context"]
        )["risk"]
    output = policy.correct_projected(
        projected, risk, selected["context"],
    )
    position = nn.functional.smooth_l1_loss(
        output["corrected_relative"][..., :2], selected["target_relative"][..., :2]
    )
    heading_delta = output["corrected_relative"][..., 2] - selected["target_relative"][..., 2]
    heading = (1.0 - heading_delta.cos()).mean()
    control_scale = output["corrected_controls"].new_tensor([6.0, 0.2])
    target_controls = torch.where(
        selected["bad"][:, None, None].bool(), selected["target_controls"],
        output["reference_controls"].detach(),
    )
    control = nn.functional.smooth_l1_loss(
        output["corrected_controls"] / control_scale,
        target_controls / control_scale,
    )
    gate = nn.functional.binary_cross_entropy_with_logits(
        output["gate_logits"], selected["bad"],
        pos_weight=output["gate_logits"].new_tensor(8.0),
    )
    physics = output["corrected_physics_violation"].mean()
    loss = position + 2.0 * heading + control + 0.25 * gate + 0.5 * physics
    return loss, {"position": position, "heading": heading, "control": control, "gate": gate, "physics": physics}


def validation_selection_score(metrics: dict) -> float:
    """Checkpoint-selection score from evaluate_policy()'s held-out metrics.

    Mirrors constrained_correction_objective's own reward shape (progress
    gain relative to tau^ref, hard-penalized for any risk regression or
    remaining physics violation -- see
    docs/reference_anchored_correction_rl.tex's Reward Design section) but
    computed deterministically over the whole validation set rather than
    from a single stochastic RL rollout on a training batch. The regression/
    physics coefficients are large relative to typical progress-gain
    magnitudes so a checkpoint that regresses risk or leaves the trajectory
    infeasible cannot be selected just for scoring well on progress.
    """
    progress_gain = metrics["progress_after"] - metrics["progress_reference"]
    risk_regret = max(0.0, metrics["risk_after"] - metrics["risk_reference"])
    return progress_gain - 4.0 * risk_regret - 2.0 * metrics["physics_after"]


def train_policy(policy, outcome, train_loader, val_loader):
    optimizer = torch.optim.AdamW(policy.parameters(), lr=POLICY_LR, weight_decay=1e-4)
    for epoch in range(WARMUP_EPOCHS):
        policy.train(); values = []
        for batch in train_loader:
            selected = correction_batch(move(batch), outcome)
            loss, _ = warmup_loss(policy, outcome, selected)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 2.0, error_if_nonfinite=True)
            optimizer.step(); values.append(float(loss.detach()))
        print(f"warmup epoch={epoch:02d} loss={np.mean(values):.6f}", flush=True)

    rl = UnifiedCorrectionRL(policy, outcome, CorrectionRLConfig()).to(DEVICE)
    best_state, best_value, best_epoch = cpu_state(policy), -float("inf"), -1
    for epoch in range(RL_EPOCHS):
        policy.train(); rewards = []; feasible = []; losses = []
        for batch in train_loader:
            selected = correction_batch(move(batch), outcome)
            # project() is parameter-free and deterministic in
            # (proposal, initial_state) -- compute tau^ref once per batch and
            # share it between the RL forward pass and the imitation anchor
            # below instead of paying for a second identical LBFGS solve (and
            # risking the two calls drifting apart as separately-optimized
            # references instead of the same tau^ref).
            projected = policy.project(
                selected["proposal"], initial_state=selected["initial"],
            )
            output = rl(
                selected["proposal"], selected["context"],
                projected=projected, sample_correction=True,
            )
            # A small imitation anchor prevents exploitation of imperfect WM
            # gradients while the constrained policy gradient drives utility.
            anchor, _ = warmup_loss(policy, outcome, selected, projected=projected)
            loss = output["loss"] + 0.10 * anchor
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 2.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            rewards.extend(output["reward"].detach().cpu().tolist())
            feasible.extend(output["feasible"].detach().cpu().tolist())
        train_reward = float(np.mean(rewards))
        # Select the checkpoint by held-out validation performance, not by
        # training-batch RL reward: the RL objective's per-sample variance
        # and the correction policy's tendency to overfit a small warm-start
        # dataset both make train-reward an unreliable "best epoch" signal.
        # evaluate_policy() already computes exactly the reference-relative
        # quantities (progress_after vs. progress_reference, risk_after vs.
        # risk_reference, physics_after) that
        # docs/reference_anchored_correction_rl.tex's Reward Design section
        # defines the correction objective against -- val_loader was already
        # threaded into this function but previously unused for selection.
        val_metrics = evaluate_policy(policy, outcome, val_loader)
        value = validation_selection_score(val_metrics)
        if value > best_value:
            best_state, best_value, best_epoch = cpu_state(policy), value, epoch
        print(
            f"rl epoch={epoch:02d} loss={np.mean(losses):.6f} "
            f"train_reward={train_reward:.6f} feasible={np.mean(feasible):.6f} "
            f"val_score={value:.6f} "
            f"val_progress_gain={val_metrics['progress_after'] - val_metrics['progress_reference']:.6f} "
            f"val_risk_regret={max(0.0, val_metrics['risk_after'] - val_metrics['risk_reference']):.6f} "
            f"val_physics_after={val_metrics['physics_after']:.6f}", flush=True,
        )
    policy.load_state_dict(best_state)
    return best_epoch, best_value


def evaluate_policy(policy, outcome, loader):
    policy.eval(); metrics = {name: [] for name in (
        "position_change", "risk_raw_proposal", "risk_reference", "risk_after",
        "progress_raw_proposal", "progress_reference", "progress_after",
        "physics_before", "physics_after", "gate",
    )}
    # no_grad(), not inference_mode(): policy.project() runs an internal
    # LBFGS solve under torch.enable_grad() (see project_to_feasible_reference
    # in unified_correction_policy.py), and enable_grad() cannot re-enable
    # autograd for tensors created inside an outer inference_mode() block --
    # it silently produces a loss with no grad_fn and crashes on backward().
    with torch.no_grad():
        for batch in loader:
            selected = correction_batch(move(batch), outcome)
            # `proposal_outcome` (tau^0) is reported for the three-way
            # (tau^0/tau^ref/tau^1) breakdown but does not condition the
            # policy. `reference` (tau^ref) is both what conditions the
            # correction and the before/after baseline: comparisons are
            # always tau^1 vs. tau^ref, never tau^1 vs. tau^0, per
            # docs/reference_anchored_correction_rl.tex's Reward Design and
            # Training-Time Critic sections.
            proposal_outcome = outcome.outcomes(selected["proposal"].flatten(-2), selected["context"])
            projected = policy.project(selected["proposal"], initial_state=selected["initial"])
            reference = outcome.outcomes(
                projected["reference_relative"].flatten(-2), selected["context"]
            )
            output = policy.correct_projected(
                projected, reference["risk"], selected["context"],
            )
            after = outcome.outcomes(output["corrected_relative"].flatten(-2), selected["context"])
            delta = output["corrected_absolute"][..., :2] - output["reference_states"][..., :2]
            values = {
                "position_change": torch.linalg.vector_norm(delta, dim=-1).mean(-1),
                "risk_raw_proposal": proposal_outcome["risk"].mean(-1),
                "risk_reference": reference["risk"].mean(-1),
                "risk_after": after["risk"].mean(-1),
                "progress_raw_proposal": proposal_outcome["progress"],
                "progress_reference": reference["progress"],
                "progress_after": after["progress"],
                "physics_before": output["reference_physics_violation"],
                "physics_after": output["corrected_physics_violation"], "gate": output["gate"],
            }
            for name, value in values.items(): metrics[name].extend(value.cpu().tolist())
    return {name: float(np.mean(value)) for name, value in metrics.items()}


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    train_loader, val_loader, test_loader, sizes = loaders()
    print(f"unified seed={SEED} device={DEVICE} scenes={sizes}", flush=True)
    outcome = ContinuousSceneOutcomeCritic().to(DEVICE)
    outcome_epoch, outcome_val = train_outcome(outcome, train_loader, val_loader)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    outcome_payload = {
        "model": cpu_state(outcome), "latent_dim": outcome.latent_dim,
        "dim": outcome.dim, "best_epoch": outcome_epoch, "best_val": outcome_val,
    }
    torch.save(outcome_payload, OUTPUT_ROOT / "outcome_critic.pt")
    if OUTCOME_ONLY:
        summary = {
            "schema_version": 1, "architecture": "continuous_scene_outcome_critic_v1",
            "seed": SEED, "scene_splits": sizes, "outcome_best_epoch": outcome_epoch,
            "outcome_best_val": outcome_val,
        }
        (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"saved outcome-only={OUTPUT_ROOT}", flush=True)
        return
    outcome.eval()
    for parameter in outcome.parameters(): parameter.requires_grad_(False)
    policy = UnifiedCorrectionPolicy(
        projection_iterations=PROJECTION_ITERATIONS
    ).to(DEVICE)
    policy_epoch, policy_reward = train_policy(policy, outcome, train_loader, val_loader)
    metrics = evaluate_policy(policy, outcome, test_loader)
    policy_payload = {
        "model": cpu_state(policy), **policy.metadata(),
        "best_epoch": policy_epoch, "best_reward": policy_reward,
        "outcome_checkpoint": str(OUTPUT_ROOT / "outcome_critic.pt"),
        "rl_config": asdict(CorrectionRLConfig()),
    }
    torch.save(policy_payload, OUTPUT_ROOT / "correction_policy.pt")
    summary = {
        "schema_version": 1, "architecture": "unified_constrained_correction_rl_v1",
        "seed": SEED, "scene_splits": sizes, "outcome_best_epoch": outcome_epoch,
        "outcome_best_val": outcome_val, "policy_best_epoch": policy_epoch,
        "policy_best_reward": policy_reward, "test": metrics,
    }
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"saved={OUTPUT_ROOT} metrics={metrics}", flush=True)


if __name__ == "__main__":
    main()
