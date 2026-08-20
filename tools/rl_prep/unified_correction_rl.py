"""Reference-anchored constrained RL objective for trajectory correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from unified_correction_policy import UnifiedCorrectionPolicy


@dataclass(frozen=True)
class CorrectionRLConfig:
    hard_constraint_penalty: float = 2.0
    trajectory_deviation_coef: float = 0.10
    control_residual_coef: float = 0.02
    correction_count_coef: float = 0.01
    entropy_coef: float = 1e-4
    semantic_margin: float = 0.005
    physics_tolerance: float = 1e-3
    cvar_fraction: float = 0.10
    cvar_coef: float = 0.25
    # Delta in the chance/tail constraint: activate the tail update only when
    # the lower-tail return falls below this auditable minimum.
    cvar_minimum: float = -0.25


def lower_tail_cvar(values: Tensor, fraction: float) -> Tensor:
    if values.ndim != 1:
        raise ValueError("CVaR values must be one-dimensional")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("CVaR fraction must be in (0, 1]")
    count = max(1, int(torch.ceil(values.new_tensor(values.numel() * fraction)).item()))
    return torch.topk(values, count, largest=False).values.mean()


def constrained_correction_objective(
    output: dict[str, Tensor],
    reference_risk: Tensor,
    corrected_risk: Tensor,
    reference_progress: Tensor,
    corrected_progress: Tensor,
    config: CorrectionRLConfig | None = None,
) -> dict[str, Tensor]:
    """Build a lexicographic REINFORCE objective and auditable diagnostics.

    Progress can never compensate a semantic-risk regression or remaining
    physical violation: infeasible samples receive a negative hard reward.

    The baseline is tau^ref (the projected physical reference), not the raw
    proposal tau^0: docs/reference_anchored_correction_rl.tex's Reward
    Design section defines the correction advantage relative to R(tau^ref),
    and its Training-Time Critic section requires C_phi(tau^1) <=
    C_phi(tau^ref) -- comparing against tau^0 instead would credit the
    correction policy for the physics-driven projection's own effect, which
    it did not produce and does not control.
    """
    config = config or CorrectionRLConfig()
    if reference_risk.shape != corrected_risk.shape or reference_risk.ndim != 2:
        raise ValueError("reference/corrected risk must have matching [B,R] shapes")
    batch = reference_risk.shape[0]
    for name, value in (
        ("reference_progress", reference_progress),
        ("corrected_progress", corrected_progress),
    ):
        if value.shape != (batch,):
            raise ValueError(f"{name} must have shape ({batch},)")

    semantic_regret = torch.relu(
        corrected_risk - reference_risk - config.semantic_margin
    ).sum(dim=-1)
    corrected_physics = output["corrected_physics_violation"]
    reference_physics = output["reference_physics_violation"]
    physics_regret = torch.relu(corrected_physics - reference_physics)
    remaining_physics = corrected_physics
    feasible = (
        (semantic_regret <= 1e-8)
        & (remaining_physics <= config.physics_tolerance)
    )

    position_delta = output["corrected_absolute"][..., :2] - output["reference_states"][..., :2]
    trajectory_deviation = torch.linalg.vector_norm(position_delta, dim=-1).mean(dim=-1)
    control_residual = output["residual_controls"].square().mean(dim=(-2, -1))
    gate = output["gate"]
    progress_gain = corrected_progress - reference_progress
    correction_cost = (
        config.trajectory_deviation_coef * trajectory_deviation
        + config.control_residual_coef * control_residual
        + config.correction_count_coef * gate
    )
    feasible_reward = progress_gain - correction_cost
    hard_violation = semantic_regret + physics_regret + remaining_physics
    reward = torch.where(
        feasible,
        feasible_reward,
        -config.hard_constraint_penalty * (1.0 + hard_violation),
    )
    advantage = reward - reward.mean()
    scale = advantage.std(unbiased=False).clamp_min(1e-4)
    advantage = advantage / scale
    policy_loss = -(advantage.detach() * output["correction_log_prob"]).mean()
    entropy_bonus = output["correction_entropy"].mean()
    cvar = lower_tail_cvar(reward, config.cvar_fraction)
    tail_count = max(
        1, int(torch.ceil(reward.new_tensor(batch * config.cvar_fraction)).item())
    )
    tail_indices = torch.topk(reward, tail_count, largest=False).indices
    # Score-function CVaR update: directly improve ordering inside the worst
    # tail. A plain differentiable `relu(-cvar)` has no policy gradient because
    # correction actions are sampled (not reparameterized).
    tail_advantage = reward[tail_indices] - reward[tail_indices].mean()
    cvar_active = (cvar.detach() < config.cvar_minimum).to(reward.dtype)
    cvar_policy_loss = -cvar_active * (
        tail_advantage.detach() * output["correction_log_prob"][tail_indices]
    ).mean()
    cvar_violation = torch.relu(reward.new_tensor(config.cvar_minimum) - cvar)
    loss = (
        policy_loss + config.cvar_coef * cvar_policy_loss
        - config.entropy_coef * entropy_bonus
    )
    return {
        "loss": loss,
        "policy_loss": policy_loss,
        "reward": reward,
        "advantage": advantage,
        "feasible": feasible,
        "semantic_regret": semantic_regret,
        "physics_regret": physics_regret,
        "remaining_physics": remaining_physics,
        "progress_gain": progress_gain,
        "trajectory_deviation": trajectory_deviation,
        "control_residual": control_residual,
        "entropy": entropy_bonus,
        "cvar": cvar,
        "cvar_violation": cvar_violation,
        "cvar_active": cvar_active,
        "cvar_policy_loss": cvar_policy_loss,
    }


class UnifiedCorrectionRL(nn.Module):
    """Single propose-condition-correct-evaluate interface.

    ``risk_critic`` is frozen but differentiable with respect to trajectories.
    It is part of the RL environment/condition, not a second planning policy.
    """

    def __init__(
        self,
        policy: UnifiedCorrectionPolicy,
        outcome_critic: nn.Module,
        config: CorrectionRLConfig | None = None,
    ):
        super().__init__()
        self.policy = policy
        self.outcome_critic = outcome_critic
        self.config = config or CorrectionRLConfig()
        self.outcome_critic.eval()
        for parameter in self.outcome_critic.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        proposal_waypoints: Tensor,
        wm_context: Tensor,
        *,
        initial_state: Tensor | None = None,
        scene_parameters: Mapping[str, Tensor | float] | None = None,
        projected: dict[str, Tensor] | None = None,
        **policy_kwargs,
    ) -> dict[str, Tensor]:
        # tau^0 -> tau^ref first (project() has no trainable parameters and
        # does not depend on risk/context), then compute risk on tau^ref --
        # the trajectory correct_projected() actually conditions on and
        # corrects around -- rather than on the raw proposal. Using the raw
        # proposal here was exactly the trajectory-level mismatch described
        # in docs/reference_anchored_correction_rl.tex Step 3's
        # implementation note; project() being a public method (not buried
        # inside forward()) is what makes computing risk in the right order
        # possible at all.
        #
        # initial_state/scene_parameters are split out explicitly (not left
        # inside **policy_kwargs) because they belong to project(), while
        # everything else in policy_kwargs (correction_trace, vlm_context,
        # deterministic_gate, sample_correction, correction_noise) belongs
        # to correct_projected() -- correct_projected() does not accept
        # initial_state (it reads it back out of the already-projected
        # dict), so leaving it in policy_kwargs raises a TypeError.
        #
        # ``projected`` lets a caller that already ran project() on this
        # exact proposal_waypoints/initial_state (e.g. train_policy() also
        # needs it for warmup_loss()'s imitation anchor on the same batch)
        # pass it straight through instead of paying for a second identical
        # LBFGS solve -- project() is parameter-free and deterministic, so
        # recomputing it for the same input is pure waste, not a second
        # independent measurement. Skipping it here also guarantees the risk
        # conditioning and the imitation anchor are evaluated against the
        # bit-identical tau^ref rather than two separately-optimized (and
        # potentially slightly different, since LBFGS is only run to a fixed
        # iteration budget rather than to convergence) reference solves.
        with torch.no_grad():
            proposal_outcome = self.outcome_critic.outcomes(
                proposal_waypoints.flatten(-2), wm_context
            )
        if projected is None:
            projected = self.policy.project(
                proposal_waypoints,
                initial_state=initial_state,
                scene_parameters=scene_parameters,
            )
        with torch.no_grad():
            reference_outcome = self.outcome_critic.outcomes(
                projected["reference_relative"].flatten(-2), wm_context
            )
        output = self.policy.correct_projected(
            projected,
            reference_outcome["risk"],
            wm_context,
            scene_parameters=scene_parameters,
            **policy_kwargs,
        )
        corrected_outcome = self.outcome_critic.outcomes(
            output["corrected_relative"].flatten(-2), wm_context
        )
        corrected_risk = corrected_outcome["risk"]
        corrected_progress = corrected_outcome["progress"]
        objective = constrained_correction_objective(
            output,
            reference_outcome["risk"],
            corrected_risk,
            reference_outcome["progress"],
            corrected_progress,
            self.config,
        )
        return {
            **output, **objective,
            "proposal_risk": proposal_outcome["risk"],
            "reference_risk": reference_outcome["risk"],
            "reference_progress": reference_outcome["progress"],
            "corrected_risk": corrected_risk,
        }
