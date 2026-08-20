"""Unified risk/physics-conditioned residual correction policy.

DriveVLA remains the proposal policy.  This module turns its incremental
waypoints into reference controls, predicts a *bounded residual* conditioned
on WM risk, analytic physics and rejected proposals, and rolls the result
through the differentiable vehicle model.  It replaces a chain of independent
ranker/adapter/filter modules with one correction-policy interface.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping

import torch
from torch import Tensor, nn

from differentiable_vehicle_model import (
    DifferentiableVehicleModel,
    VehicleModelConfig,
    absolute_to_relative_se2,
    relative_to_absolute_se2,
    smooth_clamp,
)


RISK_NAMES = ("unsafe", "ttc_failure", "dac_failure", "nc_failure")
PHYSICS_NAMES = (
    "speed_ratio",
    "acceleration_ratio",
    "jerk_ratio",
    "curvature_ratio",
    "curvature_rate_ratio",
    "lateral_acceleration_ratio",
    "friction_ratio",
)


def _wrap_angle(value: Tensor) -> Tensor:
    return torch.atan2(value.sin(), value.cos())


def infer_reference_controls(
    relative_waypoints: Tensor,
    config: VehicleModelConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Differentiably estimate initial state and controls from waypoints.

    This is deliberately an initialization, not a claim that inverse vehicle
    dynamics are uniquely identifiable from poses.  The learned policy only
    predicts residuals around it.
    """
    config = config or VehicleModelConfig()
    if relative_waypoints.shape[-2:] != (config.action_horizon, 3):
        raise ValueError(
            f"waypoints must end in ({config.action_horizon}, 3), "
            f"got {tuple(relative_waypoints.shape)}"
        )
    absolute = relative_to_absolute_se2(relative_waypoints)
    origin = absolute.new_zeros(*absolute.shape[:-2], 1, 3)
    poses = torch.cat([origin, absolute], dim=-2)
    displacement = poses[..., 1:, :2] - poses[..., :-1, :2]
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    speed = distance / config.action_dt
    heading_delta = _wrap_angle(poses[..., 1:, 2] - poses[..., :-1, 2])
    curvature = torch.where(
        distance > 1e-3,
        heading_delta / distance.clamp_min(1e-3),
        torch.zeros_like(distance),
    )
    acceleration = torch.zeros_like(speed)
    curvature_rate = torch.zeros_like(curvature)
    acceleration[..., 1:] = (speed[..., 1:] - speed[..., :-1]) / config.action_dt
    curvature_rate[..., 1:] = (
        curvature[..., 1:] - curvature[..., :-1]
    ) / config.action_dt
    controls = torch.stack(
        [
            acceleration.clamp(config.min_acceleration, config.max_acceleration),
            curvature_rate.clamp(
                -config.max_abs_curvature_rate, config.max_abs_curvature_rate
            ),
        ],
        dim=-1,
    )
    initial = relative_waypoints.new_zeros(*relative_waypoints.shape[:-2], 5)
    initial[..., 3] = speed[..., 0].clamp(config.min_speed, config.max_speed)
    initial[..., 4] = curvature[..., 0].clamp(
        -config.max_abs_curvature, config.max_abs_curvature
    )
    return initial, controls


def project_to_feasible_reference(
    proposal_waypoints: Tensor,
    config: VehicleModelConfig | None = None,
    vehicle_model: DifferentiableVehicleModel | None = None,
    initial_state: Tensor | None = None,
    scene_parameters: Mapping[str, Tensor | float] | None = None,
    num_iterations: int = 30,
    lr: float = 0.5,
    position_weight: float = 1.0,
    heading_weight: float = 0.5,
    velocity_weight: float = 0.05,
    control_weight: float = 1e-3,
    control_rate_weight: float = 1e-3,
) -> dict[str, Tensor]:
    """Project a raw waypoint proposal onto the dynamically feasible manifold.

    ``infer_reference_controls`` estimates controls via finite differences;
    re-integrating that estimate through the vehicle model is not guaranteed
    to reconstruct the original proposal, even with zero residual added on
    top -- the waypoint -> control -> waypoint round trip is lossy. This
    solves the small trajectory-tracking optimal-control problem
    min_u J_proj(F_vehicle(s_ego, u), tau^0) instead, so the reference is the
    *closest dynamically feasible realization* of the proposal rather than an
    approximate, potentially-distorting inverse.

    Box constraints (speed, curvature, acceleration, curvature rate) are
    satisfied by construction: the vehicle model's own RK4 step already
    smooth-clamps every control/state to its own independent bound (see
    differentiable_vehicle_model.py), so plain unconstrained gradient descent
    over raw controls is sufficient here -- no separate NLP/QP solver or
    scipy dependency is required.

    The 2D friction circle is only *partially* built-in this way: each
    substep clamps longitudinal control authority given the state's current
    lateral acceleration, but curvature's own bound does not shrink as speed
    grows, so an extreme enough target can still land on a reference state
    where speed^2*curvature alone exceeds the friction budget. That residual
    case is exactly what physics_condition's lateral_acceleration_ratio /
    friction_ratio channels are for, and physics_violation() is what
    penalizes it in the reward -- callers must not assume reference_states
    is friction-circle-feasible without also checking physics_violation.
    """
    config = config or VehicleModelConfig()
    vehicle_model = vehicle_model or DifferentiableVehicleModel(config)
    if proposal_waypoints.shape[-2:] != (config.action_horizon, 3):
        raise ValueError(
            f"proposal_waypoints must end in ({config.action_horizon}, 3), "
            f"got {tuple(proposal_waypoints.shape)}"
        )

    warm_initial, warm_controls = infer_reference_controls(proposal_waypoints, config)
    if initial_state is None:
        initial_state = warm_initial
    if initial_state.shape != proposal_waypoints.shape[:-2] + (5,):
        raise ValueError("initial_state has the wrong shape")

    target_absolute = relative_to_absolute_se2(proposal_waypoints)
    target_position = target_absolute[..., :2].detach()
    target_heading = target_absolute[..., 2].detach()
    origin = target_absolute.new_zeros(*target_absolute.shape[:-2], 1, 3)
    padded = torch.cat([origin, target_absolute], dim=-2)
    displacement = padded[..., 1:, :2] - padded[..., :-1, :2]
    target_speed = (
        torch.linalg.vector_norm(displacement, dim=-1) / config.action_dt
    ).detach()
    frozen_initial_state = initial_state.detach()

    with torch.enable_grad():
        controls = warm_controls.detach().clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [controls], lr=lr, max_iter=num_iterations, line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            states = vehicle_model(
                frozen_initial_state, controls, scene_parameters=scene_parameters
            )
            absolute = states[..., :3]
            position_error = (absolute[..., :2] - target_position).square().sum(-1)
            heading_error = 1.0 - torch.cos(absolute[..., 2] - target_heading)
            velocity_error = (states[..., 3] - target_speed).square()
            control_energy = controls.square().sum(-1)
            control_rate = (controls[..., 1:, :] - controls[..., :-1, :]).square().sum(-1)
            loss = (
                position_weight * position_error.sum(-1)
                + heading_weight * heading_error.sum(-1)
                + velocity_weight * velocity_error.sum(-1)
                + control_weight * control_energy.sum(-1)
                + control_rate_weight * control_rate.sum(-1)
            ).mean()
            loss.backward()
            return loss

        optimizer.step(closure)

    with torch.no_grad():
        # Note: this is the free LBFGS variable (the requested control),
        # not necessarily bit-identical to what the vehicle model applied
        # internally -- DifferentiableVehicleModel clamps box/friction-circle
        # limits per fine integration substep using each substep's own
        # evolving state, so an extremely infeasible target can make the
        # optimizer request values beyond what was actually executed.
        # reference_states below is always physically valid regardless (the
        # model enforces this on every forward pass, independent of its
        # input); treat reference_states, not reference_controls, as the
        # authoritative description of what tau^ref actually is.
        reference_controls = controls.detach()
        reference_states = vehicle_model(
            initial_state, reference_controls, scene_parameters=scene_parameters
        )
        reference_absolute = reference_states[..., :3]
        reference_relative = absolute_to_relative_se2(reference_absolute)
        projection_error = torch.linalg.vector_norm(
            reference_absolute[..., :2] - target_position, dim=-1
        ).mean(dim=-1)
        # tau^base in the spec: the projection is the fallback target only
        # while it is itself well-formed. LBFGS over a smooth, box-clamped
        # rollout should not diverge, but this is the only thing standing
        # between a solver failure and a NaN trajectory being deployed.
        # flatten(-2).all(dim=-1) instead of all(dim=(-2,-1)): tuple-valued
        # dim for .all() is not supported by every installed torch version
        # (confirmed failing on the HPC drivevla env), while single-int dim
        # after flattening the trailing two axes is universally supported
        # and reduces over the same elements.
        projection_valid = (
            torch.isfinite(reference_states).flatten(-2).all(dim=-1)
            & torch.isfinite(reference_controls).flatten(-2).all(dim=-1)
        )

    return {
        "initial_state": initial_state,
        "reference_controls": reference_controls,
        "reference_states": reference_states,
        "reference_absolute": reference_absolute,
        "reference_relative": reference_relative,
        "projection_error": projection_error,
        "projection_valid": projection_valid,
        "warm_start_controls": warm_controls.detach(),
    }


def physics_condition(
    states: Tensor,
    controls: Tensor,
    config: VehicleModelConfig | None = None,
    scene_parameters: Mapping[str, Tensor | float] | None = None,
) -> Tensor:
    """Return seven auditable, differentiable per-step physical ratios."""
    config = config or VehicleModelConfig()
    if states.shape[-2:] != (config.action_horizon, 5):
        raise ValueError("states have the wrong horizon/state dimension")
    if controls.shape[-2:] != (config.action_horizon, 2):
        raise ValueError("controls have the wrong horizon/control dimension")
    parameters = scene_parameters or {}

    def parameter(name: str, default: float) -> Tensor:
        value = torch.as_tensor(
            parameters.get(name, default), dtype=states.dtype, device=states.device
        )
        while value.ndim < states[..., 3].ndim:
            value = value.unsqueeze(-1)
        return value

    speed = states[..., 3]
    curvature = states[..., 4]
    acceleration = controls[..., 0]
    curvature_rate = controls[..., 1]
    max_speed = parameter("max_speed", config.max_speed)
    max_acceleration = parameter("max_acceleration", config.max_acceleration)
    max_deceleration = -parameter("min_acceleration", config.min_acceleration)
    acceleration_scale = torch.where(
        acceleration >= 0, max_acceleration, max_deceleration
    ).clamp_min(1e-6)
    max_curvature = parameter("max_abs_curvature", config.max_abs_curvature)
    max_curvature_rate = parameter(
        "max_abs_curvature_rate", config.max_abs_curvature_rate
    )
    mu = parameter("friction_coefficient", config.friction_coefficient)
    gravity = parameter("gravity", config.gravity)
    friction_limit = (mu * gravity).clamp_min(1e-6)
    lateral = speed.square() * curvature
    friction = torch.sqrt(acceleration.square() + lateral.square() + 1e-8)
    previous_acceleration = torch.cat(
        [torch.zeros_like(acceleration[..., :1]), acceleration[..., :-1]], dim=-1
    )
    jerk = (acceleration - previous_acceleration) / config.action_dt
    # Matches the current NAVSIM-oriented analytic dynamics audit default.
    max_abs_jerk = parameter("max_abs_jerk", 8.0)
    return torch.stack(
        [
            speed.abs() / max_speed.clamp_min(1e-6),
            acceleration.abs() / acceleration_scale,
            jerk.abs() / max_abs_jerk.clamp_min(1e-6),
            curvature.abs() / max_curvature.clamp_min(1e-6),
            curvature_rate.abs() / max_curvature_rate.clamp_min(1e-6),
            lateral.abs() / friction_limit,
            friction / friction_limit,
        ],
        dim=-1,
    )


def physics_violation(condition: Tensor, reduction: str = "mean") -> Tensor:
    violation = torch.relu(condition - 1.0)
    per_item = violation.amax(dim=(-2, -1))
    if reduction == "none":
        return per_item
    if reduction == "mean":
        return per_item.mean()
    if reduction == "sum":
        return per_item.sum()
    raise ValueError(f"unsupported reduction {reduction!r}")


class UnifiedCorrectionPolicy(nn.Module):
    """One conditioned policy for minimal bounded continuous correction.

    The policy starts close to the identity: its residual head is zero and its
    gate probability is low.  This prevents an untrained corrector from
    replacing a competent waypoint planner.
    """

    risk_names = RISK_NAMES
    physics_names = PHYSICS_NAMES

    def __init__(
        self,
        dim: int = 192,
        layers: int = 3,
        heads: int = 6,
        wm_context_dim: int = 4096,
        vlm_context_dim: int = 4096,
        max_trace_length: int = 2,
        projection_iterations: int = 30,
        config: VehicleModelConfig | None = None,
    ):
        super().__init__()
        self.config = config or VehicleModelConfig()
        self.dim = int(dim)
        self.layers = int(layers)
        self.heads = int(heads)
        self.wm_context_dim = int(wm_context_dim)
        self.vlm_context_dim = int(vlm_context_dim)
        self.max_trace_length = int(max_trace_length)
        self.projection_iterations = int(projection_iterations)
        if self.max_trace_length < 0:
            raise ValueError("max_trace_length must be non-negative")

        # proposal(3), state(5), reference control(2), physics(7), risk(4),
        # rejected-trace summary(3)
        self.local_projection = nn.Sequential(
            nn.LayerNorm(24), nn.Linear(24, dim), nn.GELU()
        )
        self.wm_projection = nn.Sequential(
            nn.LayerNorm(wm_context_dim), nn.Linear(wm_context_dim, dim), nn.GELU()
        )
        self.vlm_projection = nn.Sequential(
            nn.LayerNorm(vlm_context_dim), nn.Linear(vlm_context_dim, dim)
        )
        # Offline warm-up has no saved contextual VLM states. Starting this
        # path at zero preserves the warm checkpoint; online joint RL then
        # learns how perception context should alter corrections.
        nn.init.zeros_(self.vlm_projection[1].weight)
        nn.init.zeros_(self.vlm_projection[1].bias)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.config.action_horizon, dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.residual_head = nn.Linear(dim, 2)
        self.gate_head = nn.Linear(dim, 1)
        self.residual_log_std = nn.Parameter(torch.full((2,), -2.0))
        self.gate_log_std = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -4.0)
        self.vehicle_model = DifferentiableVehicleModel(self.config)

    def _validate_conditions(
        self,
        proposal: Tensor,
        risk: Tensor,
        wm_context: Tensor,
        correction_trace: Tensor | None,
        vlm_context: Tensor | None,
    ) -> None:
        leading = proposal.shape[:-2]
        if risk.shape != leading + (len(RISK_NAMES),):
            raise ValueError(
                f"risk_condition must have shape {leading + (len(RISK_NAMES),)}, "
                f"got {tuple(risk.shape)}"
            )
        if wm_context.shape != leading + (self.wm_context_dim,):
            raise ValueError("wm_context batch dimensions or feature dimension mismatch")
        if vlm_context is not None and vlm_context.shape != leading + (self.vlm_context_dim,):
            raise ValueError("vlm_context batch dimensions or feature dimension mismatch")
        if correction_trace is not None:
            expected_tail = (self.config.action_horizon, 3)
            if correction_trace.shape[:-3] != leading or correction_trace.shape[-2:] != expected_tail:
                raise ValueError("correction_trace must be [..., trace, horizon, 3]")
            if correction_trace.shape[-3] > self.max_trace_length:
                raise ValueError("correction_trace exceeds max_trace_length")

    def project(
        self,
        proposal_waypoints: Tensor,
        *,
        initial_state: Tensor | None = None,
        scene_parameters: Mapping[str, Tensor | float] | None = None,
    ) -> dict[str, Tensor]:
        """Deterministic physics-driven projection tau^0 -> tau^ref.

        Contains no trainable parameters and does not depend on risk/WM/VLM
        context, so it can be computed once and reused by every caller that
        needs tau^ref: risk/critic evaluation on the correct trajectory
        level, the learned correction step (correct_projected), or an
        independent frozen-reference branch. This is what
        docs/reference_anchored_correction_rl.tex Step 2 calls the
        physics-driven reference projection, kept separate from Step 3/4's
        learned correction.
        """
        config = self.config
        if proposal_waypoints.shape[-2:] != (config.action_horizon, 3):
            raise ValueError("proposal_waypoints have the wrong shape")
        projection = project_to_feasible_reference(
            proposal_waypoints,
            config,
            vehicle_model=self.vehicle_model,
            initial_state=initial_state,
            scene_parameters=scene_parameters,
            num_iterations=self.projection_iterations,
        )
        reference_physics = physics_condition(
            projection["reference_states"], projection["reference_controls"],
            config, scene_parameters,
        )
        return {
            "proposal_relative": proposal_waypoints,
            "initial_state": projection["initial_state"],
            "reference_controls": projection["reference_controls"],
            "reference_states": projection["reference_states"],
            "reference_absolute": projection["reference_absolute"],
            "reference_relative": projection["reference_relative"],
            "reference_physics": reference_physics,
            "reference_physics_violation": physics_violation(
                reference_physics, reduction="none"
            ),
            "projection_error": projection["projection_error"],
            "projection_valid": projection["projection_valid"],
        }

    def correct_projected(
        self,
        projected: dict[str, Tensor],
        risk_condition: Tensor,
        wm_context: Tensor,
        *,
        correction_trace: Tensor | None = None,
        vlm_context: Tensor | None = None,
        scene_parameters: Mapping[str, Tensor | float] | None = None,
        deterministic_gate: bool = False,
        sample_correction: bool = False,
        correction_noise: tuple[Tensor, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Learned residual correction around an already-projected reference
        (the output of ``project()``).

        ``risk_condition`` must describe tau^ref -- the trajectory this call
        actually conditions on and corrects around -- not the raw proposal
        tau^0. Computing it on tau^0 instead is exactly the trajectory-level
        mismatch documented in the tex's Step 3 implementation note: it
        silently mixes two different "reference" definitions the same way
        the reference-anchored GRPO probes and the critic before/after
        baselines did before they were fixed to use tau^ref consistently.
        """
        config = self.config
        proposal_waypoints = projected["proposal_relative"]
        reference_controls = projected["reference_controls"]
        reference_states = projected["reference_states"]
        reference_physics = projected["reference_physics"]
        initial_state = projected["initial_state"]
        self._validate_conditions(
            proposal_waypoints, risk_condition, wm_context, correction_trace,
            vlm_context,
        )
        if correction_trace is None or correction_trace.shape[-3] == 0:
            trace_summary = torch.zeros_like(proposal_waypoints)
        else:
            # Mean over rejected proposals retains a per-time-step correction
            # direction without adding a second planner network.
            trace_summary = correction_trace.mean(dim=-3)
        repeated_initial = initial_state.unsqueeze(-2).expand(
            *proposal_waypoints.shape[:-2], config.action_horizon, 5
        )
        repeated_risk = risk_condition.unsqueeze(-2).expand(
            *proposal_waypoints.shape[:-2], config.action_horizon, len(RISK_NAMES)
        )
        local = torch.cat(
            [
                proposal_waypoints,
                repeated_initial,
                reference_controls,
                reference_physics,
                repeated_risk,
                trace_summary,
            ],
            dim=-1,
        )
        hidden = self.local_projection(local)
        hidden = hidden + self.wm_projection(wm_context).unsqueeze(-2)
        if vlm_context is not None:
            hidden = hidden + self.vlm_projection(vlm_context).unsqueeze(-2)
        flat = hidden.reshape(-1, config.action_horizon, self.dim)
        flat = self.encoder(flat + self.position_embedding)
        hidden = flat.reshape(*proposal_waypoints.shape[:-2], config.action_horizon, self.dim)

        residual_mean = self.residual_head(hidden)
        gate_mean = self.gate_head(hidden).squeeze(-1).mean(dim=-1)
        residual_std = self.residual_log_std.clamp(-5.0, 1.0).exp()
        gate_std = self.gate_log_std.clamp(-5.0, 1.0).exp()
        residual_distribution = torch.distributions.Normal(residual_mean, residual_std)
        gate_distribution = torch.distributions.Normal(gate_mean, gate_std)
        if sample_correction:
            if correction_noise is None:
                residual_latent = residual_distribution.sample()
                gate_latent = gate_distribution.sample()
            else:
                residual_noise, gate_noise = correction_noise
                if residual_noise.shape != residual_mean.shape or gate_noise.shape != gate_mean.shape:
                    raise ValueError("correction noise shapes must match residual/gate means")
                residual_latent = residual_mean + residual_std * residual_noise
                gate_latent = gate_mean + gate_std * gate_noise
        else:
            residual_latent, gate_latent = residual_mean, gate_mean
        raw_residual = torch.tanh(residual_latent)
        residual_scale = raw_residual.new_tensor(
            [
                config.max_acceleration - config.min_acceleration,
                2.0 * config.max_abs_curvature_rate,
            ]
        )
        residual_controls = 0.25 * residual_scale * raw_residual
        gate_logits = gate_latent
        gate_probability = torch.sigmoid(gate_logits)
        gate = (gate_probability >= 0.5).to(gate_probability.dtype) if deterministic_gate else gate_probability
        correction_log_prob = (
            residual_distribution.log_prob(residual_latent.detach()).sum(dim=(-2, -1))
            + gate_distribution.log_prob(gate_latent.detach())
        )
        correction_entropy = (
            residual_distribution.entropy().sum(dim=(-2, -1))
            + gate_distribution.entropy()
        )
        corrected_unbounded = reference_controls + gate.unsqueeze(-1).unsqueeze(-1) * residual_controls
        acceleration = smooth_clamp(
            corrected_unbounded[..., 0],
            config.min_acceleration,
            config.max_acceleration,
            config.smooth_clamp_beta,
        )
        curvature_rate = config.max_abs_curvature_rate * torch.tanh(
            corrected_unbounded[..., 1] / config.max_abs_curvature_rate
        )
        corrected_controls = torch.stack([acceleration, curvature_rate], dim=-1)
        corrected_states = self.vehicle_model(
            initial_state, corrected_controls, scene_parameters=scene_parameters
        )
        corrected_absolute = corrected_states[..., :3]
        corrected_relative = absolute_to_relative_se2(corrected_absolute)
        corrected_physics = physics_condition(
            corrected_states, corrected_controls, config, scene_parameters
        )
        return {
            **projected,
            "risk_condition": risk_condition,
            "residual_controls": residual_controls,
            "residual_latent": residual_latent,
            "gate_logits": gate_logits,
            "gate_probability": gate_probability,
            "gate": gate,
            "correction_log_prob": correction_log_prob,
            "correction_entropy": correction_entropy,
            "corrected_controls": corrected_controls,
            "corrected_states": corrected_states,
            "corrected_absolute": corrected_absolute,
            "corrected_relative": corrected_relative,
            "corrected_physics": corrected_physics,
            "corrected_physics_violation": physics_violation(
                corrected_physics, reduction="none"
            ),
        }

    def forward(
        self,
        proposal_waypoints: Tensor,
        risk_condition: Tensor,
        wm_context: Tensor,
        *,
        initial_state: Tensor | None = None,
        correction_trace: Tensor | None = None,
        vlm_context: Tensor | None = None,
        scene_parameters: Mapping[str, Tensor | float] | None = None,
        deterministic_gate: bool = False,
        sample_correction: bool = False,
        correction_noise: tuple[Tensor, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Convenience one-shot wrapper: ``project()`` then
        ``correct_projected()``.

        ``risk_condition`` is still required and is used exactly as passed
        -- this method does not decide for the caller whether it describes
        tau^0 or tau^ref. New call sites should call ``project()`` and
        ``correct_projected()`` explicitly, computing risk_condition on the
        returned ``reference_relative`` (see ``apply_verified_correction``),
        so the trajectory level risk_condition describes is unambiguous at
        the call site rather than implicit in this wrapper.
        """
        projected = self.project(
            proposal_waypoints,
            initial_state=initial_state,
            scene_parameters=scene_parameters,
        )
        return self.correct_projected(
            projected,
            risk_condition,
            wm_context,
            correction_trace=correction_trace,
            vlm_context=vlm_context,
            scene_parameters=scene_parameters,
            deterministic_gate=deterministic_gate,
            sample_correction=sample_correction,
            correction_noise=correction_noise,
        )

    def metadata(self) -> dict:
        return {
            "schema_version": 1,
            "architecture": "risk_physics_conditioned_residual_correction_v1",
            "dim": self.dim,
            "layers": self.layers,
            "heads": self.heads,
            "wm_context_dim": self.wm_context_dim,
            "vlm_context_dim": self.vlm_context_dim,
            "max_trace_length": self.max_trace_length,
            "projection_iterations": self.projection_iterations,
            "risk_names": list(self.risk_names),
            "physics_names": list(self.physics_names),
            "vehicle_model": asdict(self.config),
        }


def apply_verified_correction(
    policy_module: UnifiedCorrectionPolicy,
    outcome_module,
    proposal_relative: Tensor,
    wm_context: Tensor,
    vlm_context: Tensor,
    *,
    deterministic: bool,
    correction_trace: Tensor | None = None,
    scene_parameters: Mapping[str, Tensor | float] | None = None,
    residual_noise: Tensor | None = None,
    gate_noise: Tensor | None = None,
) -> dict[str, Tensor]:
    """Run the full propose -> project -> correct -> verify -> fallback
    deployment stack and return tau^final, matching
    docs/reference_anchored_correction_rl.tex Step 5 and the verification
    paragraph exactly.

    This is the single implementation shared by every caller (training's
    current-policy candidates, training's frozen reference-anchored probes,
    and the standalone deployment script) precisely so the current and
    reference sides of a reference-anchored comparison cannot silently drift
    apart from each other -- they call the same function with a different
    ``policy_module``/``deterministic`` setting rather than duplicating the
    projection/gate/verification logic per call site.

    ``deterministic=True`` samples nothing: the mean residual is used and the
    gate is hard-thresholded, so the same (policy, proposal) pair always
    produces the same tau^final. This is required for the frozen reference
    branch (whose score must be a fixed quantity to serve as a stable
    advantage baseline, and which is therefore cacheable per scene/probe);
    the trainable current-policy branch instead passes
    ``deterministic=False`` with externally supplied noise so its stochastic
    action remains reproducible for the later log-prob recomputation.
    """
    with torch.no_grad():
        # tau^0 -> tau^ref first, deterministic and parameter-free (see
        # UnifiedCorrectionPolicy.project). risk_condition is then computed
        # on tau^ref, not on the raw proposal_relative: the correction step
        # below conditions on and corrects around tau^ref, so that is the
        # trajectory level its risk assessment must describe (tex Step 3
        # implementation note). This is also, conveniently, exactly the
        # quantity the semantic verifier below needs as its baseline, so it
        # is computed once and reused for both purposes.
        projected = policy_module.project(
            proposal_relative, scene_parameters=scene_parameters,
        )
        reference_risk = outcome_module.outcomes(
            projected["reference_relative"].flatten(-2), wm_context
        )["risk"]
        policy_kwargs = dict(
            correction_trace=correction_trace,
            vlm_context=vlm_context,
            scene_parameters=scene_parameters,
        )
        if deterministic:
            policy_kwargs["deterministic_gate"] = True
        else:
            policy_kwargs["sample_correction"] = True
            policy_kwargs["correction_noise"] = (residual_noise, gate_noise)
        output = policy_module.correct_projected(
            projected, reference_risk, wm_context, **policy_kwargs
        )

        # Semantic non-degradation, V(tau^1, tau^ref): a learned-proxy check
        # against the outcome critic's four RISK_NAMES channels, which are
        # already risk-form (smaller is better). Ground-truth NAVSIM scoring
        # is not available at this point -- see the Implementation note in
        # the tex's verification paragraph.
        corrected_risk = outcome_module.outcomes(
            output["corrected_relative"].flatten(-2), wm_context
        )["risk"]

    gate_selected = output["gate_probability"] >= 0.5
    projection_valid = output["projection_valid"]
    physics_ok = output["corrected_physics_violation"] <= 0.0
    semantic_ok = (corrected_risk <= reference_risk).all(dim=-1)
    correction_mask = gate_selected & physics_ok & semantic_ok

    base_relative = torch.where(
        projection_valid[..., None, None], output["reference_relative"], proposal_relative
    )
    final_relative = torch.where(
        correction_mask[..., None, None], output["corrected_relative"], base_relative
    )

    return {
        "final_relative": final_relative,
        "base_relative": base_relative,
        "projection_valid": projection_valid,
        "gate_selected": gate_selected,
        "physics_ok": physics_ok,
        "semantic_ok": semantic_ok,
        "correction_mask": correction_mask,
        "reference_risk": reference_risk,
        "corrected_risk": corrected_risk,
        "output": output,
    }
