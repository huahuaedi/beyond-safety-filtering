"""Control-space representation used by the native DriveVLA action expert.

The pretrained checkpoint has action_dim=3.  Version 1 preserves that tensor
shape for warm-start compatibility while assigning physical semantics to the
first two channels: acceleration and curvature rate.  The third channel is a
reserved zero-valued compatibility channel and is never executed.
"""

from __future__ import annotations

import torch
from torch import Tensor

from differentiable_vehicle_model import DifferentiableVehicleModel, VehicleModelConfig


def initial_state_from_relative_history(
    relative_history: Tensor,
    config: VehicleModelConfig | None = None,
) -> Tensor:
    """Estimate current vehicle state from observed relative SE(2) history.

    The same function is used during training and deployment.  It consumes no
    future trajectory or NAVSIM metric label.
    """
    config = config or VehicleModelConfig()
    if relative_history.ndim < 2 or relative_history.shape[-1] != 3:
        raise ValueError(
            f"relative history must end in [history,3], got {tuple(relative_history.shape)}"
        )
    last = relative_history[..., -1, :].float()
    distance = torch.linalg.vector_norm(last[..., :2], dim=-1)
    speed = (distance / config.action_dt).clamp(config.min_speed, config.max_speed)
    curvature = torch.where(
        distance > 1e-3,
        last[..., 2] / distance.clamp_min(1e-3),
        torch.zeros_like(distance),
    ).clamp(-config.max_abs_curvature, config.max_abs_curvature)
    zeros = torch.zeros_like(speed)
    return torch.stack([zeros, zeros, zeros, speed, curvature], dim=-1)


def controls_from_absolute_trajectory(
    absolute_trajectory: Tensor,
    initial_state: Tensor,
    config: VehicleModelConfig | None = None,
) -> Tensor:
    """State-aligned inverse kinematics for piecewise-constant controls.

    Segment-average speed and curvature are matched recursively from the
    observed initial state.  This is an analytic initialization/target; the
    differentiable rollout reconstruction loss handles integration residuals.
    """
    config = config or VehicleModelConfig()
    if absolute_trajectory.shape[-2:] != (config.action_horizon, 3):
        raise ValueError(
            f"expected [...,{config.action_horizon},3], got {tuple(absolute_trajectory.shape)}"
        )
    if initial_state.shape != absolute_trajectory.shape[:-2] + (5,):
        raise ValueError("initial state and trajectory batch dimensions must match")

    trajectory = absolute_trajectory.float()
    origin = trajectory.new_zeros(*trajectory.shape[:-2], 1, 3)
    complete = torch.cat([origin, trajectory], dim=-2)
    delta_xy = complete[..., 1:, :2] - complete[..., :-1, :2]
    distance = torch.linalg.vector_norm(delta_xy, dim=-1)
    yaw_delta_raw = complete[..., 1:, 2] - complete[..., :-1, 2]
    yaw_delta = torch.atan2(yaw_delta_raw.sin(), yaw_delta_raw.cos())
    average_speed = distance / config.action_dt
    average_curvature = torch.where(
        distance > 1e-3,
        yaw_delta / distance.clamp_min(1e-3),
        torch.zeros_like(distance),
    )

    speed = initial_state[..., 3].float()
    curvature = initial_state[..., 4].float()
    controls = []
    for step in range(config.action_horizon):
        acceleration = 2.0 * (average_speed[..., step] - speed) / config.action_dt
        curvature_rate = 2.0 * (
            average_curvature[..., step] - curvature
        ) / config.action_dt
        acceleration = acceleration.clamp(
            config.min_acceleration, config.max_acceleration
        )
        curvature_rate = curvature_rate.clamp(
            -config.max_abs_curvature_rate, config.max_abs_curvature_rate
        )
        controls.append(torch.stack([acceleration, curvature_rate], dim=-1))
        speed = (speed + acceleration * config.action_dt).clamp(
            config.min_speed, config.max_speed
        )
        curvature = (curvature + curvature_rate * config.action_dt).clamp(
            -config.max_abs_curvature, config.max_abs_curvature
        )
    return torch.stack(controls, dim=-2)


def controls_to_model_action(controls: Tensor, config: VehicleModelConfig | None = None) -> Tensor:
    config = config or VehicleModelConfig()
    if controls.shape[-1] != 2:
        raise ValueError(f"controls must end in 2, got {tuple(controls.shape)}")
    acceleration = controls[..., 0]
    acceleration_model = torch.where(
        acceleration >= 0,
        acceleration / config.max_acceleration,
        acceleration / abs(config.min_acceleration),
    )
    curvature_rate_model = controls[..., 1] / config.max_abs_curvature_rate
    reserved = torch.zeros_like(acceleration_model)
    normalized = torch.stack(
        [acceleration_model, curvature_rate_model, reserved], dim=-1
    ).clamp(-0.999, 0.999)
    # The flow predicts unconstrained logits. Smooth tanh saturation in the
    # execution path keeps gradients finite and avoids hard-clip dead zones.
    return torch.atanh(normalized)


def model_action_to_controls(action: Tensor, config: VehicleModelConfig | None = None) -> Tensor:
    config = config or VehicleModelConfig()
    if action.shape[-1] < 2:
        raise ValueError(f"model action must have at least 2 channels, got {tuple(action.shape)}")
    bounded = torch.tanh(action[..., :2])
    acceleration = torch.where(
        bounded[..., 0] >= 0,
        bounded[..., 0] * config.max_acceleration,
        bounded[..., 0] * abs(config.min_acceleration),
    )
    curvature_rate = bounded[..., 1] * config.max_abs_curvature_rate
    return torch.stack([acceleration, curvature_rate], dim=-1)


def execute_model_action(
    action: Tensor,
    initial_state: Tensor,
    config: VehicleModelConfig | None = None,
) -> dict[str, Tensor]:
    config = config or VehicleModelConfig()
    controls = model_action_to_controls(action, config)
    states = DifferentiableVehicleModel(config)(initial_state, controls)
    return {"controls": controls, "states": states, "absolute_trajectory": states[..., :3]}
