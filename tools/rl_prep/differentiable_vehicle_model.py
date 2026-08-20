"""Differentiable continuous-control trajectory generation for DriveVLA.

The state is rear-axle ``(x, y, yaw, speed, curvature)`` and the control is
``(longitudinal_acceleration, curvature_rate)``.  Controls are held constant
inside each 0.5 s DriveVLA action interval while RK4 integrates smaller steps.
This continuous-curvature kinematic model is the first auditable structural
baseline; learned residuals and high-speed dynamic-bicycle blending belong in
later ablations, not in this base implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VehicleModelConfig:
    action_horizon: int = 8
    action_dt: float = 0.5
    integration_dt: float = 0.1
    # Signed speed avoids the non-zero creep introduced by a smooth projection
    # exactly at a zero lower bound. Reverse motion is kept small and can be
    # penalized explicitly during training.
    min_speed: float = -1.0
    max_speed: float = 30.0
    max_abs_curvature: float = 0.25
    min_acceleration: float = -6.0
    max_acceleration: float = 4.0
    max_abs_curvature_rate: float = 0.20
    friction_coefficient: float = 0.90
    gravity: float = 9.81
    smooth_clamp_beta: float = 20.0

    @property
    def substeps(self) -> int:
        ratio = self.action_dt / self.integration_dt
        rounded = round(ratio)
        if rounded < 1 or not math.isclose(ratio, rounded, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("action_dt must be an integer multiple of integration_dt")
        return int(rounded)


def smooth_clamp(value: Tensor, minimum, maximum, beta: float = 20.0) -> Tensor:
    """Approximately identity in range, with smooth saturation outside it."""
    if not torch.is_tensor(minimum) and not torch.is_tensor(maximum) and not minimum < maximum:
        raise ValueError("minimum must be smaller than maximum")
    return minimum + F.softplus(value - minimum, beta=beta) - F.softplus(value - maximum, beta=beta)


def _scene_parameter(value, reference: Tensor) -> Tensor:
    """Broadcast a scalar/per-scene value across an optional horizon axis."""
    tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    while tensor.ndim < reference.ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor


def absolute_to_relative_se2(poses: Tensor) -> Tensor:
    """Convert accumulated ego-frame poses to DriveVLA incremental SE(2)."""
    if poses.ndim < 2 or poses.shape[-1] != 3:
        raise ValueError(f"expected [..., horizon, 3], got {tuple(poses.shape)}")
    origin = poses.new_zeros(*poses.shape[:-2], 1, 3)
    previous = torch.cat([origin, poses[..., :-1, :]], dim=-2)
    global_delta = poses[..., :, :2] - previous[..., :, :2]
    previous_yaw = previous[..., :, 2]
    cosine, sine = previous_yaw.cos(), previous_yaw.sin()
    local_x = cosine * global_delta[..., 0] + sine * global_delta[..., 1]
    local_y = -sine * global_delta[..., 0] + cosine * global_delta[..., 1]
    yaw_delta = poses[..., :, 2] - previous_yaw
    yaw_delta = torch.atan2(yaw_delta.sin(), yaw_delta.cos())
    return torch.stack([local_x, local_y, yaw_delta], dim=-1)


def relative_to_absolute_se2(increments: Tensor) -> Tensor:
    """Differentiable equivalent of DriveVLA's incremental pose integration."""
    if increments.ndim < 2 or increments.shape[-1] != 3:
        raise ValueError(f"expected [..., horizon, 3], got {tuple(increments.shape)}")
    state = increments.new_zeros(*increments.shape[:-2], 3)
    output = []
    for step in range(increments.shape[-2]):
        delta = increments[..., step, :]
        cosine, sine = state[..., 2].cos(), state[..., 2].sin()
        x = state[..., 0] + cosine * delta[..., 0] - sine * delta[..., 1]
        y = state[..., 1] + sine * delta[..., 0] + cosine * delta[..., 1]
        yaw_raw = state[..., 2] + delta[..., 2]
        yaw = torch.atan2(yaw_raw.sin(), yaw_raw.cos())
        state = torch.stack([x, y, yaw], dim=-1)
        output.append(state)
    return torch.stack(output, dim=-2)


class ContinuousControlHead(nn.Module):
    """Map Action Expert token states to smooth bounded physical controls."""

    def __init__(self, hidden_dim: int, config: VehicleModelConfig | None = None):
        super().__init__()
        self.config = config or VehicleModelConfig()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        # Start close to constant-speed/constant-curvature behavior without
        # killing the gradient from the controls back into Action Expert.
        nn.init.normal_(self.network[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.network[-1].bias)

    @staticmethod
    def _zero_centered_asymmetric_sigmoid(raw: Tensor, minimum, maximum) -> Tensor:
        zero_fraction = -minimum / (maximum - minimum)
        if torch.is_tensor(zero_fraction):
            bias = torch.log(zero_fraction / (1.0 - zero_fraction))
        else:
            bias = math.log(zero_fraction / (1.0 - zero_fraction))
        return minimum + (maximum - minimum) * torch.sigmoid(raw + bias)

    def forward(
        self, hidden_states: Tensor, *, range_scale: float = 1.0,
        scene_parameters: Mapping[str, Tensor | float] | None = None,
    ) -> Tensor:
        if hidden_states.ndim < 3 or hidden_states.shape[-2] != self.config.action_horizon:
            raise ValueError(
                f"expected [..., {self.config.action_horizon}, hidden_dim], "
                f"got {tuple(hidden_states.shape)}"
            )
        if not 0.0 < range_scale <= 1.0:
            raise ValueError("range_scale must be in (0, 1]")
        raw = self.network(hidden_states)
        parameters = scene_parameters or {}
        min_acceleration = _scene_parameter(
            parameters.get("min_acceleration", self.config.min_acceleration), raw[..., 0]
        )
        max_acceleration = _scene_parameter(
            parameters.get("max_acceleration", self.config.max_acceleration), raw[..., 0]
        )
        acceleration = self._zero_centered_asymmetric_sigmoid(
            raw[..., 0], min_acceleration, max_acceleration
        )
        max_curvature_rate = _scene_parameter(
            parameters.get("max_abs_curvature_rate", self.config.max_abs_curvature_rate),
            raw[..., 1],
        )
        curvature_rate = max_curvature_rate * torch.tanh(raw[..., 1])
        return range_scale * torch.stack([acceleration, curvature_rate], dim=-1)


class DifferentiableVehicleModel(nn.Module):
    """RK4 rollout of the continuous-curvature kinematic vehicle equations."""

    def __init__(self, config: VehicleModelConfig | None = None):
        super().__init__()
        self.config = config or VehicleModelConfig()
        # Validate discretization at construction time.
        _ = self.config.substeps

    @staticmethod
    def derivative(state: Tensor, control: Tensor) -> Tensor:
        yaw, speed, curvature = state[..., 2], state[..., 3], state[..., 4]
        acceleration, curvature_rate = control[..., 0], control[..., 1]
        return torch.stack(
            [
                speed * yaw.cos(),
                speed * yaw.sin(),
                speed * curvature,
                acceleration,
                curvature_rate,
            ],
            dim=-1,
        )

    def _rk4_step(
        self, state: Tensor, control: Tensor, dt: float,
        scene_parameters: Mapping[str, Tensor | float] | None = None,
    ) -> Tensor:
        parameters = scene_parameters or {}
        mu = _scene_parameter(
            parameters.get("friction_coefficient", self.config.friction_coefficient),
            state[..., 3],
        )
        gravity = _scene_parameter(
            parameters.get("gravity", self.config.gravity), state[..., 3]
        )
        lateral = state[..., 3].square() * state[..., 4]
        available_longitudinal = (
            (mu * gravity).square() - lateral.square()
        ).clamp_min(0.0).sqrt()
        # Smoothly enforce the friction circle inside the rollout. This keeps
        # gradients while making longitudinal authority depend on cornering.
        acceleration = smooth_clamp(
            control[..., 0], -available_longitudinal, available_longitudinal,
            self.config.smooth_clamp_beta,
        )
        control = torch.stack([acceleration, control[..., 1]], dim=-1)
        k1 = self.derivative(state, control)
        k2 = self.derivative(state + 0.5 * dt * k1, control)
        k3 = self.derivative(state + 0.5 * dt * k2, control)
        k4 = self.derivative(state + dt * k3, control)
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        min_speed = _scene_parameter(
            parameters.get("min_speed", self.config.min_speed), next_state[..., 3]
        )
        max_speed = _scene_parameter(
            parameters.get("max_speed", self.config.max_speed), next_state[..., 3]
        )
        max_curvature = _scene_parameter(
            parameters.get("max_abs_curvature", self.config.max_abs_curvature),
            next_state[..., 4],
        )
        speed = smooth_clamp(
            next_state[..., 3], min_speed, max_speed,
            self.config.smooth_clamp_beta,
        )
        curvature = smooth_clamp(
            next_state[..., 4], -max_curvature,
            max_curvature, self.config.smooth_clamp_beta,
        )
        yaw = torch.atan2(next_state[..., 2].sin(), next_state[..., 2].cos())
        return torch.stack(
            [next_state[..., 0], next_state[..., 1], yaw, speed, curvature], dim=-1
        )

    def forward(
        self, initial_state: Tensor, controls: Tensor, *,
        scene_parameters: Mapping[str, Tensor | float] | None = None,
    ) -> Tensor:
        """Return coarse states shaped ``[..., horizon, 5]``.

        ``initial_state`` is ``[..., 5]`` and controls are ``[..., horizon, 2]``.
        Integration is performed in float32 for numerical stability; casts are
        differentiable, so gradients still reach mixed-precision policy heads.
        """
        if initial_state.shape[-1:] != (5,):
            raise ValueError(f"initial_state must end in 5, got {tuple(initial_state.shape)}")
        if controls.shape[:-2] != initial_state.shape[:-1]:
            raise ValueError("initial_state and controls batch dimensions must match")
        if controls.shape[-2:] != (self.config.action_horizon, 2):
            raise ValueError(
                f"controls must end in ({self.config.action_horizon}, 2), got {tuple(controls.shape)}"
            )
        state, controls_fp32 = initial_state.float(), controls.float()
        states = []
        for coarse_step in range(self.config.action_horizon):
            control = controls_fp32[..., coarse_step, :]
            for _ in range(self.config.substeps):
                state = self._rk4_step(
                    state, control, self.config.integration_dt, scene_parameters
                )
            states.append(state)
        return torch.stack(states, dim=-2)


class ContinuousControlTrajectoryHead(nn.Module):
    """Action Expert hidden states -> controls -> DriveVLA-compatible trajectory."""

    def __init__(self, hidden_dim: int, config: VehicleModelConfig | None = None):
        super().__init__()
        self.config = config or VehicleModelConfig()
        self.control_head = ContinuousControlHead(hidden_dim, self.config)
        self.vehicle_model = DifferentiableVehicleModel(self.config)

    def forward(
        self,
        action_hidden_states: Tensor,
        initial_state: Tensor,
        *,
        range_scale: float = 1.0,
        scene_parameters: Mapping[str, Tensor | float] | None = None,
    ) -> dict[str, Tensor]:
        controls = self.control_head(
            action_hidden_states, range_scale=range_scale,
            scene_parameters=scene_parameters,
        )
        states = self.vehicle_model(
            initial_state, controls, scene_parameters=scene_parameters
        )
        absolute_trajectory = states[..., :3]
        relative_trajectory = absolute_to_relative_se2(absolute_trajectory)
        return {
            "controls": controls,
            "states": states,
            "absolute_trajectory": absolute_trajectory,
            "relative_trajectory": relative_trajectory,
        }
