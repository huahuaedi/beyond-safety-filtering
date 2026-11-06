"""Lightweight waypoint-to-control adapter used before VLA integration."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from typing import Mapping

from differentiable_vehicle_model import (
    ContinuousControlHead,
    DifferentiableVehicleModel,
    VehicleModelConfig,
    absolute_to_relative_se2,
)


class WaypointControlAdapter(nn.Module):
    """Convert DriveVLA incremental waypoints into executable controls.

    This standalone adapter establishes whether a control-space projection can
    preserve DriveVLA trajectories before the same control head is attached to
    expensive Action Expert hidden states.
    """

    def __init__(
        self,
        dim: int = 128,
        layers: int = 2,
        heads: int = 4,
        config: VehicleModelConfig | None = None,
    ):
        super().__init__()
        self.config = config or VehicleModelConfig()
        self.dim = int(dim)
        self.layers = int(layers)
        self.heads = int(heads)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(8), nn.Linear(8, dim), nn.GELU()
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.config.action_horizon, dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.control_head = ContinuousControlHead(dim, self.config)
        self.vehicle_model = DifferentiableVehicleModel(self.config)

    def forward(
        self,
        relative_waypoints: Tensor,
        initial_state: Tensor,
        *,
        range_scale: float = 1.0,
        scene_parameters: Mapping[str, Tensor | float] | None = None,
    ) -> dict[str, Tensor]:
        if relative_waypoints.shape[-2:] != (self.config.action_horizon, 3):
            raise ValueError(
                f"waypoints must end in ({self.config.action_horizon}, 3), "
                f"got {tuple(relative_waypoints.shape)}"
            )
        if initial_state.shape != relative_waypoints.shape[:-2] + (5,):
            raise ValueError("initial_state must match waypoint leading dimensions and end in 5")
        leading = relative_waypoints.shape[:-2]
        initial_per_step = initial_state.unsqueeze(-2).expand(
            *leading, self.config.action_horizon, 5
        )
        features = torch.cat([relative_waypoints, initial_per_step], dim=-1)
        hidden = self.input_projection(features)
        flat = hidden.reshape(-1, self.config.action_horizon, self.dim)
        flat = self.temporal_encoder(flat + self.position_embedding)
        hidden = flat.reshape(*leading, self.config.action_horizon, self.dim)
        controls = self.control_head(
            hidden, range_scale=range_scale, scene_parameters=scene_parameters
        )
        states = self.vehicle_model(
            initial_state, controls, scene_parameters=scene_parameters
        )
        absolute = states[..., :3]
        return {
            "controls": controls,
            "states": states,
            "absolute_trajectory": absolute,
            "relative_trajectory": absolute_to_relative_se2(absolute),
        }

    def metadata(self) -> dict:
        return {"dim": self.dim, "layers": self.layers, "heads": self.heads}
