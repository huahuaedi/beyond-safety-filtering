"""Continuous trajectory risk student conditioned on WM scene context."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ContinuousSceneRiskCritic(nn.Module):
    """Predict environmental risks for arbitrary continuous trajectories.

    Candidate-conditioned WM latents are averaged across the candidate set to
    form an action-independent scene context.  The candidate action is supplied
    through a separate continuous branch, so gradients are valid for newly
    generated trajectories.
    """

    risk_names = ("unsafe", "ttc_failure", "dac_failure", "nc_failure")

    def __init__(self, latent_dim: int = 4096, dim: int = 256):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.dim = int(dim)
        self.layer_attention = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, 1)
        )
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, dim), nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(24), nn.Linear(24, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(),
            nn.Linear(dim, len(self.risk_names)),
        )

    def scene_context(self, candidate_latents: Tensor) -> Tensor:
        if candidate_latents.ndim != 4 or candidate_latents.shape[-2:] != (4, self.latent_dim):
            raise ValueError(f"expected [B,C,4,{self.latent_dim}], got {tuple(candidate_latents.shape)}")
        layers = candidate_latents.mean(dim=1)
        weights = torch.softmax(self.layer_attention(layers).squeeze(-1), dim=-1)
        return (layers * weights.unsqueeze(-1)).sum(dim=-2)

    def forward(self, action: Tensor, scene_context: Tensor) -> Tensor:
        if action.shape[-1] != 24 or scene_context.shape[-1] != self.latent_dim:
            raise ValueError("action/context must end in 24/latent_dim")
        if scene_context.ndim == action.ndim - 1:
            scene_context = scene_context.unsqueeze(-2).expand(*action.shape[:-1], self.latent_dim)
        if scene_context.shape[:-1] != action.shape[:-1]:
            raise ValueError("scene context batch dimensions must match action")
        context = self.context_encoder(scene_context)
        trajectory = self.action_encoder(action)
        return self.head(torch.cat([trajectory, context], dim=-1))

    def risk(self, action: Tensor, scene_context: Tensor, weights=None) -> Tensor:
        probability = self(action, scene_context).sigmoid()
        if weights is None:
            weights = probability.new_ones(len(self.risk_names))
        else:
            weights = torch.as_tensor(weights, device=probability.device, dtype=probability.dtype)
        return (probability * weights).sum(-1) / weights.sum().clamp_min(1e-8)
