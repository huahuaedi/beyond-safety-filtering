"""Differentiable WM-conditioned outcome model for correction RL.

The critic is the frozen environment used by the unified correction policy. It
predicts both constraint probabilities and driving progress for an arbitrary
continuous trajectory. Keeping progress in the same model prevents the
degenerate "stop to look safe" solution produced by a risk-only objective.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ContinuousSceneOutcomeCritic(nn.Module):
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
        self.trunk = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU()
        )
        self.risk_head = nn.Linear(dim, len(self.risk_names))
        self.progress_head = nn.Linear(dim, 1)

    def scene_context(self, candidate_latents: Tensor) -> Tensor:
        if candidate_latents.ndim != 4 or candidate_latents.shape[-2:] != (
            4, self.latent_dim
        ):
            raise ValueError(
                f"expected [B,C,4,{self.latent_dim}], got {tuple(candidate_latents.shape)}"
            )
        layers = candidate_latents.mean(dim=1)
        weights = torch.softmax(self.layer_attention(layers).squeeze(-1), dim=-1)
        return (layers * weights.unsqueeze(-1)).sum(dim=-2)

    def forward(self, action: Tensor, scene_context: Tensor) -> dict[str, Tensor]:
        if action.shape[-1] != 24 or scene_context.shape[-1] != self.latent_dim:
            raise ValueError("action/context must end in 24/latent_dim")
        if scene_context.ndim == action.ndim - 1:
            scene_context = scene_context.unsqueeze(-2).expand(
                *action.shape[:-1], self.latent_dim
            )
        if scene_context.shape[:-1] != action.shape[:-1]:
            raise ValueError("scene context batch dimensions must match action")
        hidden = self.trunk(torch.cat([
            self.action_encoder(action), self.context_encoder(scene_context)
        ], dim=-1))
        return {
            "risk_logits": self.risk_head(hidden),
            "progress": self.progress_head(hidden).squeeze(-1).sigmoid(),
        }

    def outcomes(self, action: Tensor, scene_context: Tensor) -> dict[str, Tensor]:
        output = self(action, scene_context)
        return {
            "risk": output["risk_logits"].sigmoid(),
            "progress": output["progress"],
        }

