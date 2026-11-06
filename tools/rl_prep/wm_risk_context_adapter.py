"""Lightweight gated interface from WM latents/quality metadata to DriveVLA."""

import torch
from torch import nn


class WorldRiskContextAdapter(nn.Module):
    """Compress multi-layer WM latents into one action-conditioning token.

    The output gate is initialized to zero. Adding this token to a pretrained
    state token therefore preserves the original policy exactly at
    initialization and lets training learn how much WM context to trust.
    """

    def __init__(self, latent_dim, action_hidden_dim, condition_dim=11, bottleneck=512):
        super().__init__()
        self.layer_score = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 1),
        )
        self.latent_projector = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, bottleneck),
            nn.SiLU(),
        )
        self.condition_projector = nn.Sequential(
            nn.Linear(condition_dim, bottleneck),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(2 * bottleneck, action_hidden_dim),
            nn.LayerNorm(action_hidden_dim),
        )
        self.gate_logit = nn.Parameter(torch.tensor(-20.0))

    @property
    def gate(self):
        return torch.sigmoid(self.gate_logit)

    def forward(self, wm_latents, condition_vector):
        if wm_latents.ndim != 3:
            raise ValueError(
                f"wm_latents must be [batch, layers, dim], got {wm_latents.shape}"
            )
        if condition_vector.ndim != 2 or condition_vector.shape[0] != wm_latents.shape[0]:
            raise ValueError(
                "condition_vector must be [batch, condition_dim] with matching batch"
            )
        attention = torch.softmax(self.layer_score(wm_latents).squeeze(-1), dim=1)
        pooled = (wm_latents * attention.unsqueeze(-1)).sum(dim=1)
        latent_features = self.latent_projector(pooled)
        condition_features = self.condition_projector(condition_vector)
        token = self.output(torch.cat([latent_features, condition_features], dim=-1))
        return (self.gate * token).unsqueeze(1), attention
