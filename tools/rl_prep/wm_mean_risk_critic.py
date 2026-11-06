"""Deploy the trained WM-mean feasibility model as a differentiable critic.

The world-model latents may be precomputed and therefore do not carry a graph
back to the WM backbone.  The action/trajectory input remains continuous,
however, so a risk objective produced here can backpropagate into a trajectory
decoder or a differentiable vehicle model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from train_world_action_feasibility_transformer import (
    FeasibilityTransformer,
    LABEL_NAMES,
)


ENVIRONMENTAL_RISKS = ("unsafe", "ttc_failure", "dac_failure", "nc_failure")
_RISK_INDICES = tuple(LABEL_NAMES.index(name) for name in ENVIRONMENTAL_RISKS)


class WMMeanRiskCritic(nn.Module):
    """Candidate-wise frozen risk critic initialized from a WM-mean checkpoint.

    Inputs can contain any number of candidates; their final dimensions must be
    24 for the continuous relative trajectory and ``(4, 4096)`` for the four
    extracted WM layers.  Candidate groups are not mixed in WM-mean mode.
    """

    def __init__(self, model_dim: int = 256):
        super().__init__()
        self.model = FeasibilityTransformer("wm_mean", dim=model_dim)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        freeze: bool = True,
    ) -> "WMMeanRiskCritic":
        try:
            payload = torch.load(Path(checkpoint), map_location=map_location, weights_only=False)
        except TypeError as error:
            if "weights_only" not in str(error):
                raise
            payload = torch.load(Path(checkpoint), map_location=map_location)
        results = payload.get("results", {})
        mode = results.get("feature_mode")
        if mode != "wm_mean":
            raise ValueError(f"expected a wm_mean checkpoint, found feature_mode={mode!r}")
        critic = cls(model_dim=int(payload["model_dim"]))
        critic.model.load_state_dict(payload["model"], strict=True)
        if freeze:
            critic.freeze()
        return critic

    def freeze(self) -> "WMMeanRiskCritic":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    @staticmethod
    def _validate(action: Tensor, wm_latent: Tensor) -> None:
        if action.ndim < 2 or action.shape[-1] != 24:
            raise ValueError(f"action must end in 24 values, got {tuple(action.shape)}")
        if wm_latent.shape[:-2] != action.shape[:-1] or wm_latent.shape[-2:] != (4, 4096):
            raise ValueError(
                "wm_latent must match action leading dimensions and end in (4, 4096), "
                f"got action={tuple(action.shape)} latent={tuple(wm_latent.shape)}"
            )

    def forward(self, action: Tensor, wm_latent: Tensor) -> Tensor:
        """Return environmental-risk logits in ``ENVIRONMENTAL_RISKS`` order."""
        self._validate(action, wm_latent)
        single_candidate = action.ndim == 2
        if single_candidate:
            action = action.unsqueeze(-2)
            wm_latent = wm_latent.unsqueeze(-3)
        # The historical model accepts a quality vector but deliberately does
        # not consume it, because those fields contain post-outcome labels.
        condition = action.new_zeros(*action.shape[:-1], 0)
        logits, _ = self.model(action, wm_latent, condition)
        logits = logits[..., list(_RISK_INDICES)]
        return logits.squeeze(-2) if single_candidate else logits

    def probabilities(self, action: Tensor, wm_latent: Tensor) -> Tensor:
        return self(action, wm_latent).sigmoid()

    def named_probabilities(self, action: Tensor, wm_latent: Tensor) -> Mapping[str, Tensor]:
        probabilities = self.probabilities(action, wm_latent)
        return {name: probabilities[..., i] for i, name in enumerate(ENVIRONMENTAL_RISKS)}

    def risk_objective(
        self,
        action: Tensor,
        wm_latent: Tensor,
        *,
        weights: Sequence[float] | Tensor | None = None,
        reduction: str = "mean",
    ) -> Tensor:
        """Return a differentiable weighted expected-risk objective.

        Minimizing this value sends gradients to ``action`` (and anything that
        generated it) while the frozen critic remains unchanged.
        """
        risks = self.probabilities(action, wm_latent)
        if weights is None:
            weight_tensor = risks.new_ones(len(ENVIRONMENTAL_RISKS))
        else:
            weight_tensor = torch.as_tensor(weights, dtype=risks.dtype, device=risks.device)
            if weight_tensor.shape != (len(ENVIRONMENTAL_RISKS),):
                raise ValueError(f"weights must have shape (4,), got {tuple(weight_tensor.shape)}")
        objective = (risks * weight_tensor).sum(dim=-1) / weight_tensor.sum().clamp_min(1e-8)
        if reduction == "none":
            return objective
        if reduction == "mean":
            return objective.mean()
        if reduction == "sum":
            return objective.sum()
        raise ValueError(f"unsupported reduction: {reduction!r}")
