"""Differentiable geometric cost and cost-guided flow sampling for 8-waypoint ego plans.

The cost uses the same scene information as the selector's geometric features, and only the state observed at t=0:
agent boxes with constant-velocity forecasts, and the drivable area as a signed-distance raster in the ego frame at
t=0 (positive inside the drivable area, in metres). It is a hinge: a trajectory that keeps its clearance and stays inside
the drivable area has zero cost and zero gradient, so guidance leaves it untouched.

Guided sampling follows the policy's Euler loop (t from 1 to 0, x0_hat = z - t v for a flow with v = noise - action) and
moves z against the gradient of the cost evaluated at x0_hat, treating the velocity as constant (no backpropagation
through the network). The push is scaled by (1 - t)^power, because x0_hat is only trustworthy late in the loop.
"""
from __future__ import annotations

import math

import torch

CLEARANCE_CAP = 30.0
N_CIRCLES = 3


def to_absolute(rel):
    """Cumulative SE(2) poses [B, T, 3] from per-step increments, matching relative_to_absolute_se2 (positions use the yaw
    before the step's yaw increment)."""
    dx, dy, dyaw = rel[..., 0], rel[..., 1], rel[..., 2]
    yaw_cum = torch.cumsum(dyaw, dim=1)
    yaw_prev = torch.cat([torch.zeros_like(yaw_cum[:, :1]), yaw_cum[:, :-1]], dim=1)
    gdx = torch.cos(yaw_prev) * dx - torch.sin(yaw_prev) * dy
    gdy = torch.sin(yaw_prev) * dx + torch.cos(yaw_prev) * dy
    return torch.stack([torch.cumsum(gdx, dim=1), torch.cumsum(gdy, dim=1), yaw_cum], dim=-1)


def _unit(heading):
    return torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)


def ego_circles(poses, length, width, center_offset):
    radius = width / 2.0
    half = max(length / 2.0 - radius, 0.0)
    u = _unit(poses[..., 2])
    centers = poses[..., :2] + center_offset * u
    offsets = torch.linspace(-half, half, N_CIRCLES, dtype=poses.dtype, device=poses.device)
    return centers[..., None, :] + offsets[:, None] * u[..., None, :], radius        # [B, T, n, 2]


def agent_circles(pos, vel, heading, length, width, times):
    radius = width / 2.0                                                              # [N]
    half = torch.clamp(length / 2.0 - radius, min=0.0)
    offsets = half[:, None] * torch.linspace(-1.0, 1.0, N_CIRCLES, dtype=pos.dtype, device=pos.device)[None]
    centers = pos[:, None, :] + vel[:, None, :] * times[None, :, None]                # [N, T, 2]
    u = _unit(heading)
    return centers[:, :, None, :] + offsets[:, None, :, None] * u[:, None, None, :], radius   # [N, T, n, 2]


def agent_clearance(poses, agents, ego_dims, times):
    """Minimum circle-cover clearance to any forecast agent: [B, T] (negative = overlap)."""
    b, t = poses.shape[:2]
    if agents is None or len(agents["pos"]) == 0:
        return torch.full((b, t), CLEARANCE_CAP, dtype=poses.dtype, device=poses.device)
    ego, ego_r = ego_circles(poses, ego_dims["length"], ego_dims["width"], ego_dims["center_offset"])
    other, other_r = agent_circles(agents["pos"], agents["vel"], agents["heading"], agents["length"], agents["width"], times)
    diff = ego[:, None, :, :, None, :] - other[None, :, :, None, :, :]                # [B, N, T, ne, na, 2]
    dist = torch.sqrt((diff ** 2).sum(-1) + 1e-9).amin(dim=(-2, -1))                  # [B, N, T]
    return torch.clamp((dist - ego_r - other_r[None, :, None]).amin(dim=1), max=CLEARANCE_CAP)


def footprint_points(poses, length, width, center_offset):
    u = _unit(poses[..., 2])
    n = torch.stack([-u[..., 1], u[..., 0]], dim=-1)
    center = poses[..., :2] + center_offset * u
    lon = torch.tensor([1, 1, -1, -1, 1, -1, 0, 0, 0], dtype=poses.dtype, device=poses.device) * length / 2.0
    lat = torch.tensor([1, -1, 1, -1, 0, 0, 1, -1, 0], dtype=poses.dtype, device=poses.device) * width / 2.0
    return center[..., None, :] + lon[:, None] * u[..., None, :] + lat[:, None] * n[..., None, :]   # [B, T, 9, 2]


def sample_sdf(points, sdf, origin, res):
    """Bilinear lookup in an SDF raster sdf[row (y), col (x)] with cell (i, j) at (x0 + j res, y0 + i res)."""
    h, w = sdf.shape
    gx = 2.0 * (points[..., 0] - origin[0]) / ((w - 1) * res) - 1.0
    gy = 2.0 * (points[..., 1] - origin[1]) / ((h - 1) * res) - 1.0
    grid = torch.stack([gx, gy], dim=-1).reshape(points.shape[0], -1, 1, 2)
    out = torch.nn.functional.grid_sample(sdf[None, None].expand(points.shape[0], -1, -1, -1).to(points.dtype), grid,
                                          mode="bilinear", padding_mode="border", align_corners=True)
    return out.reshape(points.shape[:-1])


class GuidanceCost:
    """Cost of normalised policy actions [B, 8, 3] for one scene."""

    def __init__(self, agents, sdf, origin, res, ego_dims, low, high, times, clearance_margin=0.5, road_margin=0.25,
                 w_collision=1.0, w_road=1.0):
        self.agents, self.sdf, self.origin, self.res, self.ego = agents, sdf, tuple(origin), float(res), ego_dims
        self.low, self.high, self.times = low, high, times
        self.cm, self.rm, self.wc, self.wr = clearance_margin, road_margin, w_collision, w_road

    def absolute(self, action):
        rel = 0.5 * (action + 1.0) * (self.high - self.low) + self.low
        return to_absolute(rel)

    def terms(self, action):
        return self.terms_abs(self.absolute(action))

    def terms_abs(self, poses):
        clear = agent_clearance(poses, self.agents, self.ego, self.times)
        pts = footprint_points(poses, self.ego["length"], self.ego["width"], self.ego["center_offset"])
        sdf = sample_sdf(pts, self.sdf, self.origin, self.res)
        coll = torch.relu(self.cm - clear).pow(2).sum(dim=1)
        road = torch.relu(self.rm - sdf).pow(2).sum(dim=(1, 2))
        return coll, road, clear.amin(dim=1), sdf.amin(dim=(1, 2))

    def __call__(self, action):
        coll, road, _, _ = self.terms(action)
        return self.wc * coll + self.wr * road


def guided_euler(velocity_fn, z, num_steps, noise_std, cost_fn=None, lam=0.0, power=1.0, max_grad_norm=1.0, eps_fn=None):
    """Euler sampling of the flow with optional cost guidance. ``velocity_fn(z, t)`` returns v (no gradient needed)."""
    dt = -1.0 / num_steps
    sigma = noise_std * math.sqrt(abs(dt))
    t = 1.0
    z = z.clone()
    while t >= -dt / 2:
        with torch.no_grad():
            v = velocity_fn(z, t)
        push = 0.0
        if cost_fn is not None and lam > 0.0:
            x0 = (z - t * v).detach().float().requires_grad_(True)
            with torch.enable_grad():
                cost = cost_fn(x0)
                (g,) = torch.autograd.grad(cost.sum(), x0)
            norm = g.flatten(1).norm(dim=1).clamp_min(1e-12)
            g = g * (norm.clamp(max=max_grad_norm) / norm)[:, None, None]
            push = (lam * (1.0 - t) ** power * g).to(z.dtype)
        eps = eps_fn() if eps_fn is not None else torch.randn_like(z)
        z = z + dt * v + sigma * eps - push
        t += dt
    return z
