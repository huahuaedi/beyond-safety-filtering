"""Collect candidates with cost-guided sampling, reusing the unchanged collect-only loop of rl_stage1_prototype.

Run from an overlay directory that holds only this file and guidance_geometry.py, with the repository's tools/rl_prep on
PYTHONPATH, so that ``import rl_stage1_prototype`` resolves to the validated copy. Only the candidate rollouts (noise_std > 0)
are guided; the deterministic reference probes (noise_std == 0) call the original function. With GUIDE_LAMBDA=0 the original
function is called for everything, so the output is identical to an unguided collection with the same seed.

Env: GUIDE_DIR (per-scene inputs from extract_guidance_inputs.py), GUIDE_LAMBDA, GUIDE_POWER, GUIDE_MAX_GRAD,
     GUIDE_CLEARANCE_MARGIN, GUIDE_ROAD_MARGIN; everything else as for the unguided collection.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

import guidance_geometry as G
import rl_stage1_prototype as core

LAM = float(os.environ.get("GUIDE_LAMBDA", "0"))
POWER = float(os.environ.get("GUIDE_POWER", "1.0"))
MAX_GRAD = float(os.environ.get("GUIDE_MAX_GRAD", "1.0"))
CLEAR_MARGIN = float(os.environ.get("GUIDE_CLEARANCE_MARGIN", "0.5"))
ROAD_MARGIN = float(os.environ.get("GUIDE_ROAD_MARGIN", "0.25"))
GUIDE_DIR = Path(os.environ.get("GUIDE_DIR", "."))
HORIZON = 0.5 * np.arange(1, 9)

STATE = {"token": None, "costs": {}, "norm": None}
ORIGINAL_ROLLOUT = core.rollout_and_score
ORIGINAL_SPLIT = core.stable_scene_split


def norm_bounds():
    if STATE["norm"] is None:
        norm = json.loads(core.NORM_STATS_PATH.read_text())["norm_stats"]["libero"]
        STATE["norm"] = (torch.tensor(norm["q01"], dtype=torch.float32, device=core.DEVICE),
                         torch.tensor(norm["q99"], dtype=torch.float32, device=core.DEVICE))
    return STATE["norm"]


def cost_for(token):
    if token not in STATE["costs"]:
        z = np.load(GUIDE_DIR / f"{token}.npz", allow_pickle=False)
        dev = core.DEVICE
        agents = {k: torch.tensor(z["agent_" + k], dtype=torch.float32, device=dev)
                  for k in ("pos", "vel", "heading", "length", "width")}
        ego = {"length": float(z["ego_length"]), "width": float(z["ego_width"]), "center_offset": float(z["ego_center_offset"])}
        low, high = norm_bounds()
        STATE["costs"][token] = G.GuidanceCost(
            agents, torch.tensor(z["sdf"].astype(np.float32), device=dev), tuple(float(v) for v in z["sdf_origin"]),
            float(z["sdf_res"]), ego, low, high, torch.tensor(HORIZON, dtype=torch.float32, device=dev),
            clearance_margin=CLEAR_MARGIN, road_margin=ROAD_MARGIN)
    return STATE["costs"][token]


def remember_token(token):
    STATE["token"] = token                      # called once per scene by the collect-only loop, before its rollouts
    return ORIGINAL_SPLIT(token)


def guided_rollout_and_score(policy, heads, input_ids, attention_mask, pre_action, cmd, action_frames, action_dim,
                             num_steps, noise_std, initial_z=None, transition_noise=None):
    if LAM <= 0.0 or noise_std <= 0.0 or STATE["token"] is None:
        return ORIGINAL_ROLLOUT(policy, heads, input_ids, attention_mask, pre_action, cmd, action_frames, action_dim,
                                num_steps, noise_std, initial_z=initial_z, transition_noise=transition_noise)
    cost = cost_for(STATE["token"])
    batch_size = input_ids.shape[0]
    with torch.no_grad():
        vlm_h = policy.vlm.model.embed_tokens(input_ids)
        vlm_seq_len = vlm_h.shape[1]
        positions = torch.arange(vlm_seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        state_embedding = heads.state_projector(torch.cat([pre_action.reshape(batch_size, -1), cmd], dim=1)).unsqueeze(1)
        z = initial_z.clone() if initial_z is not None else torch.randn(
            batch_size, action_frames, action_dim, device=input_ids.device, dtype=vlm_h.dtype)
        dt = -1.0 / num_steps
        sigma = noise_std * (abs(dt) ** 0.5)
        steps, noise_i, t = [], 0, 1.0
        while t >= -dt / 2:
            velo, contextual_vlm = core.denoise_step_velocity(
                heads, vlm_h, state_embedding, z, t, input_ids, attention_mask, positions, vlm_seq_len, batch_size,
                return_vlm_context=True)
            mean = z + dt * velo
            eps = transition_noise[noise_i] if transition_noise is not None else torch.randn_like(z)
            x0 = (z.float() - t * velo.float()).detach().requires_grad_(True)       # estimate of the final action
            with torch.enable_grad():
                total = cost(x0)
                (grad,) = torch.autograd.grad(total.sum(), x0)
            norm = grad.flatten(1).norm(dim=1).clamp_min(1e-12)
            grad = grad * (norm.clamp(max=MAX_GRAD) / norm)[:, None, None]
            push = (LAM * (1.0 - t) ** POWER * grad).to(z.dtype)
            z_next = mean + sigma * eps - push
            steps.append({"t": t, "z": z, "z_next": z_next, "_dt": dt})
            z = z_next
            t += dt
            noise_i += 1
        if steps:
            steps[-1]["vlm_context"] = contextual_vlm.detach()
        final_cost = cost(z.float()).detach().cpu().numpy()
    print(f"guided token={STATE['token']} lam={LAM} final_cost mean={final_cost.mean():.4f} "
          f"share_violating={(final_cost > 0).mean():.3f}", flush=True)
    return z, steps, vlm_seq_len


if __name__ == "__main__":
    core.stable_scene_split = remember_token
    core.rollout_and_score = guided_rollout_and_score
    print(f"guided collection: lambda={LAM} power={POWER} max_grad={MAX_GRAD} dir={GUIDE_DIR}", flush=True)
    core.main()
