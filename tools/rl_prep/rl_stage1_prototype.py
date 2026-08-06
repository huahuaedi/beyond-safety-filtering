"""
Stage 1 of docs/rl_world_model_design.md: get the RL *mechanics* working end
to end on this local single-GPU machine, decoupled from the still-unvalidated
world-model reward (see docs/rl_world_model_design.md, "阶段 0.5 结果").

Reward source for this prototype: nearest-neighbor lookup into the 8192-anchor
trajectory vocabulary, using the *real* NAVSIM-simulator PDM sub-scores we
already extracted (data/navsim/processed_data/anchor/trainval_scene_anchor_scores.pkl).
This is an approximation (snapping a continuous sampled action to its nearest
of 8192 fixed anchors) but it is grounded in real simulator output, unlike the
world-model signal, so it lets us validate:
  - ReinFlow-style noisy Euler sampling of the flow-matching Action Expert
    with a tractable per-step Gaussian log-prob
  - GRPO-style within-scene group-relative advantage
  - a KL penalty against the frozen SFT reference policy
  - that training only moves the (small, decoupled) Action Expert weights,
    with the VLM backbone frozen

Memory note: a single Emu3Pi0 (VLM + action expert) already uses ~18GB in
bf16, so this 24GB card can't hold two full model copies. Since the VLM is
frozen and therefore identical between "policy" and "reference", we only
duplicate the small (~581M param) action-expert-side modules for the
reference snapshot and reuse the single frozen VLM's layers for both.

This is a prototype for validating plumbing, not a tuned training recipe.
"""
import copy
import hashlib
import json
import os
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.utils.checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "inference" / "vla"))  # for `config` module
sys.path.insert(0, str(REPO_ROOT / "utils"))
sys.path.insert(0, str(REPO_ROOT / "reference" / "Emu3"))
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("VLA_ACTION_TOKENIZER", str(REPO_ROOT / "pretrained_models/fast"))
os.environ.setdefault("VLA_VLM_MODEL", str(REPO_ROOT / "pretrained_models/Emu3-Stage1"))
os.environ.setdefault("VLA_NORM_STATS", str(REPO_ROOT / "configs/normalizer_navsim_trainval/norm_stats.json"))
os.environ.setdefault(
    "VLA_PATH_REPLACEMENTS",
    "/mnt/vdb1/yingyan.li/repo/VLA/data/navsim/processed_data/:"
    f"{REPO_ROOT}/data/navsim/processed_data/data/navsim/processed_data/;"
    "/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/data/navsim/processed_data/:"
    f"{REPO_ROOT}/data/navsim/processed_data/data/navsim/processed_data/;"
    "/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/data/navsim/processed_data/:"
    f"{REPO_ROOT}/data/navsim/processed_data/data/navsim/processed_data/",
)

from config import setup_paths_early  # noqa: E402
setup_paths_early()

from emu3.mllm import Emu3Tokenizer, Emu3Pi0, Emu3Pi0Config  # noqa: E402
from emu3.mllm.modeling_emu3 import Emu3Pi0SharedLayer  # noqa: E402
from utils.train_pi0 import DataArguments  # noqa: E402
from datasets import Emu3DrivingVAVADataset  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMU_HUB = REPO_ROOT / "pretrained_models/DriveVLA-W0/Emu3_Flow_Matching_Action_Expert_PDMS_87.2"
VLM_MODEL_PATH = Path(os.environ["VLA_VLM_MODEL"])
NORM_STATS_PATH = Path(os.environ["VLA_NORM_STATS"])
SCENE_PKL = REPO_ROOT / "pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
CLUSTER_PATH = REPO_ROOT / "data/navsim/processed_data/anchor/cluster_centers_8192.npy"
ANCHOR_SCORE_PATH = REPO_ROOT / "data/navsim/processed_data/anchor/trainval_scene_anchor_scores.pkl"
# Scene pool actually confirmed to resolve to a valid NAVSIM scene anchor and
# get a metric_cache built (see tools/rl_prep/cache_trainval_rl_scenes.py) —
# a different, smaller set than anchor_scores.keys() since DriveVLA-W0's own
# frame sampling doesn't align 1:1 with NAVSIM's official scene grid.
NAVSIM_SCENE_TOKENS_PATH = REPO_ROOT / "data/navsim/processed_data/anchor/trainval_rl_scene_tokens_cached.json"

# REWARD_SOURCE=anchor (default): nearest-neighbor lookup into the 8192-anchor
#   vocabulary — validated first-pass approximation, see module docstring.
# REWARD_SOURCE=navsim: real online PDM simulator via a subprocess into the
#   separate `navsim` conda env (tools/rl_prep/navsim_score_worker.py) — no
#   longer limited to the fixed 8192-anchor vocabulary, but needs a
#   navsim-env python + a metric_cache built for the trainval scenes in use.
REWARD_SOURCE = os.environ.get("RL_REWARD_SOURCE", "anchor")
NAVSIM_ENV_PYTHON = os.environ.get("NAVSIM_ENV_PYTHON", "")  # e.g. ~/.conda/envs/navsim/bin/python
NAVSIM_METRIC_CACHE_PATH = os.environ.get("NAVSIM_METRIC_CACHE_PATH", "")  # trainval metric_cache dir
NAVSIM_WORKER_SCRIPT = REPO_ROOT / "tools/rl_prep/navsim_score_worker.py"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

GROUP_SIZE = int(os.environ.get("RL_GROUP_SIZE", 8))          # G in GRPO
NUM_STEPS = int(os.environ.get("RL_NUM_STEPS", 20))            # policy update steps
NUM_INFERENCE_STEPS = int(os.environ.get("RL_DENOISE_STEPS", 10))
NOISE_STD = float(os.environ.get("RL_NOISE_STD", 0.1))         # ReinFlow-style per-step noise scale
# KL_COEF now multiplies a *per-element* (per denoise-step, per action-dim)
# k3 estimate (see http://joschu.net/blog/kl-approx.html), not the raw summed
# log-ratio — the summed version scales with NUM_INFERENCE_STEPS*action_frames*
# action_dim (240 terms by default) and grew unbounded (KL>200 by step 160 in
# the first real run), which both makes the coefficient hard to tune and risks
# overflow if ever exponentiated. 0.5 here is a fresh starting point at the
# new (much smaller) per-element scale, not comparable to the old 0.05.
KL_COEF = float(os.environ.get("RL_KL_COEF", 0.5))
LR = float(os.environ.get("RL_LR", 1e-5))
ADVANTAGE_MODE = os.environ.get("RL_ADVANTAGE_MODE", "safety_gated")
ROLLBACK_COEF = float(os.environ.get("RL_ROLLBACK_COEF", 1.0))
DEPLOYMENT_PROBES = int(os.environ.get("RL_DEPLOYMENT_PROBES", 1))
KL_BOOST = float(os.environ.get("RL_KL_BOOST", 3.0))
KL_DECAY = float(os.environ.get("RL_KL_DECAY", 0.9))

# Where to write trained weights, for a real before/after PDMS comparison
# (阶段 2) — empty string skips saving entirely (e.g. short mechanics-only
# dry runs). policy_heads/ref_heads are references into `policy`'s own
# modules, not copies (see build_policy_heads), so optimizer.step() mutates
# `policy` in place and a plain policy.save_pretrained(...) captures the
# trained action-expert weights alongside the (unchanged) frozen VLM weights
# — a complete, directly loadable checkpoint in the same format as EMU_HUB.
CHECKPOINT_DIR = os.environ.get("RL_CHECKPOINT_DIR", "")
SAVE_EVERY = int(os.environ.get("RL_SAVE_EVERY", 200))
SAVE_STEPS = {
    int(x) for x in os.environ.get("RL_SAVE_STEPS", "").split(",") if x.strip()
}

ANCHOR_METRIC_KEYS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
]


def relative_to_absolute_se2(rel_traj: np.ndarray) -> np.ndarray:
    """Forward direction (matches EmuVLAAgent.convert_relative_to_absolute_se2_trajectory)."""
    H = rel_traj.shape[0]
    abs_traj = np.zeros_like(rel_traj)
    x = y = yaw = 0.0
    for i in range(H):
        dx, dy, dyaw = rel_traj[i]
        gdx = np.cos(yaw) * dx - np.sin(yaw) * dy
        gdy = np.sin(yaw) * dx + np.cos(yaw) * dy
        x += gdx
        y += gdy
        yaw = (yaw + dyaw + np.pi) % (2 * np.pi) - np.pi
        abs_traj[i] = [x, y, yaw]
    return abs_traj


def fixed_scene_latent(scene_token: str, probe_idx: int, action_frames: int, action_dim: int):
    """Stable per-scene deployment probe. Python's hash() is process-randomized,
    so use SHA-256 to make local/HPC runs and reference-cache entries identical."""
    digest = hashlib.sha256(f"{scene_token}:deployment_probe:{probe_idx}".encode()).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**31)
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    return torch.randn(
        1, action_frames, action_dim, generator=generator, device=DEVICE, dtype=DTYPE
    )


@dataclass
class ActionHeads:
    """Everything needed to run the flow-matching denoising step, other than
    the (frozen, shared) VLM embeddings/layers themselves."""
    tau_emb: torch.nn.Module
    action_projector: torch.nn.Module
    state_projector: torch.nn.Module
    action_decoder: torch.nn.Module
    action_expert_norm: torch.nn.Module
    shared_layers: List[Emu3Pi0SharedLayer]


def build_policy_heads(policy: Emu3Pi0) -> ActionHeads:
    return ActionHeads(
        tau_emb=policy.tau_emb,
        action_projector=policy.action_projector,
        state_projector=policy.state_projector,
        action_decoder=policy.action_decoder,
        action_expert_norm=policy.action_expert.norm,
        shared_layers=policy.shared_layers,
    )


def build_frozen_reference_heads(policy: Emu3Pi0) -> ActionHeads:
    """Deep-copies only the small action-expert-side modules (~581M params);
    reuses policy.vlm.model.layers (frozen, unchanged) for the VLM side of
    each shared layer instead of loading a whole second model."""
    ref_action_expert = copy.deepcopy(policy.action_expert).to(DEVICE)
    ref_tau_emb = copy.deepcopy(policy.tau_emb).to(DEVICE)
    ref_action_projector = copy.deepcopy(policy.action_projector).to(DEVICE)
    ref_state_projector = copy.deepcopy(policy.state_projector).to(DEVICE)
    ref_action_decoder = copy.deepcopy(policy.action_decoder).to(DEVICE)
    for m in (ref_action_expert, ref_tau_emb, ref_action_projector, ref_state_projector, ref_action_decoder):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    ref_shared_layers = [
        Emu3Pi0SharedLayer(vlm_layer, action_layer)
        for vlm_layer, action_layer in zip(policy.vlm.model.layers, ref_action_expert.layers)
    ]
    return ActionHeads(
        tau_emb=ref_tau_emb,
        action_projector=ref_action_projector,
        state_projector=ref_state_projector,
        action_decoder=ref_action_decoder,
        action_expert_norm=ref_action_expert.norm,
        shared_layers=ref_shared_layers,
    )


def freeze_vlm(policy: Emu3Pi0):
    for p in policy.vlm.parameters():
        p.requires_grad_(False)
    trainable = []
    for name, p in policy.named_parameters():
        if not name.startswith("vlm."):
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


def denoise_step_velocity(heads: ActionHeads, vlm_h, state_token_embedding, z, t_scalar,
                           input_ids, attention_mask, vlm_position_ids, vlm_seq_len, batch_size):
    """Reproduces the per-step computation inside Emu3Pi0.sample_actions, exposed
    so we can call it either under no_grad (rollout) or with grad (log-prob
    recomputation for the policy-gradient loss)."""
    t_tensor = torch.full((batch_size,), t_scalar, dtype=z.dtype, device=z.device)
    tau_emb = heads.tau_emb(t_tensor).to(z.dtype)
    tau_emb_expanded = tau_emb.unsqueeze(1).expand(-1, z.shape[1], -1)
    action_hidden_states_no_state = heads.action_projector(z, tau_emb_expanded)
    current_action_h = torch.cat([state_token_embedding, action_hidden_states_no_state], dim=1)

    action_seq_len_with_state = current_action_h.shape[1]
    combined_attention_mask_4d = Emu3Pi0.create_causal_style_attention_mask(
        vlm_seq_len, action_seq_len_with_state, attention_mask, input_ids,
        batch_size, z.device, vlm_h.dtype,
    )

    current_vlm_h = vlm_h
    for shared_layer in heads.shared_layers:
        if torch.is_grad_enabled():
            # Per-layer checkpointing (matches Emu3Pi0.forward's own use of
            # gradient checkpointing): checkpointing the whole 32-layer loop
            # as one unit still materializes all 32 layers' activations
            # during the single recompute, which is what actually OOM'd —
            # only per-layer granularity keeps peak memory to ~1 layer.
            current_vlm_h, current_action_h = torch.utils.checkpoint.checkpoint(
                shared_layer.__call__,
                current_vlm_h, current_action_h, vlm_position_ids,
                combined_attention_mask_4d, vlm_seq_len, action_seq_len_with_state, batch_size,
                use_reentrant=False,
            )
        else:
            current_vlm_h, current_action_h = shared_layer(
                current_vlm_h, current_action_h, vlm_position_ids,
                combined_attention_mask_4d, vlm_seq_len, action_seq_len_with_state, batch_size,
            )
    final_action_hidden_for_decode = heads.action_expert_norm(current_action_h)
    velo_t_pred = heads.action_decoder(final_action_hidden_for_decode[:, 1:, :], tau_emb_expanded)
    return velo_t_pred


def rollout_and_score(policy: Emu3Pi0, heads: ActionHeads, input_ids, attention_mask, pre_action, cmd,
                       action_frames, action_dim, num_steps, noise_std, initial_z=None):
    """No-grad rollout using the policy's current heads. Returns the per-step
    (z_t -> z_{t+dt}) transitions needed to later recompute log-probs with
    grad, plus the final denoised action."""
    batch_size = input_ids.shape[0]
    with torch.no_grad():
        vlm_h = policy.vlm.model.embed_tokens(input_ids)
        vlm_seq_len = vlm_h.shape[1]
        vlm_position_ids = torch.arange(vlm_seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        state_input = torch.cat([pre_action.reshape(batch_size, -1), cmd], dim=1)
        state_token_embedding = heads.state_projector(state_input).unsqueeze(1)

        z = initial_z.clone() if initial_z is not None else torch.randn(
            batch_size, action_frames, action_dim, device=input_ids.device, dtype=vlm_h.dtype
        )
        dt = -1.0 / num_steps
        sigma = noise_std * (abs(dt) ** 0.5)

        steps = []
        t = 1.0
        while t >= -dt / 2:
            velo = denoise_step_velocity(
                heads, vlm_h, state_token_embedding, z, t,
                input_ids, attention_mask, vlm_position_ids, vlm_seq_len, batch_size,
            )
            mean = z + dt * velo
            eps = torch.randn_like(z)
            z_next = mean + sigma * eps
            steps.append({"t": t, "z": z, "z_next": z_next, "_dt": dt})
            z = z_next
            t += dt

    return z, steps, vlm_seq_len


def reference_path_rollback_loss(policy: Emu3Pi0, policy_heads: ActionHeads, ref_heads: ActionHeads,
                                 input_ids, attention_mask, pre_action, cmd, steps, vlm_seq_len):
    """Pull the deployed zero-transition-noise ODE path back toward the frozen
    SFT velocity field after NAVSIM detects a reference-relative safety
    regression.  Unlike the global KL, this is activated only on a scene whose
    current deployment trajectory is less safe than its own SFT reference."""
    batch_size = input_ids.shape[0]
    vlm_h = policy.vlm.model.embed_tokens(input_ids)
    vlm_position_ids = torch.arange(vlm_seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
    state_input = torch.cat([pre_action.reshape(batch_size, -1), cmd], dim=1)
    policy_state = policy_heads.state_projector(state_input).unsqueeze(1)
    with torch.no_grad():
        ref_state = ref_heads.state_projector(state_input).unsqueeze(1)

    losses = []
    for step in steps:
        z, t = step["z"], step["t"]
        policy_velocity = denoise_step_velocity(
            policy_heads, vlm_h, policy_state, z, t, input_ids, attention_mask,
            vlm_position_ids, vlm_seq_len, batch_size,
        )
        with torch.no_grad():
            ref_velocity = denoise_step_velocity(
                ref_heads, vlm_h, ref_state, z, t, input_ids, attention_mask,
                vlm_position_ids, vlm_seq_len, batch_size,
            )
        losses.append(torch.mean((policy_velocity.float() - ref_velocity.float()) ** 2))
    return torch.stack(losses).mean()


def recompute_log_prob(policy: Emu3Pi0, heads: ActionHeads, input_ids, attention_mask, pre_action, cmd,
                        steps, sigma, vlm_seq_len):
    """Re-forward through the stored (z_t -> z_{t+dt}) transitions using the
    given heads (policy = grad-enabled, reference = no_grad by caller) to get
    log-prob of the already-sampled trajectory under those params."""
    batch_size = input_ids.shape[0]
    vlm_h = policy.vlm.model.embed_tokens(input_ids)
    vlm_position_ids = torch.arange(vlm_seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
    state_input = torch.cat([pre_action.reshape(batch_size, -1), cmd], dim=1)
    state_token_embedding = heads.state_projector(state_input).unsqueeze(1)

    log_prob = torch.zeros(batch_size, device=input_ids.device, dtype=torch.float32)
    for step in steps:
        z = step["z"]
        t = step["t"]
        z_next = step["z_next"]
        dt = step["_dt"]
        # Checkpointing happens per-layer inside denoise_step_velocity (see
        # its internal use of torch.utils.checkpoint) — wrapping this whole
        # multi-layer call as one extra outer checkpoint unit would still
        # materialize all 32 layers' activations during a single recompute,
        # which is what actually caused the earlier OOM.
        velo = denoise_step_velocity(
            heads, vlm_h, state_token_embedding, z, t,
            input_ids, attention_mask, vlm_position_ids, vlm_seq_len, batch_size,
        )
        mean = z + dt * velo
        var = sigma ** 2
        diff = (z_next.float() - mean.float())
        per_dim = -0.5 * (diff ** 2 / var + torch.log(torch.tensor(2 * np.pi * var, device=diff.device)))
        log_prob = log_prob + per_dim.sum(dim=(-1, -2))
    return log_prob


SUB_SCORE_KEYS = [
    "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
    "time_to_collision_within_bound", "comfort", "driving_direction_compliance",
]


def score_with_navsim_simulator(scene_token: str, abs_trajectories: np.ndarray) -> dict:
    """Cross-env call: scores a whole group of candidate trajectories for one
    scene via the real NAVSIM PDM simulator, running in the separate `navsim`
    conda env (incompatible torch/numpy pins vs `drivevla`, can't import
    in-process). One subprocess call per training step, not per candidate,
    so simulator/metric-cache loading is paid once per group.

    abs_trajectories: [G, 8, 3] absolute (x, y, heading) poses in the ego's
    initial local frame (same format relative_to_absolute_se2() produces).
    Returns: dict of [G] arrays, one per SUB_SCORE_KEYS entry — raw PDM
    sub-scores, not a single reduced reward. The caller (compute_safety_gated_advantage)
    decides how to turn these into an advantage; we stopped collapsing to a
    single weighted scalar here after 阶段 2 showed that any fixed weighting
    (including the naive TTC:progress 8:2 re-weight, still in
    navsim_score_worker.py's `train_reward` field for comparison) just changes
    the exchange rate rather than stopping policy from trading safety for
    progress. All-zero on any per-candidate scoring failure.
    """
    import subprocess
    import tempfile

    assert NAVSIM_ENV_PYTHON, "NAVSIM_ENV_PYTHON must be set when RL_REWARD_SOURCE=navsim"
    assert NAVSIM_METRIC_CACHE_PATH, "NAVSIM_METRIC_CACHE_PATH must be set when RL_REWARD_SOURCE=navsim"

    G = abs_trajectories.shape[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        traj_path = f"{tmpdir}/candidates.npy"
        out_path = f"{tmpdir}/scores.json"
        np.save(traj_path, abs_trajectories.astype(np.float32))

        result = subprocess.run(
            [
                NAVSIM_ENV_PYTHON, str(NAVSIM_WORKER_SCRIPT),
                "--metric-cache-path", NAVSIM_METRIC_CACHE_PATH,
                "--scene-token", scene_token,
                "--trajectories-npy", traj_path,
                "--output-json", out_path,
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"navsim_score_worker failed for {scene_token}: {result.stderr[-2000:]}", flush=True)
            return {k: np.zeros(G, dtype=np.float32) for k in SUB_SCORE_KEYS}

        with open(out_path) as f:
            payload = json.load(f)
        if "error" in payload:
            print(f"navsim_score_worker error for {scene_token}: {payload['error']}", flush=True)
            return {k: np.zeros(G, dtype=np.float32) for k in SUB_SCORE_KEYS}
        return {k: np.array([r[k] for r in payload["results"]], dtype=np.float32) for k in SUB_SCORE_KEYS}


# Hard safety gate — a candidate must clear ALL of these to be eligible for
# positive advantage at all, no matter how much progress it makes. NC/DAC are
# near-binary in NAVSIM's own scorer (any at-fault collision or drivable-area
# violation anywhere in the trajectory zeroes them out), so "== 1.0" is the
# correct threshold, not an arbitrary cutoff. TTC is graded — 0.5 is a LOOSE
# gate (held-out baseline TTC is 0.993), chosen to validate the hard-gate
# mechanism itself first without also fighting an aggressive threshold. This
# run therefore tests "does a loose hard gate stop severe TTC collapse",
# not "does it preserve baseline TTC" — don't over-read the result. A
# threshold sweep (0.5 / 0.9 / 0.99), or better, replacing the fixed
# threshold with a primal-dual expectation constraint, is planned follow-up
# (see docs/rl_world_model_design.md 阶段 3) — that stricter/adaptive version
# is what should be called "Constrained Flow-GRPO"; this one is
# "Safety-Gated GRPO", a heuristic hard gate, not a formal constrained
# optimization — keep that terminology distinction in anything written up.
SAFE_NC_THRESHOLD = 1.0
SAFE_DAC_THRESHOLD = 1.0
SAFE_TTC_THRESHOLD = 0.5
UNSAFE_ADVANTAGE = -1.0  # fixed penalty; strictly below the typical safe-group advantage range


def compute_safety_gated_advantage(sub_scores: dict) -> np.ndarray:
    """Safety-Gated GRPO advantage (阶段 3, see docs/rl_world_model_design.md):
    replaces plain group-relative advantage on a single weighted reward
    scalar, which 阶段 2 showed lets progress simply buy back a safety
    violation regardless of the exchange rate. Here safety is a hard gate,
    not a price:
      - partition the group into safe / unsafe by SAFE_*_THRESHOLD
      - among safe candidates, standard GRPO group-relative advantage on
        ego_progress + driving_direction_compliance (mean of the two)
      - unsafe candidates get a fixed negative advantage regardless of how
        much progress they made — as long as the group contains at least
        one safe candidate, no unsafe candidate can ever outrank it
      - the whole group's advantage is then re-centered to zero mean. Without
        this, a group with e.g. 6/16 safe has raw mean -0.625 (safe subgroup
        already averages ~0, each unsafe candidate contributes a flat -1) —
        that's a systematic downward shift applied to *every* candidate's
        log-prob regardless of relative quality, not just a safe-vs-unsafe
        preference signal. Re-centering removes that global suppression
        while providing that the mean is always > -1 whenever n_safe>=1
        (proof: mean = -unsafe_frac > -1 for unsafe_frac<1), so re-centered
        unsafe advantage (UNSAFE_ADVANTAGE - mean) stays strictly negative.
      - if there are 0 or 1 safe candidates, relative progress-ranking isn't
        meaningful, so fall back (0-safe case only) to ranking by (negated)
        constraint violation magnitude within the group — "less bad" gets
        relatively encouraged, "more bad" relatively discouraged. This means
        an all-unsafe group's least-bad candidate CAN get positive advantage
        (a minimum-violation recovery signal) — the "unsafe never gets
        positive advantage" guarantee only holds when the group has at least
        one genuinely safe candidate to gate against.
    """
    nc = sub_scores["no_at_fault_collisions"]
    dac = sub_scores["drivable_area_compliance"]
    ttc = sub_scores["time_to_collision_within_bound"]
    # anchor-mode's pre-extracted scores don't include driving_direction_compliance
    # (ANCHOR_METRIC_KEYS never had it) — fall back to ego_progress alone rather
    # than averaging against a fake zero.
    if "driving_direction_compliance" in sub_scores:
        progress = (sub_scores["ego_progress"] + sub_scores["driving_direction_compliance"]) / 2.0
    else:
        progress = sub_scores["ego_progress"]

    is_safe = (nc >= SAFE_NC_THRESHOLD) & (dac >= SAFE_DAC_THRESHOLD) & (ttc >= SAFE_TTC_THRESHOLD)
    n_safe = int(is_safe.sum())
    G = len(nc)
    adv = np.zeros(G, dtype=np.float32)

    if n_safe >= 2:
        safe_progress = progress[is_safe]
        mu, sigma = safe_progress.mean(), safe_progress.std() + 1e-6
        adv[is_safe] = (safe_progress - mu) / sigma
        adv[~is_safe] = UNSAFE_ADVANTAGE
        adv -= adv.mean()  # remove the systematic downward shift, see docstring
    elif n_safe == 1:
        adv[is_safe] = 1.0
        adv[~is_safe] = UNSAFE_ADVANTAGE
        adv -= adv.mean()
    else:
        # nobody in this group is safe — group-relative advantage on
        # (negated) violation magnitude, same normalization convention as
        # the safe branch: "less bad" is relatively encouraged, "more bad"
        # relatively discouraged, within this group only. Already zero-mean
        # by construction (it's a z-score), no re-centering needed — and
        # unlike the branches above, this one has no safe candidate to gate
        # against, so it's expected/correct that the least-bad candidate
        # here can land above zero.
        violation = (1.0 - nc) + (1.0 - dac) + np.maximum(0.0, SAFE_TTC_THRESHOLD - ttc)
        mu, sigma = violation.mean(), violation.std() + 1e-6
        adv = -(violation - mu) / sigma

    return adv, is_safe


def compute_reference_anchored_advantage(sub_scores: dict, reference_scores: dict):
    """Lexicographic group advantage relative to the frozen SFT policy on the
    same scene.  NC/DAC/TTC are NAVSIM binary feasibility indicators, so no
    progress gain may compensate for dropping any indicator below reference.
    Feasible candidates compete on progress/DDC; infeasible candidates remain
    negative and are ordered by total reference-relative violation."""
    nc = sub_scores["no_at_fault_collisions"]
    dac = sub_scores["drivable_area_compliance"]
    ttc = sub_scores["time_to_collision_within_bound"]
    progress = (
        sub_scores["ego_progress"]
        + 0.25 * sub_scores["driving_direction_compliance"]
    )
    ref_nc = float(reference_scores["no_at_fault_collisions"])
    ref_dac = float(reference_scores["drivable_area_compliance"])
    ref_ttc = float(reference_scores["time_to_collision_within_bound"])
    ref_utility = float(reference_scores["ego_progress"]) + 0.25 * float(
        reference_scores["driving_direction_compliance"]
    )

    violation = (
        np.maximum(0.0, ref_nc - nc)
        + np.maximum(0.0, ref_dac - dac)
        + np.maximum(0.0, ref_ttc - ttc)
    )
    feasible = violation <= 1e-8
    adv = np.zeros(len(nc), dtype=np.float32)
    n_feasible = int(feasible.sum())

    if n_feasible >= 2:
        improvement = progress[feasible] - ref_utility
        mu, sigma = improvement.mean(), improvement.std() + 1e-6
        adv[feasible] = (improvement - mu) / sigma
        # Severity ordering keeps a recovery signal while guaranteeing every
        # reference-regressing candidate stays below every feasible one.
        if (~feasible).any():
            bad = violation[~feasible]
            scaled = bad / (bad.max() + 1e-6)
            adv[~feasible] = -1.0 - scaled
        adv -= adv.mean()
    elif n_feasible == 1:
        adv[feasible] = 1.0
        if (~feasible).any():
            bad = violation[~feasible]
            adv[~feasible] = -1.0 - bad / (bad.max() + 1e-6)
        adv -= adv.mean()
    else:
        mu, sigma = violation.mean(), violation.std() + 1e-6
        adv = -(violation - mu) / sigma

    return adv.astype(np.float32), feasible, violation.astype(np.float32)


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    print("Loading tokenizer + policy ...", flush=True)
    tokenizer = Emu3Tokenizer.from_pretrained(
        str(VLM_MODEL_PATH), model_max_length=1400, padding_side="right", use_fast=False,
    )

    model_config = Emu3Pi0Config.from_pretrained(os.path.join(str(EMU_HUB), "config.json"))
    policy, _ = Emu3Pi0.from_pretrained(
        str(EMU_HUB), config=model_config, pretrain_vlm_path=str(VLM_MODEL_PATH),
        attn_implementation="sdpa", torch_dtype=DTYPE, trust_remote_code=True, output_loading_info=True,
    )
    policy = policy.to(DEVICE)
    trainable_params = freeze_vlm(policy)
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"Trainable params: {n_trainable / 1e6:.1f}M / {n_total / 1e6:.1f}M total", flush=True)

    policy_heads = build_policy_heads(policy)
    ref_heads = None
    if KL_COEF > 0:
        print("Building frozen reference snapshot (action-expert modules only, VLM reused) ...", flush=True)
        ref_heads = build_frozen_reference_heads(policy)
    else:
        print("RL_KL_COEF=0, skipping reference snapshot to save memory.", flush=True)
    if ADVANTAGE_MODE == "reference_anchored" and ref_heads is None:
        raise ValueError("reference_anchored mode requires RL_KL_COEF > 0 to build frozen reference heads")
    print(f"GPU memory after setup: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated", flush=True)

    # Default AdamW now that we're not squeezed onto a 24GB card (AdamW's
    # fp32 momentum/variance state needs ~4.6GB extra for 581M trainable
    # params — trivial on a 48GB+ GPU). RL_OPTIMIZER=sgd still available for
    # constrained hardware (see docs/rl_world_model_design.md 阶段 1 结果).
    if os.environ.get("RL_OPTIMIZER", "adamw").lower() == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=LR, momentum=0.0)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=LR)

    with open(NORM_STATS_PATH) as f:
        norm_cfg = json.load(f)["norm_stats"]["libero"]
    low = np.array(norm_cfg["q01"], dtype=np.float32)
    high = np.array(norm_cfg["q99"], dtype=np.float32)

    cluster = np.load(CLUSTER_PATH)  # [8192, 8, 3] absolute SE2
    cluster_flat = cluster.reshape(cluster.shape[0], -1)

    print(
        f"REWARD_SOURCE={REWARD_SOURCE} ADVANTAGE_MODE={ADVANTAGE_MODE} "
        f"ROLLBACK_COEF={ROLLBACK_COEF} DEPLOYMENT_PROBES={DEPLOYMENT_PROBES} "
        f"KL_BOOST={KL_BOOST} KL_DECAY={KL_DECAY}", flush=True,
    )
    if ADVANTAGE_MODE == "reference_anchored" and REWARD_SOURCE != "navsim":
        raise ValueError("reference_anchored mode currently requires RL_REWARD_SOURCE=navsim")
    anchor_scores = None
    navsim_scene_tokens = None
    if REWARD_SOURCE == "navsim":
        assert NAVSIM_ENV_PYTHON and Path(NAVSIM_ENV_PYTHON).exists(), \
            f"NAVSIM_ENV_PYTHON not set or not found: {NAVSIM_ENV_PYTHON!r}"
        assert NAVSIM_METRIC_CACHE_PATH and Path(NAVSIM_METRIC_CACHE_PATH).exists(), \
            f"NAVSIM_METRIC_CACHE_PATH not set or not found: {NAVSIM_METRIC_CACHE_PATH!r}"
        with open(NAVSIM_SCENE_TOKENS_PATH) as f:
            navsim_scene_tokens = set(json.load(f))
        print(f"{len(navsim_scene_tokens)} scenes with a cached NAVSIM metric_cache available", flush=True)
    else:
        with open(ANCHOR_SCORE_PATH, "rb") as f:
            anchor_scores = pickle.load(f)
        print(f"{len(anchor_scores)} scenes with real PDM anchor scores available", flush=True)

    data_args = DataArguments(
        data_path=str(SCENE_PKL),
        actions=True, driving=True, use_previous_actions=True, actions_format="fast",
        action_tokenizer_path=os.environ["VLA_ACTION_TOKENIZER"],
        frames=1, action_frames=8, action_dim=3, cur_frame_idx=3, pre_action_frames=3,
        video_format=None, use_flip=False,
    )
    dataset = Emu3DrivingVAVADataset(data_args, tokenizer=tokenizer)
    token_to_idx = {scene["token"]: i for i, scene in enumerate(dataset.data)}
    token_pool = navsim_scene_tokens if REWARD_SOURCE == "navsim" else anchor_scores
    usable_indices = [token_to_idx[tok] for tok in token_pool if tok in token_to_idx]
    print(f"{len(usable_indices)} scenes usable for RL rollouts", flush=True)
    if not usable_indices:
        print("No usable scenes — aborting.")
        return

    action_frames, action_dim = 8, 3
    dt = -1.0 / NUM_INFERENCE_STEPS
    sigma = NOISE_STD * (abs(dt) ** 0.5)

    reward_history = []
    # Reference heads never change. With stable per-scene probe latents, their
    # NAVSIM vector scores are immutable and can be cached after first use.
    reference_score_cache = {}
    adaptive_kl_multiplier = 1.0
    risk_event_count = 0

    for step_i in range(NUM_STEPS):
        idx = random.choice(usable_indices)
        scene = dataset.data[idx]
        sample = dataset[idx]
        input_ids_1 = sample["input_ids"].unsqueeze(0).to(DEVICE)
        attention_mask_1 = sample["attention_mask"].unsqueeze(0).to(DEVICE)
        pre_action_1 = sample["pre_action"].unsqueeze(0).to(DEVICE).to(DTYPE)
        cmd_1 = sample["cmd"].unsqueeze(0).to(DEVICE).to(DTYPE)

        # ---- phase 1: G no-grad rollouts, one at a time (batch_size=1 keeps
        # peak memory flat regardless of group size — this card can't hold
        # more than ~1 sample's worth of activations at once alongside the
        # ~18GB of model weights) ----
        if REWARD_SOURCE == "anchor":
            entry = anchor_scores[scene["token"]]
            traj_scores = entry["trajectory_scores"][0]
            stacked = {k: np.asarray(traj_scores[k]) for k in ANCHOR_METRIC_KEYS}

        rollouts = []
        abs_trajs = np.zeros((GROUP_SIZE, action_frames, 3), dtype=np.float32)
        for g in range(GROUP_SIZE):
            z_final, steps, vlm_seq_len = rollout_and_score(
                policy, policy_heads, input_ids_1, attention_mask_1, pre_action_1, cmd_1,
                action_frames, action_dim, NUM_INFERENCE_STEPS, NOISE_STD,
            )
            rollouts.append((steps, vlm_seq_len))

            rel_raw = 0.5 * (z_final[0].float().cpu().numpy() + 1.0) * (high - low) + low
            abs_trajs[g] = relative_to_absolute_se2(rel_raw)

        if REWARD_SOURCE == "navsim":
            if ADVANTAGE_MODE == "reference_anchored":
                token = scene["token"]
                current_probe_finals, current_probe_steps = [], []
                probe_vlm_seq_lens = []
                reference_finals, missing_reference_indices = [], []
                cache_hit = token in reference_score_cache

                for probe_idx in range(DEPLOYMENT_PROBES):
                    deployment_z = fixed_scene_latent(token, probe_idx, action_frames, action_dim)
                    current_final, det_steps, det_seq_len = rollout_and_score(
                        policy, policy_heads, input_ids_1, attention_mask_1, pre_action_1, cmd_1,
                        action_frames, action_dim, NUM_INFERENCE_STEPS, 0.0,
                        initial_z=deployment_z,
                    )
                    current_probe_finals.append(current_final)
                    current_probe_steps.append(det_steps)
                    probe_vlm_seq_lens.append(det_seq_len)
                    if not cache_hit:
                        reference_final, _, _ = rollout_and_score(
                            policy, ref_heads, input_ids_1, attention_mask_1, pre_action_1, cmd_1,
                            action_frames, action_dim, NUM_INFERENCE_STEPS, 0.0,
                            initial_z=deployment_z,
                        )
                        reference_finals.append(reference_final)
                        missing_reference_indices.append(probe_idx)

                extra_finals = current_probe_finals + reference_finals
                extra_abs = np.zeros((len(extra_finals), action_frames, action_dim), dtype=np.float32)
                for extra_i, z_final in enumerate(extra_finals):
                    rel_raw = 0.5 * (z_final[0].float().cpu().numpy() + 1.0) * (high - low) + low
                    extra_abs[extra_i] = relative_to_absolute_se2(rel_raw)
                all_scores = score_with_navsim_simulator(
                    token, np.concatenate([abs_trajs, extra_abs], axis=0)
                )
                sub_scores = {k: v[:GROUP_SIZE] for k, v in all_scores.items()}
                current_probe_scores = [
                    {k: float(v[GROUP_SIZE + probe_idx]) for k, v in all_scores.items()}
                    for probe_idx in range(DEPLOYMENT_PROBES)
                ]
                if not cache_hit:
                    ref_offset = GROUP_SIZE + DEPLOYMENT_PROBES
                    reference_score_cache[token] = [
                        {k: float(v[ref_offset + probe_idx]) for k, v in all_scores.items()}
                        for probe_idx in range(DEPLOYMENT_PROBES)
                    ]
                reference_probe_scores = reference_score_cache[token]

                # Candidate feasibility uses the conservative safety envelope
                # across fixed reference probes; utility uses their mean.
                reference_scores = {
                    k: (
                        max(x[k] for x in reference_probe_scores)
                        if k in ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")
                        else float(np.mean([x[k] for x in reference_probe_scores]))
                    )
                    for k in SUB_SCORE_KEYS
                }
            else:
                # One subprocess call scores the whole group — real PDM simulator,
                # not limited to the fixed 8192-anchor vocabulary.
                sub_scores = score_with_navsim_simulator(scene["token"], abs_trajs)
        else:
            sub_scores = {k: np.zeros(GROUP_SIZE, dtype=np.float32) for k in ANCHOR_METRIC_KEYS}
            for g in range(GROUP_SIZE):
                dists = np.linalg.norm(cluster_flat - abs_trajs[g].reshape(-1)[None, :], axis=1)
                nearest = np.argmin(dists)
                for k in ANCHOR_METRIC_KEYS:
                    sub_scores[k][g] = stacked[k][nearest]

        if ADVANTAGE_MODE == "reference_anchored":
            adv, is_safe, violation = compute_reference_anchored_advantage(sub_scores, reference_scores)
            probe_regrets = []
            for current_scores, ref_scores in zip(current_probe_scores, reference_probe_scores):
                nc_regret = max(0.0, ref_scores["no_at_fault_collisions"] - current_scores["no_at_fault_collisions"])
                dac_regret = max(0.0, ref_scores["drivable_area_compliance"] - current_scores["drivable_area_compliance"])
                ttc_regret = max(0.0, ref_scores["time_to_collision_within_bound"] - current_scores["time_to_collision_within_bound"])
                probe_regrets.append((nc_regret + dac_regret + ttc_regret, nc_regret, dac_regret, ttc_regret))
            worst_probe_idx = int(np.argmax([x[0] for x in probe_regrets]))
            det_violation, det_nc_regret, det_dac_regret, det_ttc_regret = probe_regrets[worst_probe_idx]
            if det_violation > 0.0:
                adaptive_kl_multiplier = KL_BOOST
                risk_event_count += 1
            else:
                adaptive_kl_multiplier = 1.0 + KL_DECAY * (adaptive_kl_multiplier - 1.0)
        else:
            adv, is_safe = compute_safety_gated_advantage(sub_scores)
            violation = np.zeros(GROUP_SIZE, dtype=np.float32)
            det_violation = 0.0
            det_nc_regret = det_dac_regret = det_ttc_regret = 0.0
            adaptive_kl_multiplier = 1.0
            worst_probe_idx = 0
            cache_hit = False
        effective_kl_coef = KL_COEF * adaptive_kl_multiplier
        if ADVANTAGE_MODE == "reference_anchored" and det_violation > 0.0:
            rl_weight = 0.0
        else:
            rl_weight = 1.0 / adaptive_kl_multiplier

        # ---- phase 2: gradient-accumulate the per-sample REINFORCE (+ KL)
        # loss across the group, one sample at a time, then a single
        # optimizer step — equivalent to a batched group update but with
        # batch_size=1 peak memory. ----
        optimizer.zero_grad()
        pg_loss_sum, kl_sum = 0.0, 0.0
        for g in range(GROUP_SIZE):
            steps, vlm_seq_len = rollouts[g]
            logp_policy = recompute_log_prob(policy, policy_heads, input_ids_1, attention_mask_1, pre_action_1, cmd_1, steps, sigma, vlm_seq_len)
            pg_loss_g = -rl_weight * (float(adv[g]) * logp_policy).mean()
            if ref_heads is not None:
                with torch.no_grad():
                    logp_ref = recompute_log_prob(policy, ref_heads, input_ids_1, attention_mask_1, pre_action_1, cmd_1, steps, sigma, vlm_seq_len)
                # k3 estimator (http://joschu.net/blog/kl-approx.html): always
                # >=0, lower variance than the raw log-ratio. Computed on the
                # *per-element* mean log-ratio (divide by steps*frames*dim)
                # so it stays O(1) and safe to exponentiate — the raw summed
                # log-ratio reached ~200 in the first run, which would have
                # overflowed exp() directly.
                n_elements = len(steps) * action_frames * action_dim
                log_ratio_mean = (logp_policy - logp_ref).mean() / n_elements
                log_ratio_mean = log_ratio_mean.clamp(-20.0, 20.0)
                ratio = torch.exp(log_ratio_mean)
                kl_g = (ratio - 1) - log_ratio_mean
                loss_g = pg_loss_g + effective_kl_coef * kl_g
                kl_sum += kl_g.item()
            else:
                loss_g = pg_loss_g
            (loss_g / GROUP_SIZE).backward()
            pg_loss_sum += pg_loss_g.item()

        rollback_value = 0.0
        if ADVANTAGE_MODE == "reference_anchored" and det_violation > 0.0:
            rollback_loss = reference_path_rollback_loss(
                policy, policy_heads, ref_heads, input_ids_1, attention_mask_1,
                pre_action_1, cmd_1, current_probe_steps[worst_probe_idx],
                probe_vlm_seq_lens[worst_probe_idx],
            )
            (ROLLBACK_COEF * det_violation * rollback_loss).backward()
            rollback_value = float(rollback_loss.item())

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        safe_frac = float(is_safe.mean())
        ttc_mean = float(sub_scores["time_to_collision_within_bound"].mean())
        progress_mean = float(sub_scores["ego_progress"].mean())
        reward_history.append({"safe_frac": safe_frac, "ttc_mean": ttc_mean, "progress_mean": progress_mean})
        print(
            f"step {step_i:03d} scene={scene['token']} "
            f"safe_frac={safe_frac:.4f} ttc_mean={ttc_mean:.4f} progress_mean={progress_mean:.4f} "
            f"adv_mean={adv.mean():.4f} adv_std={adv.std():.4f} "
            f"pg_loss={pg_loss_sum / GROUP_SIZE:.4f} kl={kl_sum / GROUP_SIZE:.4f} "
            f"rl_weight={rl_weight:.4f} kl_mult={adaptive_kl_multiplier:.4f} effective_kl={effective_kl_coef:.4f} "
            f"det_violation={det_violation:.4f} det_nc_regret={det_nc_regret:.4f} "
            f"det_dac_regret={det_dac_regret:.4f} det_ttc_regret={det_ttc_regret:.4f} "
            f"rollback={rollback_value:.6f} risk_events={risk_event_count} cache_hit={int(cache_hit)} "
            f"ref_ttc={reference_probe_scores[worst_probe_idx]['time_to_collision_within_bound'] if ADVANTAGE_MODE == 'reference_anchored' else -1:.4f} "
            f"det_ttc={current_probe_scores[worst_probe_idx]['time_to_collision_within_bound'] if ADVANTAGE_MODE == 'reference_anchored' else -1:.4f} "
            f"grad_norm={grad_norm:.4f}",
            flush=True,
        )

        should_save = (step_i + 1) in SAVE_STEPS or ((step_i + 1) % SAVE_EVERY == 0)
        if CHECKPOINT_DIR and should_save:
            # Separate subdirectory per step — a single shared path would
            # have each save silently overwrite the previous one, losing
            # every intermediate checkpoint needed to see *where* along
            # training PDMS starts moving (up or down).
            step_dir = f"{CHECKPOINT_DIR}/step_{step_i + 1:06d}"
            print(f"Saving checkpoint at step {step_i} to {step_dir} ...", flush=True)
            policy.save_pretrained(step_dir)

    if CHECKPOINT_DIR:
        final_dir = f"{CHECKPOINT_DIR}/step_{NUM_STEPS:06d}_final"
        print(f"Saving final checkpoint to {final_dir} ...", flush=True)
        policy.save_pretrained(final_dir)

    print("\n=== Summary ===")
    first5_safe = [h["safe_frac"] for h in reward_history[:5]]
    last5_safe = [h["safe_frac"] for h in reward_history[-5:]]
    first5_ttc = [h["ttc_mean"] for h in reward_history[:5]]
    last5_ttc = [h["ttc_mean"] for h in reward_history[-5:]]
    print(f"safe_frac over training: first 5 = {first5_safe}, last 5 = {last5_safe}")
    print(f"ttc_mean over training: first 5 = {first5_ttc}, last 5 = {last5_ttc}")


if __name__ == "__main__":
    main()
