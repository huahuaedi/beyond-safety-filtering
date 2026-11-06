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

from risk_curriculum import (
    parse_source_mix,
    select_curriculum_memory,
    tempered_priority_weights,
)
from rollout_diagnostics import compute_rollout_diagnostics
from risk_replay_experiment import step_policy
from positive_safety_guard import guard_positive_advantage
from policy_optimization import (
    compute_reference_anchored_advantage as reference_anchored_advantage,
    shrinkage_centered_advantage,
    update_adaptive_kl_coefficient,
)
from adaptive_reward import validate_constraint_weights
from legacy_advantage import legacy_reference_anchored_advantage
from build_pseudo_control_sidecars import estimate_pseudo_controls
from trajectory_dynamics import analyze_pose_trajectories
from waypoint_control_adapter import WaypointControlAdapter
from differentiable_vehicle_model import VehicleModelConfig
from continuous_scene_outcome_critic import ContinuousSceneOutcomeCritic
from unified_correction_policy import UnifiedCorrectionPolicy, apply_verified_correction

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
ASSET_ROOT = Path(os.environ.get("DRIVEVLA_ASSET_ROOT", REPO_ROOT))
EMU_HUB = Path(os.environ.get("RL_POLICY_CHECKPOINT", ASSET_ROOT / "pretrained_models/DriveVLA-W0/Emu3_Flow_Matching_Action_Expert_PDMS_87.2"))
VLM_MODEL_PATH = Path(os.environ["VLA_VLM_MODEL"])
NORM_STATS_PATH = Path(os.environ["VLA_NORM_STATS"])
SCENE_PKL = ASSET_ROOT / "pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
CLUSTER_PATH = ASSET_ROOT / "data/navsim/processed_data/anchor/cluster_centers_8192.npy"
ANCHOR_SCORE_PATH = ASSET_ROOT / "data/navsim/processed_data/anchor/trainval_scene_anchor_scores.pkl"
# Scene pool actually confirmed to resolve to a valid NAVSIM scene anchor and
# get a metric_cache built (see tools/rl_prep/cache_trainval_rl_scenes.py) —
# a different, smaller set than anchor_scores.keys() since DriveVLA-W0's own
# frame sampling doesn't align 1:1 with NAVSIM's official scene grid.
NAVSIM_SCENE_TOKENS_PATH = Path(os.environ.get("RL_TRAIN_TOKENS", ASSET_ROOT / "data/navsim/processed_data/anchor/trainval_rl_scene_tokens_cached.json"))

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
# Optional desired-vs-executed trace per NAVSIM scoring call (see
# tools/rl_prep/execution_trace.py). Off by default; scores are unaffected.
EXECUTION_TRACE_DIR = os.environ.get("RL_EXECUTION_TRACE_DIR", "")
EXECUTION_TRACE_VERIFY = os.environ.get("RL_EXECUTION_TRACE_VERIFY", "0") == "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
SEED = int(os.environ.get("RL_SEED", 0))

GROUP_SIZE = int(os.environ.get("RL_GROUP_SIZE", 8))          # G in GRPO
ROLLOUT_MICRO_BATCH = int(os.environ.get("RL_ROLLOUT_MICRO_BATCH", 1))
if ROLLOUT_MICRO_BATCH < 1:
    raise ValueError("RL_ROLLOUT_MICRO_BATCH must be >= 1")
NUM_STEPS = int(os.environ.get("RL_NUM_STEPS", 20))            # policy update steps
NUM_INFERENCE_STEPS = int(os.environ.get("RL_DENOISE_STEPS", 10))
NOISE_STD = float(os.environ.get("RL_NOISE_STD", 0.1))         # ReinFlow-style per-step noise scale
DIAGNOSTICS = os.environ.get("RL_DIAGNOSTICS", "0") == "1"
DIAGNOSTIC_TRAJECTORY_EPS = float(os.environ.get("RL_DIAGNOSTIC_TRAJECTORY_EPS", 0.05))
DIAGNOSTIC_SCORE_EPS = float(os.environ.get("RL_DIAGNOSTIC_SCORE_EPS", 1e-4))
# KL_COEF now multiplies a *per-element* (per denoise-step, per action-dim)
# k3 estimate (see http://joschu.net/blog/kl-approx.html), not the raw summed
# log-ratio — the summed version scales with NUM_INFERENCE_STEPS*action_frames*
# action_dim (240 terms by default) and grew unbounded (KL>200 by step 160 in
# the first real run), which both makes the coefficient hard to tune and risks
# overflow if ever exponentiated. 0.5 here is a fresh starting point at the
# new (much smaller) per-element scale, not comparable to the old 0.05.
KL_COEF = float(os.environ.get("RL_KL_COEF", 0.5))
# Opt-in ablation: legacy behavior remains the default.
KL_MODE = os.environ.get("RL_KL_MODE", "legacy")
POSITIVE_SAFETY_GUARD = os.environ.get("RL_POSITIVE_SAFETY_GUARD", "0") == "1"
if KL_MODE not in ("legacy", "transition"):
    raise ValueError("RL_KL_MODE must be legacy or transition")
if KL_MODE == "transition":
    if "RL_KL_COEF" not in os.environ or not np.isfinite(KL_COEF) or KL_COEF <= 0:
        raise ValueError("transition KL requires an explicit positive RL_KL_COEF; scales differ")
    if os.environ.get("RL_ADAPTIVE_KL", "0") == "1" and "RL_TARGET_KL" not in os.environ:
        raise ValueError("transition KL requires an explicit RL_TARGET_KL for adaptive control")
ADVANTAGE_NOISE_FLOOR = float(os.environ.get("RL_ADVANTAGE_NOISE_FLOOR", 0.0))
ADAPTIVE_KL = os.environ.get("RL_ADAPTIVE_KL", "0") == "1"
TARGET_KL = float(os.environ.get("RL_TARGET_KL", 5e-4))
KL_ADAPTATION_RATE = float(os.environ.get("RL_KL_ADAPTATION_RATE", 0.1))
KL_COEF_MIN = float(os.environ.get("RL_KL_COEF_MIN", 0.05))
KL_COEF_MAX = float(os.environ.get("RL_KL_COEF_MAX", 5.0))
KL_HARD_LIMIT = float(os.environ.get("RL_KL_HARD_LIMIT", 0.0))
LR = float(os.environ.get("RL_LR", 1e-5))
ADVANTAGE_MODE = os.environ.get("RL_ADVANTAGE_MODE", "safety_gated")
# "current" = hard-sign gate (post 2026-09-23); "legacy_centered" = pre-repair group-centred z-score, for A/B only
ADVANTAGE_VERSION = os.environ.get("RL_ADVANTAGE_VERSION", "current")
if ADVANTAGE_VERSION not in ("current", "legacy_centered"):
    raise ValueError("RL_ADVANTAGE_VERSION must be current or legacy_centered")
ROLLBACK_COEF = float(os.environ.get("RL_ROLLBACK_COEF", 1.0))
DEPLOYMENT_PROBES = int(os.environ.get("RL_DEPLOYMENT_PROBES", 1))
KL_BOOST = float(os.environ.get("RL_KL_BOOST", 3.0))
KL_DECAY = float(os.environ.get("RL_KL_DECAY", 0.9))
RISK_REPLAY_MODE = os.environ.get("RL_RISK_REPLAY_MODE", "reference")
RISK_FIXED_SCHEDULE = os.environ.get("RL_RISK_FIXED_SCHEDULE", "")
NAVSIM_SCORE_CALLS = 0
NAVSIM_SCORED_TRAJECTORIES = 0
RISK_BUFFER_SIZE = int(os.environ.get("RL_RISK_BUFFER_SIZE", 32))
RISK_REPLAY_EVERY = int(os.environ.get("RL_RISK_REPLAY_EVERY", 1))
RISK_REPLAY_COEF = float(os.environ.get("RL_RISK_REPLAY_COEF", 1.0))
RISK_EXPOSURE_THRESHOLD = float(os.environ.get("RL_RISK_EXPOSURE_THRESHOLD", 0.05))
RISK_REPLAY_SAMPLING = os.environ.get("RL_RISK_REPLAY_SAMPLING", "priority").lower()
RISK_MANIFEST = os.environ.get("RL_RISK_MANIFEST", "")
RISK_COVERAGE_MANIFEST = os.environ.get("RL_RISK_COVERAGE_MANIFEST", "")
RISK_ONLINE_INSERT = os.environ.get("RL_RISK_ONLINE_INSERT", "1") == "1"
RISK_CURRICULUM = os.environ.get("RL_RISK_CURRICULUM", "0") == "1"
RISK_SOURCE_MIX_SPEC = os.environ.get(
    "RL_RISK_SOURCE_MIX", "manifest:0.5,coverage:0.25,online:0.25"
)
RISK_TYPE_BALANCE = os.environ.get("RL_RISK_TYPE_BALANCE", "1") == "1"
RISK_REDUNDANCY_COEF = float(os.environ.get("RL_RISK_REDUNDANCY_COEF", 0.25))
RISK_PRIORITY_POWER = float(os.environ.get("RL_RISK_PRIORITY_POWER", 1.0))
DYNAMIC_CONSTRAINT_WEIGHTS = os.environ.get("RL_DYNAMIC_CONSTRAINT_WEIGHTS", "0") == "1"
DYNAMIC_WEIGHT_MIN = float(os.environ.get("RL_DYNAMIC_WEIGHT_MIN", 0.5))
DYNAMIC_WEIGHT_MAX = float(os.environ.get("RL_DYNAMIC_WEIGHT_MAX", 2.0))
RISK_SOURCE_MIX = parse_source_mix(RISK_SOURCE_MIX_SPEC)
if RISK_REDUNDANCY_COEF < 0:
    raise ValueError("RL_RISK_REDUNDANCY_COEF must be >= 0")
if RISK_PRIORITY_POWER < 0:
    raise ValueError("RL_RISK_PRIORITY_POWER must be >= 0")
if not 0.0 < DYNAMIC_WEIGHT_MIN <= 1.0 <= DYNAMIC_WEIGHT_MAX:
    raise ValueError("dynamic weight bounds must satisfy 0 < min <= 1 <= max")
if RISK_REPLAY_SAMPLING not in {"priority", "uniform"}:
    raise ValueError(
        "RL_RISK_REPLAY_SAMPLING must be 'priority' or 'uniform', got "
        f"{RISK_REPLAY_SAMPLING!r}"
    )
if ADVANTAGE_NOISE_FLOOR < 0:
    raise ValueError("RL_ADVANTAGE_NOISE_FLOOR must be >= 0")
if ADAPTIVE_KL and not (TARGET_KL > 0 and KL_ADAPTATION_RATE >= 0):
    raise ValueError("adaptive KL requires RL_TARGET_KL > 0 and RL_KL_ADAPTATION_RATE >= 0")
if not 0 < KL_COEF_MIN <= KL_COEF_MAX:
    raise ValueError("KL coefficient bounds must satisfy 0 < min <= max")
if KL_HARD_LIMIT < 0:
    raise ValueError("RL_KL_HARD_LIMIT must be >= 0")
COLLECT_ONLY = os.environ.get("RL_COLLECT_ONLY", "0") == "1"
COLLECTION_DIR = os.environ.get("RL_COLLECTION_DIR", "")
COLLECTION_SHARD_ID = int(os.environ.get("RL_COLLECTION_SHARD_ID", 0))
COLLECTION_NUM_SHARDS = int(os.environ.get("RL_COLLECTION_NUM_SHARDS", 1))
if COLLECTION_NUM_SHARDS < 1 or not 0 <= COLLECTION_SHARD_ID < COLLECTION_NUM_SHARDS:
    raise ValueError(
        "collection shard must satisfy 0 <= RL_COLLECTION_SHARD_ID < "
        f"RL_COLLECTION_NUM_SHARDS, got {COLLECTION_SHARD_ID}/{COLLECTION_NUM_SHARDS}"
    )

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
CONTROL_ADAPTER_CHECKPOINT = os.environ.get("RL_CONTROL_ADAPTER_CHECKPOINT", "")
CONTROL_ADAPTER_MODE = os.environ.get("RL_CONTROL_ADAPTER_MODE", "infeasible_only")
CONTROL_ADAPTER_MAX_POSITION_RMSE = float(
    os.environ.get("RL_CONTROL_ADAPTER_MAX_POSITION_RMSE", "inf")
)
UNIFIED_CORRECTION_CHECKPOINT = os.environ.get(
    "RL_UNIFIED_CORRECTION_CHECKPOINT", ""
)
UNIFIED_OUTCOME_CHECKPOINT = os.environ.get("RL_UNIFIED_OUTCOME_CHECKPOINT", "")
UNIFIED_LATENT_ROOT = Path(os.environ.get(
    "RL_UNIFIED_LATENT_ROOT",
    REPO_ROOT / "navsim_data/exp/world_action_latents_full_v1",
))
UNIFIED_CORRECTION_LOGP_COEF = float(
    os.environ.get("RL_UNIFIED_CORRECTION_LOGP_COEF", 1.0)
)
# 'Projection + Correction-only' Table 1/4 arm: Action Expert stays fixed at
# its warm-start weights and only the correction policy is optimized by the
# joint RL loop. Isolates the correction policy's own contribution from the
# Action Expert's, the same way RL_UNIFIED_CORRECTION_CHECKPOINT left unset
# isolates the reverse (Action Expert trained, no correction at all).
FREEZE_ACTION_EXPERT = os.environ.get("RL_FREEZE_ACTION_EXPERT", "0") == "1"
# Risk-aware on-policy scene sampling (see load_scene_risk_weights):
# opt-in, default off, so every already-validated run (baseline/Direct RL/
# earlier unified-correction runs) keeps its exact prior behavior unless
# explicitly re-enabled.
RISK_WEIGHTED_SAMPLING = os.environ.get("RL_RISK_WEIGHTED_SAMPLING", "0") == "1"
RISK_SCENE_DATA_ROOT = Path(os.environ.get(
    "RL_RISK_SCENE_DATA_ROOT", REPO_ROOT / "navsim_data/exp/world_action_full_v1",
))
RISK_SAMPLING_EPSILON = float(os.environ.get("RL_RISK_SAMPLING_EPSILON", 0.1))
if bool(UNIFIED_CORRECTION_CHECKPOINT) != bool(UNIFIED_OUTCOME_CHECKPOINT):
    raise ValueError(
        "RL_UNIFIED_CORRECTION_CHECKPOINT and RL_UNIFIED_OUTCOME_CHECKPOINT "
        "must be set together"
    )
if UNIFIED_CORRECTION_CHECKPOINT and CONTROL_ADAPTER_CHECKPOINT:
    raise ValueError("unified correction and the legacy control adapter are mutually exclusive")
if FREEZE_ACTION_EXPERT and not UNIFIED_CORRECTION_CHECKPOINT:
    raise ValueError(
        "RL_FREEZE_ACTION_EXPERT requires RL_UNIFIED_CORRECTION_CHECKPOINT -- "
        "freezing the Action Expert with no correction policy would leave "
        "nothing trainable"
    )
if CONTROL_ADAPTER_MODE not in ("infeasible_only", "all"):
    raise ValueError("RL_CONTROL_ADAPTER_MODE must be infeasible_only or all")

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


def stable_scene_split(scene_token: str) -> str:
    """Deterministic scene-disjoint 80/10/10 split for world-action data."""
    bucket = int.from_bytes(hashlib.sha256(scene_token.encode()).digest()[:8], "little") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def load_scene_risk_weights(data_root: Path, epsilon: float) -> dict[str, float]:
    """Per-scene risk-aware sampling weight from pre-verified closed-loop
    safety outcomes, not a new model: world_action_full_v1 already stores 64
    candidate trajectories per scene, each scored against the real NAVSIM
    simulator.

    r_scene = p_NC + p_TTC + p_DAC + p_DDC, where each p_X is the fraction
    of that scene's 64 candidates failing metric X (score < 1). Deliberately
    the four hard safety-compliance sub-scores only (not ego_progress/
    comfort, and not reference_violation) for a first version whose signal
    is unambiguous: does risk-aware sampling visibly help correction-policy
    learning before adding a physical-violation term as a follow-up
    ablation. This directly targets the diagnosed root cause -- the primary
    on-policy training loop (usable_indices below) previously sampled
    scenes uniformly at random from the whole ~7171-scene pool, so a
    correction policy training run's small RL_NUM_STEPS budget touched only
    a handful of genuinely risky scenes by chance.

    Returns raw (un-normalized) weight = epsilon + r_scene; epsilon keeps
    every scene reachable so normal scenes are not starved, not just
    up-weighting the risky minority. Missing/unreadable files are skipped
    (caller falls back to epsilon for scenes with no weight entry) rather
    than failing the whole run over one bad file.
    """
    weights: dict[str, float] = {}
    paths = sorted(data_root.glob("*/*.npz"))
    if not paths:
        raise FileNotFoundError(f"no world-action candidate data under {data_root}")
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            token = str(data["scene_token"])
            failure_rate = lambda key: float(np.mean(data[key] < 1.0))
            risk = (
                failure_rate("score__no_at_fault_collisions")
                + failure_rate("score__time_to_collision_within_bound")
                + failure_rate("score__drivable_area_compliance")
                + failure_rate("score__driving_direction_compliance")
            )
        weights[token] = epsilon + risk
    return weights


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
                           input_ids, attention_mask, vlm_position_ids, vlm_seq_len, batch_size,
                           return_vlm_context=False):
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
    if return_vlm_context:
        mask = attention_mask.to(current_vlm_h.dtype).unsqueeze(-1)
        pooled_vlm = (current_vlm_h * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return velo_t_pred, pooled_vlm
    return velo_t_pred


def rollout_and_score(policy: Emu3Pi0, heads: ActionHeads, input_ids, attention_mask, pre_action, cmd,
                       action_frames, action_dim, num_steps, noise_std, initial_z=None,
                       transition_noise=None):
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
        noise_i = 0
        t = 1.0
        while t >= -dt / 2:
            velo, contextual_vlm = denoise_step_velocity(
                heads, vlm_h, state_token_embedding, z, t,
                input_ids, attention_mask, vlm_position_ids, vlm_seq_len, batch_size,
                return_vlm_context=True,
            )
            mean = z + dt * velo
            eps = (
                transition_noise[noise_i]
                if transition_noise is not None
                else torch.randn_like(z)
            )
            if eps.shape != z.shape:
                raise ValueError(
                    f"transition noise must match z={tuple(z.shape)}, got {tuple(eps.shape)}"
                )
            z_next = mean + sigma * eps
            steps.append({"t": t, "z": z, "z_next": z_next, "_dt": dt})
            z = z_next
            t += dt
            noise_i += 1

        if steps:
            steps[-1]["vlm_context"] = contextual_vlm.detach()

        if transition_noise is not None and noise_i != len(transition_noise):
            raise ValueError(
                f"received {len(transition_noise)} transition noises, consumed {noise_i}"
            )

    return z, steps, vlm_seq_len


def split_batched_rollout(final_z, steps, vlm_seq_len):
    """Return batch-size-one records consumed by the unchanged grad loop."""
    records = []
    for batch_i in range(final_z.shape[0]):
        member_steps = [
            {
                "t": step["t"],
                "z": step["z"][batch_i:batch_i + 1],
                "z_next": step["z_next"][batch_i:batch_i + 1],
                "_dt": step["_dt"],
                **(
                    {"vlm_context": step["vlm_context"][batch_i:batch_i + 1]}
                    if "vlm_context" in step else {}
                ),
            }
            for step in steps
        ]
        records.append((final_z[batch_i:batch_i + 1], member_steps, vlm_seq_len))
    return records


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


def correction_path_rollback_loss(
    correction_policy,
    reference_correction_policy,
    outcome_critic,
    proposal_waypoints,
    wm_context,
    vlm_context,
):
    """Safety-only rehearsal for the correction half of the joint policy.

    Risk-memory scenes pull the correction distribution (gate and residual
    control mean) back to its frozen initialization on the same frozen
    reference proposal. This closes the previous rehearsal asymmetry where
    only the waypoint Action Expert received a rollback gradient.
    """
    with torch.no_grad():
        risk = outcome_critic.outcomes(
            proposal_waypoints.flatten(-2), wm_context
        )["risk"]
        reference = reference_correction_policy(
            proposal_waypoints, risk, wm_context, vlm_context=vlm_context
        )
    current = correction_policy(
        proposal_waypoints, risk, wm_context, vlm_context=vlm_context
    )
    residual = torch.nn.functional.smooth_l1_loss(
        current["residual_latent"], reference["residual_latent"]
    )
    gate = torch.nn.functional.smooth_l1_loss(
        current["gate_logits"], reference["gate_logits"]
    )
    controls = torch.nn.functional.smooth_l1_loss(
        current["corrected_controls"], reference["corrected_controls"]
    )
    return residual + gate + 0.1 * controls


def recompute_log_prob(policy: Emu3Pi0, heads: ActionHeads, input_ids, attention_mask, pre_action, cmd,
                        steps, sigma, vlm_seq_len, return_means=False):
    """Re-forward through the stored (z_t -> z_{t+dt}) transitions using the
    given heads (policy = grad-enabled, reference = no_grad by caller) to get
    log-prob of the already-sampled trajectory under those params."""
    batch_size = input_ids.shape[0]
    vlm_h = policy.vlm.model.embed_tokens(input_ids)
    vlm_position_ids = torch.arange(vlm_seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
    state_input = torch.cat([pre_action.reshape(batch_size, -1), cmd], dim=1)
    state_token_embedding = heads.state_projector(state_input).unsqueeze(1)

    means = [] if return_means else None
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
        if return_means:
            means.append(mean.float())
        var = sigma ** 2
        diff = (z_next.float() - mean.float())
        per_dim = -0.5 * (diff ** 2 / var + torch.log(torch.tensor(2 * np.pi * var, device=diff.device)))
        log_prob = log_prob + per_dim.sum(dim=(-1, -2))
    return (log_prob, torch.stack(means, dim=1)) if return_means else log_prob


SUB_SCORE_KEYS = [
    "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
    "time_to_collision_within_bound", "comfort", "driving_direction_compliance",
]


def execution_trace_args(scene_token, n_rows, trace_meta):
    if not EXECUTION_TRACE_DIR or trace_meta is None:
        return []
    if len(trace_meta["row_kinds"]) != n_rows:
        raise ValueError(f"row_kinds has {len(trace_meta['row_kinds'])} entries for {n_rows} rows")
    meta = dict(trace_meta, seed=SEED, policy_checkpoint=str(EMU_HUB),
                control_adapter_checkpoint=CONTROL_ADAPTER_CHECKPOINT,
                unified_correction_checkpoint=UNIFIED_CORRECTION_CHECKPOINT)
    path = Path(EXECUTION_TRACE_DIR) / f"{scene_token}_step{int(trace_meta['step']):06d}.npz"
    args = ["--trace-npz", str(path), "--trace-meta-json", json.dumps(meta)]
    return args + (["--verify-trace-scores"] if EXECUTION_TRACE_VERIFY else [])


def score_with_navsim_simulator(scene_token: str, abs_trajectories: np.ndarray, trace_meta=None) -> dict:
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
    progress. Scoring errors abort training.

    trace_meta: when RL_EXECUTION_TRACE_DIR is set, a JSON-able dict (must
    contain "step" and "row_kinds") stored with the execution trace written
    to RL_EXECUTION_TRACE_DIR/<token>_step<step>.npz.
    """
    import subprocess
    import tempfile

    assert NAVSIM_ENV_PYTHON, "NAVSIM_ENV_PYTHON must be set when RL_REWARD_SOURCE=navsim"
    assert NAVSIM_METRIC_CACHE_PATH, "NAVSIM_METRIC_CACHE_PATH must be set when RL_REWARD_SOURCE=navsim"

    global NAVSIM_SCORE_CALLS, NAVSIM_SCORED_TRAJECTORIES
    G = abs_trajectories.shape[0]
    NAVSIM_SCORE_CALLS += 1
    NAVSIM_SCORED_TRAJECTORIES += G
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
            ] + execution_trace_args(scene_token, G, trace_meta),
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"NAVSIM scoring process failed for {scene_token}: {result.stderr[-2000:]}")

        with open(out_path) as f:
            payload = json.load(f)
        if "error" in payload:
            raise RuntimeError(f"NAVSIM scoring failed for {scene_token}: {payload['error']}")
        rows = payload["results"]
        if len(rows) != G or any("error" in row for row in rows):
            raise RuntimeError(f"Invalid NAVSIM scoring batch for {scene_token}: {rows}")
        scores = {k: np.array([r[k] for r in rows], dtype=np.float32) for k in SUB_SCORE_KEYS}
        if not all(np.isfinite(v).all() for v in scores.values()):
            raise RuntimeError(f"Nonfinite NAVSIM scores for {scene_token}")
        return scores


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
        adv[is_safe] = shrinkage_centered_advantage(
            safe_progress, noise_floor=ADVANTAGE_NOISE_FLOOR
        )
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
        adv = -shrinkage_centered_advantage(
            violation, noise_floor=ADVANTAGE_NOISE_FLOOR
        )

    return adv, is_safe


def compute_reference_anchored_advantage(
    sub_scores: dict, reference_scores: dict, constraint_weights=None
):
    """Use the tested hard-sign gate with the configured observation-noise floor."""
    if ADVANTAGE_VERSION == "legacy_centered":
        return legacy_reference_anchored_advantage(
            sub_scores, reference_scores, constraint_weights,
            noise_floor=ADVANTAGE_NOISE_FLOOR,
        )
    return reference_anchored_advantage(
        sub_scores, reference_scores, constraint_weights,
        noise_floor=ADVANTAGE_NOISE_FLOOR,
    )


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

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
    if FREEZE_ACTION_EXPERT:
        # freeze_vlm() already froze policy.vlm; this additionally freezes
        # the Action Expert modules it left trainable (action_expert/
        # tau_emb/action_projector/state_projector/action_decoder), so the
        # Action Expert stays fixed at its warm-start weights for the whole
        # run and only the correction policy (added to trainable_params
        # below once loaded) receives gradient updates.
        for parameter in trainable_params:
            parameter.requires_grad_(False)
        trainable_params = []
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

    control_adapter = None
    if CONTROL_ADAPTER_CHECKPOINT:
        checkpoint_path = Path(CONTROL_ADAPTER_CHECKPOINT)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        try:
            adapter_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError as error:
            if "weights_only" not in str(error):
                raise
            adapter_payload = torch.load(checkpoint_path, map_location="cpu")
        control_adapter = WaypointControlAdapter(
            dim=int(adapter_payload.get("dim", 128)),
            layers=int(adapter_payload.get("layers", 2)),
            heads=int(adapter_payload.get("heads", 4)),
        ).to(DEVICE)
        control_adapter.load_state_dict(adapter_payload["model"])
        control_adapter.eval()
        for parameter in control_adapter.parameters():
            parameter.requires_grad_(False)
        print(
            f"Loaded frozen control adapter={checkpoint_path} mode={CONTROL_ADAPTER_MODE} "
            f"max_position_rmse={CONTROL_ADAPTER_MAX_POSITION_RMSE}", flush=True,
        )

    unified_policy = None
    unified_outcome = None
    unified_reference = None
    if UNIFIED_CORRECTION_CHECKPOINT:
        def load_payload(path):
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except TypeError as error:
                if "weights_only" not in str(error):
                    raise
                return torch.load(path, map_location="cpu")

        correction_payload = load_payload(Path(UNIFIED_CORRECTION_CHECKPOINT))
        outcome_payload = load_payload(Path(UNIFIED_OUTCOME_CHECKPOINT))
        unified_policy = UnifiedCorrectionPolicy(
            dim=int(correction_payload.get("dim", 192)),
            layers=int(correction_payload.get("layers", 3)),
            heads=int(correction_payload.get("heads", 6)),
            wm_context_dim=int(correction_payload.get("wm_context_dim", 4096)),
            vlm_context_dim=int(correction_payload.get("vlm_context_dim", 4096)),
            max_trace_length=int(correction_payload.get("max_trace_length", 2)),
            projection_iterations=int(correction_payload.get("projection_iterations", 30)),
        ).to(DEVICE)
        unified_policy.load_state_dict(correction_payload["model"])
        unified_reference = copy.deepcopy(unified_policy).to(DEVICE).eval()
        for parameter in unified_reference.parameters():
            parameter.requires_grad_(False)
        unified_outcome = ContinuousSceneOutcomeCritic(
            latent_dim=int(outcome_payload.get("latent_dim", 4096)),
            dim=int(outcome_payload.get("dim", 256)),
        ).to(DEVICE)
        unified_outcome.load_state_dict(outcome_payload["model"])
        unified_outcome.eval()
        for parameter in unified_outcome.parameters():
            parameter.requires_grad_(False)
        # This is the same executed policy as the Action Expert: its log-prob
        # is added to the flow log-prob and receives the identical GRPO
        # advantage, rather than being trained as a post-hoc adapter.
        # Eval mode disables Transformer dropout so the stored sampled action
        # has exactly the same distribution during log-prob recomputation;
        # gradients remain enabled for all correction parameters.
        unified_policy.eval()
        trainable_params.extend(unified_policy.parameters())
        print(
            f"Loaded joint correction policy={UNIFIED_CORRECTION_CHECKPOINT} "
            f"outcome={UNIFIED_OUTCOME_CHECKPOINT} logp_coef={UNIFIED_CORRECTION_LOGP_COEF}",
            flush=True,
        )

    def unified_scene_context(scene_token):
        if unified_outcome is None:
            return None
        path = UNIFIED_LATENT_ROOT / stable_scene_split(scene_token) / f"{scene_token}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing unified WM latent for scene {scene_token}: {path}"
            )
        with np.load(path, allow_pickle=False) as data:
            latent = torch.from_numpy(data["eoa_latents"].astype(np.float32)).to(DEVICE)
        with torch.no_grad():
            return unified_outcome.scene_context(latent.unsqueeze(0))

    def save_unified_components(directory):
        if unified_policy is None:
            return
        directory = Path(directory)
        torch.save(
            {"model": {k: v.detach().cpu() for k, v in unified_policy.state_dict().items()},
             **unified_policy.metadata()},
            directory / "unified_correction_policy.pt",
        )
        # Store the fixed environment beside the policy so this directory is
        # a self-contained inference checkpoint.
        torch.save(
            {
                "model": {k: v.detach().cpu() for k, v in unified_outcome.state_dict().items()},
                "latent_dim": unified_outcome.latent_dim,
                "dim": unified_outcome.dim,
            },
            directory / "unified_outcome_critic.pt",
        )

    def execute_control_projection(relative_batch):
        """Deterministic safety wrapper used for rewards and deployment probes."""
        relative_batch = np.asarray(relative_batch, dtype=np.float32)
        absolute_batch = np.stack([relative_to_absolute_se2(x) for x in relative_batch])
        if control_adapter is None:
            return relative_batch, absolute_batch, 0, 0
        original_feasible = analyze_pose_trajectories(absolute_batch, dt=0.5)["dynamics_feasible"]
        attempted_mask = np.ones(len(relative_batch), dtype=bool) if CONTROL_ADAPTER_MODE == "all" else ~original_feasible
        indices = np.flatnonzero(attempted_mask)
        if not len(indices):
            return relative_batch, absolute_batch, 0, 0
        config = VehicleModelConfig()
        initial, _, _ = estimate_pseudo_controls(absolute_batch[indices], config.action_dt, config)
        with torch.inference_mode():
            projected = control_adapter(
                torch.from_numpy(relative_batch[indices]).to(DEVICE),
                torch.from_numpy(initial).to(DEVICE),
            )
        projected_rel = projected["relative_trajectory"].cpu().numpy()
        projected_abs = projected["absolute_trajectory"].cpu().numpy()
        projected_feasible = analyze_pose_trajectories(projected_abs, dt=0.5)["dynamics_feasible"]
        delta = np.linalg.norm(projected_abs[..., :2] - absolute_batch[indices, ..., :2], axis=-1)
        position_rmse = np.sqrt(np.mean(delta**2, axis=1))
        accepted = projected_feasible & (position_rmse <= CONTROL_ADAPTER_MAX_POSITION_RMSE)
        accepted_indices = indices[accepted]
        relative_batch = relative_batch.copy(); absolute_batch = absolute_batch.copy()
        relative_batch[accepted_indices] = projected_rel[accepted]
        absolute_batch[accepted_indices] = projected_abs[accepted]
        return relative_batch, absolute_batch, int(len(indices)), int(accepted.sum())

    def save_execution_metadata(directory):
        """Keep the deploy-time execution contract beside every policy save."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "reference_anchored_execution.json", "w") as file:
            json.dump(
                {
                    "schema_version": 1,
                    "control_adapter_checkpoint": CONTROL_ADAPTER_CHECKPOINT,
                    "control_adapter_mode": CONTROL_ADAPTER_MODE,
                    "control_adapter_max_position_rmse": CONTROL_ADAPTER_MAX_POSITION_RMSE,
                    "unified_correction": bool(unified_policy is not None),
                    "unified_correction_logp_coef": UNIFIED_CORRECTION_LOGP_COEF,
                    "reward_source": REWARD_SOURCE,
                    "advantage_mode": ADVANTAGE_MODE,
                    "risk_replay_mode": RISK_REPLAY_MODE,
                    "fixed_schedule": RISK_FIXED_SCHEDULE,
                    "reference_advantage_semantics": ("hard_sign_bounded_v1" if ADVANTAGE_VERSION == "current" else "legacy_centered_zscore_v0"),
                    "advantage_version": ADVANTAGE_VERSION,
                    "kl_semantics": ("conditional_gaussian_mean_per_coordinate" if KL_MODE == "transition" else "normalized_mean_log_ratio_surrogate"),
                    "positive_safety_guard": POSITIVE_SAFETY_GUARD,
                    "advantage_noise_floor": ADVANTAGE_NOISE_FLOOR,
                    "adaptive_kl": ADAPTIVE_KL,
                    "target_kl": TARGET_KL,
                    "kl_hard_limit": KL_HARD_LIMIT,
                },
                file,
                indent=2,
            )

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
        f"SEED={SEED} REWARD_SOURCE={REWARD_SOURCE} ADVANTAGE_MODE={ADVANTAGE_MODE} ADVANTAGE_VERSION={ADVANTAGE_VERSION} "
        f"ROLLBACK_COEF={ROLLBACK_COEF} DEPLOYMENT_PROBES={DEPLOYMENT_PROBES} "
        f"KL_BOOST={KL_BOOST} KL_DECAY={KL_DECAY} "
        f"KL_SEMANTICS=normalized_mean_log_ratio_surrogate "
        f"RISK_BUFFER_SIZE={RISK_BUFFER_SIZE} RISK_REPLAY_EVERY={RISK_REPLAY_EVERY} "
        f"RISK_REPLAY_COEF={RISK_REPLAY_COEF} "
        f"RISK_EXPOSURE_THRESHOLD={RISK_EXPOSURE_THRESHOLD} "
        f"RISK_REPLAY_SAMPLING={RISK_REPLAY_SAMPLING} "
        f"RISK_MANIFEST={RISK_MANIFEST!r} RISK_COVERAGE_MANIFEST={RISK_COVERAGE_MANIFEST!r} "
        f"RISK_ONLINE_INSERT={int(RISK_ONLINE_INSERT)} RISK_CURRICULUM={int(RISK_CURRICULUM)} "
        f"RISK_SOURCE_MIX={RISK_SOURCE_MIX_SPEC!r} RISK_TYPE_BALANCE={int(RISK_TYPE_BALANCE)} "
        f"RISK_REDUNDANCY_COEF={RISK_REDUNDANCY_COEF} RISK_PRIORITY_POWER={RISK_PRIORITY_POWER} "
        f"DYNAMIC_CONSTRAINT_WEIGHTS={int(DYNAMIC_CONSTRAINT_WEIGHTS)} "
        f"DYNAMIC_WEIGHT_BOUNDS=({DYNAMIC_WEIGHT_MIN},{DYNAMIC_WEIGHT_MAX}) "
        f"ADVANTAGE_NOISE_FLOOR={ADVANTAGE_NOISE_FLOOR} "
        f"KL_MODE={KL_MODE} POSITIVE_SAFETY_GUARD={int(POSITIVE_SAFETY_GUARD)} ADAPTIVE_KL={int(ADAPTIVE_KL)} TARGET_KL={TARGET_KL} "
        f"KL_COEF_BOUNDS=({KL_COEF_MIN},{KL_COEF_MAX}) KL_HARD_LIMIT={KL_HARD_LIMIT} "
        f"ROLLOUT_MICRO_BATCH={ROLLOUT_MICRO_BATCH} "
        f"DIAGNOSTICS={int(DIAGNOSTICS)} "
        f"CONTROL_ADAPTER={CONTROL_ADAPTER_CHECKPOINT!r} CONTROL_MODE={CONTROL_ADAPTER_MODE} "
        f"UNIFIED_CORRECTION={UNIFIED_CORRECTION_CHECKPOINT!r} "
        f"UNIFIED_OUTCOME={UNIFIED_OUTCOME_CHECKPOINT!r} "
        f"UNIFIED_LOGP_COEF={UNIFIED_CORRECTION_LOGP_COEF} "
        f"COLLECT_ONLY={int(COLLECT_ONLY)} COLLECTION_DIR={COLLECTION_DIR!r}", flush=True,
    )
    if COLLECT_ONLY and not COLLECTION_DIR:
        raise ValueError("RL_COLLECT_ONLY=1 requires RL_COLLECTION_DIR")
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
    # token_pool is a set in NAVSIM mode. Sorting is required: a seeded
    # random.choice over an unsorted set-derived list maps the same RNG draw
    # to different scenes across Python processes because of hash randomization.
    usable_indices = [token_to_idx[tok] for tok in sorted(token_pool) if tok in token_to_idx]
    collection_indices = [token_to_idx[tok] for tok in sorted(token_pool) if tok in token_to_idx]
    collection_indices = collection_indices[COLLECTION_SHARD_ID::COLLECTION_NUM_SHARDS]
    print(f"{len(usable_indices)} scenes usable for RL rollouts", flush=True)
    scene_sample_weights = None
    if RISK_WEIGHTED_SAMPLING:
        print(f"Loading risk-aware scene weights from {RISK_SCENE_DATA_ROOT} ...", flush=True)
        scene_risk_weights = load_scene_risk_weights(RISK_SCENE_DATA_ROOT, RISK_SAMPLING_EPSILON)
        scene_sample_weights = [
            scene_risk_weights.get(dataset.data[idx]["token"], RISK_SAMPLING_EPSILON)
            for idx in usable_indices
        ]
        covered = sum(
            1 for idx in usable_indices if dataset.data[idx]["token"] in scene_risk_weights
        )
        weights_array = np.asarray(scene_sample_weights)
        print(
            f"risk-aware sampling enabled: {covered}/{len(usable_indices)} scenes have "
            f"verified risk weights, mean_weight={weights_array.mean():.4f} "
            f"max_weight={weights_array.max():.4f} epsilon={RISK_SAMPLING_EPSILON}",
            flush=True,
        )
    if not usable_indices:
        raise RuntimeError("No usable training scenes")

    action_frames, action_dim = 8, 3
    dt = -1.0 / NUM_INFERENCE_STEPS
    sigma = NOISE_STD * (abs(dt) ** 0.5)

    reward_history = []
    control_attempted_total = 0
    control_accepted_total = 0
    # Reference heads never change. With stable per-scene probe latents, their
    # NAVSIM vector scores are immutable and can be cached after first use.
    reference_score_cache = {}
    adaptive_kl_multiplier = 1.0
    controlled_kl_coef = KL_COEF
    previous_observed_kl = 0.0
    risk_event_count = 0
    # V3 safety rehearsal memory. Dataset indices keep the buffer bounded and
    # avoid retaining GPU tensors. Priority comes from the whole stochastic
    # candidate group, not only V2's sparse deterministic-probe failures.
    risk_buffer = {}
    replay_update_count = 0
    policy_optimizer_steps = 0
    manifest_replay_count = 0
    curriculum_source_counts = {"manifest": 0, "coverage": 0, "online": 0, "fallback": 0}
    # Keep rehearsal sampling from perturbing the seeded training-scene
    # sequence. Otherwise V3 and its ablations would see different scenes
    # solely because one method consumes extra values from Python's RNG.
    scene_rng = random.Random(SEED)
    replay_rng = random.Random(SEED + 2027)

    def upsert_risk_memory(
        token, priority, probe_idx, last_seen, source, label_counts=None
    ):
        if RISK_BUFFER_SIZE <= 0:
            return
        previous = risk_buffer.get(token)
        sources = set(previous.get("sources", ())) if previous else set()
        sources.add(source)
        merged_counts = dict(previous.get("label_counts", {})) if previous else {}
        for risk, count in (label_counts or {}).items():
            merged_counts[risk] = max(int(count), int(merged_counts.get(risk, 0)))
        risk_buffer[token] = {
            "idx": token_to_idx[token],
            "priority": max(float(priority), previous["priority"] if previous else 0.0),
            "probe_idx": int(probe_idx),
            "last_seen": int(last_seen),
            "source": "+".join(sorted(sources)),
            "sources": sources,
            "label_counts": merged_counts,
            "replay_count": int(previous.get("replay_count", 0)) if previous else 0,
        }
        if len(risk_buffer) > RISK_BUFFER_SIZE:
            evict_token = min(
                risk_buffer,
                key=lambda tok: (
                    risk_buffer[tok]["priority"],
                    risk_buffer[tok]["last_seen"],
                    tok,
                ),
            )
            del risk_buffer[evict_token]

    manifest_selector = "none"
    manifest_verified_events = 0
    manifest_budget = 0
    coverage_selector = "none"
    coverage_verified_events = 0
    scene_constraint_weight_map = {}

    def load_risk_manifest(path_string, source):
        nonlocal manifest_selector, manifest_budget, manifest_verified_events
        nonlocal coverage_selector, coverage_verified_events
        if not path_string:
            return
        manifest_path = Path(path_string)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"risk manifest not found: {manifest_path}")
        with open(manifest_path) as file:
            manifest = json.load(file)
        if manifest.get("schema_version") != 1:
            raise ValueError(f"unsupported risk manifest schema: {manifest.get('schema_version')}")
        selector = str(manifest.get("selector", "unknown"))
        budget = int(manifest.get("verification_budget_per_scene", 0))
        if source == "manifest":
            manifest_selector = selector
            manifest_budget = budget
        else:
            coverage_selector = selector
        skipped_unverified = skipped_unusable = 0
        loaded_events = 0
        for entry in manifest.get("entries", []):
            token = str(entry["scene_token"])
            if "dynamic_constraint_weights" in entry:
                weights = validate_constraint_weights(
                    entry["dynamic_constraint_weights"],
                    minimum=DYNAMIC_WEIGHT_MIN,
                    maximum=DYNAMIC_WEIGHT_MAX,
                )
                previous_weights = scene_constraint_weight_map.get(token)
                if previous_weights is not None and previous_weights != weights:
                    raise ValueError(f"conflicting dynamic weights for scene {token}")
                scene_constraint_weight_map[token] = weights
            verified_count = len(entry.get("verified_unsafe_indices", []))
            if verified_count == 0:
                skipped_unverified += 1
                continue
            if token not in token_to_idx or token not in token_pool:
                skipped_unusable += 1
                continue
            priority = float(entry.get("risk_priority", 0.0))
            if priority <= 0.0:
                raise ValueError(f"verified manifest scene {token} has non-positive priority")
            upsert_risk_memory(
                token,
                priority,
                probe_idx=0,
                last_seen=-1,
                source=source,
                label_counts=entry.get("verified_label_counts", {}),
            )
            loaded_events += verified_count
        if source == "manifest":
            manifest_verified_events += loaded_events
        else:
            coverage_verified_events += loaded_events
        print(
            f"Loaded risk manifest source={source} selector={selector} budget={budget} "
            f"buffered={len(risk_buffer)} verified_events={loaded_events} "
            f"skipped_unverified={skipped_unverified} skipped_unusable={skipped_unusable}",
            flush=True,
        )

    load_risk_manifest(RISK_MANIFEST, "manifest")
    load_risk_manifest(RISK_COVERAGE_MANIFEST, "coverage")

    if COLLECT_ONLY:
        collection_root = Path(COLLECTION_DIR)
        for split in ("train", "val", "test"):
            (collection_root / split).mkdir(parents=True, exist_ok=True)
        print(
            f"Collection mode: {len(collection_indices)} deterministic scenes, "
            f"{GROUP_SIZE} candidates/scene, resume enabled at {collection_root}",
            flush=True,
        )

    fixed_schedule = None
    if RISK_REPLAY_MODE != "reference":
        if RISK_REPLAY_MODE not in ("matched_reference", "rescored_rl", "rescored_rl_reference"):
            raise ValueError(f"Unknown risk replay mode: {RISK_REPLAY_MODE}")
        if (COLLECT_ONLY or RISK_ONLINE_INSERT or RISK_CURRICULUM or ADAPTIVE_KL
                or unified_policy is not None or control_adapter is not None):
            raise ValueError("Fixed replay experiment requires collection/online insertion/curriculum/adaptive KL/adapters off")
        with open(RISK_FIXED_SCHEDULE) as file:
            fixed_schedule = json.load(file)
        if len(fixed_schedule) < NUM_STEPS:
            raise ValueError("Schedule shorter than training")
        for item in fixed_schedule[:NUM_STEPS]:
            step_policy(RISK_REPLAY_MODE, item["kind"])
            if item["token"] not in token_pool or item["token"] not in token_to_idx:
                raise ValueError("Scheduled token outside training pool")
            if item["kind"] == "risk" and item["token"] not in risk_buffer:
                raise ValueError("Scheduled risk scene missing from fixed buffer")
    loop_steps = min(NUM_STEPS, len(collection_indices)) if COLLECT_ONLY else NUM_STEPS
    for step_i in range(loop_steps):
        idx = (
            collection_indices[step_i]
            if COLLECT_ONLY
            else (
                scene_rng.choices(usable_indices, weights=scene_sample_weights, k=1)[0]
                if scene_sample_weights is not None
                else scene_rng.choice(usable_indices)
            )
        )
        step_kind = "main"
        train_on_group = True
        scheduled_rehearsal = False
        if fixed_schedule is not None:
            item = fixed_schedule[step_i]
            idx = token_to_idx[item["token"]]
            step_kind = item["kind"]
            controls = step_policy(RISK_REPLAY_MODE, step_kind)
            train_on_group, scheduled_rehearsal = controls["train_on_group"], controls["rehearse"]
            # Reset all rollout RNG streams per slot to keep candidate-noise
            # budgets aligned across mechanisms despite different backwards.
            slot_seed = SEED * 1000003 + step_i + 12345
            random.seed(slot_seed); np.random.seed(slot_seed)
            torch.manual_seed(slot_seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(slot_seed)
        scene = dataset.data[idx]
        if COLLECT_ONLY:
            split = stable_scene_split(scene["token"])
            collection_path = collection_root / split / f"{scene['token']}.npz"
            if collection_path.exists():
                print(
                    f"collect {step_i:03d} scene={scene['token']} split={split} "
                    "status=exists",
                    flush=True,
                )
                continue
        sample = dataset[idx]
        input_ids_1 = sample["input_ids"].unsqueeze(0).to(DEVICE)
        attention_mask_1 = sample["attention_mask"].unsqueeze(0).to(DEVICE)
        pre_action_1 = sample["pre_action"].unsqueeze(0).to(DEVICE).to(DTYPE)
        cmd_1 = sample["cmd"].unsqueeze(0).to(DEVICE).to(DTYPE)

        # ---- phase 1: no-grad rollout micro-batches. Noise is generated in
        # the legacy candidate-major order before stacking, so changing the
        # micro-batch size does not change seeded candidate samples. The
        # phase-2 grad loop below intentionally remains batch-size one. ----
        if REWARD_SOURCE == "anchor":
            entry = anchor_scores[scene["token"]]
            traj_scores = entry["trajectory_scores"][0]
            stacked = {k: np.asarray(traj_scores[k]) for k in ANCHOR_METRIC_KEYS}

        rollouts = []
        rel_trajs = np.zeros((GROUP_SIZE, action_frames, 3), dtype=np.float32)
        abs_trajs = np.zeros((GROUP_SIZE, action_frames, 3), dtype=np.float32)
        noise_bundles = []
        for _ in range(GROUP_SIZE):
            initial = torch.randn(
                1, action_frames, action_dim, device=DEVICE, dtype=DTYPE
            )
            transitions = [torch.randn_like(initial) for _ in range(NUM_INFERENCE_STEPS)]
            noise_bundles.append((initial, transitions))
        for start in range(0, GROUP_SIZE, ROLLOUT_MICRO_BATCH):
            stop = min(start + ROLLOUT_MICRO_BATCH, GROUP_SIZE)
            batch_size = stop - start
            initial = torch.cat([noise_bundles[g][0] for g in range(start, stop)])
            transition_noise = [
                torch.cat([
                    noise_bundles[g][1][noise_i] for g in range(start, stop)
                ])
                for noise_i in range(NUM_INFERENCE_STEPS)
            ]
            z_final, steps, vlm_seq_len = rollout_and_score(
                policy, policy_heads,
                input_ids_1.expand(batch_size, -1),
                attention_mask_1.expand(batch_size, -1),
                pre_action_1.expand(batch_size, *pre_action_1.shape[1:]),
                cmd_1.expand(batch_size, -1),
                action_frames, action_dim, NUM_INFERENCE_STEPS, NOISE_STD,
                initial_z=initial, transition_noise=transition_noise,
            )
            for offset, (member_final, member_steps, member_seq_len) in enumerate(
                split_batched_rollout(z_final, steps, vlm_seq_len)
            ):
                g = start + offset
                rollouts.append((member_steps, member_seq_len))
                rel_raw = 0.5 * (
                    member_final[0].float().cpu().numpy() + 1.0
                ) * (high - low) + low
                rel_trajs[g] = rel_raw
                abs_trajs[g] = relative_to_absolute_se2(rel_raw)

        # A single joint policy now owns both distributions. The Action Expert
        # proposes waypoints; its trainable correction branch samples bounded
        # residual controls conditioned on the frozen WM outcome context and
        # analytic physics. NAVSIM scores only the resulting executed paths.
        unified_records = None
        if unified_policy is not None:
            proposal_rel_trajs = rel_trajs.copy()
            scene_context = unified_scene_context(scene["token"])
            expanded_context = scene_context.expand(GROUP_SIZE, -1)
            proposal_tensor = torch.from_numpy(proposal_rel_trajs).to(DEVICE)
            vlm_context = torch.cat([
                rollouts[g][0][-1]["vlm_context"] for g in range(GROUP_SIZE)
            ]).float()
            residual_noise = torch.randn(GROUP_SIZE, action_frames, 2, device=DEVICE)
            gate_noise = torch.randn(GROUP_SIZE, device=DEVICE)
            # tau^base / tau^final via the shared verified-correction stack
            # (docs/reference_anchored_correction_rl.tex Step 5 + the
            # verification paragraph) -- the same function is used for the
            # frozen reference-anchored probes below (deterministic=True),
            # so the current and reference sides of every comparison run
            # through identical projection/gate/verification logic and
            # cannot silently drift apart from each other. abs_trajs is
            # regenerated from rel_trajs by execute_control_projection
            # immediately below, so it is not recomputed here.
            applied = apply_verified_correction(
                unified_policy, unified_outcome, proposal_tensor,
                expanded_context, vlm_context, deterministic=False,
                residual_noise=residual_noise, gate_noise=gate_noise,
            )
            corrected = applied["output"]
            rel_trajs = applied["final_relative"].cpu().numpy()
            unified_records = {
                "proposal": proposal_tensor.detach(),
                "risk": corrected["risk_condition"].detach(),
                "context": expanded_context.detach(),
                "vlm_context": vlm_context.detach(),
                "residual_noise": residual_noise.detach(),
                "gate_noise": gate_noise.detach(),
                "physics_before": corrected["reference_physics_violation"].cpu().numpy(),
                "physics_after": corrected["corrected_physics_violation"].cpu().numpy(),
                "gate": corrected["gate_probability"].cpu().numpy(),
                "residual_abs": corrected["residual_controls"].abs().mean(
                    dim=(-2, -1)
                ).cpu().numpy(),
            }

        rel_trajs, abs_trajs, control_attempted, control_accepted = execute_control_projection(
            rel_trajs
        )

        if REWARD_SOURCE == "navsim":
            if ADVANTAGE_MODE == "reference_anchored":
                token = scene["token"]
                current_probe_finals, current_probe_steps = [], []
                probe_vlm_seq_lens = []
                reference_finals, missing_reference_indices = [], []
                cache_hit = token in reference_score_cache
                deployment_latents = []
                current_probe_noise = []
                reference_probe_noise = []
                for probe_idx in range(DEPLOYMENT_PROBES):
                    deployment_z = fixed_scene_latent(token, probe_idx, action_frames, action_dim)
                    deployment_latents.append(deployment_z)
                    # Preserve the old global RNG consumption even though
                    # sigma=0 makes these tensors irrelevant to the ODE path.
                    current_probe_noise.append([
                        torch.randn_like(deployment_z)
                        for _ in range(NUM_INFERENCE_STEPS)
                    ])
                    if not cache_hit:
                        reference_probe_noise.append([
                            torch.randn_like(deployment_z)
                            for _ in range(NUM_INFERENCE_STEPS)
                        ])
                        missing_reference_indices.append(probe_idx)

                probe_batch = DEPLOYMENT_PROBES
                probe_initial = torch.cat(deployment_latents)
                probe_inputs = (
                    input_ids_1.expand(probe_batch, -1),
                    attention_mask_1.expand(probe_batch, -1),
                    pre_action_1.expand(probe_batch, *pre_action_1.shape[1:]),
                    cmd_1.expand(probe_batch, -1),
                )
                current_final, det_steps, det_seq_len = rollout_and_score(
                    policy, policy_heads, *probe_inputs,
                    action_frames, action_dim, NUM_INFERENCE_STEPS, 0.0,
                    initial_z=probe_initial,
                    transition_noise=[
                        torch.cat([current_probe_noise[p][i] for p in range(probe_batch)])
                        for i in range(NUM_INFERENCE_STEPS)
                    ],
                )
                for final, steps, seq_len in split_batched_rollout(
                    current_final, det_steps, det_seq_len
                ):
                    current_probe_finals.append(final)
                    current_probe_steps.append(steps)
                    probe_vlm_seq_lens.append(seq_len)
                if not cache_hit:
                    reference_final, reference_steps, reference_seq_len = rollout_and_score(
                        policy, ref_heads, *probe_inputs,
                        action_frames, action_dim, NUM_INFERENCE_STEPS, 0.0,
                        initial_z=probe_initial,
                        transition_noise=[
                            torch.cat([reference_probe_noise[p][i] for p in range(probe_batch)])
                            for i in range(NUM_INFERENCE_STEPS)
                        ],
                    )
                    reference_finals = [
                        record[0] for record in split_batched_rollout(
                            reference_final, reference_steps, reference_seq_len
                        )
                    ]

                extra_finals = current_probe_finals + reference_finals
                extra_abs = np.zeros((len(extra_finals), action_frames, action_dim), dtype=np.float32)
                for extra_i, z_final in enumerate(extra_finals):
                    rel_raw = 0.5 * (z_final[0].float().cpu().numpy() + 1.0) * (high - low) + low
                    extra_abs[extra_i] = relative_to_absolute_se2(rel_raw)
                extra_relative = np.stack([
                    0.5 * (z_final[0].float().cpu().numpy() + 1.0) * (high - low) + low
                    for z_final in extra_finals
                ])
                deployment_proposal_relative = None
                if unified_policy is not None:
                    current_relative = torch.from_numpy(
                        extra_relative[:DEPLOYMENT_PROBES]
                    ).to(DEVICE)
                    deployment_proposal_relative = current_relative
                    current_context = scene_context.expand(DEPLOYMENT_PROBES, -1)
                    # Deterministic: probes measure a fixed deployment
                    # decision (mean residual, thresholded gate), matching
                    # apply_unified_correction_policy.py's inference
                    # behavior -- not a fresh stochastic sample every time
                    # this scene happens to be probed.
                    applied_current = apply_verified_correction(
                        unified_policy, unified_outcome, current_relative,
                        current_context, det_steps[-1]["vlm_context"].float(),
                        deterministic=True,
                    )
                    extra_relative[:DEPLOYMENT_PROBES] = (
                        applied_current["final_relative"].cpu().numpy()
                    )
                    if not cache_hit:
                        # Reference-anchored comparison requires both sides
                        # to run through the *same* deployment pipeline
                        # (docs/reference_anchored_correction_rl.tex): the
                        # frozen policy's own proposal is independently
                        # projected and corrected by the frozen
                        # unified_reference copy, never by unified_policy
                        # and never conditioned on the current-side
                        # trajectory/context computed above. scene_context is
                        # shared because it depends only on the scene token,
                        # not on which policy produced a candidate. This
                        # branch only runs on a cache miss: once computed,
                        # the frozen deployment stack is deterministic, so
                        # its NAVSIM score is cached below exactly like the
                        # old raw-waypoint reference score was.
                        reference_relative_tensor = torch.from_numpy(
                            extra_relative[DEPLOYMENT_PROBES:DEPLOYMENT_PROBES * 2]
                        ).to(DEVICE)
                        applied_reference = apply_verified_correction(
                            unified_reference, unified_outcome, reference_relative_tensor,
                            current_context, reference_steps[-1]["vlm_context"].float(),
                            deterministic=True,
                        )
                        extra_relative[DEPLOYMENT_PROBES:DEPLOYMENT_PROBES * 2] = (
                            applied_reference["final_relative"].cpu().numpy()
                        )
                _, extra_abs, probe_control_attempted, probe_control_accepted = execute_control_projection(
                    extra_relative
                )
                control_attempted += probe_control_attempted
                control_accepted += probe_control_accepted
                all_scores = score_with_navsim_simulator(
                    token, np.concatenate([abs_trajs, extra_abs], axis=0),
                    trace_meta=dict(
                        step=step_i, step_kind=step_kind, collect_only=COLLECT_ONLY,
                        row_kinds=(["candidate"] * GROUP_SIZE
                                   + ["current_probe"] * len(current_probe_finals)
                                   + ["reference_probe"] * len(reference_finals)),
                    ),
                )
                sub_scores = {k: v[:GROUP_SIZE] for k, v in all_scores.items()}
                current_probe_scores = [
                    {k: float(v[GROUP_SIZE + probe_idx]) for k, v in all_scores.items()}
                    for probe_idx in range(DEPLOYMENT_PROBES)
                ]
                if not cache_hit:
                    # When unified_policy is active, this caches NAVSIM
                    # scores for the frozen policy's *full deployed stack*
                    # (projection + frozen correction + verification), not
                    # the raw frozen-AE waypoint -- the two are different
                    # quantities. reference_score_cache is in-memory and
                    # freshly empty at the start of every run, so there is
                    # no on-disk cache from an older code version to worry
                    # about invalidating.
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
                sub_scores = score_with_navsim_simulator(
                    scene["token"], abs_trajs,
                    trace_meta=dict(step=step_i, step_kind=step_kind, collect_only=COLLECT_ONLY,
                                    row_kinds=["candidate"] * GROUP_SIZE),
                )
        else:
            sub_scores = {k: np.zeros(GROUP_SIZE, dtype=np.float32) for k in ANCHOR_METRIC_KEYS}
            for g in range(GROUP_SIZE):
                dists = np.linalg.norm(cluster_flat - abs_trajs[g].reshape(-1)[None, :], axis=1)
                nearest = np.argmin(dists)
                for k in ANCHOR_METRIC_KEYS:
                    sub_scores[k][g] = stacked[k][nearest]

        control_attempted_total += control_attempted
        control_accepted_total += control_accepted

        if COLLECT_ONLY:
            if ADVANTAGE_MODE != "reference_anchored":
                raise ValueError("world-action collection requires reference_anchored mode")
            _, collect_feasible, collect_violation = compute_reference_anchored_advantage(
                sub_scores, reference_scores
            )
            payload = {
                "schema_version": np.asarray(1, dtype=np.int32),
                "scene_token": np.asarray(scene["token"]),
                "split": np.asarray(split),
                "relative_trajectories": rel_trajs.astype(np.float32),
                "absolute_trajectories": abs_trajs.astype(np.float32),
                "expert_action_normalized": np.asarray(
                    scene["action"][3:3 + action_frames], dtype=np.float32
                ),
                "reference_feasible": collect_feasible.astype(np.bool_),
                "reference_violation": collect_violation.astype(np.float32),
                "candidate_seed": np.asarray(0, dtype=np.int64),
            }
            for key, values in sub_scores.items():
                payload[f"score__{key}"] = np.asarray(values, dtype=np.float32)
            for key in SUB_SCORE_KEYS:
                payload[f"reference_probe__{key}"] = np.asarray(
                    [x[key] for x in reference_probe_scores], dtype=np.float32
                )
                payload[f"current_probe__{key}"] = np.asarray(
                    [x[key] for x in current_probe_scores], dtype=np.float32
                )
            # Unique sequential scenes mean reference_finals are present on
            # the first collection pass. Store their verified trajectories so
            # later latent-risk work can form same-scene policy/reference pairs.
            payload["current_probe_absolute_trajectories"] = extra_abs[
                :DEPLOYMENT_PROBES
            ].astype(np.float32)
            payload["reference_probe_absolute_trajectories"] = extra_abs[
                DEPLOYMENT_PROBES:
            ].astype(np.float32)

            tmp_path = collection_path.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                np.savez_compressed(f, **payload)
            os.replace(tmp_path, collection_path)
            print(
                f"collect {step_i:03d} scene={scene['token']} split={split} "
                f"candidates={GROUP_SIZE} unsafe_frac={float((~collect_feasible).mean()):.4f} "
                f"violation_mean={float(collect_violation.mean()):.4f} "
                f"path={collection_path}",
                flush=True,
            )
            continue

        scene_constraint_weights = {"nc": 1.0, "dac": 1.0, "ttc": 1.0}
        if DYNAMIC_CONSTRAINT_WEIGHTS:
            scene_constraint_weights = scene_constraint_weight_map.get(
                scene["token"], scene_constraint_weights
            )
        if ADVANTAGE_MODE == "reference_anchored":
            adv, is_safe, violation = compute_reference_anchored_advantage(
                sub_scores, reference_scores, scene_constraint_weights
            )
            probe_regrets = []
            for current_scores, ref_scores in zip(current_probe_scores, reference_probe_scores):
                nc_regret = max(0.0, ref_scores["no_at_fault_collisions"] - current_scores["no_at_fault_collisions"])
                dac_regret = max(0.0, ref_scores["drivable_area_compliance"] - current_scores["drivable_area_compliance"])
                ttc_regret = max(0.0, ref_scores["time_to_collision_within_bound"] - current_scores["time_to_collision_within_bound"])
                weighted_regret = (
                    scene_constraint_weights["nc"] * nc_regret
                    + scene_constraint_weights["dac"] * dac_regret
                    + scene_constraint_weights["ttc"] * ttc_regret
                )
                probe_regrets.append((weighted_regret, nc_regret, dac_regret, ttc_regret))
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
        if POSITIVE_SAFETY_GUARD and ADVANTAGE_MODE != "reference_anchored":
            raise ValueError("positive safety guard is scoped to reference_anchored mode")
        adv, positive_guard_stats = guard_positive_advantage(
            adv, sub_scores, enabled=POSITIVE_SAFETY_GUARD,
            ttc_threshold=SAFE_TTC_THRESHOLD,
        )
        # Random candidates expose substantially more safety drift than the
        # two deployment probes. Preserve scenes with either a deterministic
        # regression, high mean violation, or a large infeasible fraction.
        candidate_risk = float(np.maximum(violation, 0.0).mean())
        unsafe_fraction = float((~is_safe).mean())
        risk_priority = max(float(det_violation), candidate_risk, unsafe_fraction)
        if (
            ADVANTAGE_MODE == "reference_anchored"
            and RISK_ONLINE_INSERT
            and RISK_BUFFER_SIZE > 0
            and risk_priority >= RISK_EXPOSURE_THRESHOLD
        ):
            token = scene["token"]
            upsert_risk_memory(
                token,
                risk_priority,
                probe_idx=worst_probe_idx if det_violation > 0.0 else 0,
                last_seen=step_i,
                source="online",
            )
        effective_kl_coef = controlled_kl_coef * adaptive_kl_multiplier
        kl_guard_active = bool(
            KL_HARD_LIMIT > 0.0 and previous_observed_kl > KL_HARD_LIMIT
        )
        if ADVANTAGE_MODE == "reference_anchored" and det_violation > 0.0:
            rl_weight = 0.0
        elif kl_guard_active:
            # The observed reference KL is already outside the configured
            # region. Suppress the outward policy-gradient term for one
            # update while retaining the KL restoration gradient.
            rl_weight = 0.0
        else:
            rl_weight = 1.0 / adaptive_kl_multiplier

        # ---- phase 2: gradient-accumulate the per-sample REINFORCE (+ KL)
        # loss across the group, one sample at a time, then a single
        # optimizer step — equivalent to a batched group update but with
        # batch_size=1 peak memory. ----
        optimizer.zero_grad()
        pg_loss_sum, kl_sum = 0.0, 0.0
        for g in range(GROUP_SIZE if train_on_group else 0):
            steps, vlm_seq_len = rollouts[g]
            if KL_MODE == "transition":
                logp_policy, policy_means = recompute_log_prob(policy, policy_heads, input_ids_1, attention_mask_1, pre_action_1, cmd_1, steps, sigma, vlm_seq_len, return_means=True)
            else:
                logp_policy = recompute_log_prob(policy, policy_heads, input_ids_1, attention_mask_1, pre_action_1, cmd_1, steps, sigma, vlm_seq_len)
            joint_logp = logp_policy
            if unified_records is not None:
                correction = unified_policy(
                    unified_records["proposal"][g:g + 1],
                    unified_records["risk"][g:g + 1],
                    unified_records["context"][g:g + 1],
                    vlm_context=unified_records["vlm_context"][g:g + 1],
                    sample_correction=True,
                    correction_noise=(
                        unified_records["residual_noise"][g:g + 1],
                        unified_records["gate_noise"][g:g + 1],
                    ),
                )
                joint_logp = (
                    logp_policy
                    + UNIFIED_CORRECTION_LOGP_COEF
                    * correction["correction_log_prob"]
                )
            pg_loss_g = -rl_weight * (float(adv[g]) * joint_logp).mean()
            if ref_heads is not None:
                if KL_MODE == "transition":
                    from transition_kl import equal_variance_transition_kl
                    with torch.no_grad():
                        _, reference_means = recompute_log_prob(policy, ref_heads, input_ids_1, attention_mask_1, pre_action_1, cmd_1, steps, sigma, vlm_seq_len, return_means=True)
                    kl_g = equal_variance_transition_kl(policy_means, reference_means, sigma)
                else:
                    with torch.no_grad():
                        logp_ref = recompute_log_prob(policy, ref_heads, input_ids_1, attention_mask_1, pre_action_1, cmd_1, steps, sigma, vlm_seq_len)
                    # Nonnegative surrogate on the normalized mean log-ratio.
                    # The nonlinear transform follows averaging and clipping;
                    # this is NOT an exact policy KL or an unbiased KL estimator.
                    # Keep its scale unchanged in this correctness repair.
                    n_elements = len(steps) * action_frames * action_dim
                    log_ratio_mean = (logp_policy - logp_ref).mean() / n_elements
                    log_ratio_mean = log_ratio_mean.clamp(-20.0, 20.0)
                    ratio = torch.exp(log_ratio_mean)
                    kl_g = (ratio - 1) - log_ratio_mean
                if unified_records is not None:
                    with torch.no_grad():
                        reference_correction = unified_reference(
                            unified_records["proposal"][g:g + 1],
                            unified_records["risk"][g:g + 1],
                            unified_records["context"][g:g + 1],
                            vlm_context=unified_records["vlm_context"][g:g + 1],
                            sample_correction=True,
                            correction_noise=(
                                unified_records["residual_noise"][g:g + 1],
                                unified_records["gate_noise"][g:g + 1],
                            ),
                        )
                    correction_log_ratio = (
                        correction["correction_log_prob"]
                        - reference_correction["correction_log_prob"]
                    ) / (action_frames * 2 + 1)
                    correction_log_ratio = correction_log_ratio.clamp(-20.0, 20.0)
                    correction_ratio = correction_log_ratio.exp()
                    correction_kl = (
                        correction_ratio - 1.0 - correction_log_ratio
                    ).mean()
                    kl_g = kl_g + correction_kl
                loss_g = pg_loss_g + effective_kl_coef * kl_g
                kl_sum += kl_g.item()
            else:
                loss_g = pg_loss_g
            (loss_g / GROUP_SIZE).backward()
            pg_loss_sum += pg_loss_g.item()

        rollback_value = 0.0
        rollback_correction_value = 0.0
        if train_on_group and ADVANTAGE_MODE == "reference_anchored" and det_violation > 0.0:
            rollback_loss = reference_path_rollback_loss(
                policy, policy_heads, ref_heads, input_ids_1, attention_mask_1,
                pre_action_1, cmd_1, current_probe_steps[worst_probe_idx],
                probe_vlm_seq_lens[worst_probe_idx],
            )
            total_rollback_loss = rollback_loss
            # Close the same rehearsal asymmetry as correction_path_rollback_loss,
            # but for the immediate same-step rollback trigger rather than the
            # deferred risk-buffer replay: a deterministic-probe regression must
            # also pull the correction policy's gate/residual back toward its
            # frozen reference, not just the waypoint Action Expert.
            if unified_policy is not None and deployment_proposal_relative is not None:
                worst_proposal = deployment_proposal_relative[worst_probe_idx:worst_probe_idx + 1]
                worst_vlm_context = current_probe_steps[worst_probe_idx][-1]["vlm_context"].float()
                rollback_correction_loss = correction_path_rollback_loss(
                    unified_policy, unified_reference, unified_outcome,
                    worst_proposal, scene_context, worst_vlm_context,
                )
                total_rollback_loss = total_rollback_loss + rollback_correction_loss
                rollback_correction_value = float(rollback_correction_loss.item())
            (ROLLBACK_COEF * det_violation * total_rollback_loss).backward()
            rollback_value = float(rollback_loss.item())

        grad_norm = 0.0
        if train_on_group:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            policy_optimizer_steps += 1
        observed_kl = kl_sum / GROUP_SIZE
        previous_observed_kl = observed_kl
        if ADAPTIVE_KL:
            controlled_kl_coef = update_adaptive_kl_coefficient(
                controlled_kl_coef,
                observed_kl,
                target_kl=TARGET_KL,
                adaptation_rate=KL_ADAPTATION_RATE,
                minimum=KL_COEF_MIN,
                maximum=KL_COEF_MAX,
            )

        # Apply safety rehearsal as its own optimizer step. If it shared the
        # GRPO backward pass, the small reference-consistency gradient would
        # be suppressed by the policy gradient before global clipping.
        replay_value = 0.0
        replay_correction_value = 0.0
        replay_correction_grad_norm = 0.0
        replay_grad_norm = 0.0
        replay_token = "none"
        replay_source = "none"
        replay_bucket = "none"
        replay_label_counts = {}
        if (
            ADVANTAGE_MODE == "reference_anchored"
            and RISK_REPLAY_EVERY > 0
            and risk_buffer
            and (step_i + 1) % RISK_REPLAY_EVERY == 0
            and (fixed_schedule is None or scheduled_rehearsal)
        ):
            # Dataset transforms and model forwards may consume global
            # Python/NumPy/CPU/CUDA RNG even though the rehearsal ODE uses a
            # fixed latent and zero transition noise. Preserve all streams so
            # enabling replay cannot change later on-policy candidate noise.
            python_rng_state = random.getstate()
            numpy_rng_state = np.random.get_state()
            torch_rng_state = torch.random.get_rng_state()
            cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            replay_tokens = sorted(risk_buffer)
            if fixed_schedule is not None:
                replay_token = scene["token"]
                replay_bucket = "fixed_manifest"
            elif RISK_CURRICULUM:
                replay_token, replay_bucket = select_curriculum_memory(
                    risk_buffer,
                    replay_rng,
                    RISK_SOURCE_MIX,
                    type_balance=RISK_TYPE_BALANCE,
                    redundancy_coef=RISK_REDUNDANCY_COEF,
                    priority_power=RISK_PRIORITY_POWER,
                )
                curriculum_source_counts[replay_bucket] += 1
            elif RISK_REPLAY_SAMPLING == "priority":
                # alpha=1 reproduces legacy priority sampling; alpha=0 is
                # exactly uniform. Intermediate values avoid double-counting
                # risk when the manifest selector already enriches hard cases.
                replay_weights = tempered_priority_weights(
                    [risk_buffer[t]["priority"] for t in replay_tokens],
                    RISK_PRIORITY_POWER,
                )
                replay_token = replay_rng.choices(
                    replay_tokens, weights=replay_weights, k=1
                )[0]
                replay_bucket = "legacy_priority"
            else:
                replay_token = replay_rng.choice(replay_tokens)
                replay_bucket = "legacy_uniform"
            memory = risk_buffer[replay_token]
            replay_source = memory["source"]
            replay_label_counts = memory.get("label_counts", {})
            replay_sample = dataset[memory["idx"]]
            replay_input_ids = replay_sample["input_ids"].unsqueeze(0).to(DEVICE)
            replay_attention_mask = replay_sample["attention_mask"].unsqueeze(0).to(DEVICE)
            replay_pre_action = replay_sample["pre_action"].unsqueeze(0).to(DEVICE).to(DTYPE)
            replay_cmd = replay_sample["cmd"].unsqueeze(0).to(DEVICE).to(DTYPE)
            replay_z = fixed_scene_latent(
                replay_token, memory["probe_idx"], action_frames, action_dim
            )
            replay_reference_final, reference_steps, replay_vlm_seq_len = rollout_and_score(
                policy, ref_heads, replay_input_ids, replay_attention_mask,
                replay_pre_action, replay_cmd, action_frames, action_dim,
                NUM_INFERENCE_STEPS, 0.0, initial_z=replay_z,
            )
            optimizer.zero_grad()
            replay_loss = reference_path_rollback_loss(
                policy, policy_heads, ref_heads, replay_input_ids,
                replay_attention_mask, replay_pre_action, replay_cmd,
                reference_steps, replay_vlm_seq_len,
            )
            total_replay_loss = replay_loss
            if unified_policy is not None:
                replay_relative = 0.5 * (
                    replay_reference_final[0].float() + 1.0
                ) * torch.from_numpy(high - low).to(DEVICE) + torch.from_numpy(low).to(DEVICE)
                replay_relative = replay_relative.unsqueeze(0)
                replay_context = unified_scene_context(replay_token)
                replay_vlm_context = reference_steps[-1]["vlm_context"].float()
                replay_correction_loss = correction_path_rollback_loss(
                    unified_policy, unified_reference, unified_outcome,
                    replay_relative, replay_context, replay_vlm_context,
                )
                total_replay_loss = total_replay_loss + replay_correction_loss
                replay_correction_value = float(replay_correction_loss.item())
            (RISK_REPLAY_COEF * total_replay_loss).backward()
            if unified_policy is not None:
                correction_grad_sq = sum(
                    float(parameter.grad.detach().float().square().sum().item())
                    for parameter in unified_policy.parameters()
                    if parameter.grad is not None
                )
                replay_correction_grad_norm = correction_grad_sq ** 0.5
            replay_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0, error_if_nonfinite=True)
            )
            optimizer.step()
            replay_value = float(replay_loss.item())
            replay_update_count += 1
            memory["replay_count"] += 1
            if "manifest" in replay_source:
                manifest_replay_count += 1
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

        if DIAGNOSTICS:
            diagnostic_progress = np.asarray(sub_scores["ego_progress"])
            if ADVANTAGE_MODE == "reference_anchored":
                diagnostic_progress = diagnostic_progress + 0.25 * np.asarray(
                    sub_scores["driving_direction_compliance"]
                )
            diagnostic = compute_rollout_diagnostics(
                abs_trajs,
                sub_scores,
                adv,
                violation,
                is_safe,
                progress_values=diagnostic_progress,
                trajectory_epsilon=DIAGNOSTIC_TRAJECTORY_EPS,
                score_epsilon=DIAGNOSTIC_SCORE_EPS,
            )
            print(
                f"diagnostics step={step_i:03d} scene={scene['token']} "
                + " ".join(f"{key}={value:.6g}" for key, value in diagnostic.items()),
                flush=True,
            )

        safe_frac = float(is_safe.mean())
        ttc_mean = float(sub_scores["time_to_collision_within_bound"].mean())
        progress_mean = float(sub_scores["ego_progress"].mean())
        unified_gate_mean = (
            float(np.mean(unified_records["gate"]))
            if unified_records is not None else 0.0
        )
        unified_physics_before = (
            float(np.mean(unified_records["physics_before"]))
            if unified_records is not None else 0.0
        )
        unified_physics_after = (
            float(np.mean(unified_records["physics_after"]))
            if unified_records is not None else 0.0
        )
        unified_residual_abs = (
            float(np.mean(unified_records["residual_abs"]))
            if unified_records is not None else 0.0
        )
        reward_history.append({"safe_frac": safe_frac, "ttc_mean": ttc_mean, "progress_mean": progress_mean})
        print(
            f"step {step_i:03d} scene={scene['token']} "
            f"safe_frac={safe_frac:.4f} ttc_mean={ttc_mean:.4f} progress_mean={progress_mean:.4f} "
            f"adv_mean={adv.mean():.4f} adv_std={adv.std():.4f} "
            f"pg_loss={pg_loss_sum / GROUP_SIZE:.4f} kl={kl_sum / GROUP_SIZE:.8g} "
            f"rl_weight={rl_weight:.4f} kl_mult={adaptive_kl_multiplier:.4f} "
            f"effective_kl={effective_kl_coef:.4f} next_kl_coef={controlled_kl_coef:.4f} "
            f"kl_guard={int(kl_guard_active)} "
            f"det_violation={det_violation:.4f} det_nc_regret={det_nc_regret:.4f} "
            f"det_dac_regret={det_dac_regret:.4f} det_ttc_regret={det_ttc_regret:.4f} "
            f"w_nc={scene_constraint_weights['nc']:.4f} "
            f"w_dac={scene_constraint_weights['dac']:.4f} "
            f"w_ttc={scene_constraint_weights['ttc']:.4f} "
            f"control_attempted={control_attempted} control_accepted={control_accepted} "
            f"unified_gate={unified_gate_mean:.4f} "
            f"unified_residual_abs={unified_residual_abs:.6f} "
            f"unified_physics_before={unified_physics_before:.6f} "
            f"unified_physics_after={unified_physics_after:.6f} "
            f"rollback={rollback_value:.6f} "
            f"rollback_correction={rollback_correction_value:.6f} "
            f"risk_priority={risk_priority:.4f} "
            f"risk_buffer={len(risk_buffer)} replay={replay_value:.6f} "
            f"replay_correction={replay_correction_value:.6f} "
            f"replay_correction_grad_norm={replay_correction_grad_norm:.6f} "
            f"replay_scene={replay_token} replay_source={replay_source} replay_bucket={replay_bucket} "
            f"replay_grad_norm={replay_grad_norm:.4f} "
            f"replay_unsafe={int(replay_label_counts.get('unsafe', 0))} "
            f"replay_ttc={int(replay_label_counts.get('ttc_failure', 0))} "
            f"replay_dac={int(replay_label_counts.get('dac_failure', 0))} "
            f"replay_nc={int(replay_label_counts.get('nc_failure', 0))} "
            f"risk_events={risk_event_count} cache_hit={int(cache_hit)} "
            f"ref_ttc={reference_probe_scores[worst_probe_idx]['time_to_collision_within_bound'] if ADVANTAGE_MODE == 'reference_anchored' else -1:.4f} "
            f"det_ttc={current_probe_scores[worst_probe_idx]['time_to_collision_within_bound'] if ADVANTAGE_MODE == 'reference_anchored' else -1:.4f} "
            f"grad_norm={grad_norm:.4f}",
            flush=True,
        )

        if CHECKPOINT_DIR:
            Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
            with open(Path(CHECKPOINT_DIR) / "optimization_audit.jsonl", "a") as audit_file:
                audit_file.write(json.dumps(dict(
                    step=step_i, scene=scene["token"], kl_mode=KL_MODE,
                    positive_guard=POSITIVE_SAFETY_GUARD, **positive_guard_stats,
                    active_changed_candidates=(positive_guard_stats["changed_candidates"]
                        if train_on_group and rl_weight > 0 else 0),
                    pg_weight=float(rl_weight) if train_on_group else 0.0,
                    observed_kl=float(observed_kl), effective_kl_coef=float(effective_kl_coef),
                    policy_optimizer_steps=policy_optimizer_steps,
                    rehearsal_optimizer_steps=replay_update_count,
                    peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                ), allow_nan=False) + "\n")

        if fixed_schedule is not None and CHECKPOINT_DIR:
            Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
            with open(Path(CHECKPOINT_DIR) / "replay_updates.jsonl", "a") as file:
                file.write(json.dumps(dict(step=step_i, kind=step_kind, scene=scene["token"],
                    mode=RISK_REPLAY_MODE, train_on_group=train_on_group,
                    rehearsal=scheduled_rehearsal, policy_optimizer_steps=policy_optimizer_steps,
                    rehearsal_optimizer_steps=replay_update_count, pg_weight=float(rl_weight) if train_on_group else 0.0,
                    pg_loss=pg_loss_sum / GROUP_SIZE, replay_loss=replay_value,
                    grad_norm=float(grad_norm), replay_grad_norm=replay_grad_norm,
                    score_calls=NAVSIM_SCORE_CALLS, scored_trajectories=NAVSIM_SCORED_TRAJECTORIES), allow_nan=False) + "\n")

        should_save = (step_i + 1) in SAVE_STEPS or ((step_i + 1) % SAVE_EVERY == 0)
        if CHECKPOINT_DIR and should_save and step_i + 1 != NUM_STEPS:
            # Separate subdirectory per step — a single shared path would
            # have each save silently overwrite the previous one, losing
            # every intermediate checkpoint needed to see *where* along
            # training PDMS starts moving (up or down).
            step_dir = f"{CHECKPOINT_DIR}/step_{step_i + 1:06d}"
            print(f"Saving checkpoint at step {step_i} to {step_dir} ...", flush=True)
            policy.save_pretrained(step_dir)
            save_unified_components(step_dir)
            save_execution_metadata(step_dir)

    if CHECKPOINT_DIR:
        final_dir = f"{CHECKPOINT_DIR}/step_{NUM_STEPS:06d}_final"
        print(f"Saving final checkpoint to {final_dir} ...", flush=True)
        policy.save_pretrained(final_dir)
        save_unified_components(final_dir)
        save_execution_metadata(final_dir)

    print("\n=== Summary ===")
    first5_safe = [h["safe_frac"] for h in reward_history[:5]]
    last5_safe = [h["safe_frac"] for h in reward_history[-5:]]
    first5_ttc = [h["ttc_mean"] for h in reward_history[:5]]
    last5_ttc = [h["ttc_mean"] for h in reward_history[-5:]]
    print(f"safe_frac over training: first 5 = {first5_safe}, last 5 = {last5_safe}")
    print(f"ttc_mean over training: first 5 = {first5_ttc}, last 5 = {last5_ttc}")
    print(
        f"control projection: attempted={control_attempted_total}; "
        f"accepted={control_accepted_total}; "
        f"checkpoint={CONTROL_ADAPTER_CHECKPOINT!r}; mode={CONTROL_ADAPTER_MODE}"
    )
    print(
        f"risk buffer final size: {len(risk_buffer)}; "
        f"safety rehearsal updates: {replay_update_count}; "
        f"manifest rehearsal updates: {manifest_replay_count}; "
        f"manifest selector: {manifest_selector}; manifest budget: {manifest_budget}; "
        f"manifest verified events: {manifest_verified_events}; "
        f"coverage selector: {coverage_selector}; coverage verified events: {coverage_verified_events}; "
        f"curriculum source counts: {curriculum_source_counts}"
    )


if __name__ == "__main__":
    main()
