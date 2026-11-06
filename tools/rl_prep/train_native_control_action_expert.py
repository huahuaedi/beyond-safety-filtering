"""Warm-start DriveVLA's flow Action Expert in native continuous-control space.

The VLM stays frozen.  Valid pseudo-controls supervise flow matching, while a
differentiable vehicle rollout supplies position/heading reconstruction losses.
Version 1 retains action_dim=3 for checkpoint compatibility; channel semantics
are (acceleration logit, curvature-rate logit, reserved zero channel).
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "inference" / "vla"))
sys.path.insert(0, str(REPO_ROOT / "utils"))
sys.path.insert(0, str(REPO_ROOT / "reference" / "Emu3"))
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("VLA_ACTION_TOKENIZER", str(REPO_ROOT / "pretrained_models/fast"))
os.environ.setdefault("VLA_VLM_MODEL", str(REPO_ROOT / "pretrained_models/Emu3-Stage1"))
os.environ.setdefault(
    "VLA_NORM_STATS", str(REPO_ROOT / "configs/normalizer_navsim_trainval/norm_stats.json")
)
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

from emu3.mllm import Emu3Pi0, Emu3Pi0Config, Emu3Tokenizer  # noqa: E402
from datasets import Emu3DrivingVAVADataset  # noqa: E402
from utils.train_pi0 import DataArguments  # noqa: E402

from differentiable_vehicle_model import VehicleModelConfig  # noqa: E402
from native_control_expert import (  # noqa: E402
    controls_from_absolute_trajectory,
    controls_to_model_action,
    execute_model_action,
    initial_state_from_relative_history,
)
from rl_stage1_prototype import build_policy_heads, denoise_step_velocity, freeze_vlm  # noqa: E402


EMU_HUB = Path(os.environ.get(
    "NATIVE_CONTROL_INIT_CHECKPOINT",
    REPO_ROOT / "pretrained_models/DriveVLA-W0/Emu3_Flow_Matching_Action_Expert_PDMS_87.2",
))
SCENE_PKL = Path(os.environ.get(
    "NATIVE_CONTROL_SCENE_PKL",
    REPO_ROOT / "pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl",
))
DATA_ROOT = Path(os.environ.get(
    "NATIVE_CONTROL_DATA_ROOT", REPO_ROOT / "navsim_data/exp/world_action_full_v1"
))
CONTROL_ROOT = Path(os.environ.get(
    "NATIVE_CONTROL_SIDECAR_ROOT", REPO_ROOT / "navsim_data/exp/world_action_controls_full_v1"
))
OUTPUT_ROOT = Path(os.environ.get(
    "NATIVE_CONTROL_OUTPUT", REPO_ROOT / "output/native_control_action_expert_v1"
))
SEED = int(os.environ.get("NATIVE_CONTROL_SEED", 0))
EPOCHS = int(os.environ.get("NATIVE_CONTROL_EPOCHS", 1))
MAX_SCENES = int(os.environ.get("NATIVE_CONTROL_MAX_SCENES", 16))
MAX_UPDATES = int(os.environ.get("NATIVE_CONTROL_MAX_UPDATES", 0))
LR = float(os.environ.get("NATIVE_CONTROL_LR", 1e-5))
TRAJECTORY_COEF = float(os.environ.get("NATIVE_CONTROL_TRAJECTORY_COEF", 1.0))
HEADING_COEF = float(os.environ.get("NATIVE_CONTROL_HEADING_COEF", 2.0))
RESERVED_COEF = float(os.environ.get("NATIVE_CONTROL_RESERVED_COEF", 0.1))
CANDIDATE_SELECTION = os.environ.get("NATIVE_CONTROL_CANDIDATE_SELECTION", "quality_top1")
if CANDIDATE_SELECTION not in ("quality_top1", "random"):
    raise ValueError("NATIVE_CONTROL_CANDIDATE_SELECTION must be quality_top1 or random")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16


def sample_flow_time(scheduler, batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample the repository's flow timestep without Beta/Dirichlet kernels.

    DriveVLA uses ``1 - X`` with ``X ~ Beta(1.5, 1)``.  For Beta(a, 1),
    inverse-transform sampling is exactly ``X = U ** (1 / a)``.  Keeping this
    operation in float32 also avoids accelerator/version-specific failures in
    ``torch.distributions.Dirichlet`` before the result is cast by the caller.
    """
    if scheduler.sample_method == "beta":
        uniform = torch.rand(batch_size, device=device, dtype=torch.float32)
        return float(scheduler.s) * (1.0 - uniform.pow(1.0 / 1.5))
    if scheduler.sample_method == "uniform":
        return float(scheduler.s) * torch.rand(
            batch_size, device=device, dtype=torch.float32
        )
    raise ValueError(f"unsupported flow timestep distribution: {scheduler.sample_method}")


def discover_records(split: str):
    records = []
    for sidecar in sorted((CONTROL_ROOT / split).glob("*.npz")):
        data_path = DATA_ROOT / split / sidecar.name
        if data_path.is_file():
            records.append((sidecar.stem, data_path, sidecar))
    if MAX_SCENES > 0:
        records = records[:MAX_SCENES]
    return records


def cpu_state(module):
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def load_candidate(record, rng: np.random.Generator):
    _, data_path, sidecar_path = record
    with np.load(data_path, allow_pickle=False) as data:
        absolute = data["absolute_trajectories"].astype(np.float32)
        feasible = data["reference_feasible"].astype(bool)
        violation = data["reference_violation"].astype(np.float32)
        progress = data["score__ego_progress"].astype(np.float32)
        direction = data["score__driving_direction_compliance"].astype(np.float32)
    with np.load(sidecar_path, allow_pickle=False) as data:
        initial = data["initial_state"].astype(np.float32)
        controls = data["pseudo_controls"].astype(np.float32)
        valid = data["control_supervision_valid"].astype(bool)
    candidates = np.flatnonzero(valid)
    if not len(candidates):
        return None
    if CANDIDATE_SELECTION == "random":
        candidate = int(rng.choice(candidates))
    else:
        safe = candidates[feasible[candidates]]
        pool = safe if len(safe) else candidates
        utility = progress[pool] + 0.25 * direction[pool]
        if not len(safe):
            # Lexicographic recovery: minimize verified violation first, then
            # maximize progress/direction among equally safe candidates.
            minimum = violation[pool].min()
            pool = pool[np.isclose(violation[pool], minimum, atol=1e-8)]
            utility = progress[pool] + 0.25 * direction[pool]
        candidate = int(pool[np.argmax(utility)])
    return initial[candidate], controls[candidate], absolute[candidate], candidate


def native_control_loss(policy, heads, sample, target, action_low, action_high):
    _, _, absolute_np, _ = target
    input_ids = sample["input_ids"].unsqueeze(0).to(DEVICE)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(DEVICE)
    pre_action = sample["pre_action"].unsqueeze(0).to(DEVICE, dtype=DTYPE)
    cmd = sample["cmd"].unsqueeze(0).to(DEVICE, dtype=DTYPE)
    target_absolute = torch.from_numpy(absolute_np).unsqueeze(0).to(DEVICE)
    history = pre_action.float()
    history_denorm = 0.5 * (history + 1.0) * (action_high - action_low) + action_low
    initial_state = initial_state_from_relative_history(history_denorm)
    controls = controls_from_absolute_trajectory(target_absolute, initial_state)
    action = controls_to_model_action(controls).to(DTYPE)

    noise = torch.randn_like(action)
    tau = sample_flow_time(policy.rf, 1, DEVICE).to(dtype=DTYPE)
    noisy_action = policy.rf.add_noise(action, noise, tau)
    vlm_h = policy.vlm.model.embed_tokens(input_ids)
    vlm_seq_len = vlm_h.shape[1]
    position_ids = torch.arange(vlm_seq_len, device=DEVICE).unsqueeze(0)
    state = torch.cat([pre_action.reshape(1, -1), cmd], dim=1)
    state_embedding = heads.state_projector(state).unsqueeze(1)
    velocity = denoise_step_velocity(
        heads, vlm_h, state_embedding, noisy_action, float(tau.item()),
        input_ids, attention_mask, position_ids, vlm_seq_len, 1,
    )
    flow_target = noise - action
    flow = nn.functional.mse_loss(velocity.float(), flow_target.float())

    predicted_action = noise.float() - velocity.float()
    execution = execute_model_action(predicted_action, initial_state)
    predicted_absolute = execution["absolute_trajectory"]
    position = nn.functional.smooth_l1_loss(
        predicted_absolute[..., :2], target_absolute[..., :2]
    )
    heading_delta = predicted_absolute[..., 2] - target_absolute[..., 2]
    heading = (1.0 - heading_delta.cos()).mean()
    reserved = predicted_action[..., 2].square().mean()
    total = flow + TRAJECTORY_COEF * position + HEADING_COEF * heading + RESERVED_COEF * reserved
    return total, {
        "loss": float(total.detach()), "flow": float(flow.detach()),
        "position": float(position.detach()), "heading": float(heading.detach()),
        "reserved": float(reserved.detach()),
    }


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    train_records = discover_records("train")
    val_records = discover_records("val")
    if not train_records or not val_records:
        raise ValueError(f"missing records: train={len(train_records)} val={len(val_records)}")

    tokenizer = Emu3Tokenizer.from_pretrained(
        str(os.environ["VLA_VLM_MODEL"]), model_max_length=1400,
        padding_side="right", use_fast=False,
    )
    config = Emu3Pi0Config.from_pretrained(str(EMU_HUB / "config.json"))
    policy, loading = Emu3Pi0.from_pretrained(
        str(EMU_HUB), config=config,
        pretrain_vlm_path=os.environ["VLA_VLM_MODEL"],
        attn_implementation="sdpa", torch_dtype=DTYPE,
        trust_remote_code=True, output_loading_info=True,
    )
    policy = policy.to(DEVICE)
    trainable = freeze_vlm(policy)
    policy.vlm.eval()
    heads = build_policy_heads(policy)

    data_args = DataArguments(
        data_path=str(SCENE_PKL), actions=True, driving=True,
        use_previous_actions=True, actions_format="fast",
        action_tokenizer_path=os.environ["VLA_ACTION_TOKENIZER"],
        frames=1, action_frames=8, action_dim=3, cur_frame_idx=3,
        pre_action_frames=3, video_format=None, use_flip=False,
    )
    dataset = Emu3DrivingVAVADataset(data_args, tokenizer=tokenizer)
    norm_cfg = json.loads(Path(os.environ["VLA_NORM_STATS"]).read_text())
    action_low = torch.tensor(
        norm_cfg["norm_stats"]["libero"]["q01"], device=DEVICE, dtype=torch.float32
    )
    action_high = torch.tensor(
        norm_cfg["norm_stats"]["libero"]["q99"], device=DEVICE, dtype=torch.float32
    )
    token_to_index = {scene["token"]: i for i, scene in enumerate(dataset.data)}
    train_records = [r for r in train_records if r[0] in token_to_index]
    val_records = [r for r in val_records if r[0] in token_to_index]
    print(
        f"native-control seed={SEED} device={DEVICE} train={len(train_records)} "
        f"val={len(val_records)} epochs={EPOCHS} max_updates={MAX_UPDATES}", flush=True,
    )
    print(
        f"loading missing={len(loading.get('missing_keys', []))} "
        f"unexpected={len(loading.get('unexpected_keys', []))}", flush=True,
    )

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    best_state = None; best_value = float("inf"); updates = 0
    for epoch in range(EPOCHS):
        policy.train(); policy.vlm.eval()
        order = rng.permutation(len(train_records))
        values = []
        for record_index in order:
            target = load_candidate(train_records[int(record_index)], rng)
            if target is None:
                continue
            sample = dataset[token_to_index[train_records[int(record_index)][0]]]
            loss, metrics = native_control_loss(
                policy, heads, sample, target, action_low, action_high
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            values.append(metrics); updates += 1
            if updates % 10 == 0 or updates == 1:
                print(
                    f"epoch={epoch:02d} update={updates:05d} "
                    + " ".join(f"{k}={v:.6f}" for k, v in metrics.items()),
                    flush=True,
                )
            if MAX_UPDATES > 0 and updates >= MAX_UPDATES:
                break
        if not values:
            raise RuntimeError("no valid control candidates were loaded")
        mean_loss = float(np.mean([x["loss"] for x in values]))
        if mean_loss < best_value:
            best_value = mean_loss
            # Save only trainable action-side modules; the base checkpoint is
            # recorded explicitly and remains the source for the frozen VLM.
            best_state = {
                "action_expert": cpu_state(policy.action_expert),
                "tau_emb": cpu_state(policy.tau_emb),
                "action_projector": cpu_state(policy.action_projector),
                "state_projector": cpu_state(policy.state_projector),
                "action_decoder": cpu_state(policy.action_decoder),
            }
        print(f"epoch={epoch:02d} mean_loss={mean_loss:.6f}", flush=True)
        if MAX_UPDATES > 0 and updates >= MAX_UPDATES:
            break

    output = OUTPUT_ROOT / f"seed_{SEED}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "representation": "native_acceleration_curvature_rate_v2",
        "tensor_shape": [8, 3],
        "executed_channels": ["acceleration", "curvature_rate"],
        "reserved_channel": 2,
        "base_checkpoint": str(EMU_HUB),
        "seed": SEED, "updates": updates, "epochs": EPOCHS,
        "candidate_selection": CANDIDATE_SELECTION,
        "initial_state_source": "observed_pre_action",
        "best_train_epoch_loss": best_value,
        "loss_weights": {
            "flow": 1.0, "trajectory": TRAJECTORY_COEF,
            "heading": HEADING_COEF, "reserved": RESERVED_COEF,
        },
        "vehicle_model": VehicleModelConfig().__dict__,
    }
    torch.save({"action_state_dict": best_state, "metadata": metadata}, output / "model.pt")
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
