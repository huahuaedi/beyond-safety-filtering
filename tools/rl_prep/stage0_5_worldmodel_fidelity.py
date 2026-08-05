"""
Stage 0.5 sanity check for docs/rl_world_model_design.md.

Question: does the NuPlan-pretrained world model's predictive fidelity for a
candidate future frame correlate with that candidate trajectory's *real*
NAVSIM PDM sub-scores (from formatted_pdm_score_8192.npy)?

For a handful of test scenes, and a stratified sample of the 8192 anchor
trajectories per scene:
  1. condition the world model (Emu3MoE, Emu3_NuPlan_Pretrain_Cktps) on
     [current frame tokens] + [candidate 8-step action, FAST-tokenized]
  2. score the negative log-likelihood the model assigns to the *real*
     recorded frame 4s (8 steps) ahead, teacher-forced
  3. correlate that NLL against the candidate's real PDM sub-scores

Deviation from the original Stage-1 training format (documented in
docs/rl_world_model_design.md): the real pipeline interleaves 6 frame/action
pairs at 10-step spacing; here we only use a single [frame_t, action, frame_t+8]
triple per candidate to keep the per-candidate cost to one forward pass.
Previous-action history conditioning is dropped since it is constant across
candidates within a scene and should not affect the within-scene ranking
we care about.
"""
import os
import sys
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "reference" / "Emu3"))

from transformers import AutoProcessor
from emu3.mllm import Emu3Tokenizer, Emu3MoE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUPLAN_CKPT = REPO_ROOT / "pretrained_models/DriveVLA-W0/Emu3_NuPlan_Pretrain_Cktps"
FAST_TOKENIZER_PATH = REPO_ROOT / "pretrained_models/fast"
NORM_STATS_PATH = REPO_ROOT / "configs/normalizer_navsim_trainval/norm_stats.json"
CLUSTER_PATH = REPO_ROOT / "data/navsim/processed_data/anchor/cluster_centers_8192.npy"
# Pre-extracted small subset produced by extract_relevant_anchor_scores.py —
# avoids ever loading the full 47GB formatted_pdm_score_8192.npy again.
PDM_SCORE_PATH = REPO_ROOT / "data/navsim/processed_data/anchor/trainval_scene_anchor_scores.pkl"
# The anchor scores are keyed against the TRAINVAL split (scene["token"]),
# not the test split — see extract_relevant_anchor_scores.py docstring.
SCENE_PKL = REPO_ROOT / "pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
VQ_ROOT_CANDIDATES = [
    REPO_ROOT / "data/navsim/processed_data/data/navsim/processed_data",
]

NUM_SCENES = int(os.environ.get("S05_NUM_SCENES", 8))
CANDIDATES_PER_SCENE = int(os.environ.get("S05_CANDIDATES_PER_SCENE", 40))
CUR_IDX = 3          # matches cur_frame_idx used everywhere else in this repo
HORIZON = 8           # matches action_frames / cluster trajectory length
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

ANCHOR_METRIC_KEYS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
]


_PATH_PREFIXES = (
    "/mnt/vdb1/yingyan.li/repo/VLA/data/navsim/processed_data/",
    "/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/data/navsim/processed_data/",
    "/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/data/navsim/processed_data/",
)


def resolve_vq_path(raw_path: str) -> str:
    for old in _PATH_PREFIXES:
        if raw_path.startswith(old):
            return raw_path.replace(old, str(VQ_ROOT_CANDIDATES[0]) + "/")
    return raw_path


def load_vq_tokens(raw_path: str) -> np.ndarray:
    resolved = resolve_vq_path(raw_path)
    return np.load(resolved).reshape(1, 18, 32)


def absolute_to_relative_se2(abs_traj: np.ndarray) -> np.ndarray:
    """Inverse of EmuVLAAgent.convert_relative_to_absolute_se2_trajectory.

    abs_traj: [H, 3] absolute (x, y, yaw), starting implicitly from (0,0,0).
    returns: [H, 3] relative (dx, dy, dyaw) in the local frame of the
    previous step.
    """
    H = abs_traj.shape[0]
    rel = np.zeros_like(abs_traj)
    prev = np.array([0.0, 0.0, 0.0])
    for i in range(H):
        cur = abs_traj[i]
        dx_g = cur[0] - prev[0]
        dy_g = cur[1] - prev[1]
        dyaw = (cur[2] - prev[2] + np.pi) % (2 * np.pi) - np.pi
        yaw_prev = prev[2]
        dx_l = np.cos(yaw_prev) * dx_g + np.sin(yaw_prev) * dy_g
        dy_l = -np.sin(yaw_prev) * dx_g + np.cos(yaw_prev) * dy_g
        rel[i] = [dx_l, dy_l, dyaw]
        prev = cur
    return rel


def build_scene_token(image_path: str) -> str:
    return Path(image_path).stem


def main():
    random.seed(0)
    np.random.seed(0)

    print(f"Loading tokenizer/model from {NUPLAN_CKPT} ...")
    tokenizer = Emu3Tokenizer.from_pretrained(
        str(NUPLAN_CKPT), model_max_length=4000, padding_side="right", use_fast=False,
    )
    model = Emu3MoE.from_pretrained(
        str(NUPLAN_CKPT), attn_implementation="sdpa", torch_dtype=DTYPE,
    ).to(DEVICE).eval()

    action_tokenizer = AutoProcessor.from_pretrained(str(FAST_TOKENIZER_PATH), trust_remote_code=True)

    with open(NORM_STATS_PATH) as f:
        norm_cfg = json.load(f)["norm_stats"]["libero"]
    low = np.array(norm_cfg["q01"], dtype=np.float32)
    high = np.array(norm_cfg["q99"], dtype=np.float32)

    cluster = np.load(CLUSTER_PATH)  # [8192, 8, 3] absolute SE2

    print(f"Loading pre-extracted PDM anchor scores from {PDM_SCORE_PATH} ...")
    with open(PDM_SCORE_PATH, "rb") as f:
        metric_raw = pickle.load(f)
    print(f"  loaded {len(metric_raw)} scene entries")

    with open(SCENE_PKL, "rb") as f:
        scene_data = pickle.load(f)
    by_token = {scene["token"]: scene for scene in scene_data}

    # extract_relevant_anchor_scores.py already sampled which scene tokens
    # to keep — just look them up here.
    eligible = [(tok, by_token[tok]) for tok in metric_raw if tok in by_token]
    print(f"{len(eligible)} anchor-score entries have a matching scene in {SCENE_PKL.name}")
    if not eligible:
        print("No scene token matched — dumping a few keys for debugging:")
        print("  score dict keys sample:", list(metric_raw.keys())[:5])
        print("  scene token sample:", list(by_token.keys())[:5])
        return

    random.shuffle(eligible)
    eligible = eligible[:NUM_SCENES]

    boa_id = tokenizer.encode(tokenizer.boa_token)[0]
    eoa_id = tokenizer.encode(tokenizer.eoa_token)[0]
    bov_id = model.bov_token_id
    eov_id = model.eov_token_id

    def format_video_prompt(vq_tokens: np.ndarray) -> str:
        frames, h, w = vq_tokens.shape
        rows = []
        for frame in vq_tokens:
            row_strs = []
            for row in frame:
                row_strs.append("".join(f"<|visual token {t:0>6d}|>" for t in row))
            rows.append(tokenizer.eol_token.join(row_strs))
        videostr = tokenizer.eof_token.join(rows)
        return (
            tokenizer.boi_token + f"{frames}*{h}*{w}" + tokenizer.img_token
            + videostr + tokenizer.eof_token + tokenizer.eoi_token
        )

    def tokenize_candidate_action(rel_action_norm: np.ndarray) -> list:
        action_ids = action_tokenizer(torch.tensor(rel_action_norm, dtype=torch.float32))[0]
        last_vocab_idx = tokenizer.pad_token_id - 1
        action_ids = [last_vocab_idx - i for i in action_ids]
        return [boa_id] + action_ids + [eoa_id]

    all_nll = []
    all_scores = {k: [] for k in ANCHOR_METRIC_KEYS}
    all_l2_to_expert = []
    all_scene_idx = []

    with torch.no_grad():
        for scene_i, (tok, scene) in enumerate(eligible):
            entry = metric_raw[tok]
            traj_scores = entry["trajectory_scores"][0]
            stacked = {k: np.asarray(traj_scores[k]) for k in ANCHOR_METRIC_KEYS}
            num_traj = len(stacked[ANCHOR_METRIC_KEYS[0]])
            assert num_traj == cluster.shape[0], f"anchor count mismatch: {num_traj} vs {cluster.shape[0]}"

            combined = 5 * stacked["time_to_collision_within_bound"] + 2 * stacked["comfort"] + 5 * stacked["ego_progress"]
            combined *= stacked["no_at_fault_collisions"].clip(min=1e-6)
            combined *= stacked["drivable_area_compliance"].clip(min=1e-6)

            order = np.argsort(combined)
            n = len(order)
            strata = [order[int(f * n):int(f * n) + max(1, CANDIDATES_PER_SCENE // 4)]
                      for f in (0.0, 0.33, 0.66, 0.95)]
            cand_idx = np.unique(np.concatenate(strata))[:CANDIDATES_PER_SCENE]

            try:
                cur_frame = load_vq_tokens(scene["image"][CUR_IDX])
                target_frame = load_vq_tokens(scene["image"][CUR_IDX + HORIZON])
            except FileNotFoundError as e:
                print(f"  [scene {scene_i}] skipping, missing vq file: {e}")
                continue

            prompt_text = scene["text"][CUR_IDX]
            context_str = tokenizer.bos_token + prompt_text + format_video_prompt(cur_frame)
            context_ids = tokenizer(context_str, add_special_tokens=False)["input_ids"]

            target_str = format_video_prompt(target_frame)
            target_ids = tokenizer(target_str, add_special_tokens=False)["input_ids"]

            expert_action_norm = scene["action"][CUR_IDX:CUR_IDX + HORIZON]  # already normalized rel actions

            for ci in cand_idx:
                abs_traj = cluster[ci]
                rel_action_raw = absolute_to_relative_se2(abs_traj)
                rel_action_norm = 2 * (rel_action_raw - low) / (high - low) - 1
                rel_action_norm = np.clip(rel_action_norm, -1.0, 1.0)

                action_id_seq = tokenize_candidate_action(rel_action_norm)

                full_ids = context_ids + action_id_seq + target_ids
                input_ids = torch.tensor([full_ids], dtype=torch.long, device=DEVICE)
                labels = torch.full_like(input_ids, -100)
                target_len = len(target_ids)
                labels[0, -target_len:] = input_ids[0, -target_len:]

                out = model(input_ids=input_ids, labels=None)
                logits = out.logits[:, :-1, :].float()
                shift_labels = labels[:, 1:]
                mask = shift_labels != -100
                token_nll = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    shift_labels.reshape(-1).clamp(min=0),
                    reduction="none",
                ).reshape(shift_labels.shape)
                nll = (token_nll * mask).sum() / mask.sum()

                all_nll.append(nll.item())
                for k in ANCHOR_METRIC_KEYS:
                    all_scores[k].append(float(stacked[k][ci]))
                l2 = np.linalg.norm((rel_action_norm - expert_action_norm).reshape(-1))
                all_l2_to_expert.append(l2)
                all_scene_idx.append(scene_i)

            print(f"  [scene {scene_i}] token={tok} scored {len(cand_idx)} candidates")

    all_nll = np.array(all_nll)
    combined_score = (
        5 * np.array(all_scores["time_to_collision_within_bound"])
        + 2 * np.array(all_scores["comfort"])
        + 5 * np.array(all_scores["ego_progress"])
    ) * np.clip(all_scores["no_at_fault_collisions"], 1e-6, None) * np.clip(all_scores["drivable_area_compliance"], 1e-6, None)

    print("\n=== Results ===")
    print(f"N candidates scored: {len(all_nll)} across {len(eligible)} scenes")

    def report(name, x, y):
        sp, sp_p = spearmanr(x, y)
        pr, pr_p = pearsonr(x, y)
        print(f"  world-model NLL vs {name}: spearman={sp:.3f} (p={sp_p:.3g})  pearson={pr:.3f} (p={pr_p:.3g})")

    report("combined PDM-like score", all_nll, combined_score)
    for k in ANCHOR_METRIC_KEYS:
        report(k, all_nll, np.array(all_scores[k]))
    print("\n  --- confound check ---")
    report("L2 distance to expert action (normalized space)", all_nll, np.array(all_l2_to_expert))

    out_path = REPO_ROOT / "docs/stage0_5_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "nll": all_nll.tolist(),
            "combined_score": combined_score.tolist(),
            "scores": all_scores,
            "l2_to_expert": all_l2_to_expert,
            "scene_idx": all_scene_idx,
        }, f)
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    main()
