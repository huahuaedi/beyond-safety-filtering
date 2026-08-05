"""
阶段 0.5-bis (see docs/rl_world_model_design.md, 阶段 1.5之后的追加验证).

Stage 0.5 tested whether the world model's own predictive *fidelity* for a
candidate future frame (teacher-forced NLL) correlates with that candidate's
real PDM score — it does not (spearman~=0.05). This script tests a *different*
signal, closer to what VLA-RFT's "verified reward" actually does: treat the
world model as a forward simulator and measure how far a candidate action's
*imagined rollout* diverges from the expert action's imagined rollout, in
hidden-state space — not "does this look plausible" but "does imagining this
action lead somewhere different from imagining the real expert action."

Cheap proxy for full autoregressive frame generation (which would be
prohibitively slow per-candidate): a single forward pass conditioned on
[context frame] + [candidate action] already encodes the model's belief about
what happens next in its hidden state at the end of the action span — we
don't need to actually decode the future frame tokens to compare that belief
against the same position's hidden state when conditioned on the *expert*
action instead. Both forwards share the identical context prefix; only the
action differs, so the hidden-state divergence isolates the effect of the
action choice under the model's own dynamics.

Distance metric: 1 - cosine_similarity between the two hidden vectors at the
end-of-action-token position (found dynamically, not by a fixed offset — FAST
action tokenization is variable-length, so the two sequences' EOA positions
generally differ).

Reuses the same NuPlan-pretrain checkpoint, anchor-score ground truth, and
reporting format as stage0_5_worldmodel_fidelity.py so the two signals'
correlation with real PDM scores are directly comparable.
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
# Config (mirrors stage0_5_worldmodel_fidelity.py)
# ---------------------------------------------------------------------------
NUPLAN_CKPT = REPO_ROOT / "pretrained_models/DriveVLA-W0/Emu3_NuPlan_Pretrain_Cktps"
FAST_TOKENIZER_PATH = REPO_ROOT / "pretrained_models/fast"
NORM_STATS_PATH = REPO_ROOT / "configs/normalizer_navsim_trainval/norm_stats.json"
CLUSTER_PATH = REPO_ROOT / "data/navsim/processed_data/anchor/cluster_centers_8192.npy"
PDM_SCORE_PATH = REPO_ROOT / "data/navsim/processed_data/anchor/trainval_scene_anchor_scores.pkl"
SCENE_PKL = REPO_ROOT / "pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
VQ_ROOT_CANDIDATES = [
    REPO_ROOT / "data/navsim/processed_data/data/navsim/processed_data",
]

NUM_SCENES = int(os.environ.get("S05B_NUM_SCENES", 8))
CANDIDATES_PER_SCENE = int(os.environ.get("S05B_CANDIDATES_PER_SCENE", 40))
CUR_IDX = 3
HORIZON = 8
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

    eligible = [(tok, by_token[tok]) for tok in metric_raw if tok in by_token]
    print(f"{len(eligible)} anchor-score entries have a matching scene in {SCENE_PKL.name}")
    if not eligible:
        print("No scene token matched.")
        return

    random.shuffle(eligible)
    eligible = eligible[:NUM_SCENES]

    boa_id = tokenizer.encode(tokenizer.boa_token)[0]
    eoa_id = tokenizer.encode(tokenizer.eoa_token)[0]

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

    def tokenize_action(rel_action_norm: np.ndarray) -> list:
        action_ids = action_tokenizer(torch.tensor(rel_action_norm, dtype=torch.float32))[0]
        last_vocab_idx = tokenizer.pad_token_id - 1
        action_ids = [last_vocab_idx - i for i in action_ids]
        return [boa_id] + action_ids + [eoa_id]

    def hidden_state_at_eoa(context_ids: list, action_id_seq: list) -> torch.Tensor:
        """Forward [context, action] through the decoder only (no LM head, no
        target frame) and return the hidden vector at the action span's EOA
        token — found by position, not a fixed offset (FAST tokenization is
        variable-length)."""
        full_ids = context_ids + action_id_seq
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=DEVICE)
        eoa_pos = len(full_ids) - 1  # eoa_id is always the last token we appended
        assert full_ids[eoa_pos] == eoa_id
        outputs = model.model(input_ids=input_ids)
        h = outputs[0]  # [1, seq_len, hidden_dim], pre-LM-head
        return h[0, eoa_pos].float()

    all_dist_cos = []
    all_dist_l2 = []
    all_scores = {k: [] for k in ANCHOR_METRIC_KEYS}
    all_l2_to_expert_action = []
    all_scene_idx = []

    with torch.no_grad():
        for scene_i, (tok, scene) in enumerate(eligible):
            entry = metric_raw[tok]
            traj_scores = entry["trajectory_scores"][0]
            stacked = {k: np.asarray(traj_scores[k]) for k in ANCHOR_METRIC_KEYS}
            num_traj = len(stacked[ANCHOR_METRIC_KEYS[0]])
            assert num_traj == cluster.shape[0]

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
            except FileNotFoundError as e:
                print(f"  [scene {scene_i}] skipping, missing vq file: {e}")
                continue

            prompt_text = scene["text"][CUR_IDX]
            context_str = tokenizer.bos_token + prompt_text + format_video_prompt(cur_frame)
            context_ids = tokenizer(context_str, add_special_tokens=False)["input_ids"]

            expert_action_norm = scene["action"][CUR_IDX:CUR_IDX + HORIZON]
            expert_action_seq = tokenize_action(expert_action_norm)
            h_expert = hidden_state_at_eoa(context_ids, expert_action_seq)

            for ci in cand_idx:
                abs_traj = cluster[ci]
                rel_action_raw = absolute_to_relative_se2(abs_traj)
                rel_action_norm = 2 * (rel_action_raw - low) / (high - low) - 1
                rel_action_norm = np.clip(rel_action_norm, -1.0, 1.0)

                action_seq = tokenize_action(rel_action_norm)
                h_cand = hidden_state_at_eoa(context_ids, action_seq)

                cos_sim = F.cosine_similarity(h_cand.unsqueeze(0), h_expert.unsqueeze(0)).item()
                all_dist_cos.append(1.0 - cos_sim)
                all_dist_l2.append(torch.norm(h_cand - h_expert).item())

                for k in ANCHOR_METRIC_KEYS:
                    all_scores[k].append(float(stacked[k][ci]))
                l2_action = np.linalg.norm((rel_action_norm - expert_action_norm).reshape(-1))
                all_l2_to_expert_action.append(l2_action)
                all_scene_idx.append(scene_i)

            print(f"  [scene {scene_i}] token={tok} scored {len(cand_idx)} candidates")

    all_dist_cos = np.array(all_dist_cos)
    all_dist_l2 = np.array(all_dist_l2)
    combined_score = (
        5 * np.array(all_scores["time_to_collision_within_bound"])
        + 2 * np.array(all_scores["comfort"])
        + 5 * np.array(all_scores["ego_progress"])
    ) * np.clip(all_scores["no_at_fault_collisions"], 1e-6, None) * np.clip(all_scores["drivable_area_compliance"], 1e-6, None)

    print("\n=== Results ===")
    print(f"N candidates scored: {len(all_dist_cos)} across {len(eligible)} scenes")

    def report(name, x, y):
        sp, sp_p = spearmanr(x, y)
        pr, pr_p = pearsonr(x, y)
        print(f"  {name}: spearman={sp:.3f} (p={sp_p:.3g})  pearson={pr:.3f} (p={pr_p:.3g})")

    print("--- cosine hidden-state distance to expert-action rollout ---")
    report("vs combined PDM-like score", all_dist_cos, combined_score)
    for k in ANCHOR_METRIC_KEYS:
        report(f"vs {k}", all_dist_cos, np.array(all_scores[k]))

    print("--- L2 hidden-state distance to expert-action rollout ---")
    report("vs combined PDM-like score", all_dist_l2, combined_score)

    print("\n  --- confound check ---")
    report("cos_dist vs L2 distance to expert action (normalized action space)", all_dist_cos, np.array(all_l2_to_expert_action))

    out_path = REPO_ROOT / "docs/stage0_5b_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "dist_cos": all_dist_cos.tolist(),
            "dist_l2": all_dist_l2.tolist(),
            "combined_score": combined_score.tolist(),
            "scores": all_scores,
            "l2_to_expert_action": all_l2_to_expert_action,
            "scene_idx": all_scene_idx,
        }, f)
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    main()
