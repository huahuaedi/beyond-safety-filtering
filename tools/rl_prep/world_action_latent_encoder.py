"""Reusable action-conditioned NuPlan world-model latent encoder."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "reference" / "Emu3"))

from emu3.mllm import Emu3MoE, Emu3Tokenizer  # noqa: E402


PATH_PREFIXES = (
    "/mnt/vdb1/yingyan.li/repo/VLA/data/navsim/processed_data/",
    "/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/data/navsim/processed_data/",
    "/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/data/navsim/processed_data/",
)


class WorldActionLatentEncoder:
    """Encode arbitrary relative trajectories with their scene context."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        scene_pkl: str | Path,
        fast_tokenizer: str | Path,
        norm_stats: str | Path,
        vq_root: str | Path,
        layers=(-1, -9, -17, -25),
        batch_size: int = 8,
        device: str | torch.device = "cuda",
    ):
        self.checkpoint = Path(checkpoint)
        self.vq_root = Path(vq_root)
        self.layers = tuple(int(x) for x in layers)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        for name in ("config.json", "tokenizer_config.json", "model.safetensors.index.json"):
            if not (self.checkpoint / name).is_file():
                raise FileNotFoundError(self.checkpoint / name)
        self.tokenizer = Emu3Tokenizer.from_pretrained(
            str(self.checkpoint), model_max_length=4000,
            padding_side="right", use_fast=False,
        )
        self.model = Emu3MoE.from_pretrained(
            str(self.checkpoint), attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        ).to(self.device).eval()
        self.action_tokenizer = AutoProcessor.from_pretrained(
            str(fast_tokenizer), trust_remote_code=True
        )
        with Path(norm_stats).open() as handle:
            stats = json.load(handle)["norm_stats"]["libero"]
        self.low = np.asarray(stats["q01"], dtype=np.float32)
        self.high = np.asarray(stats["q99"], dtype=np.float32)
        with Path(scene_pkl).open("rb") as handle:
            scenes = pickle.load(handle)
        self.scenes = {scene["token"]: scene for scene in scenes}
        self.boa_id = self.tokenizer.encode(self.tokenizer.boa_token)[0]
        self.eoa_id = self.tokenizer.encode(self.tokenizer.eoa_token)[0]

    def _resolve_vq_path(self, raw_path: str) -> Path:
        for old in PATH_PREFIXES:
            if raw_path.startswith(old):
                return Path(raw_path.replace(old, str(self.vq_root) + "/"))
        return Path(raw_path)

    def _video_prompt(self, vq_tokens: np.ndarray) -> str:
        frames, height, width = vq_tokens.shape
        frame_strings = []
        for frame in vq_tokens:
            rows = [
                "".join(f"<|visual token {token:0>6d}|>" for token in row)
                for row in frame
            ]
            frame_strings.append(self.tokenizer.eol_token.join(rows))
        video = self.tokenizer.eof_token.join(frame_strings)
        return (
            self.tokenizer.boi_token + f"{frames}*{height}*{width}"
            + self.tokenizer.img_token + video + self.tokenizer.eof_token
            + self.tokenizer.eoi_token
        )

    def encode(self, scene_token: str, relative_trajectories: np.ndarray) -> torch.Tensor:
        relative = np.asarray(relative_trajectories, dtype=np.float32)
        if relative.ndim != 3 or relative.shape[1:] != (8, 3):
            raise ValueError(f"expected [N,8,3], got {relative.shape}")
        if scene_token not in self.scenes:
            raise KeyError(f"scene token absent from WM metadata: {scene_token}")
        scene = self.scenes[scene_token]
        vq = np.load(self._resolve_vq_path(scene["image"][3])).reshape(1, 18, 32)
        context = (
            self.tokenizer.bos_token + scene["text"][3] + self._video_prompt(vq)
        )
        context_ids = self.tokenizer(context, add_special_tokens=False)["input_ids"]
        normalized = 2.0 * (relative - self.low) / (self.high - self.low) - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)
        last_vocab_idx = self.tokenizer.pad_token_id - 1
        sequences, positions = [], []
        for action in normalized:
            ids = self.action_tokenizer(torch.tensor(action, dtype=torch.float32))[0]
            sequence = context_ids + [self.boa_id] + [last_vocab_idx - int(x) for x in ids] + [self.eoa_id]
            sequences.append(sequence)
            positions.append(len(sequence) - 1)

        chunks = []
        resolved_layers = None
        with torch.inference_mode():
            for start in range(0, len(sequences), self.batch_size):
                batch = sequences[start:start + self.batch_size]
                batch_positions = positions[start:start + self.batch_size]
                width = max(map(len, batch))
                input_ids = torch.full(
                    (len(batch), width), self.tokenizer.pad_token_id,
                    dtype=torch.long, device=self.device,
                )
                attention_mask = torch.zeros_like(input_ids)
                for row, sequence in enumerate(batch):
                    input_ids[row, :len(sequence)] = torch.tensor(sequence, device=self.device)
                    attention_mask[row, :len(sequence)] = 1
                output = self.model.model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, use_cache=False,
                )
                hidden = output.hidden_states
                if resolved_layers is None:
                    resolved_layers = [x if x >= 0 else len(hidden) + x for x in self.layers]
                    if any(x < 0 or x >= len(hidden) for x in resolved_layers):
                        raise ValueError(f"invalid layers {self.layers} for {len(hidden)} states")
                rows = torch.arange(len(batch), device=self.device)
                pos = torch.tensor(batch_positions, device=self.device)
                chunks.append(torch.stack([hidden[layer][rows, pos] for layer in resolved_layers], dim=1))
        return torch.cat(chunks, dim=0).float()
