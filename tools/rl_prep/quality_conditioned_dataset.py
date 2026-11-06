"""Paired trajectory/WM-latent/quality-sidecar dataset interface."""

from pathlib import Path

import numpy as np


class QualityConditionedWorldActionDataset:
    """Flatten scenes into candidate examples while keeping files lazy."""

    def __init__(self, data_root, quality_root, latent_root=None, splits=("train",)):
        self.data_root = Path(data_root)
        self.quality_root = Path(quality_root)
        self.latent_root = Path(latent_root) if latent_root is not None else None
        self.scenes = []
        self.index = []
        for split in splits:
            for source in sorted((self.data_root / split).glob("*.npz")):
                quality = self.quality_root / split / source.name
                latent = self.latent_root / split / source.name if self.latent_root else None
                if not quality.is_file():
                    raise FileNotFoundError(f"missing quality sidecar: {quality}")
                if latent is not None and not latent.is_file():
                    raise FileNotFoundError(f"missing WM latent: {latent}")
                with np.load(source, allow_pickle=False) as data:
                    candidates = int(data["relative_trajectories"].shape[0])
                scene_id = len(self.scenes)
                self.scenes.append((source, quality, latent))
                self.index.extend((scene_id, candidate) for candidate in range(candidates))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        scene_id, candidate = self.index[item]
        source_path, quality_path, latent_path = self.scenes[scene_id]
        with np.load(source_path, allow_pickle=False) as source:
            result = {
                "scene_token": str(source["scene_token"]),
                "split": str(source["split"]),
                "candidate_index": candidate,
                "action": source["relative_trajectories"][candidate].astype(np.float32),
            }
        with np.load(quality_path, allow_pickle=False) as quality:
            result.update(
                quality_tier=int(quality["quality_tier"][candidate]),
                advantage=float(quality["advantage"][candidate]),
                advantage_bin=int(quality["advantage_bin"][candidate]),
                risk_mask=quality["risk_mask"][candidate].astype(np.bool_),
                condition_vector=quality["condition_vector"][candidate].astype(np.float32),
            )
        if latent_path is not None:
            with np.load(latent_path, allow_pickle=False) as latent:
                result["wm_latent"] = latent["eoa_latents"][candidate].astype(np.float32)
        return result
