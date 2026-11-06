"""Feature-block ablations that decide whether the world model adds risk signal.

The frozen WM latent of a candidate mixes (a) scene identity, (b) the candidate
action and (c) whatever the WM "imagines" about consequences.  Only (c) can make
a WM worth using, so the blocks separate them and add a strong non-WM baseline:

  action    the 24-d relative trajectory (identical to the existing baseline)
  geom      constant-velocity clearance + drivable-area features (no WM)
  layer1    raw WM latent of one depth (as in the existing layer1 mode)
  scene1    the scene mean of the latent: action-independent scene context
  resid1    latent minus scene mean: the action-specific part of the WM

Modes combine blocks; the decisive contrast is ``action_geom`` vs
``action_geom_wm1`` (does the WM help once a cheap geometric baseline exists?).

Folds, inner selection split, metrics and artifact layout are shared with
``train_world_action_risk_ranker.py`` so the outputs feed the existing mining
and manifest tools unchanged.  Optionally each trained model is also scored on
a disjoint "fresh" scene set (``WM_FRESH_*``), which is the cleanest test.
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import geometric_features as gf
import train_world_action_risk_ranker as base
from risk_ranker_splits import inner_scene_split

MODES = {
    "action": ("action",),
    "geom": ("geom",),
    "action_geom": ("action", "geom"),
    "action_scene1": ("action", "scene1"),
    "resid1": ("resid1",),
    "action_geom_wm1": ("action", "geom", "layer1"),
    "action_geom_resid1": ("action", "geom", "resid1"),
}
LATENT_BLOCKS = ("layer1", "scene1", "resid1")
LATENT_DIM = 4096
NUM_CANDIDATES = 64

MODE = os.environ.get("WM_BASELINE_MODE", "")
LATENT_LAYER = int(os.environ.get("WM_BASELINE_LAYER", 1))  # same depth as the existing "layer1"
GEOM_ROOT = Path(os.environ.get("WM_GEOM_ROOT", base.REPO_ROOT / "navsim_data/exp/world_action_geom_v1"))
FRESH_DATA_ROOT = os.environ.get("WM_FRESH_DATA_ROOT")  # contains split dirs, e.g. world_action_full_v1
FRESH_LATENT_ROOT = os.environ.get("WM_FRESH_LATENT_ROOT")
FRESH_GEOM_ROOT = os.environ.get("WM_FRESH_GEOM_ROOT")
FRESH_SPLIT = os.environ.get("WM_FRESH_SPLIT", "test")


def load_scenes(label_files, latent_root, geom_root, blocks, layer=LATENT_LAYER):
    """Stack per-scene arrays for the requested blocks: dict of [S, 64, *] arrays."""
    need_latent = any(b in blocks for b in LATENT_BLOCKS)
    need_geom = "geom" in blocks
    stacks = {"action": [], "labels": [], "latent": [], "geom": []}
    names = []
    for path in sorted(label_files):
        split = path.parent.name
        with np.load(path, allow_pickle=False) as data:
            action = data["relative_trajectories"].reshape(-1, 24).astype(np.float32)
            labels = base.label_matrix(data)
        if len(action) != NUM_CANDIDATES:
            raise ValueError(f"{path}: expected {NUM_CANDIDATES} candidates, got {len(action)}")
        stacks["action"].append(action)
        stacks["labels"].append(labels)
        if need_latent:
            with np.load(Path(latent_root) / split / path.name, allow_pickle=False) as data:
                latent = data["eoa_latents"]
            if latent.shape != (NUM_CANDIDATES, 4, LATENT_DIM):
                raise ValueError(f"{path.name}: unexpected latent shape {latent.shape}")
            stacks["latent"].append(latent[:, layer].astype(np.float32))
        if need_geom:
            with np.load(Path(geom_root) / split / path.name, allow_pickle=False) as data:
                if tuple(str(n) for n in data["feature_names"]) != gf.FEATURE_NAMES:
                    raise ValueError(f"{path.name}: geometric feature names differ from geometric_features.py")
                if str(data["scene_token"]) != path.stem:
                    raise ValueError(f"{path.name}: geometric features belong to {data['scene_token']}")
                features = data["geom_features"].astype(np.float32)
            if features.shape != (NUM_CANDIDATES, len(gf.FEATURE_NAMES)) or not np.isfinite(features).all():
                raise ValueError(f"{path.name}: invalid geometric features {features.shape}")
            stacks["geom"].append(features)
        names.append(path.stem)
    if not names:
        raise ValueError("no scene files supplied")
    out = {k: np.stack(v) for k, v in stacks.items() if v}
    out["names"] = names
    return out


class BlockRanker(nn.Module):
    """Per-block encoders -> fusion -> one head per risk label (as the V2 ranker)."""

    def __init__(self, blocks):
        super().__init__()
        unknown = [b for b in blocks if b not in ("action", "geom") + LATENT_BLOCKS]
        if unknown:
            raise ValueError(f"unknown blocks {unknown}")
        self.blocks = tuple(blocks)
        encoders, width = {}, 0
        for block in self.blocks:
            if block == "action":
                encoders[block], out = base.Encoder(24, 128), 128
            elif block == "geom":
                # Features are already bounded; per-sample LayerNorm would erase their scale.
                encoders[block] = nn.Sequential(
                    nn.Linear(len(gf.FEATURE_NAMES), 64), nn.GELU(), nn.Linear(64, 128), nn.GELU()
                )
                out = 128
            else:
                encoders[block], out = base.Encoder(LATENT_DIM, 256), 256
            width += out
        self.encoders = nn.ModuleDict(encoders)
        self.fusion = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Dropout(0.1))
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1)) for _ in base.LABEL_NAMES]
        )

    @staticmethod
    def block_input(block, batch):
        if block == "action":
            return batch["action"]
        if block == "geom":
            return batch["geom"]
        latent = batch["latent"]
        scene_mean = latent.mean(dim=1, keepdim=True)  # over the 64 candidates of a scene
        if block == "layer1":
            return latent
        if block == "scene1":
            return scene_mean.expand_as(latent)
        return latent - scene_mean  # resid1

    def forward(self, batch):
        hidden = self.fusion(torch.cat([self.encoders[b](self.block_input(b, batch)) for b in self.blocks], dim=-1))
        return torch.cat([head(hidden) for head in self.heads], dim=-1)  # [B, C, L]


def tensor_keys(blocks):
    keys = []
    if "action" in blocks:
        keys.append("action")
    if "geom" in blocks:
        keys.append("geom")
    if any(b in blocks for b in LATENT_BLOCKS):
        keys.append("latent")
    return keys + ["labels"]


def predict(model, data, device, scenes_per_batch=16):
    keys = tensor_keys(model.blocks)[:-1]
    out = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data["names"]), scenes_per_batch):
            batch = {k: torch.from_numpy(data[k][start:start + scenes_per_batch]).to(device) for k in keys}
            out.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(out)


def evaluate(model, data, device):
    return base.metrics_from_probability(data["labels"], predict(model, data, device))


def train_model(mode_blocks, train, val, device, seed, epochs, log=print):
    model = BlockRanker(mode_blocks).to(device)
    positives = train["labels"].sum(axis=(0, 1))
    total = train["labels"].shape[0] * train["labels"].shape[1]
    pos_weight = torch.tensor((total - positives) / np.maximum(positives, 1.0), dtype=torch.float32, device=device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=1e-4)
    keys = tensor_keys(mode_blocks)
    loader = DataLoader(
        TensorDataset(*[torch.from_numpy(train[k]) for k in keys]),
        batch_size=base.SCENES_PER_BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    best_state, best_score, best_epoch = None, -float("inf"), -1
    for epoch in range(epochs):
        model.train()
        losses = []
        for tensors in loader:
            tensors = [t.to(device) for t in tensors]
            batch, targets = dict(zip(keys[:-1], tensors[:-1])), tensors[-1]
            logits = model(batch)
            loss = bce_loss(logits, targets) + base.RANKING_COEF * base.pairwise_ranking_loss(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        score = base.selection_score(evaluate(model, val, device))
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(f"epoch={epoch:02d} loss={np.mean(losses):.6f} val_selection={score:.6f}")
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, best_state, best_score, best_epoch


def fresh_files(dev_names):
    """Scenes of the fresh split that never appear in the development pool."""
    if not FRESH_DATA_ROOT:
        return []
    files = [p for p in sorted((Path(FRESH_DATA_ROOT) / FRESH_SPLIT).glob("*.npz")) if p.stem not in dev_names]
    if not files:
        raise ValueError(f"no fresh scenes under {FRESH_DATA_ROOT}/{FRESH_SPLIT} outside the development pool")
    return files


def main():
    if MODE not in MODES:
        raise SystemExit(f"WM_BASELINE_MODE must be one of {sorted(MODES)}, got {MODE!r}")
    if base.CV_FOLD < 0 or base.CV_FOLD >= base.CV_FOLDS:
        raise SystemExit("WM_RISK_CV_FOLD must select an outer fold in [0, WM_RISK_CV_FOLDS)")
    blocks = MODES[MODE]
    random.seed(base.SEED)
    np.random.seed(base.SEED)
    torch.manual_seed(base.SEED)
    device = base.DEVICE
    output_dir = base.OUTPUT_ROOT / MODE / f"seed_{base.SEED}" / f"fold_{base.CV_FOLD}"
    if (output_dir / "metrics.json").exists():
        raise SystemExit(f"refusing to overwrite {output_dir}")
    print(f"mode={MODE} blocks={blocks} seed={base.SEED} fold={base.CV_FOLD} device={device}", flush=True)

    paths, assignments = base.stratified_development_folds()
    outer_train = [p for p, f in zip(paths, assignments) if f != base.CV_FOLD]
    outer_paths = [p for p, f in zip(paths, assignments) if f == base.CV_FOLD]
    train_paths, val_paths = inner_scene_split(outer_train, seed=base.SEED)
    load = lambda files: load_scenes(files, base.LATENT_ROOT, GEOM_ROOT, blocks)
    train, val = load(train_paths), load(val_paths)

    model, best_state, best_score, best_epoch = train_model(
        blocks, train, val, device, base.SEED, base.EPOCHS, log=lambda m: print(m, flush=True)
    )
    outer = load(outer_paths)  # outer labels are touched only after selection
    results = {
        "feature_mode": MODE, "blocks": list(blocks), "latent_layer": LATENT_LAYER, "seed": base.SEED,
        "cv_fold": base.CV_FOLD, "cv_folds": base.CV_FOLDS,
        "selection_protocol": "inner_scene_holdout_v1",
        "training_scenes": train["names"], "selection_scenes": val["names"], "outer_scenes": outer["names"],
        "ranking_coef": base.RANKING_COEF, "ranking_margin": base.RANKING_MARGIN,
        "best_epoch": best_epoch, "best_val_selection": best_score,
        "val": evaluate(model, val, device), "outer": evaluate(model, outer, device),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outer_probability = predict(model, outer, device)
    np.savez_compressed(
        output_dir / "oof_predictions.npz",
        scene_names=np.asarray(outer["names"]), probabilities=outer_probability, labels=outer["labels"],
        selection_protocol=np.asarray("inner_scene_holdout_v1"),
        training_scene_names=np.asarray(train["names"]), selection_scene_names=np.asarray(val["names"]),
    )

    dev_names = {p.stem for p in paths}
    fresh = fresh_files(dev_names)
    if fresh:
        fresh_data = load_scenes(fresh, FRESH_LATENT_ROOT or base.LATENT_ROOT, FRESH_GEOM_ROOT or GEOM_ROOT, blocks)
        if set(fresh_data["names"]) & set(train["names"] + val["names"] + outer["names"]):
            raise RuntimeError("fresh scenes overlap the development pool")
        fresh_probability = predict(model, fresh_data, device)
        np.savez_compressed(
            output_dir / "fresh_predictions.npz",
            scene_names=np.asarray(fresh_data["names"]), probabilities=fresh_probability, labels=fresh_data["labels"],
        )
        results["fresh"] = base.metrics_from_probability(fresh_data["labels"], fresh_probability)
        results["fresh_scenes"] = len(fresh_data["names"])

    torch.save({"model": best_state, "config": {"feature_mode": MODE, "blocks": list(blocks),
                                                "seed": base.SEED, "best_epoch": best_epoch}},
               output_dir / "ranker.pt")
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=True))
    print(json.dumps(results["outer"], indent=2), flush=True)
    print(f"saved={output_dir}", flush=True)


if __name__ == "__main__":
    main()
