"""Controlled scene-level diagnostics; existing 200 scenes are exploratory data."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def transform_features(arrays, condition, seed):
    action, l1, l2, labels, names = arrays
    if condition == 'shuffled_fusion':
        l1, l2 = l1.copy(), l2.copy()
        for i, name in enumerate(names):
            value = int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'little')
            order = np.random.default_rng(value).permutation(l1.shape[1])
            l1[i], l2[i] = l1[i, order], l2[i, order]
    return action, l1, l2, labels, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[32, 64, 128])
    parser.add_argument('--conditions', nargs='+', default=['action', 'layer1', 'fusion', 'linear_layer1', 'shuffled_fusion'])
    args = parser.parse_args()
    import torch
    import train_world_action_risk_ranker as ranker
    from risk_ranker_splits import inner_scene_split
    original_load, original_model = ranker.load_paths, ranker.RiskRanker
    ranker.CV_FOLD, ranker.CV_FOLDS, ranker.SEED = args.fold, 5, args.seed
    # Keep inner selection scenes AND nested training subsets identical across seeds.
    def fixed_split(paths, seed=0):
        train, val = inner_scene_split(paths, seed=0)
        train = sorted(train, key=lambda p: hashlib.sha256(p.stem.encode()).digest())
        if size > len(train):
            raise ValueError(f'requested {size} training scenes; only {len(train)} available')
        return sorted(train[:size]), val
    ranker.inner_scene_split = fixed_split
    class LinearLayer(torch.nn.Module):
        def __init__(self, mode):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.LayerNorm(4096, elementwise_affine=False), torch.nn.Linear(4096, 4))
        def forward(self, action, layer1, layer2):
            return self.net(layer1)
    for size in args.sizes:
        for condition in args.conditions:
            if condition not in ('action', 'layer1', 'fusion', 'linear_layer1', 'shuffled_fusion'):
                raise ValueError(condition)
            ranker.FEATURE_MODE = 'fusion' if condition == 'shuffled_fusion' else 'layer1' if condition == 'linear_layer1' else condition
            ranker.RiskRanker = LinearLayer if condition == 'linear_layer1' else original_model
            ranker.load_paths = lambda paths: transform_features(original_load(paths), condition, args.seed)
            ranker.OUTPUT_ROOT = args.output / condition / f'n_{size}'
            ranker.main()
            directory = ranker.OUTPUT_ROOT / ranker.FEATURE_MODE / f'seed_{args.seed}' / f'fold_{args.fold}'
            (directory/'diagnostic_config.json').write_text(json.dumps(dict(condition=condition, train_size=size, seed=args.seed, fold=args.fold, split_seed=0, interpretation='exploratory; previously inspected scenes'), indent=2))


if __name__ == '__main__':
    main()
