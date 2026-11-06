"""Turn the policy's scene-level context vectors into per-candidate feature blocks for fit_and_apply_selector.py (--*-extra-root pv=DIR).

PCA is fitted on the training pool's contexts only and applied to the application cohort. The reduced vector is repeated for every candidate of
a scene (the boosted selector sees raw as well as scene-centred features, so a scene constant can interact with trajectory features).
Usage: make_policy_visual_blocks.py --contexts DIR_POOL --candidate-root POOL --out POOL_PV --fit
       make_policy_visual_blocks.py --contexts DIR_NAVTEST --candidate-root COLLECTION --out NAVTEST_PV --pca PCA.npz
"""
import argparse
from pathlib import Path

import numpy as np


def load_contexts(directory):
    tokens, vectors = [], []
    for p in sorted(Path(directory).glob("context_shard*.npz")):
        with np.load(p, allow_pickle=False) as z:
            tokens += [str(t) for t in z["tokens"]]
            vectors.append(z["contexts"].astype(np.float32))
    if not vectors:
        raise FileNotFoundError(f"no context_shard*.npz under {directory}")
    if len(set(tokens)) != len(tokens):
        raise ValueError("a scene appears in two shards")
    return tokens, np.concatenate(vectors)


def fit_pca(x, k):
    mean = x.mean(0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    comps = vt[:k]
    scale = s[:k] / np.sqrt(len(x) - 1)
    return {"mean": mean, "components": comps, "scale": scale}


def project(pca, x):
    return ((x - pca["mean"]) @ pca["components"].T) / np.maximum(pca["scale"], 1e-6)   # unit variance per component on the pool


def scene_files(root):
    return [p for split in ("train", "val", "test") for p in sorted((Path(root) / split).glob("*.npz"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, required=True)
    ap.add_argument("--candidate-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--fit", action="store_true", help="fit PCA on these contexts (training pool) and save it next to --out")
    ap.add_argument("--pca", type=Path, default=None, help="PCA saved by --fit")
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    tokens, x = load_contexts(args.contexts)
    if args.fit:
        pca = fit_pca(x, args.k)
        args.out.mkdir(parents=True)
        np.savez(args.out / "pca.npz", **pca)
    else:
        if args.pca is None:
            raise SystemExit("--pca is required unless --fit")
        with np.load(args.pca) as z:
            pca = {k: z[k] for k in z.files}
        args.out.mkdir(parents=True)
    reduced = dict(zip(tokens, project(pca, x)))
    n = 0
    for p in scene_files(args.candidate_root):
        if p.stem not in reduced:
            raise KeyError(f"no context for scene {p.stem}")
        with np.load(p, allow_pickle=False) as z:
            c = z["relative_trajectories"].shape[0]
        d = args.out / p.parent.name
        d.mkdir(exist_ok=True)
        np.savez(d / p.name, features=np.repeat(reduced[p.stem][None], c, axis=0).astype(np.float32))
        n += 1
    print(f"wrote {n} scene blocks of dimension {pca['components'].shape[0]} to {args.out}")


if __name__ == "__main__":
    main()
