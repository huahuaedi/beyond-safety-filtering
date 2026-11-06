"""Low-dimensional world-model features for the candidate selector (CPU, high memory).

Reads the cached action-conditioned latents ``<latent-root>/<split>/<token>.npz``
(``eoa_latents`` [64, 4, 4096], layers -1/-9/-17/-25) and writes, per layer,
``<out-root>/L<i>/<split>/<token>.npz`` with ``features`` [64, k]: the projection of the
scene-centred latent onto the top-k principal axes.

The PCA is fitted ONLY on scenes in the training manifest (log-disjoint from every other
scene), on latents centred within each scene.  Centring removes what is common to the 64
candidates (the visual context), so the features describe how a candidate's action changes
the world-model state, which is the information the selector could not get from the
trajectory or the geometry.  No layer is chosen by held-out performance: all requested
layers are written and reported.

Usage:
    python extract_wm_pca_features.py --latent-root L --manifest M --out-root O \
        [--layers 0,1,2,3] [--k 32] [--sample-per-scene 16] [--limit N]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")


def load_centred(path, layers):
    with np.load(path, allow_pickle=False) as z:
        eoa = z["eoa_latents"]
    if eoa.ndim != 3:
        raise ValueError(f"{path}: expected [C, L, D] eoa_latents, got {eoa.shape}")
    eoa = eoa.astype(np.float32)
    eoa = eoa - eoa.mean(axis=0, keepdims=True)
    return eoa[:, layers, :]


def fit_pca(sample, k, seed):
    from sklearn.decomposition import PCA
    sd = sample.std(axis=0) + 1e-6
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed).fit(sample / sd)
    return pca, sd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--layers", default="0,1,2,3", help="indices into the stored layer axis")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--sample-per-scene", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="use at most this many scenes (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    layers = [int(v) for v in args.layers.split(",")]
    rng = np.random.default_rng(args.seed)

    train_tokens = {r["token"] for r in json.loads(args.manifest.read_text())["train"]}
    files = sorted(p for s in SPLITS for p in (args.latent_root / s).glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no latent files under {args.latent_root}")
    fit_files = [p for p in files if p.stem in train_tokens]
    if not fit_files:
        raise SystemExit("no training-manifest scenes among the latent files")

    rows = [[] for _ in layers]
    for p in fit_files:
        x = load_centred(p, layers)
        idx = rng.choice(len(x), min(args.sample_per_scene, len(x)), replace=False)
        for j in range(len(layers)):
            rows[j].append(x[idx, j])
    models = []
    for j, layer in enumerate(layers):
        sample = np.concatenate(rows[j])
        pca, sd = fit_pca(sample, min(args.k, len(sample) - 1, sample.shape[1]), args.seed)
        models.append((pca, sd))
        print(f"layer idx {layer}: fitted on {len(sample)} rows from {len(fit_files)} train scenes, "
              f"explained variance {pca.explained_variance_ratio_.sum():.3f}", flush=True)
    del rows

    for n, p in enumerate(files, 1):
        x = load_centred(p, layers)
        for j, layer in enumerate(layers):
            pca, sd = models[j]
            feats = pca.transform(x[:, j] / sd).astype(np.float32)
            feats /= np.sqrt(pca.explained_variance_[None, :] + 1e-9)       # unit scale per axis
            out = args.out_root / f"L{layer}" / p.parent.name / p.name
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, features=feats, scene_token=np.asarray(p.stem))
            os.replace(tmp, out)
        if n % 500 == 0:
            print(f"{n}/{len(files)} scenes written", flush=True)
    print(json.dumps({"scenes": len(files), "fit_scenes": len(fit_files), "layers": layers, "k": args.k}))


if __name__ == "__main__":
    main()
