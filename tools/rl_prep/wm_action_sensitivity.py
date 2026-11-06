"""How much does the world-model latent actually respond to the candidate action?

The WM was trained on expert logs, i.e. one action per scene, so it may never
have learned the *consequences* of alternative actions.  This diagnostic uses
the already-extracted end-of-action latents (64 candidates per scene, 4 depths)
and asks, per depth:

* ``within_scene_variance_fraction``: share of latent variance that varies
  between candidates of one scene (the rest is scene identity).
* ``action_latent_distance``: within-scene Spearman between pairwise action
  distance and pairwise latent distance (and the same with the latents shuffled
  across candidates as a null).
* ``near_pair_latent_ratio``: latent distance of the closest-action pairs
  relative to a typical pair.  ~1 means the latent is not smooth in the action.
* ``action_decodability_r2``: held-out-scene R^2 of a ridge probe from the
  scene-centred latent to the scene-centred action.
* ``latent_from_action_r2``: held-out R^2 of predicting the scene-centred latent
  from the scene-centred action (linear and quadratic).
* ``outcome_gap``: does latent distance predict the NAVSIM outcome gap between
  two candidates *beyond* what action distance already does (partial Spearman)?
* ``effective_rank``: participation ratio of the within-scene latent covariance.

Nothing here trains a risk model; it characterises the representation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORE_KEYS = (
    "score__no_at_fault_collisions",
    "score__drivable_area_compliance",
    "score__ego_progress",
    "score__time_to_collision_within_bound",
    "score__comfort",
)


def rankdata(values):
    """Average ranks (1-based), ties handled."""
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denominator = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denominator) if denominator > 0 else float("nan")


def partial_from_rhos(r_xy, r_xz, r_yz):
    """Partial correlation of x and y given z from the three pairwise rhos."""
    denominator = np.sqrt(max(1.0 - r_xz ** 2, 0.0) * max(1.0 - r_yz ** 2, 0.0))
    return float((r_xy - r_xz * r_yz) / denominator) if denominator > 1e-9 else float("nan")


def pairwise_l2(x):
    """Condensed upper-triangle Euclidean distances of the rows of x."""
    x = np.asarray(x, dtype=np.float64)
    squared = (x ** 2).sum(axis=1)
    gram = x @ x.T
    d2 = np.maximum(squared[:, None] + squared[None, :] - 2.0 * gram, 0.0)
    return np.sqrt(d2[np.triu_indices(len(x), k=1)])


def pairwise_absdiff(v):
    v = np.asarray(v, dtype=np.float64)
    return np.abs(v[:, None] - v[None, :])[np.triu_indices(len(v), k=1)]


def pdm_like_score(scores):
    """PDMS-shaped scalar from the per-candidate NAVSIM sub-scores."""
    nc, dac, ep, ttc, comfort = (np.asarray(scores[k], dtype=np.float64) for k in SCORE_KEYS)
    return nc * dac * (5.0 * ttc + 5.0 * ep + 2.0 * comfort) / 12.0


def select_scenes(paths, max_scenes, seed):
    ordered = sorted(paths, key=lambda p: hashlib.sha256(f"{seed}:{p.stem}".encode()).digest())
    return ordered[:max_scenes] if max_scenes else ordered


def load_scenes(data_root, latent_root, max_scenes, seed):
    files = sorted(p for split in ("train", "val", "test") for p in (data_root / split).glob("*.npz"))
    if not files:
        raise SystemExit(f"no scene files under {data_root}")
    chosen = select_scenes(files, max_scenes, seed)
    actions, latents, outcomes, names = [], [], [], []
    for path in chosen:
        latent_path = latent_root / path.parent.name / path.name
        with np.load(path, allow_pickle=False) as label, np.load(latent_path, allow_pickle=False) as lat:
            action = label["relative_trajectories"].reshape(len(label["relative_trajectories"]), -1)
            z = lat["eoa_latents"]
            if z.ndim != 3 or len(z) != len(action):
                raise ValueError(f"{path.name}: latent {z.shape} does not match {len(action)} candidates")
            outcomes.append(pdm_like_score({k: label[k] for k in SCORE_KEYS}))
            actions.append(action.astype(np.float32))
            latents.append(z.astype(np.float32))
            names.append(path.stem)
    return np.stack(actions), np.stack(latents), np.stack(outcomes), names


def within_scene_variance_fraction(z):
    """z: [S, C, D]. Fraction of total variance that is within-scene."""
    z = np.asarray(z, dtype=np.float64)
    within = ((z - z.mean(axis=1, keepdims=True)) ** 2).sum()
    total = ((z - z.mean(axis=(0, 1), keepdims=True)) ** 2).sum()
    return float(within / total)


def effective_rank(residual):
    """Participation ratio of the covariance spectrum of [N, D] residuals."""
    x = residual - residual.mean(axis=0, keepdims=True)
    n, d = x.shape
    gram = (x @ x.T if n < d else x.T @ x).astype(np.float64)
    eigenvalues = np.clip(np.linalg.eigvalsh(gram), 0.0, None)
    total = eigenvalues.sum()
    return float(total ** 2 / (eigenvalues ** 2).sum()) if total > 0 else float("nan")


def ridge_fit(x, y, alpha):
    gram = x.T @ x
    scale = np.trace(gram) / gram.shape[0]
    return np.linalg.solve(gram + alpha * scale * np.eye(gram.shape[0]), x.T @ y)


def r2_score(y, prediction):
    sse = ((y - prediction) ** 2).sum()
    sst = ((y - y.mean(axis=0, keepdims=True)) ** 2).sum()
    return float(1.0 - sse / sst)


def scene_split(num_scenes, seed, fractions=(0.6, 0.2)):
    order = np.random.default_rng(seed).permutation(num_scenes)
    a = int(fractions[0] * num_scenes)
    b = a + int(fractions[1] * num_scenes)
    return order[:a], order[a:b], order[b:]


def held_out_ridge_r2(x, y, seed, alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Scene-grouped ridge probe.  x: [S, C, Dx], y: [S, C, Dy] (both centred).

    Alpha is picked on a validation split of scenes, R^2 is reported on
    untouched test scenes.  Features are standardised with training statistics.
    """
    train, val, test = scene_split(len(x), seed)
    flat = lambda a, idx: a[idx].reshape(-1, a.shape[-1]).astype(np.float64)
    xt, yt = flat(x, train), flat(y, train)
    mean, std = xt.mean(axis=0), xt.std(axis=0) + 1e-6
    norm = lambda a: (a - mean) / std
    xt = norm(xt)
    best_alpha, best_score = None, -np.inf
    for alpha in alphas:
        weights = ridge_fit(xt, yt - yt.mean(axis=0), alpha)
        score = r2_score(flat(y, val), norm(flat(x, val)) @ weights + yt.mean(axis=0))
        if score > best_score:
            best_alpha, best_score = alpha, score
    weights = ridge_fit(xt, yt - yt.mean(axis=0), best_alpha)
    prediction = norm(flat(x, test)) @ weights + yt.mean(axis=0)
    return {"r2": r2_score(flat(y, test), prediction), "alpha": best_alpha, "val_r2": float(best_score)}


def quadratic_features(a):
    """[.., d] -> [.., d + d(d+1)/2] with a standardised linear + product basis."""
    d = a.shape[-1]
    i, j = np.triu_indices(d)
    return np.concatenate([a, a[..., i] * a[..., j]], axis=-1)


def distance_statistics(actions, latents, outcomes, seed, near_fraction=0.05):
    """Per-scene pairwise-distance analysis for one latent depth."""
    rng = np.random.default_rng(seed)
    rho_al, rho_null, near_ratio = [], [], []
    rho_zg, rho_ag, partial = [], [], []
    for a, z, o in zip(actions, latents, outcomes):
        d_action, d_latent = pairwise_l2(a), pairwise_l2(z)
        gap = pairwise_absdiff(o)
        r_al = spearman(d_action, d_latent)
        rho_al.append(r_al)
        rho_null.append(spearman(d_action, pairwise_l2(z[rng.permutation(len(z))])))
        k = max(int(near_fraction * len(d_action)), 1)
        near = np.argsort(d_action, kind="stable")[:k]
        near_ratio.append(float(np.median(d_latent[near]) / np.median(d_latent)))
        r_zg, r_ag = spearman(d_latent, gap), spearman(d_action, gap)
        rho_zg.append(r_zg)
        rho_ag.append(r_ag)
        partial.append(partial_from_rhos(r_zg, r_al, r_ag))
    summary = lambda v: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)), "median": float(np.nanmedian(v))}
    return {
        "action_latent_distance_spearman": summary(rho_al),
        "action_latent_distance_spearman_shuffled_null": summary(rho_null),
        "fraction_scenes_rho_gt_0.3": float(np.nanmean(np.asarray(rho_al) > 0.3)),
        "near_pair_latent_ratio": summary(near_ratio),
        "outcome_gap": {
            "latent_distance_spearman": summary(rho_zg),
            "action_distance_spearman": summary(rho_ag),
            "latent_partial_given_action": summary(partial),
        },
    }


def analyse_layer(actions, latents, outcomes, seed):
    """actions [S,C,24], latents [S,C,D], outcomes [S,C]."""
    centred_latent = latents - latents.mean(axis=1, keepdims=True)
    centred_action = actions - actions.mean(axis=1, keepdims=True)
    result = {
        "within_scene_variance_fraction": within_scene_variance_fraction(latents),
        "effective_rank": effective_rank(centred_latent.reshape(-1, latents.shape[-1])),
    }
    result.update(distance_statistics(actions, latents, outcomes, seed))
    result["action_decodability_r2"] = held_out_ridge_r2(centred_latent, centred_action, seed)
    result["latent_from_action_r2"] = {
        "linear": held_out_ridge_r2(centred_action, centred_latent, seed),
        "quadratic": held_out_ridge_r2(quadratic_features(centred_action), centred_latent, seed),
    }
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get(
        "WM_ACTION_DATA_ROOT", REPO_ROOT / "navsim_data/exp/world_action_v1")))
    ap.add_argument("--latent-root", type=Path, default=Path(os.environ.get(
        "WM_LATENT_ROOT", REPO_ROOT / "navsim_data/exp/world_action_latents_v1")))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-scenes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    actions, latents, outcomes, names = load_scenes(args.data_root, args.latent_root, args.max_scenes, args.seed)
    if len(names) < 10:
        raise SystemExit(f"need at least 10 scenes for grouped probes, got {len(names)}")
    print(f"scenes={len(names)} actions={actions.shape} latents={latents.shape}", flush=True)
    results = {"scenes": len(names), "seed": args.seed, "scene_names": names, "layers": {}}
    for layer in range(latents.shape[2]):
        results["layers"][str(layer)] = analyse_layer(actions, latents[:, :, layer, :], outcomes, args.seed)
        print(f"layer {layer}: " + json.dumps({
            k: results["layers"][str(layer)][k] for k in ("within_scene_variance_fraction", "effective_rank")
        }), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=True))
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
