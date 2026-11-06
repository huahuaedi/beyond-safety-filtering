"""Auditable, scene-local labels for offline waypoint flow matching.

Unlike online group-centered advantage, POSITIVE means absolute safety AND
verified utility improvement over the same scene's frozen-reference probes.
No positive is manufactured when a scene contains no improving safe action.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

UNCONDITIONAL, NEGATIVE, POSITIVE = 0, 1, 2
SAFETY = ("no_at_fault_collisions", "time_to_collision_within_bound",
          "drivable_area_compliance", "driving_direction_compliance")


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, parts)).encode()).digest()[:8], 'little') % (2**31)


def absolute_from_relative(relative):
    relative = np.asarray(relative, dtype=np.float64)
    out = np.zeros_like(relative)
    x = np.zeros(relative.shape[:-2]); y = x.copy(); yaw = x.copy()
    for t in range(relative.shape[-2]):
        dx, dy, da = np.moveaxis(relative[..., t, :], -1, 0)
        x, y = x + np.cos(yaw)*dx - np.sin(yaw)*dy, y + np.sin(yaw)*dx + np.cos(yaw)*dy
        yaw = (yaw + da + np.pi) % (2*np.pi) - np.pi
        out[..., t, :] = np.stack([x, y, yaw], -1)
    return out.astype(np.float32)


def normalize_action(action, low, high):
    low, high = np.asarray(low), np.asarray(high)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(high <= low):
        raise ValueError('invalid action normalization bounds')
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-2:] != (8, 3) or not np.isfinite(action).all():
        raise ValueError('action must be finite with trailing shape [8,3]')
    # Do not clip: that would silently train on a different trajectory from
    # the one that NAVSIM actually scored, especially for failure examples.
    return (2*(action-low)/(high-low)-1).astype(np.float32)


def verified_labels(scores, reference, margin=0.01):
    if not np.isfinite(margin) or margin < 0:
        raise ValueError('margin must be finite and nonnegative')
    progress = np.asarray(scores['ego_progress'], dtype=np.float64)
    if progress.ndim != 1 or not len(progress):
        raise ValueError('progress must be nonempty vector')
    safe = np.ones(len(progress), dtype=bool)
    for name in (*SAFETY, 'ego_progress'):
        values = np.asarray(scores[name], dtype=np.float64)
        if values.shape != progress.shape or not np.isfinite(values).all() or np.any((values < -1e-6) | (values > 1+1e-6)):
            raise ValueError(f'invalid metric: {name}')
        if name in SAFETY:
            safe &= values >= 1-1e-6
    ref_progress = np.asarray(reference['ego_progress'], dtype=np.float64)
    if ref_progress.ndim != 1 or not len(ref_progress) or not np.isfinite(ref_progress).all() or np.any((ref_progress < 0) | (ref_progress > 1)):
        raise ValueError('invalid reference progress')
    # DDC is a hard condition, not a progress compensation term.
    gain = progress - ref_progress.mean()
    positive = safe & (gain > margin)
    labels = np.where(positive, POSITIVE, NEGATIVE).astype(np.int64)
    return {'labels': labels, 'safe': safe, 'gain': gain.astype(np.float32)}


def pdms(nc, dac, ep, ttc, comfort):
    """NAVSIM v1 PDMS, as in the scorer shipped with this repo."""
    return nc * dac * (5.0 * ep + 5.0 * ttc + 2.0 * comfort) / 12.0


def topk_safe_labels(scores, reference, k=2, margin=0.01):
    """POSITIVE = the k highest-PDMS candidates that are not worse than the
    reference on any safety metric and beat the reference PDMS by > margin.

    Reference safety is the worse of the two reference probes (lenient, the
    same rule the headroom analysis used). If no candidate qualifies the scene
    has no positive; nothing is forced.
    """
    if k < 1 or not np.isfinite(margin) or margin < 0:
        raise ValueError('k must be >= 1 and margin finite and nonnegative')
    need = (*SAFETY, 'ego_progress', 'comfort')
    for name in need:
        if name not in scores or name not in reference:
            raise ValueError(f'topk_safe_labels needs {name} in scores and reference')
    cand = {n: np.asarray(scores[n], dtype=np.float64) for n in need}
    ref = {n: np.asarray(reference[n], dtype=np.float64) for n in need}
    size = len(cand['ego_progress'])
    for n in need:
        if cand[n].shape != (size,) or not size or not np.isfinite(cand[n]).all() \
                or np.any((cand[n] < -1e-6) | (cand[n] > 1 + 1e-6)):
            raise ValueError(f'invalid candidate metric: {n}')
        if ref[n].ndim != 1 or not len(ref[n]) or not np.isfinite(ref[n]).all() \
                or np.any((ref[n] < -1e-6) | (ref[n] > 1 + 1e-6)):
            raise ValueError(f'invalid reference metric: {n}')
    ref_mean = {n: float(ref[n].mean()) for n in need}
    ref_pdms = pdms(ref_mean['no_at_fault_collisions'], ref_mean['drivable_area_compliance'],
                    ref_mean['ego_progress'], ref_mean['time_to_collision_within_bound'],
                    ref_mean['comfort'])
    c_pdms = pdms(cand['no_at_fault_collisions'], cand['drivable_area_compliance'],
                  cand['ego_progress'], cand['time_to_collision_within_bound'], cand['comfort'])
    not_worse = np.ones(size, dtype=bool)
    for n in SAFETY:
        not_worse &= cand[n] >= ref[n].min() - 1e-6
    eligible = not_worse & (c_pdms > ref_pdms + margin)
    labels = np.full(size, NEGATIVE, dtype=np.int64)
    idx = np.flatnonzero(eligible)
    top = idx[np.argsort(-c_pdms[idx], kind='stable')[:k]]
    labels[top] = POSITIVE
    return {'labels': labels, 'eligible': eligible, 'gain': (c_pdms - ref_pdms).astype(np.float32)}


def build_schedule(records, label_fn, *, mode, candidates_per_scene, epochs, seed,
                   positive_fraction=None):
    """Deterministic (record, candidate, epoch) schedule.

    ``positive_fraction=None`` and a non-filtered mode reproduces the original
    random sampling. Otherwise each scene contributes about that share of
    POSITIVE candidates (when it has any), so top-k labels are actually seen.
    ``filtered`` trains only on POSITIVE candidates and skips scenes without one.
    conditioned and unconditional share the exact same schedule.
    """
    rng = np.random.default_rng(seed)
    out = []
    legacy = positive_fraction is None and mode != 'filtered'
    for epoch in range(epochs):
        for j in rng.permutation(len(records)):
            r = records[int(j)]
            if legacy:
                chosen = rng.permutation(r['candidates'])[:candidates_per_scene]
            else:
                labels = np.asarray(label_fn(r))
                pos = rng.permutation(np.flatnonzero(labels == POSITIVE))
                neg = rng.permutation(np.flatnonzero(labels != POSITIVE))
                if mode == 'filtered':
                    chosen = pos[:candidates_per_scene]
                else:
                    want = int(round(candidates_per_scene * (positive_fraction or 0.0)))
                    n_pos = min(len(pos), max(1, want)) if len(pos) else 0
                    chosen = rng.permutation(
                        np.concatenate([pos[:n_pos], neg[:candidates_per_scene - n_pos]]))
            out.extend((r, int(c), epoch) for c in chosen)
    return out


def read_scene(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        token, split = str(data['scene_token']), str(data['split'])
        if token != path.stem or split != path.parent.name:
            raise ValueError(f'scene identity mismatch: {path}')
        relative = data['relative_trajectories'].astype(np.float32)
        absolute = data['absolute_trajectories'].astype(np.float32)
        if relative.ndim != 3 or relative.shape[1:] != (8,3) or absolute.shape != relative.shape:
            raise ValueError(f'trajectory shape mismatch: {path}')
        if not np.isfinite(relative).all() or not np.isfinite(absolute).all():
            raise ValueError(f'nonfinite trajectory: {path}')
        reconstructed = absolute_from_relative(relative)
        if not np.allclose(reconstructed[...,:2], absolute[...,:2], atol=2e-4, rtol=1e-5):
            raise ValueError(f'scored trajectory differs from training target: {path}')
        heading_delta = reconstructed[...,2] - absolute[...,2]
        if np.max(np.abs(np.arctan2(np.sin(heading_delta), np.cos(heading_delta)))) > 2e-4:
            raise ValueError(f'scored heading differs from training target: {path}')
        names = (*SAFETY, 'ego_progress')
        scores = {k: data['score__'+k].astype(np.float32) for k in names}
        reference = {'ego_progress': data['reference_probe__ego_progress'].astype(np.float32)}
        # optional fields, needed only by topk_safe_labels
        if 'score__comfort' in data.files:
            scores['comfort'] = data['score__comfort'].astype(np.float32)
        for k in (*SAFETY, 'comfort'):
            if 'reference_probe__' + k in data.files:
                reference[k] = data['reference_probe__' + k].astype(np.float32)
    if len(scores['ego_progress']) != len(relative):
        raise ValueError('candidate/score count mismatch')
    return relative, scores, reference


def split_records(root, forbidden=()):
    root = Path(root)
    records = {s: sorted((root/s).glob('*.npz')) for s in ('train','val','test')}
    tokens = {s: {p.stem for p in paths} for s,paths in records.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if tokens[a] & tokens[b]:
            raise ValueError(f'overlapping {a}/{b} scenes')
    excluded = tokens['train'] & set(forbidden)
    records['train'] = [p for p in records['train'] if p.stem not in excluded]
    if not records['train'] or not records['val']:
        raise ValueError('empty training or validation split')
    return records, sorted(excluded)
