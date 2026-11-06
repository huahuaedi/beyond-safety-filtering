"""Paired scene-weighted estimates with optional driving-log cluster bootstrap."""
import json
from pathlib import Path
import numpy as np

GATES = ('no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound')


def pass_status(rows, tokens, gates=GATES):
    values = np.asarray([[rows[t][g] for g in gates] for t in tokens], dtype=float)
    if not len(tokens) or not np.isfinite(values).all():
        raise ValueError('empty or nonfinite gate scores')
    return (values >= 1 - 1e-6).all(axis=1)


def conversions(base, arm, tokens, gates=GATES):
    b, a = pass_status(base, tokens, gates), pass_status(arm, tokens, gates)
    return (~b & a).astype(float), (b & ~a).astype(float)


def log_labels(tokens, mapping_path=None, cache_dir=None):
    if mapping_path:
        mapping = json.loads(Path(mapping_path).read_text())
    elif cache_dir:
        mapping = {}
        for p in Path(cache_dir).glob('*/*/*/metric_cache.pkl'):
            token, log = p.parent.name, p.parents[2].name
            if mapping.setdefault(token, log) != log:
                raise ValueError(f'conflicting log mapping for {token}')
    else:
        return None
    missing = [t for t in tokens if t not in mapping or not isinstance(mapping[t], str) or not mapping[t]]
    if missing:
        raise ValueError(f'{len(missing)} scenes have no valid log mapping, e.g. {missing[0]}')
    labels = [mapping[t] for t in tokens]
    if len(set(labels)) < 2:
        raise ValueError('at least two logs required for cluster bootstrap')
    return labels


def bootstrap(delta, rng, labels=None, n=4000):
    delta = np.asarray(delta, float)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError('invalid paired deltas')
    if labels is None:
        labels = np.arange(len(delta))
    if len(labels) != len(delta):
        raise ValueError('log labels must match scenes')
    _, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=delta)
    means = []
    for start in range(0, n, 128):
        idx = rng.integers(0, len(counts), size=(min(128, n-start), len(counts)))
        means.extend((sums[idx].sum(axis=1)/counts[idx].sum(axis=1)).tolist())
    return np.percentile(means, [2.5, 97.5]).tolist()
