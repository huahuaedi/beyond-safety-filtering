"""Label-independent inner scene split for outer-fold model selection."""
import hashlib


def inner_scene_split(paths, *, seed=0, validation_fraction=0.2):
    """Split whole scenes, never candidates; no outer-fold labels are inputs.

    This is scene isolation, not driving-log isolation: the candidate cache
    does not provide a log grouping here. Correlated scenes may remain.
    """
    paths = list(paths)
    if len(paths) < 2 or not 0 < validation_fraction < 1:
        raise ValueError("need >=2 training scenes and a validation fraction in (0, 1)")
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("duplicate scene tokens")
    ordered = sorted(paths, key=lambda p: hashlib.sha256(f"{seed}:{p.stem}".encode()).digest())
    count = min(len(paths) - 1, max(1, round(len(paths) * validation_fraction)))
    return sorted(ordered[count:]), sorted(ordered[:count])
