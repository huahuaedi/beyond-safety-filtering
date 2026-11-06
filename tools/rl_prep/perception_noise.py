"""Synthetic perception errors for the scene state that the geometric features read (a sensitivity analysis).

The ego pose and the candidate trajectories are left untouched. Only what a perception stack would have to provide is perturbed:
agent positions, headings, velocities and sizes, missed agents, and a rigid offset of the drivable-area map. The levels are round
numbers chosen to span "good" to "poor" perception; they are not calibrated to a particular detector.
"""
import hashlib

import numpy as np

# level: position sigma (m), heading sigma (rad), velocity sigma (m/s), relative size sigma, miss rate, map offset sigma (m)
LEVELS = {
    0: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    1: (0.3, 0.05, 0.3, 0.05, 0.03, 0.2),
    2: (0.6, 0.10, 0.6, 0.08, 0.08, 0.5),
    3: (1.0, 0.20, 1.0, 0.12, 0.15, 1.0),
}


def scene_rng(token, level, seed=0):
    digest = hashlib.sha256(f"{token}:perception:{level}:{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def perturb_agents(agents, level, rng):
    """Return a perturbed copy of the agent dict (global frame), or None when every agent is missed."""
    if agents is None or level == 0:
        return agents
    pos_s, head_s, vel_s, size_s, miss, _ = LEVELS[level]
    n = len(agents["pos"])
    keep = rng.random(n) >= miss
    if not keep.any():
        return None
    out = {}
    out["pos"] = np.asarray(agents["pos"], float)[keep] + rng.normal(0, pos_s, (int(keep.sum()), 2))
    out["vel"] = np.asarray(agents["vel"], float)[keep] + rng.normal(0, vel_s, (int(keep.sum()), 2))
    out["heading"] = np.asarray(agents["heading"], float)[keep] + rng.normal(0, head_s, int(keep.sum()))
    scale = lambda a: np.asarray(a, float)[keep] * np.clip(1.0 + rng.normal(0, size_s, int(keep.sum())), 0.5, 1.5)
    out["length"], out["width"] = scale(agents["length"]), scale(agents["width"])
    return out


def map_offset(level, rng):
    return rng.normal(0.0, LEVELS[level][5], 2)


def shift_predicate(inside, offset):
    """The same drivable-area predicate with the map displaced by ``offset`` (query points are moved by +offset)."""
    offset = np.asarray(offset, float)
    return lambda points: inside(np.asarray(points, float) + offset)
