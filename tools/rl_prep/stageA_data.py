"""Data side of the Stage A candidate evaluator (NumPy only): per-scene candidate inputs and labels, the visual-token shards, the
within-split cross-log shuffle used by the mismatched-token control, and the log-level tune/decision split. See docs/stageA_protocol.md."""
from pathlib import Path

import numpy as np

import analyze_trajectory_selector as A

EPS = 1e-6
GATES = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")
# continuous geometry targets: columns of the geometry block (12 base features + 3 indicators):
# 0 min_clearance, 2 mean_clearance, 8 offroad_frac_max, 9 offroad_frac_mean, 11 t_first_offroad
CONT_COLUMNS = (0, 2, 8, 9, 11)


def load_scene_data(path, geom_root):
    """One scene: trajectory inputs [C, 40], privileged geometry [C, 15], labels, and the reference probes' score."""
    path = Path(path)
    traj, ref_traj, scores, ref = A.load_scene(path)
    geom = A.geom_block(geom_root, path)
    ref_mean = ref_traj.mean(axis=0)
    dev = traj[..., :2] - ref_mean[None, :, :2]
    x = np.concatenate([traj.reshape(len(traj), -1), dev.reshape(len(traj), -1)], axis=1)
    return {
        "token": path.stem, "traj": x.astype(np.float32), "geom": geom.astype(np.float32),
        "pdms": A.scene_target(scores).astype(np.float32),
        "gates": np.stack([(scores[k] >= 1 - EPS) for k in GATES], axis=1).astype(np.float32),
        "cont": geom[:, list(CONT_COLUMNS)].astype(np.float32),
        "ref_pdms": float(A.scene_ref_pdms(ref)),
        "ref_gate_fail": np.array([float(ref[k].min() < 1 - EPS) for k in GATES], dtype=np.float32),
    }


def load_token_shards(directory):
    """tokens -> row of h [N, 1152, 128] (float16), plus the (frame, row, col) grid shared by all scenes."""
    names, arrays, grid, fingerprint = [], [], None, None
    for p in sorted(Path(directory).glob("tokens_shard*.npz")):
        with np.load(p, allow_pickle=False) as z:
            if grid is None:
                grid, fingerprint = z["grid"], float(z["projection_fingerprint"])
            elif not np.array_equal(grid, z["grid"]) or fingerprint != float(z["projection_fingerprint"]):
                raise ValueError(f"{p} has a different grid or projection from the other shards")
            names += [str(t) for t in z["tokens"]]
            arrays.append(z["h"])
    if not arrays:
        raise FileNotFoundError(f"no tokens_shard*.npz under {directory}")
    if len(set(names)) != len(names):
        raise ValueError("a scene appears in two shards")
    return {t: i for i, t in enumerate(names)}, np.concatenate(arrays), grid.astype(np.int64), fingerprint


def split_logs(logs, fraction, seed):
    """Deterministic split of the DISTINCT logs: returns (set of logs in the first part, set in the second)."""
    uniq = sorted(set(logs))
    order = np.random.default_rng(seed).permutation(len(uniq))
    cut = int(round(fraction * len(uniq)))
    first = {uniq[i] for i in order[:cut]}
    return first, set(uniq) - first


def cross_log_derangement(logs, seed, tries=1000):
    """For every scene a different scene of ANOTHER log, within the given list: the visual tokens of the mismatched-token control.
    Returns an index array j with logs[j[i]] != logs[i]; raises if there are fewer than two logs."""
    logs = np.asarray(logs)
    if len(set(logs.tolist())) < 2:
        raise ValueError("need at least two logs for a cross-log shuffle")
    rng = np.random.default_rng(seed)
    n = len(logs)
    j = rng.permutation(n)
    for _ in range(tries):
        bad = np.flatnonzero(logs[j] == logs)
        if not len(bad):
            return j
        # swap every offending position with a random position whose log differs from both
        for i in bad:
            candidates = np.flatnonzero((logs != logs[i]) & (logs[j] != logs[i]) & (logs != logs[j[i]]))
            if len(candidates):
                k = rng.choice(candidates)
                j[i], j[k] = j[k], j[i]
    raise RuntimeError("could not build a cross-log derangement")


def standardizer(arrays):
    x = np.concatenate(arrays)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)
