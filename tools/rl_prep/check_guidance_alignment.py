"""Do the guidance inputs line up with the simulator? Compare the geometric cost of stored candidates with their scores.

For every scene of a collection that has guidance inputs, the road term should be positive for candidates whose drivable-area
score is below one and zero for the others, and the collision term should be positive for candidates that collide (no-at-fault
collision score below one). Reports agreement as AUC-like rates; a swapped axis or a flipped sign shows up as chance level.
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch

import guidance_geometry as G

TIMES = 0.5 * np.arange(1, 9)


def load_cost(path, device="cpu"):
    z = np.load(path, allow_pickle=False)
    agents = {k: torch.tensor(z["agent_" + k], dtype=torch.float64, device=device) for k in ("pos", "vel", "heading", "length", "width")}
    ego = {"length": float(z["ego_length"]), "width": float(z["ego_width"]), "center_offset": float(z["ego_center_offset"])}
    return G.GuidanceCost(agents, torch.tensor(z["sdf"].astype(np.float64), device=device), tuple(float(v) for v in z["sdf_origin"]),
                          float(z["sdf_res"]), ego, torch.zeros(3), torch.ones(3), torch.tensor(TIMES, dtype=torch.float64, device=device))


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", type=Path, required=True)
    ap.add_argument("--guide-dir", type=Path, required=True)
    args = ap.parse_args()
    road_fail, road_ok, col_fail, col_ok = [], [], [], []
    scenes = 0
    for f in sorted(glob.glob(str(args.collection / "*" / "*.npz"))):
        token = os.path.basename(f)[:-4]
        g = args.guide_dir / f"{token}.npz"
        if not g.exists():
            continue
        cost = load_cost(g)
        with np.load(f, allow_pickle=False) as z:
            poses = torch.tensor(z["absolute_trajectories"].astype(np.float64))
            dac, nc = z["score__drivable_area_compliance"], z["score__no_at_fault_collisions"]
        coll, road, _, _ = cost.terms_abs(poses)
        road, coll = road.numpy(), coll.numpy()
        road_fail += list(road[dac < 1]); road_ok += list(road[dac >= 1])
        col_fail += list(coll[nc < 1]); col_ok += list(coll[nc >= 1])
        scenes += 1
    print(f"scenes {scenes}; candidates with DAC failure {len(road_fail)}, with collision {len(col_fail)}")
    print("road term > 0:  among DAC failures %.3f   among passing candidates %.3f" % (
        np.mean(np.array(road_fail) > 0) if road_fail else float("nan"), np.mean(np.array(road_ok) > 0) if road_ok else float("nan")))
    print("collision term > 0: among colliding candidates %.3f   among the others %.3f" % (
        np.mean(np.array(col_fail) > 0) if col_fail else float("nan"), np.mean(np.array(col_ok) > 0) if col_ok else float("nan")))
    print("AUC road term vs DAC failure:", auc(road_fail, road_ok), "| AUC collision term vs collision:", auc(col_fail, col_ok))


if __name__ == "__main__":
    main()
