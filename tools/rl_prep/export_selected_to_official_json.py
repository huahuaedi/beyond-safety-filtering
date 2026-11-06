"""Write the candidates chosen by a selector in the format of the released inference (one <token>.json per scene with the [8,3]
relative action), so they can be scored by the OFFICIAL pipeline (run_pdm_score.py + emu_vla_agent) instead of the collection's own
scorer. Before writing, the relative action is converted to an absolute trajectory with the agent's own formula and compared with the
absolute trajectory stored with the candidate; any disagreement aborts.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def agent_absolute(rel):
    """emu_vla_agent.convert_relative_to_absolute_se2_trajectory, reproduced (heading wrapped to [-pi, pi])."""
    out = np.zeros_like(rel, dtype=np.float64)
    x = y = yaw = 0.0
    for i, (dx, dy, dyaw) in enumerate(np.asarray(rel, dtype=np.float64)):
        x += np.cos(yaw) * dx - np.sin(yaw) * dy
        y += np.sin(yaw) * dx + np.cos(yaw) * dy
        yaw = (yaw + dyaw + np.pi) % (2 * np.pi) - np.pi
        out[i] = (x, y, yaw)
    return out


def find_npz(root, token):
    for split in ("train", "val", "test"):
        p = Path(root) / split / f"{token}.npz"
        if p.exists():
            return p
    raise FileNotFoundError(f"no collected scene {token} under {root}")


def export(per_scene, collection_root, out_dir, tolerance=1e-3):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    for row in per_scene:
        with np.load(find_npz(collection_root, row["token"]), allow_pickle=False) as z:
            rel = z["relative_trajectories"][int(row["pick"])].astype(np.float64)
            ab = z["absolute_trajectories"][int(row["pick"])].astype(np.float64)
        recon = agent_absolute(rel)
        dev = np.abs(recon[:, :2] - ab[:, :2]).max()
        dyaw = np.abs(np.arctan2(np.sin(recon[:, 2] - ab[:, 2]), np.cos(recon[:, 2] - ab[:, 2]))).max()
        if dev > tolerance or dyaw > tolerance:
            raise ValueError(f"{row['token']}: stored absolute trajectory differs from the agent's conversion "
                             f"(xy {dev:.2e}, yaw {dyaw:.2e})")
        (out_dir / f"{row['token']}.json").write_text(json.dumps({"action": rel.tolist()}))
    return len(per_scene)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector-json", type=Path, required=True)
    ap.add_argument("--model", choices=["ridge", "hgb"], default="hgb")
    ap.add_argument("--collection-root", type=Path, required=True)
    ap.add_argument("--tokens-json", type=Path, default=None, help="every one of these tokens must be exported")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    report = json.loads(args.selector_json.read_text())
    per_scene = report["models"][args.model]["per_scene"]
    if args.tokens_json:
        need = set(json.loads(args.tokens_json.read_text()))
        have = {r["token"] for r in per_scene}
        if need != have:
            raise SystemExit(f"selector covers {len(have)} scenes, token list {len(need)}; missing {len(need - have)}")
    n = export(per_scene, args.collection_root, args.out_dir)
    print(f"wrote {n} json files to {args.out_dir}")


if __name__ == "__main__":
    main()
