"""Offline diagnostic: does the correction policy's gate activate more on
risky-reference tokens, and does it actually improve them when it does?

Judgment-tree first layer (per docs discussion): for one already-completed
eval run's unified-correction output, split tokens into a risky bucket (top
half by max reference_risk) and a normal bucket, then report per bucket:
  gate_selected rate (gamma), mean risk delta, mean progress delta,
  regression rate (risk got worse after correction), feasibility rate.

This only reads a directory apply_unified_correction_policy.py already
produced -- no new training or data collection.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unified-dir", type=Path, required=True,
        help="output/<experiment_name>_unified_correction/ from apply_unified_correction_policy.py",
    )
    parser.add_argument(
        "--physics-tolerance", type=float, default=1e-3,
        help="matches CorrectionRLConfig.physics_tolerance",
    )
    parser.add_argument(
        "--semantic-margin", type=float, default=0.005,
        help="matches CorrectionRLConfig.semantic_margin",
    )
    return parser.parse_args()


def main():
    args = arguments()
    tokens = sorted(
        path.stem for path in args.unified_dir.glob("*.json")
        if path.stem != "unified_correction_summary"
        and not path.stem.endswith("_agent_human_traj_diff")
    )
    if not tokens:
        raise ValueError(f"no per-token JSON found under {args.unified_dir}")

    rows = []
    for token in tokens:
        unified = json.loads((args.unified_dir / f"{token}.json").read_text())["unified_correction"]
        reference_risk = unified["reference_risk"]
        corrected_risk = unified["corrected_risk"]
        rows.append({
            "token": token,
            "risk_severity": max(reference_risk),
            "gate_selected": bool(unified["gate_selected"]),
            "risk_delta": sum(reference_risk) - sum(corrected_risk),  # >0 = risk improved
            "progress_delta": unified["corrected_progress"] - unified["reference_progress"],
            # Same definition as constrained_correction_objective's
            # semantic_regret: risk got worse by more than the margin.
            "regressed": (
                sum(corrected_risk) - sum(reference_risk) > args.semantic_margin
            ),
            "feasible": unified["corrected_physics_violation"] <= args.physics_tolerance,
        })

    severities = sorted(row["risk_severity"] for row in rows)
    median_severity = severities[len(severities) // 2]
    risky = [row for row in rows if row["risk_severity"] >= median_severity]
    normal = [row for row in rows if row["risk_severity"] < median_severity]

    def report(label: str, bucket: list[dict]) -> None:
        if not bucket:
            print(f"{label}: n=0")
            return
        n = len(bucket)
        gamma = sum(row["gate_selected"] for row in bucket) / n
        risk_delta = statistics.mean(row["risk_delta"] for row in bucket)
        progress_delta = statistics.mean(row["progress_delta"] for row in bucket)
        regression_rate = sum(row["regressed"] for row in bucket) / n
        feasible_rate = sum(row["feasible"] for row in bucket) / n
        print(
            f"{label}: n={n} gamma_selected={gamma:.4f} "
            f"mean_risk_delta={risk_delta:+.6f} mean_progress_delta={progress_delta:+.6f} "
            f"regression_rate={regression_rate:.4f} feasible_rate={feasible_rate:.4f}"
        )

    print(f"tokens analyzed: {len(rows)}  (median risk_severity split at {median_severity:.6f})")
    report("risky (>= median reference_risk)", risky)
    report("normal (< median reference_risk)", normal)
    report("all", rows)


if __name__ == "__main__":
    main()
