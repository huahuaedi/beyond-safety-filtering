"""Paired, scene-level uncertainty analysis for two NAVSIM score outputs.

Each experiment directory must contain exactly one PDM-score CSV (recursively).
The synthetic ``average`` row is discarded and rows are paired by scene token.
Bootstrap resampling is therefore performed over scenes, never over aggregate
scores or metric columns.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


NON_METRICS = {"", "token", "valid"}


def find_csv(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(path.rglob("*.csv"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one CSV below {path}, found {len(candidates)}: "
            f"{[str(candidate) for candidate in candidates]}"
        )
    return candidates[0]


def load_scores(path: Path) -> tuple[Path, dict[str, dict[str, float]]]:
    csv_path = find_csv(path)
    rows: dict[str, dict[str, float]] = {}
    with csv_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            token = row.get("token", "")
            if not token or token == "average":
                continue
            if token in rows:
                raise RuntimeError(f"duplicate token {token!r} in {csv_path}")
            if str(row.get("valid", "")).lower() not in {"true", "1"}:
                raise RuntimeError(f"invalid scene {token!r} in {csv_path}")
            values = {}
            for key, value in row.items():
                if key in NON_METRICS or value in {None, ""}:
                    continue
                try:
                    numeric = float(value)
                except ValueError:
                    continue
                if np.isfinite(numeric):
                    values[key] = numeric
            rows[token] = values
    if not rows:
        raise RuntimeError(f"no valid per-scene rows in {csv_path}")
    return csv_path, rows


def analyze(
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    draws: int,
    seed: int,
) -> dict:
    baseline_tokens = set(baseline)
    candidate_tokens = set(candidate)
    if baseline_tokens != candidate_tokens:
        raise RuntimeError(
            "token mismatch: "
            f"baseline_only={len(baseline_tokens - candidate_tokens)} "
            f"candidate_only={len(candidate_tokens - baseline_tokens)}"
        )
    tokens = sorted(baseline_tokens)
    common_metrics = set.intersection(
        *(set(baseline[token]) & set(candidate[token]) for token in tokens)
    )
    if "score" not in common_metrics:
        raise RuntimeError("paired CSVs do not contain a numeric 'score' column")

    rng = np.random.default_rng(seed)
    # Reuse one paired resampling plan for all metrics. Chunking bounds memory
    # even when this tool is later used on the full trainval scene pool.
    sample_means = {metric: [] for metric in sorted(common_metrics)}
    deltas = {
        metric: np.asarray(
            [candidate[token][metric] - baseline[token][metric] for token in tokens],
            dtype=np.float64,
        )
        for metric in sorted(common_metrics)
    }
    remaining = draws
    while remaining:
        chunk = min(remaining, 2000)
        indices = rng.integers(0, len(tokens), size=(chunk, len(tokens)))
        for metric, delta in deltas.items():
            sample_means[metric].append(delta[indices].mean(axis=1))
        remaining -= chunk

    metrics = {}
    for metric, delta in deltas.items():
        boot = np.concatenate(sample_means[metric])
        probability_le_zero = float(np.mean(boot <= 0))
        probability_ge_zero = float(np.mean(boot >= 0))
        metrics[metric] = {
            "baseline_mean": float(
                np.mean([baseline[token][metric] for token in tokens])
            ),
            "candidate_mean": float(
                np.mean([candidate[token][metric] for token in tokens])
            ),
            "mean_delta": float(delta.mean()),
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "two_sided_p": float(min(1.0, 2 * min(probability_le_zero, probability_ge_zero))),
            "wins": int(np.sum(delta > 1e-12)),
            "ties": int(np.sum(np.abs(delta) <= 1e-12)),
            "losses": int(np.sum(delta < -1e-12)),
        }
    return {"scenes": len(tokens), "bootstrap_draws": draws, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")

    baseline_csv, baseline = load_scores(args.baseline)
    candidate_csv, candidate = load_scores(args.candidate)
    result = analyze(baseline, candidate, args.bootstrap, args.seed)
    result["baseline_csv"] = str(baseline_csv)
    result["candidate_csv"] = str(candidate_csv)

    score = result["metrics"]["score"]
    print(f"paired scenes: {result['scenes']}")
    print(
        "score: "
        f"baseline={score['baseline_mean']:.6f} "
        f"candidate={score['candidate_mean']:.6f} "
        f"delta={score['mean_delta']:+.6f} "
        f"ci95=[{score['ci95'][0]:+.6f}, {score['ci95'][1]:+.6f}] "
        f"p={score['two_sided_p']:.6g} "
        f"W/T/L={score['wins']}/{score['ties']}/{score['losses']}"
    )
    for metric, values in result["metrics"].items():
        if metric == "score":
            continue
        print(
            f"{metric}: delta={values['mean_delta']:+.6f} "
            f"ci95=[{values['ci95'][0]:+.6f}, {values['ci95'][1]:+.6f}]"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"saved={args.output}")


if __name__ == "__main__":
    main()
