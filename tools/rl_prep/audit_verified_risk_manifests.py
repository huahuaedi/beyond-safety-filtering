"""Audit NAVSIM-verified manifest precision and paired uncertainty.

The manifest selectors operate on the same scenes and candidate budget, so
uncertainty is bootstrapped by scene (paired across selectors), not by treating
the individual candidates as independent observations.
"""

import argparse
import json
from pathlib import Path

import numpy as np


LABELS = ("unsafe", "ttc_failure", "dac_failure", "nc_failure")
FILES = {
    "random": "random_topk_budget8.json",
    "action": "action_topk_budget8.json",
    "world_model": "world_model_routed_topk_budget8.json",
}


def load_manifest(path):
    with open(path) as file:
        payload = json.load(file)
    budget = int(payload["verification_budget_per_scene"])
    rows = {}
    for entry in payload["entries"]:
        token = str(entry["scene_token"])
        selected = list(map(int, entry["selected_indices"]))
        verified = list(map(int, entry["verified_unsafe_indices"]))
        counts = entry["verified_label_counts"]
        if len(selected) != budget or len(set(selected)) != budget:
            raise ValueError(f"{path}: invalid fixed budget for {token}")
        if not set(verified).issubset(selected):
            raise ValueError(f"{path}: verified indices not selected for {token}")
        if int(counts["unsafe"]) != len(verified):
            raise ValueError(f"{path}: unsafe count mismatch for {token}")
        rows[token] = {
            "selected": selected,
            "counts": np.asarray([counts[label] for label in LABELS], dtype=np.int64),
            "priority": float(entry["risk_priority"]),
        }
    if int(payload["verification_calls"]) != len(rows) * budget:
        raise ValueError(f"{path}: verification-call count mismatch")
    return payload, rows


def scene_file(data_root, token):
    matches = list(data_root.glob(f"*/{token}.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one NAVSIM label file for {token}, found {matches}")
    return matches[0]


def independently_verify(rows, data_root):
    for token, row in rows.items():
        with np.load(scene_file(data_root, token), allow_pickle=False) as data:
            selected = np.asarray(row["selected"], dtype=np.int64)
            observed = np.asarray(
                [
                    (~data["reference_feasible"].astype(bool)[selected]).sum(),
                    (data["score__time_to_collision_within_bound"][selected] < 1.0 - 1e-8).sum(),
                    (data["score__drivable_area_compliance"][selected] < 1.0 - 1e-8).sum(),
                    (data["score__no_at_fault_collisions"][selected] < 1.0 - 1e-8).sum(),
                ],
                dtype=np.int64,
            )
        if not np.array_equal(observed, row["counts"]):
            raise ValueError(
                f"NAVSIM label mismatch for {token}: manifest={row['counts']} source={observed}"
            )


def retained_rows(rows, cap):
    # Mirror rl_stage1_prototype.py's insertion-time eviction exactly,
    # including its deterministic token tie-break.
    retained = {}
    for token, row in rows.items():
        if row["counts"][0] <= 0:
            continue
        retained[token] = row
        if len(retained) > cap:
            evicted = min(retained, key=lambda key: (retained[key]["priority"], -1, key))
            del retained[evicted]
    return retained


def paired_bootstrap(matrices, names, draws, seed):
    rng = np.random.default_rng(seed)
    n = matrices[names[0]].shape[0]
    samples = rng.integers(0, n, size=(draws, n))
    output = {}
    reference = matrices["random"]
    for name in names:
        if name == "random":
            continue
        delta = matrices[name][samples].mean(axis=1) - reference[samples].mean(axis=1)
        output[name] = {
            label: {
                "delta": float(matrices[name][:, i].mean() - reference[:, i].mean()),
                "ci95": [float(x) for x in np.quantile(delta[:, i], [0.025, 0.975])],
                "p_delta_le_zero": float((delta[:, i] <= 0).mean()),
            }
            for i, label in enumerate(LABELS)
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--buffer-cap", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifests = {name: load_manifest(args.manifest_dir / filename) for name, filename in FILES.items()}
    common = sorted(set.intersection(*(set(rows) for _, rows in manifests.values())))
    if any(len(rows) != len(common) for _, rows in manifests.values()):
        raise ValueError("selectors do not contain the same scene set")

    if args.data_root:
        for _, rows in manifests.values():
            independently_verify(rows, args.data_root)
        print(f"independent NAVSIM-label verification: OK ({len(common)} scenes)")

    matrices = {}
    summary = {}
    for name, (payload, rows) in manifests.items():
        matrix = np.stack([rows[token]["counts"] for token in common]).astype(np.float64)
        matrices[name] = matrix / int(payload["verification_budget_per_scene"])
        retained = retained_rows(rows, args.buffer_cap)
        retained_counts = np.stack([row["counts"] for row in retained.values()]).sum(axis=0)
        summary[name] = {
            "calls": int(payload["verification_calls"]),
            "risk_scenes": int((matrix[:, 0] > 0).sum()),
            "candidate_true_positives": {label: int(matrix[:, i].sum()) for i, label in enumerate(LABELS)},
            "candidate_precision": {
                label: float(matrix[:, i].sum() / payload["verification_calls"])
                for i, label in enumerate(LABELS)
            },
            "retained_scenes": len(retained),
            "retained_true_positives": {
                label: int(retained_counts[i]) for i, label in enumerate(LABELS)
            },
        }
        print(f"\n===== {name} =====")
        print(json.dumps(summary[name], indent=2))

    bootstrap = paired_bootstrap(matrices, list(manifests), args.bootstrap, args.seed)
    print("\n===== paired scene-bootstrap delta vs random (precision) =====")
    print(json.dumps(bootstrap, indent=2))
    result = {"summary": summary, "paired_bootstrap_vs_random": bootstrap}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as file:
            json.dump(result, file, indent=2)
        print(f"saved={args.output}")


if __name__ == "__main__":
    main()
