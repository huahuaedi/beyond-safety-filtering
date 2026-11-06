"""Where do NAVSIM failures happen: in the planned trajectory or in execution?

Reads the ``*.npz`` traces written by ``navsim_score_worker.py --trace-npz``
(via ``RL_EXECUTION_TRACE_DIR``) and writes:

- ``rows.csv``: one line per scored trajectory (scores, open-loop desired
  scores, tracking/margin/actuation summaries);
- ``report.json``: sanity checks, failure attribution per metric and row kind
  with scene-bootstrap intervals, and the continuous-signal statistics that
  the stage-1 decision rules in docs/archive/execution_trace_stage1.md refer to.

Attribution compares the desired trajectory scored open-loop (no tracker,
finite-difference velocities) with the official executed score:
planned / execution_induced / execution_rescued. Candidates of one scene are
correlated, so intervals resample scenes, not rows.

Usage:
    python analyze_execution_traces.py --trace-dir DIR --output-dir OUT
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import execution_trace as et

METRICS = {
    "nc": "no_at_fault_collisions",
    "dac": "drivable_area_compliance",
    "ttc": "time_to_collision_within_bound",
}
SCORE_KEYS = et.SCORE_KEYS
NEAR_MISS_MARGIN = 0.3  # m; "passed DAC but a corner came within 30 cm"
NON_DRIVABLE = 1  # EgoAreaIndex.NON_DRIVABLE_AREA


def load_rows(trace_dir):
    rows = []
    files = sorted(Path(trace_dir).rglob("*.npz"))
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            if int(z["schema_version"]) != et.TRACE_SCHEMA_VERSION:
                raise ValueError(f"{path}: schema {int(z['schema_version'])}")
            meta = json.loads(str(z["meta_json"]))
            data = {k: z[k] for k in z.files}
        n = len(data["row_ok"])
        kinds = meta.get("row_kinds") or ["candidate"] * n
        if len(kinds) != n:
            raise ValueError(f"{path}: {len(kinds)} row kinds for {n} rows")
        summary = et.summarize(data)
        dac_flag = data["executed_ego_areas"][..., NON_DRIVABLE].any(-1)
        for i in range(n):
            row = {
                "file": path.name, "scene_token": str(data["scene_token"]),
                "step": meta.get("step", -1), "step_kind": meta.get("step_kind", ""),
                "row": i, "row_kind": kinds[i], "row_ok": bool(data["row_ok"][i]),
                "open_loop_ok": bool(data["row_ok"][i]) and not str(data["open_loop_error"][i]),
                "scorer_nondrivable_flag": bool(dac_flag[i]),
                "progress_raw": float(data["progress_raw"][i]),
                "collision_t": float(data["collision_time_idx"][i]) * et.DT,
                "ttc_t": float(data["ttc_time_idx"][i]) * et.DT,
            }
            for key in SCORE_KEYS:
                row[key] = float(data[f"score__{key}"][i])
            for short, key in METRICS.items():
                row[f"desired_{short}"] = float(data[f"desired__{key}"][i])
            for key in et.SUMMARY_KEYS:
                row[key] = float(summary[key][i])
            rows.append(row)
    return files, rows


def column(rows, key, dtype=np.float64):
    return np.asarray([r[key] for r in rows], dtype=dtype)


def bootstrap_ratio(scenes, num, den, n_boot=2000, seed=0):
    """Scene-resampled 95% interval of sum(num)/sum(den)."""
    uniq, inv = np.unique(scenes, return_inverse=True)
    num_s = np.bincount(inv, weights=num, minlength=len(uniq))
    den_s = np.bincount(inv, weights=den, minlength=len(uniq))
    point = float(num_s.sum() / den_s.sum()) if den_s.sum() > 0 else float("nan")
    if den_s.sum() == 0:
        return point, [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    d = den_s[draws].sum(1)
    ratios = np.where(d > 0, num_s[draws].sum(1) / np.maximum(d, 1e-12), np.nan)
    return point, [float(np.nanpercentile(ratios, 2.5)), float(np.nanpercentile(ratios, 97.5))]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[keep])).astype(float)
    rb = np.argsort(np.argsort(b[keep])).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def attribution_block(rows):
    ok = column(rows, "open_loop_ok", bool)
    scenes = np.asarray([r["scene_token"] for r in rows])
    out = {"n_rows": len(rows), "n_open_loop_ok": int(ok.sum()),
           "n_scenes": int(len(np.unique(scenes)))}
    for short, key in METRICS.items():
        executed_pass = column(rows, key) >= 1.0
        desired_pass = column(rows, f"desired_{short}") >= 1.0
        labels = et.attribute(desired_pass[ok], executed_pass[ok])
        counts = {label: int((labels == label).sum()) for label in et.ATTRIBUTION_LABELS}
        induced = (labels == "execution_induced").astype(float)
        failed = (labels == "execution_induced") | (labels == "planned")
        share, ci = bootstrap_ratio(scenes[ok], induced, failed.astype(float))
        scene_any = {s for s, lab in zip(scenes[ok], labels) if lab == "execution_induced"}
        out[short] = {
            "counts": counts,
            "executed_failure_rate": float((~executed_pass[ok]).mean()) if ok.any() else float("nan"),
            "execution_induced_share_of_failures": share,
            "execution_induced_share_ci95_scene_bootstrap": ci,
            "scenes_with_execution_induced_failure": len(scene_any),
        }
    return out


def continuous_block(rows):
    """Does execution information explain failures, and does the margin carry
    signal that binary DAC does not?"""
    ok = column(rows, "open_loop_ok", bool)
    r = [x for x, keep in zip(rows, ok) if keep]
    if not r:
        return {}
    dac_fail = column(r, "drivable_area_compliance") < 1.0
    desired_dac_pass = column(r, "desired_dac") >= 1.0
    out = {"auc_executed_dac_failure": {
        # Higher score = more risk, so margins are negated.
        "neg_desired_min_margin": et.roc_auc(-column(r, "desired_min_margin"), dac_fail),
        "neg_executed_min_margin": et.roc_auc(-column(r, "executed_min_margin"), dac_fail),
        "max_abs_lat_err": et.roc_auc(column(r, "max_abs_lat_err"), dac_fail),
        "max_abs_heading_err": et.roc_auc(column(r, "max_abs_heading_err"), dac_fail),
        "max_abs_lat_acc": et.roc_auc(column(r, "max_abs_lat_acc"), dac_fail),
    }}
    # Among trajectories whose plan is in bounds, which execution quantities
    # separate the ones the tracker pushes out?
    sub = [x for x, keep in zip(r, desired_dac_pass) if keep]
    sub_fail = column(sub, "drivable_area_compliance") < 1.0 if sub else np.zeros(0, bool)
    out["auc_execution_induced_dac_given_desired_pass"] = {
        key: et.roc_auc(column(sub, key), sub_fail) if sub else float("nan")
        for key in ("max_abs_lat_err", "max_abs_heading_err", "rms_pos_err",
                    "max_abs_steering_rate", "max_abs_lat_acc")
    }
    out["auc_execution_induced_dac_given_desired_pass"]["neg_desired_min_margin"] = (
        et.roc_auc(-column(sub, "desired_min_margin"), sub_fail) if sub else float("nan")
    )
    out["n_desired_dac_pass"] = len(sub)
    out["n_execution_induced_dac"] = int(sub_fail.sum())

    # Near misses invisible to binary DAC.
    ex_margin = column(r, "executed_min_margin")
    passing = ~dac_fail
    out["dac_pass_near_miss_fraction"] = (
        float((ex_margin[passing] < NEAR_MISS_MARGIN).mean()) if passing.any() else float("nan")
    )
    # Scene level: do scenes whose DAC-passing candidates run close to the
    # boundary have more DAC failures among the other candidates?
    by_scene = {}
    for x, fail in zip(r, dac_fail):
        if x["row_kind"] == "candidate":
            by_scene.setdefault(x["scene_token"], []).append((fail, x["executed_min_margin"]))
    med_margin, fail_rate, uniform_pass_spread = [], [], []
    for items in by_scene.values():
        fails = np.array([f for f, _ in items])
        margins = np.array([m for _, m in items])
        if (~fails).any():
            med_margin.append(float(np.median(margins[~fails])))
            fail_rate.append(float(fails.mean()))
        if not fails.any() and len(margins) > 1:
            uniform_pass_spread.append(float(margins.max() - margins.min()))
    out["scene_spearman_passing_margin_vs_dac_failure_rate"] = spearman(med_margin, fail_rate)
    out["n_scenes_for_spearman"] = len(med_margin)
    out["all_pass_groups"] = {
        "n": len(uniform_pass_spread),
        # Groups where DAC gives zero within-group signal but margins differ.
        "median_margin_spread_m": float(np.median(uniform_pass_spread)) if uniform_pass_spread else float("nan"),
        "frac_spread_gt_0p5m": float(np.mean(np.array(uniform_pass_spread) > 0.5)) if uniform_pass_spread else float("nan"),
    }
    return out


def sanity_block(rows):
    ok = column(rows, "row_ok", bool)
    r = [x for x, keep in zip(rows, ok) if keep]
    dac_fail = column(r, "drivable_area_compliance") < 1.0
    flag = column(r, "scorer_nondrivable_flag", bool)
    margin_out = column(r, "executed_min_margin") < 0.0
    return {
        "n_rows": len(rows),
        "n_row_errors": int((~ok).sum()),
        "n_open_loop_errors": int(sum(1 for x in rows if x["row_ok"] and not x["open_loop_ok"])),
        # Must be exactly 1.0: DAC is defined by this flag.
        "scorer_flag_vs_dac_agreement": float((flag == dac_fail).mean()) if r else float("nan"),
        # Should be ~1.0; disagreements are corners within float/edge tolerance.
        "margin_sign_vs_dac_agreement": float((margin_out == dac_fail).mean()) if r else float("nan"),
        "margin_sign_vs_dac_disagreements": int((margin_out != dac_fail).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    files, rows = load_rows(args.trace_dir)
    if not rows:
        raise SystemExit(f"no traces under {args.trace_dir}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rows.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    kinds = sorted({r["row_kind"] for r in rows})
    report = {
        "trace_dir": str(args.trace_dir),
        "n_files": len(files),
        "sanity": sanity_block(rows),
        "attribution": {k: attribution_block([r for r in rows if r["row_kind"] == k]) for k in kinds},
        "continuous_candidates": continuous_block([r for r in rows if r["row_kind"] == "candidate"]),
        "near_miss_margin_m": NEAR_MISS_MARGIN,
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
