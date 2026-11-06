#!/usr/bin/env python3
"""Build Table 1 (main results) from a manifest of eval runs, matching
ICLR_2025_Conference_Submissions.pdf's columns: NAVSIM Score, Delta-R-bar,
Improve Rate, Regression Rate, Feasible, CVaR_0.1.

This does NOT invent numbers: any column it can't compute from what you give
it is left as "-", same as the paper's current placeholder, rather than
guessing.

Manifest format (JSON), one entry per Table-1 row:
{
  "DriveVLA-W0": {
    "eval_log": "/home/huamin/eval_subset-XXXXX.log"
  },
  "+ Projection": {
    "eval_log": "/home/huamin/eval_subset-XXXXX.log",
    "per_scene_csv": "/path/to/final.csv",
    "reference_csv": "/path/to/reference.csv"
  },
  ...
}

- "eval_log": required for the NAVSIM Score column -- parsed from the
  "Final average score of valid results: X" line.
- "per_scene_csv" / "reference_csv": optional. If both given, computes
  Delta-R-bar, Improve Rate, Regression Rate, CVaR_0.1 from a per-scene join
  on a shared scene/token id column. Column names are auto-detected (see
  detect_columns); if detection looks wrong the script prints what it found
  instead of silently computing garbage -- check that output before trusting
  the numbers.
- "feasible_rate": optional float in [0,1], if you already have it from a
  training-log "saved=" line (phase-1) or eval JSON -- there's no single
  reliable place this always lives, so it's taken as a manual input rather
  than guessed.

Usage:
    python3 build_results_table.py --manifest manifest.json --out table1.csv
Also writes table1.tex (a LaTeX tabular snippet matching Table 1's columns,
to paste into wherever the paper's real source lives).
"""
import argparse
import csv
import json
import re
from pathlib import Path


def parse_final_score(eval_log_path):
    text = Path(eval_log_path).read_text()
    m = re.search(r"Final average score of valid results:\s*([0-9.]+)", text)
    return float(m.group(1)) if m else None


ID_CANDIDATES = ["token", "scene_token", "log_name", "scenario_token", "id"]
SCORE_CANDIDATES = ["score", "pdm_score", "final_score", "total_score", "PDMS"]


def detect_columns(fieldnames):
    id_col = next((c for c in ID_CANDIDATES if c in fieldnames), None)
    score_col = next((c for c in SCORE_CANDIDATES if c in fieldnames), None)
    return id_col, score_col


def load_scores(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        id_col, score_col = detect_columns(fieldnames)
        if id_col is None or score_col is None:
            print(f"  [WARN] could not auto-detect id/score columns in {csv_path}")
            print(f"         header was: {fieldnames}")
            print(f"         edit ID_CANDIDATES/SCORE_CANDIDATES in this script and re-run")
            return None, None, None
        rows = list(reader)
        scores = {row[id_col]: float(row[score_col]) for row in rows if row.get(score_col)}
        return scores, id_col, score_col


def paired_stats(per_scene_csv, reference_csv):
    final_scores, id_col, score_col = load_scores(per_scene_csv)
    if final_scores is None:
        return {}
    ref_scores, _, _ = load_scores(reference_csv)
    if ref_scores is None:
        return {}
    common = sorted(set(final_scores) & set(ref_scores))
    if not common:
        print(f"  [WARN] no overlapping ids between {per_scene_csv} and {reference_csv} "
              f"using id column '{id_col}' -- check these are really the same scene set")
        return {}
    deltas = [final_scores[k] - ref_scores[k] for k in common]
    n = len(deltas)
    mean_delta = sum(deltas) / n
    improve_rate = sum(1 for d in deltas if d > 0) / n
    regress_rate = sum(1 for d in deltas if d < 0) / n
    worst_n = max(1, round(0.1 * n))
    cvar_10 = sum(sorted(deltas)[:worst_n]) / worst_n
    return {
        "delta_r_mean": mean_delta,
        "improve_rate": improve_rate,
        "regression_rate": regress_rate,
        "cvar_0.1": cvar_10,
        "n_matched": n,
    }


def fmt(value, pct=False):
    if value is None:
        return "-"
    return f"{100*value:.1f}\\%" if pct else f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True, help="output CSV path (also writes a .tex next to it)")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    rows = []
    for name, spec in manifest.items():
        print(f"=== {name} ===")
        navsim_score = None
        if spec.get("eval_log"):
            navsim_score = parse_final_score(spec["eval_log"])
            print(f"  NAVSIM score: {navsim_score}")
        stats = {}
        if spec.get("per_scene_csv") and spec.get("reference_csv"):
            stats = paired_stats(spec["per_scene_csv"], spec["reference_csv"])
            print(f"  paired stats: {stats}")
        rows.append({
            "method": name,
            "navsim_score": navsim_score,
            "delta_r_mean": stats.get("delta_r_mean"),
            "improve_rate": stats.get("improve_rate"),
            "regression_rate": stats.get("regression_rate"),
            "feasible_rate": spec.get("feasible_rate"),
            "cvar_0.1": stats.get("cvar_0.1"),
        })

    out_csv = Path(args.out)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "navsim_score", "delta_r_mean", "improve_rate",
            "regression_rate", "feasible_rate", "cvar_0.1",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_csv}")

    tex_path = out_csv.with_suffix(".tex")
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & NAVSIM Score $\uparrow$ & $\overline{\Delta R}$ $\uparrow$ & "
        r"Improve Rate $\uparrow$ & Regression Rate $\downarrow$ & Feasible $\uparrow$ & "
        r"CVaR$_{0.1}$ $\uparrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['method']} & "
            f"{fmt(row['navsim_score'])} & "
            f"{fmt(row['delta_r_mean'])} & "
            f"{fmt(row['improve_rate'], pct=True)} & "
            f"{fmt(row['regression_rate'], pct=True)} & "
            f"{fmt(row['feasible_rate'], pct=True)} & "
            f"{fmt(row['cvar_0.1'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    tex_path.write_text("\n".join(lines))
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
