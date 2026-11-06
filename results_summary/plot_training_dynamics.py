#!/usr/bin/env python3
"""Parse a unified_drivevla_rl training log and produce the training-dynamics
figures requested by ICLR_2025_Conference_Submissions.pdf Section 6.9 /
Appendix I:
  - outcome-critic training and validation loss
  - correction-policy training reward (RL phase)
  - correction-policy validation Delta-R (risky_risk_delta / risky_progress_gain)
  - physical-feasibility rate
  - correction activation rate (needs_correction_rate -- the label rate; NOT
    gate_selected, which lives in the eval CSVs, not these logs)
  - gate_prob_needs_correction vs. gate_prob_safe (this investigation's own
    diagnostic -- not in the original Appendix I list, but directly answers
    Q4: does gate discriminate needs_correction at all)
  - sup_weight schedule (persistent supervised-anchor decay)

Usage (run on HPC where the log lives):
    python3 plot_training_dynamics.py \
        --log ~/unified_drivevla-58977582_4294967294.log \
        --label seed11_full_model \
        --out-dir ~/DriveVLA-W0/results_summary/figures_paper

Produces one PNG+PDF per figure, named <label>_<figure>.{png,pdf}, plus a
<label>_parsed.json with every parsed row (for re-plotting / table-building
without re-parsing the raw log).
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_kv_line(line, prefix_pattern):
    """A log line is `<prefix> key=value key=value ...`. Returns
    (epoch_or_step:int, {key: float}) or None if the line doesn't match.
    Non-numeric values (e.g. replay_scene=none, kl_guard=0/1 as int) are
    coerced where possible and skipped otherwise.
    """
    m = re.match(prefix_pattern, line)
    if m is None:
        return None
    index = int(m.group(1))
    rest = line[m.end():]
    values = {}
    for key, value in re.findall(r"(\w+)=([^\s]+)", rest):
        try:
            values[key] = float(value)
        except ValueError:
            continue
    return index, values


def parse_log(path):
    text = Path(path).read_text()
    outcome, warmup, rl = [], [], []
    for line in text.splitlines():
        row = parse_kv_line(line, r"outcome epoch=(\d+)")
        if row is not None:
            outcome.append({"epoch": row[0], **row[1]})
            continue
        row = parse_kv_line(line, r"warmup epoch=(\d+)")
        if row is not None:
            warmup.append({"epoch": row[0], **row[1]})
            continue
        row = parse_kv_line(line, r"rl epoch=(\d+)")
        if row is not None:
            rl.append({"epoch": row[0], **row[1]})
            continue
    return {"outcome": outcome, "warmup": warmup, "rl": rl}


def series(rows, key):
    xs = [r["epoch"] for r in rows if key in r]
    ys = [r[key] for r in rows if key in r]
    return xs, ys


def save(fig, out_dir, label, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{label}_{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_outcome_critic_loss(parsed, out_dir, label):
    rows = parsed["outcome"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(*series(rows, "train"), marker="o", label="train loss")
    ax.plot(*series(rows, "val"), marker="s", label="val loss")
    ax.set_xlabel("outcome-critic epoch")
    ax.set_ylabel("loss")
    ax.set_title("Outcome critic training/validation loss")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, out_dir, label, "outcome_critic_loss")


def plot_correction_training_reward(parsed, out_dir, label):
    rows = parsed["rl"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(*series(rows, "train_reward"), marker="o", color="tab:blue")
    ax.set_xlabel("RL epoch")
    ax.set_ylabel("mean training reward")
    ax.set_title("Correction-policy training reward")
    ax.grid(alpha=0.3)
    save(fig, out_dir, label, "correction_training_reward")


def plot_validation_delta_r(parsed, out_dir, label):
    rows = parsed["rl"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(*series(rows, "risky_risk_delta"), marker="o", label=r"risky $\Delta$risk (test-set proxy)")
    ax.plot(*series(rows, "risky_progress_gain"), marker="s", label=r"risky $\Delta$progress")
    ax.axhline(0.0, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("RL epoch")
    ax.set_ylabel(r"validation $\Delta R$ components")
    ax.set_title(r"Correction-policy validation $\Delta R$ (risky subset)")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, out_dir, label, "validation_delta_r")


def plot_feasibility_and_regression(parsed, out_dir, label):
    rows = parsed["rl"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(*series(rows, "feasible_rate"), marker="o", label="feasible_rate")
    ax.plot(*series(rows, "regression_rate"), marker="s", label="regression_rate")
    ax.set_xlabel("RL epoch")
    ax.set_ylabel("rate")
    ax.set_title("Physical feasibility and regression rate")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, out_dir, label, "feasibility_regression")


def plot_needs_correction_rate(parsed, out_dir, label):
    warmup_rows, rl_rows = parsed["warmup"], parsed["rl"]
    if not warmup_rows and not rl_rows:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    wx, wy = series(warmup_rows, "needs_correction_rate")
    rx, ry = series(rl_rows, "needs_correction_rate")
    rx_shifted = [x + (max(wx) + 1 if wx else 0) for x in rx]
    ax.plot(wx, wy, marker="o", color="tab:orange", label="warmup phase")
    ax.plot(rx_shifted, ry, marker="o", color="tab:blue", label="RL phase")
    if wx:
        ax.axvline(max(wx) + 0.5, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("epoch (warmup then RL, concatenated)")
    ax.set_ylabel("needs_correction_rate")
    ax.set_title("Verified-candidate label rate (should be ~constant)")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, out_dir, label, "needs_correction_rate")


def plot_gate_probability_split(parsed, out_dir, label):
    warmup_rows, rl_rows = parsed["warmup"], parsed["rl"]
    if "gate_prob_needs_correction" not in "".join(str(r) for r in warmup_rows + rl_rows):
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    wx, wy_need = series(warmup_rows, "gate_prob_needs_correction")
    _, wy_safe = series(warmup_rows, "gate_prob_safe")
    rx, ry_need = series(rl_rows, "gate_prob_needs_correction")
    _, ry_safe = series(rl_rows, "gate_prob_safe")
    offset = (max(wx) + 1) if wx else 0
    rx_shifted = [x + offset for x in rx]
    ax.plot(wx, wy_need, marker="o", color="tab:red", label="needs_correction=True")
    ax.plot(wx, wy_safe, marker="o", color="tab:green", label="needs_correction=False")
    ax.plot(rx_shifted, ry_need, marker="s", color="tab:red")
    ax.plot(rx_shifted, ry_safe, marker="s", color="tab:green")
    ax.axhline(0.5, color="black", linewidth=1, linestyle=":", label="deployment threshold (0.5)")
    if wx:
        ax.axvline(max(wx) + 0.5, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("epoch (warmup then RL, concatenated)")
    ax.set_ylabel("mean gate_probability")
    ax.set_yscale("log")
    ax.set_title("Gate discrimination: needs_correction vs. safe subset")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, out_dir, label, "gate_probability_split")


def plot_sup_weight(parsed, out_dir, label):
    rows = parsed["rl"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(*series(rows, "sup_weight"), marker="o", color="tab:purple")
    ax.set_xlabel("RL epoch")
    ax.set_ylabel(r"$\lambda_{\mathrm{sup}}$")
    ax.set_title("Persistent supervised-anchor weight schedule")
    ax.grid(alpha=0.3)
    save(fig, out_dir, label, "sup_weight_schedule")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="path to unified_drivevla-<JOBID>_*.log")
    parser.add_argument("--label", required=True, help="short tag used as filename prefix, e.g. seed11_full_model")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    parsed = parse_log(args.log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.label}_parsed.json").write_text(json.dumps(parsed, indent=2))

    plot_outcome_critic_loss(parsed, out_dir, args.label)
    plot_correction_training_reward(parsed, out_dir, args.label)
    plot_validation_delta_r(parsed, out_dir, args.label)
    plot_feasibility_and_regression(parsed, out_dir, args.label)
    plot_needs_correction_rate(parsed, out_dir, args.label)
    plot_gate_probability_split(parsed, out_dir, args.label)
    plot_sup_weight(parsed, out_dir, args.label)

    print(f"parsed: outcome={len(parsed['outcome'])} rows, "
          f"warmup={len(parsed['warmup'])} rows, rl={len(parsed['rl'])} rows")
    print(f"figures written to {out_dir}")


if __name__ == "__main__":
    main()
