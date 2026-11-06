"""Figures for paper_analysis.tex. Numerical data are loaded from paper_evidence.json. Values are transcribed from local summaries; latest HPC raw reports have not been reverified.
Run: python docs/make_figures.py   (writes docs/figures/*.pdf and *.png)
Palette: blue = verifier ranking / ours, orange = training methods, gray = reference / baselines (validated, categorical slots 1-2).
"""
from pathlib import Path
import json
DATA = json.loads(Path(__file__).with_name("paper_evidence.json").read_text())["figure_data"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).with_name("figures")
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GRAY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#8a8a85", "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def forest(ax, rows, xlabel, zero_line=True, xlim=None, digits=2):
    """rows: (label, estimate, lo or None, hi or None, color). Dot + whisker; no interval -> dot only."""
    for y, (label, est, lo, hi, color) in enumerate(rows[::-1]):
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.plot([est], [y], "o", color=color, ms=7, mec="white", mew=1.5, zorder=3)
        txt = f"{est:+.{digits}f}" if lo is None else f"{est:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"
        ax.annotate(txt, (est, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(rows) - 0.1 + 0.55)
    if zero_line:
        ax.axvline(0, color=INK2, lw=1)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)


# ---- Fig 1: net gate passes, same 1000 scenes (frozen navtest subset, 136 logs): ledger of training methods vs verifier ranking ----
rows = DATA["rows_1"]
fig, ax = plt.subplots(figsize=(6.6, 3.4))
for y, (lab, v, c) in enumerate(rows[::-1]):
    ax.barh(y, v, color=c, height=0.55)
    ax.text(v + (1.0 if v >= 0 else -1.0), y, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK2)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
ax.axvline(0, color=INK2, lw=1); ax.grid(axis="y", visible=False); ax.set_xlim(-40, 32)
ax.set_xlabel("Net gate passes (NC + DAC + TTC); 1,000-scene subset")
ax.set_title("Training objectives and ranking yield different net gate repairs", loc="left", fontsize=9.5)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)],
          labels=["training update", "test-time ranking"], loc="upper right", frameon=False, fontsize=8)
fig.text(0.01, -0.14, "Evaluator baselines differ across families; gate events are not distinct scenes. See manuscript table.", fontsize=7, color=INK2)
save(fig, "fig1_ledger_subset")

# ---- Fig 2: official full navtest (12,146 scenes), ours vs published ----
fig, ax = plt.subplots(figsize=(6.6, 3.0))
models = DATA["models_1"]
for y, (lab, ours, pub, c) in enumerate(models[::-1]):
    ax.plot([ours], [y], "o", color=c, ms=7, mec="white", mew=1.2, zorder=5)
    ax.annotate(f"{ours:.2f}", (ours, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    if pub is not None:
        ax.plot([pub], [y], "o", mfc="white", mec=INK2, ms=8, mew=1.5, zorder=3)
        ax.plot([min(ours, pub), max(ours, pub)], [y, y], color=GRID, lw=2, zorder=1)
ax.set_yticks(range(len(models))); ax.set_yticklabels([m[0] for m in models[::-1]]); ax.grid(axis="y", visible=False)
ax.set_xlim(85.5, 91.5); ax.set_ylim(-0.6, len(models) - 0.1 + 0.5); ax.set_xlabel("PDMS on official navtest (12,146 scenes)")
ax.plot([], [], "o", color=GRAY, label="reproduced here"); ax.plot([], [], "o", mfc="white", mec=INK2, label="published")
ax.legend(loc="lower right", frameon=False, fontsize=8)
ax.set_title("Official-protocol reproduction", loc="left", fontsize=9.5, pad=10)
save(fig, "fig2_official_baselines")

# ---- Fig 3: choices among 16 samples, full navtest: gain over reference probes (x100 PDMS). Intervals only where recorded. ----
rows = DATA["rows_2"]
fig, ax = plt.subplots(figsize=(6.6, 3.9))
forest(ax, rows, "Gain over the reference probes (PDMS points), 12,146 navtest scenes", xlim=(-2.6, 6.9))
ax.set_title("What the ranking uses matters: geometry carries most of the gain", loc="left", fontsize=9.5)
fig.text(0.01, -0.02, "Whiskers: whole-log 95% intervals where recorded. Paired differences: boosting - rule +1.41 [1.10, 1.73]; "
         "- geometry only +0.71 [0.43, 1.01]; - trajectory only +2.40 [1.94, 2.92].", fontsize=7, color=INK2, ha="left")
save(fig, "fig3_selector_choices")

# ---- Fig 4: per-gate ledger vs the independent SFT evaluation (official full navtest) ----
gates = DATA["gates_1"]
fixed = DATA["fixed_1"]
broken = DATA["broken_1"]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
for ax, (name, color) in zip(axes, (("Boosting", BLUE), ("Ridge", BLUE))):
    for i, g in enumerate(gates):
        ax.barh(i, fixed[name][i], color=color, height=0.5)
        ax.barh(i, -broken[name][i], color=ORANGE, height=0.5)
        ax.text(fixed[name][i] + 8, i, f"+{fixed[name][i]}", va="center", fontsize=8, color=INK2)
        ax.text(-broken[name][i] - 8, i, f"-{broken[name][i]}", va="center", ha="right", fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, lw=1); ax.grid(axis="y", visible=False); ax.set_xlim(-260, 470)
    ax.set_title(name, loc="left", fontsize=9); ax.set_xlabel("scenes")
axes[0].set_yticks(range(3)); axes[0].set_yticklabels(["NC (207 fail)", "DAC (615 fail)", "TTC (571 fail)"]); axes[0].invert_yaxis()
axes[1].plot([], []); fig.suptitle("Repairs (right, blue) and new failures (left, orange) per gate, official navtest", x=0.01, y=1.06, ha="left", fontsize=9.5)
save(fig, "fig4_gate_ledger")

# ---- Fig 5: perception noise, boosting, frozen 1,000-scene subset (flow-matching base) ----
fig, ax = plt.subplots(figsize=(5.2, 3.1))
lv = DATA["lv_1"]
clean = DATA["clean_1"]
noisy = DATA["noisy_1"]
for series, color, label, dx in ((clean, BLUE, "trained on clean geometry", -0.05), (noisy, ORANGE, "trained with noisy geometry", 0.05)):
    for x, pt in zip(lv, series):
        if pt is None:
            continue
        est, lo, hi = pt
        if lo is not None:
            ax.plot([x + dx] * 2, [lo, hi], color=color, lw=2, zorder=2)
        ax.plot([x + dx], [est], "o", color=color, ms=7, mec="white", mew=1.5, zorder=3)
    ax.plot([], [], "o", color=color, label=label)
ax.axhline(0, color=INK2, lw=1); ax.set_xticks(lv); ax.set_xticklabels(["clean", "level 1", "level 2", "level 3"])
ax.set_ylabel("Gain over SFT (PDMS points)"); ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("The ranking degrades with perception error (1,000-scene subset)", loc="left", fontsize=9.5)
save(fig, "fig5_noise")

# ---- Fig 6: KL-budgeted advantage-weighted flow matching, 600 held-out scenes (237 logs), vs SFT baseline 91.373 ----
rows = DATA["rows_3"]
fig, ax = plt.subplots(figsize=(6.6, 3.0))
forest(ax, rows, "PDMS change vs SFT, 600 held-out scenes (whole-log 95% interval)", xlim=(-0.9, 1.1), digits=3)
ax.set_title("Budgeted fitting: held-out score changes versus SFT", loc="left", fontsize=9.5, pad=10)
save(fig, "fig6_awfm")

# ---- Fig 7: label rule decides the drift: margin winners minus PDMS winners (frozen subset, training-script evaluator) ----
rows = DATA["rows_4"]
fig, ax = plt.subplots(figsize=(5.4, 2.0))
forest(ax, rows, "Margin winners minus PDMS winners (points)", xlim=(-2.4, 3.8))
ax.set_title("Changing winner labels changes the observed trade-off", loc="left", fontsize=9.5, pad=10)
fig.text(0.01, -0.22, "Progress interval omitted pending source verification.", fontsize=7, color=INK2)
save(fig, "fig7_label_rule")

# ---- Fig 8: our inference vs the released outputs, same checkpoint, 12,146 scenes: wins / ties / losses ----
fig, ax = plt.subplots(figsize=(5.4, 1.5))
w, t, l = DATA["w_1"]
ax.barh(0, w, color=BLUE, height=0.5); ax.barh(0, t, left=w, color=GRID, height=0.5); ax.barh(0, l, left=w + t, color=ORANGE, height=0.5)
for x0, n, lab in ((0, w, "ours better"), (w, t, "tied"), (w + t, l, "released better")):
    ax.text(x0 + n / 2, 0, f"{lab}\n{n}", ha="center", va="center", fontsize=8, color=INK if lab == "tied" else "white")
ax.set_yticks([]); ax.set_xlim(0, w + t + l); ax.grid(False); ax.spines["left"].set_visible(False)
ax.set_xlabel("scenes (PDMS difference -0.26, whole-log 95% interval [-0.52, +0.002])")
ax.set_title("Reproduction comparison: the source of the gap is unresolved", loc="left", fontsize=9.5)
save(fig, "fig8_sampling_noise")

# ---- Fig 9: which privileged information the ranking needs (full navtest, gain over reference probes). Boosting filled, ridge open. ----
subsets = DATA["subsets_1"]
fig, ax = plt.subplots(figsize=(6.6, 2.9))
for y, (lab, h, r) in enumerate(subsets[::-1]):
    ax.plot([h], [y], "o", color=BLUE, ms=8, mec="white", mew=1.4, zorder=3)
    ax.plot([r], [y], "o", mfc="white", mec=BLUE, ms=8, mew=1.6, zorder=3)
    ax.plot([min(h, r), max(h, r)], [y, y], color=GRID, lw=2, zorder=1)
    ax.annotate(f"{h:+.2f}", (h, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(subsets))); ax.set_yticklabels([x[0] for x in subsets[::-1]]); ax.grid(axis="y", visible=False)
ax.axvline(1.59, color=GRAY, lw=1, ls=(0, (4, 3))); ax.text(1.62, len(subsets) - 0.55, "rule +1.59", fontsize=7.5, color=INK2)
ax.axvline(0, color=INK2, lw=1); ax.set_xlim(-1.6, 3.6); ax.set_ylim(-0.6, len(subsets) - 0.1 + 0.4)
ax.set_xlabel("Gain over the reference probes (PDMS points), 12,146 navtest scenes")
ax.plot([], [], "o", color=BLUE, label="boosting"); ax.plot([], [], "o", mfc="white", mec=BLUE, label="ridge")
ax.legend(loc="lower right", frameon=False, fontsize=8)
ax.set_title("Feature ablations: trajectory + map retains most of the gain", loc="left", fontsize=9.5, pad=10)
save(fig, "fig9_feature_subsets")

# ---- Fig 10: errors in the drivable-area map only (rigid offset), selector trained on clean features, full navtest ----
sig = DATA["sig_1"]
full = DATA["full_1"]
tm = DATA["tm_1"]
rule = DATA["rule_1"]
mp = DATA["mp_1"]
fig, ax = plt.subplots(figsize=(5.6, 3.3))
ax.plot(sig, [f[0] for f in full], "-o", color=BLUE, lw=2, ms=7, mec="white", mew=1.4, label="boosting, all features", zorder=4)
for x, f in zip(sig, full):
    ax.plot([x, x], [f[1], f[2]], color=BLUE, lw=2, alpha=0.45, zorder=3)
ax.plot(sig, tm, "-s", color=BLUE, lw=1.5, ms=6, mfc="white", mec=BLUE, label="boosting, trajectory + map", zorder=3)
ax.plot(sig, rule, "-o", color=GRAY, lw=2, ms=6, mec="white", mew=1.2, label="non-learned rule", zorder=3)
ax.plot(sig, mp, "--", color=GRAY, lw=1.2, label="boosting, map only", zorder=2)
ax.axhline(0, color=INK2, lw=1); ax.set_xticks(sig); ax.set_xticklabels(["clean", "0.2 m", "0.5 m", "1.0 m"])
ax.set_xlabel("Rigid offset applied to the drivable-area map (standard deviation)")
ax.set_ylabel("Gain over reference probes (PDMS points)"); ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.set_title("Ranking under rigid map translations", loc="left", fontsize=9.5, pad=8)
save(fig, "fig10_map_noise")


# Fig 11: matched full-cohort comparison, each against its independent SFT.
cb = DATA['crossbase']
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={'width_ratios': [1, 1.45]})
colors = ['#315b7d', '#9b6744']
for i, (row, color) in enumerate(zip(cb, colors)):
    y = 1-i
    lo, hi = row['ci']
    axes[0].errorbar(row['delta'], y, xerr=[[row['delta']-lo], [hi-row['delta']]], fmt='o', color=color, capsize=4, markersize=7)
    axes[0].annotate(f"+{row['delta']:.2f} [{lo:.2f}, {hi:.2f}]", (row['delta'],y), xytext=(0,13), textcoords='offset points', ha='center', fontsize=8)
    net = [a-b for a,b in zip(row['fixed'], row['broken'])]
    positions = [j + (-.18 if i==0 else .18) for j in range(3)]
    axes[1].bar(positions, net, width=.32, color=color, label=row['base'])
    for x,v in zip(positions,net):
        axes[1].annotate(f'{v:+d}', (x,v), xytext=(0,4 if v>=0 else -12), textcoords='offset points', ha='center', fontsize=8)
axes[0].set_yticks([1,0]); axes[0].set_yticklabels([r['base'] for r in cb]); axes[0].set_ylim(-.65,1.65)
axes[0].set_xlim(0,4); axes[0].axvline(0,color=INK2,lw=.8); axes[0].grid(axis='y',visible=False)
axes[0].set_title('(a) Both selectors improve total score',loc='left',fontsize=10)
axes[0].set_xlabel('PDMS change (points); 95% log-cluster CI')
axes[1].set_xticks(range(3)); axes[1].set_xticklabels(['NC','DAC','TTC']); axes[1].axhline(0,color=INK2,lw=.8)
axes[1].set_ylim(-190,415); axes[1].set_ylabel('Repaired minus newly failing scenes')
axes[1].set_title('(b) Their safety profiles differ',loc='left',fontsize=10)
axes[1].legend(frameon=False,loc='upper right',fontsize=8)
fig.text(.5,.015,'Full navtest: 12,146 scenes. Each base has its own fitted HGB selector and independent SFT reference.\nPrivileged geometry; gate counts are descriptive (no gate uncertainty intervals in the reviewed summary).',ha='center',fontsize=8,color=INK2)
fig.tight_layout(rect=(0,.14,1,1),w_pad=2.5)
save(fig,'fig11_crossbase_tradeoff')
print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
