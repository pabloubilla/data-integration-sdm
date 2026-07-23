"""
Hyperparameter Sensitivity Analysis
Usage:
    python sensitivity_analysis.py results.csv [--save output.png]

Expects columns named param_* and a target column (default: mean_auc).
"""

import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("csv", help="Path to results CSV")
parser.add_argument("--target", default="mean_auc")
parser.add_argument("--save", metavar="PATH", default=None, help="Save figure to this path instead of showing")
args = parser.parse_args()

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(args.csv).apply(pd.to_numeric, errors="coerce")
TARGET = args.target
PARAMS = [c for c in df.columns if c.startswith("param_")]

if TARGET not in df.columns:
    sys.exit(f"Column '{TARGET}' not found. Available: {list(df.columns)}")

print(f"Loaded {len(df)} rows | {len(PARAMS)} params | target: {TARGET}\n")

# ── one-way sensitivity ───────────────────────────────────────────────────────
sens = {p: df.groupby(p)[TARGET].mean().pipe(lambda s: s.max() - s.min()) for p in PARAMS}
sens_series = pd.Series(sens).sort_values(ascending=False)

# ── spearman correlation ──────────────────────────────────────────────────────
corr = df[PARAMS + [TARGET]].corr(method="spearman")[TARGET].drop(TARGET)

# ── print summary ─────────────────────────────────────────────────────────────
print("── Sensitivity (max−min mean AUC) ──")
print(sens_series.to_string(), "\n")
print("── Spearman ρ ──")
print(corr.reindex(sens_series.index).to_string(), "\n")

print("── Two-way interactions ──")
for p1, p2 in combinations(PARAMS, 2):
    if df[p1].nunique() > 1 and df[p2].nunique() > 1:
        tbl = df.pivot_table(index=p1, columns=p2, values=TARGET, aggfunc="mean")
        if tbl.size > 1:
            print(f"{p1} × {p2}\n{tbl.to_string()}\n")

# ── plot ──────────────────────────────────────────────────────────────────────
COLORS = ["#378ADD", "#1D9E75", "#D85A30", "#D4537E", "#7F77DD"]
POS, NEG = "#378ADD", "#E24B4A"

fig = plt.figure(figsize=(14, 9))
fig.suptitle("Hyperparameter Sensitivity Analysis", fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# tornado
ax0 = fig.add_subplot(gs[0, 0])
bars = ax0.barh([p.replace("param_", "") for p in sens_series.index], sens_series.values,
                color=[POS if v >= 0 else NEG for v in sens_series], height=0.6)
ax0.set_xlabel("Max − Min mean AUC", fontsize=10)
ax0.set_title("Tornado: one-way sensitivity", fontsize=11)
for bar, val in zip(bars, sens_series.values):
    ax0.text(val + 1e-5, bar.get_y() + bar.get_height() / 2, f"{val:.5f}", va="center", fontsize=8)

# spearman
ax1 = fig.add_subplot(gs[0, 1])
corr_sorted = corr.reindex(sens_series.index)
ax1.barh([p.replace("param_", "") for p in corr_sorted.index], corr_sorted.values,
         color=[POS if v >= 0 else NEG for v in corr_sorted], height=0.6)
ax1.axvline(0, color="gray", linewidth=0.8, linestyle="--")
ax1.set_xlabel("Spearman ρ", fontsize=10)
ax1.set_title("Correlation with mean AUC", fontsize=11)
ax1.set_xlim(-1, 1)

# mean AUC per param value
ax2 = fig.add_subplot(gs[1, :])
x_offset, xtick_pos, xtick_labels = 0, [], []
for i, p in enumerate(sens_series.index):
    grp = df.groupby(p)[TARGET].mean().reset_index()
    xs = np.arange(len(grp)) + x_offset
    ax2.bar(xs, grp[TARGET], color=COLORS[i % len(COLORS)], width=0.6,
            label=p.replace("param_", ""), alpha=0.85)
    for x, (_, row) in zip(xs, grp.iterrows()):
        ax2.text(x, row[TARGET] + 2e-4, f"{row[TARGET]:.4f}",
                 ha="center", va="bottom", fontsize=7.5, rotation=45)
        xtick_pos.append(x)
        xtick_labels.append(row[p])
    x_offset += len(grp) + 0.8

ax2.set_xticks(xtick_pos)
ax2.set_xticklabels(xtick_labels, fontsize=8, rotation=30, ha="right")
ax2.set_ylabel("Mean AUC", fontsize=10)
ax2.set_title("Mean AUC by parameter value", fontsize=11)
ax2.legend(fontsize=8, loc="lower right", title="param", title_fontsize=8)
ax2.set_ylim(df[TARGET].min() - 0.01, df[TARGET].max() + 0.015)

if args.save:
    plt.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.save}")
else:
    plt.show()