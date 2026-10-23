"""
Test results of the split sweep, as shown in the paper.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from isdm.metrics import harmonic_mean

DATASET = "GeoPlant"
REGIONS = ["bene", "denmark", "france", "sparse_pa", "agg_regions"]
SPLIT_TYPES = ["geographical", "environmental"]
WINNERS_FROM = "agg_regions"  # best loss per source is chosen here and used for every region
FIG_DIR = Path("outputs/figures/split_sweep_results")
TAB_DIR = Path("outputs/tables/split_sweep_results") 

# run key -> (source, label), in table order
RUNS = {
    "po_deep_maxent": ("PO", "DeepMaxent"),
    "po_bce": ("PO", "BCE"),
    "po_balanced_bce": ("PO", "Bal. BCE"),
    "pa_bce": ("PA", "BCE"),
    "pa_balanced_bce": ("PA", "Bal. BCE"),
    "po_deep_maxent_pa_deep_maxent": ("PO & PA", "DeepMaxent (both)"),
    "po_bce_pa_bce": ("PO & PA", "BCE (both)"),
    "po_balanced_bce_pa_balanced_bce": ("PO & PA", "Bal. BCE (both)"),
    "po_deep_maxent_pa_bce": ("PO & PA", "DeepMaxent + BCE (Sigmoid)"),
    "po_deep_maxent_pa_bce_ippp": ("PO & PA", "DeepMaxent + BCE (IPP)"),
    "po_deep_maxent_pa_balanced_bce": ("PO & PA", "DeepMaxent + Bal. BCE (Sigmoid)"),
    "po_deep_maxent_pa_balanced_bce_ippp": ("PO & PA", "DeepMaxent + Bal. BCE (IPP)"),
}
SOURCE_COLORS = {s: plt.colormaps[c](0.75) for s, c in
                 [("PO", "Oranges"), ("PA", "Purples"), ("PO & PA", "Greens")]}
REGION_LABELS = {"bene": "BeNe", "denmark": "Denmark", "france": "Cont. France",
                 "sparse_pa": "CEE", "agg_regions": "all regions"}
REGION_MARKERS = {"denmark": "o", "bene": "s", "france": "P", "sparse_pa": "D"}  # order by extent
OPTIONS = {"closest": "close", "middle": "medium", "farthest": "far"}

HM = "harmonic_auc"
METRICS = {"avg_auc_species": "Average AUC-species", "avg_auc_site": "Average AUC-site"}
TABLE_METRICS = {"avg_auc_species": "AUC species", "avg_auc_site": "AUC site", HM: "H(AUC)"}
DISTANCE_COL, DISTANCE_LABEL = "distance_km", "Distance (km)"


def results_dir(region, split_type):
    return Path(f"outputs/split_sweep/{DATASET}/{region}_bands/{split_type}/intersect")


def load(region, split_type):
    path = results_dir(region, split_type) / "summary_common.csv"
    if not path.exists():
        print(f"[{region}/{split_type}] {path} not found, skipping.")
        return None
    df = pd.read_csv(path)
    df = df[df["run"].isin(RUNS)].copy()
    df[HM] = harmonic_mean(df["avg_auc_species"], df["avg_auc_site"])
    return df


def split_keys(df):
    """Identify one split; in the aggregated file test numbers repeat across regions."""
    return ["country", "test_number"] if "country" in df.columns else ["test_number"]


def find_winners(df):
    """Best run per source on the harmonic mean, averaged over all splits and distances."""
    means = df.groupby("run")[HM].mean()
    return {s: means[[r for r in means.index if RUNS[r][0] == s]].idxmax() for s in SOURCE_COLORS}


def save(fig, paths):
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=350, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
def loss_table(region, dfs, paths):
    """All losses x (split type x metric), averaged over splits and distances, best in bold."""
    means = {st: df.groupby("run")[list(TABLE_METRICS)].mean() for st, df in dfs.items()}
    # round
    means = {st: m.round(3) for st, m in means.items()}
    n_metrics = len(TABLE_METRICS)

    lines = [r"\begin{table}[htbp]", r"\centering",
             rf"\caption{{Comparison of losses across splits in {REGION_LABELS[region]}.}}",
             rf"\label{{tab:losses-auc-{region}}}", r"\small", r"\setlength{\tabcolsep}{4pt}",
             rf"\begin{{tabular}}{{ll{'c' * n_metrics * len(dfs)}}}", r"\toprule",
             " & ".join(["", ""] + [rf"\multicolumn{{{n_metrics}}}{{c}}{{{st.capitalize()}}}" for st in dfs])
             + r" \\",
             " ".join(rf"\cmidrule(lr){{{3 + i * n_metrics}-{2 + (i + 1) * n_metrics}}}" for i in range(len(dfs))),
             " & ".join(["Source", "Loss(es)"] + list(TABLE_METRICS.values()) * len(dfs)) + r" \\"]

    for source in SOURCE_COLORS:
        runs = [r for r, (s, _) in RUNS.items() if s == source]
        lines.append(r"\midrule")
        source_tex = source.replace("&", r"\&")
        for i, run in enumerate(runs):
            cells = [rf"\multirow{{{len(runs)}}}{{*}}{{{source_tex}}}" if i == 0 else "", RUNS[run][1]]
            for st, m in means.items():
                for metric in TABLE_METRICS:
                    if run not in m.index:
                        cells.append("--")
                        continue
                    v = m.loc[run, metric]
                    cells.append(rf"\textbf{{{v:.3f}}}" if np.isclose(v, m[metric].max(), atol = 1e-4) else f"{v:.3f}")
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines))
        print(f"Saved {path}")


def winners_boxplot(df, winners, paths, ymin=0.4, w=0.5, gap=0.2):
    """Rows = metrics. Left: PA and PO & PA per distance. Middle: averaged over distance. Right: PO."""
    by_split = lambda run, metric: df[df["run"] == run].groupby(split_keys(df))[metric].mean().values
    dist_x = {s: [k * (2 * w + gap) + i * w for k in range(len(OPTIONS))] for i, s in enumerate(["PA", "PO & PA"])}
    avg_x = {"PA": 0, "PO & PA": w}
    spans = [dist_x["PO & PA"][-1] + 2 * w, 3 * w, 2 * w]

    fig, axes = plt.subplots(len(METRICS), 3, figsize=(0.9 * sum(spans) + 1, 2.6 * len(METRICS)),
                             squeeze=False, sharey="row", gridspec_kw=dict(width_ratios=spans, wspace=0.08))

    def box(ax, data, x, source):
        bp = ax.boxplot([data], positions=[x], widths=w, patch_artist=True, manage_ticks=False,
                        medianprops=dict(color="black", linewidth=1),
                        whiskerprops=dict(linewidth=0.9, color="#555"), capprops=dict(linewidth=0.9, color="#555"),
                        flierprops=dict(marker=".", markersize=3, alpha=0.4, color="#888"),
                        boxprops=dict(linewidth=0.8))
        bp["boxes"][0].set(facecolor=SOURCE_COLORS[source], alpha=0.85)

    for row, (metric, label) in enumerate(METRICS.items()):
        ax_dist, ax_avg, ax_po = axes[row]
        for source in ["PA", "PO & PA"]:
            for x, opt in zip(dist_x[source], OPTIONS):
                sub = df[(df["run"] == winners[source]) & (df["option"] == opt)]
                box(ax_dist, sub[metric].values, x, source)
            box(ax_avg, by_split(winners[source], metric), avg_x[source], source)
        box(ax_po, by_split(winners["PO"], metric), 0, "PO")

        ax_dist.set_xlim(-w, spans[0] - w)
        ax_avg.set_xlim(-w, 2 * w)
        ax_po.set_xlim(-w, w)
        ax_dist.set_ylabel(label, fontsize=10)

    yticks = np.arange(ymin, 1.001, 0.05)
    for ax in axes.ravel():
        ax.set_ylim(ymin, 1)
        ax.set_yticks(yticks)
        ax.set_xticks([])
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_yticklabels([f"{y:.1f}" if np.isclose(y * 10, round(y * 10)) else "" for y in yticks], fontsize=8)
    axes[-1, 0].set_xticks([x + w / 2 for x in dist_x["PA"]], list(OPTIONS.values()), fontsize=9)
    axes[-1, 1].set_xticks([w / 2], ["Avg. over distance"], fontsize=9)
    for ax in axes[-1]:
        ax.tick_params(axis="x", length=0)

    handles = [mpatches.Patch(facecolor=SOURCE_COLORS[s], alpha=0.85, label=f"{s}: {RUNS[winners[s]][1]}")
               for s in SOURCE_COLORS]
    fig.subplots_adjust(bottom=0.13)
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0), fontsize=8.5, frameon=False)
    save(fig, paths)


def delta_scatter(df, winners, paths):
    """Per split and distance: metric of the best PO & PA model minus the best PA-only model."""
    keys = split_keys(df) + ["option"]
    pa = df[df["run"] == winners["PA"]]
    popa = df[df["run"] == winners["PO & PA"]]
    merged = pa.merge(popa, on=keys, suffixes=("_pa", "_popa")).sample(frac=1, random_state=123)
    print("Points per region:", merged["country"].value_counts().to_dict())

    cmap = plt.colormaps["Dark2"]
    style = {c: dict(color=cmap(i), marker=m) for i, (c, m) in enumerate(REGION_MARKERS.items())}

    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.5 * len(METRICS), 3.2), sharey=True)
    fig.subplots_adjust(wspace=0.4)
    for ax, (metric, label) in zip(axes, METRICS.items()):
        delta = merged[f"{metric}_popa"] - merged[f"{metric}_pa"]
        for (_, r), d in zip(merged.iterrows(), delta):  # one by one so regions interleave
            ax.scatter(r[f"{DISTANCE_COL}_pa"], d, s=20, edgecolors="white", linewidths=0.4, **style[r["country"]])
        ax.axhline(0, color="#988", linewidth=1, linestyle="--", zorder=0)
        ax.set_xlabel(DISTANCE_LABEL, fontsize=10)
        ax.set_ylabel(f"Δ {label}\n(PO&PA − PA)", fontsize=10)
        ax.yaxis.set_tick_params(labelleft=True)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [Line2D([], [], linestyle="", markersize=6, label=REGION_LABELS[c], markeredgecolor="white",
                      markerfacecolor=s["color"], marker=s["marker"]) for c, s in style.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=len(handles),
               fontsize=9, edgecolor="#cccccc", fancybox=False)
    save(fig, paths)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", default=",".join(REGIONS))
    parser.add_argument("--split_types", default=",".join(SPLIT_TYPES))
    args = parser.parse_args()
    regions, split_types = args.regions.split(","), args.split_types.split(",")

    winners = {st: find_winners(load(WINNERS_FROM, st)) for st in split_types}
    for st, w in winners.items():
        print(f"Best loss per source ({st}): " + ", ".join(f"{s} = {RUNS[r][1]}" for s, r in w.items()))

    for region in regions:
        dfs = {st: df for st in split_types if (df := load(region, st)) is not None}
        if not dfs:
            continue
        name = f"loss_table_{region}.tex"
        loss_table(region, dfs, [results_dir(region, split_types[0]).parent.parent / name, TAB_DIR / name])

        for st, df in dfs.items():
            name = f"winners_boxplot_{region}_{st}.png"
            winners_boxplot(df, winners[st], [results_dir(region, st) / "plots" / name, FIG_DIR / name])
            if region == "agg_regions":
                name = f"delta_vs_distance_{region}_{st}.png"
                delta_scatter(df, winners[st], [results_dir(region, st) / "plots" / name, FIG_DIR / name])


if __name__ == "__main__":
    main()