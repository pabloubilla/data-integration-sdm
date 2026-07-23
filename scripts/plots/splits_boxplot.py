import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np
import matplotlib.patheffects as pe


def _plot_group(ax, data_by_run, runs, run_colors, positions, box_width):
    """Draw one boxplot group (one subplot) and return the winner index."""
    means = [np.mean(d) if len(d) else np.nan for d in data_by_run]
    winner_idx = int(np.nanargmax(means))

    bp = ax.boxplot(
        data_by_run, positions=positions, widths=box_width, patch_artist=True,
        medianprops=dict(color="white", linewidth=0),
        whiskerprops=dict(linewidth=0.9, color="#555"),
        capprops=dict(linewidth=0.9, color="#555"),
        flierprops=dict(marker=".", markersize=3, alpha=0.4, color="#888"),
        boxprops=dict(linewidth=0.8), manage_ticks=False,
    )

    for patch, run in zip(bp["boxes"], runs):
        patch.set_facecolor(run_colors[run])
        patch.set_alpha(0.78)

    for i, (x, d, mean) in enumerate(zip(positions, data_by_run, means)):
        if len(d):
            ax.plot([x - box_width / 2 + 0.05, x + box_width / 2 - 0.05],
                    [np.median(d), np.median(d)], color="white", lw=1.8, zorder=5)
        ax.scatter(x, mean, marker="D", s=28, facecolors="white",
                   edgecolors="#222", linewidths=0.8, zorder=6)
        if i == winner_idx:
            top = np.max(d) if len(d) else mean
            ax.text(x, top + 0.02, "★", ha="center", va="bottom", fontsize=9,
                    color="#FFD700", zorder=7,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="#888")])

    return winner_idx


def plot_auc_boxplots(
    path: str,
    show_xlabels: bool = True,
    run_name_map: dict | None = None,
    run_order: list | None = None,
    box_width: float = 0.6,
    group_spacing: float = 0.05,
    metric="avg_auc_species",
    metric_map: dict | None = None,
    add_average: bool = False,
):

    df = pd.read_csv(os.path.join(path, "summary_common.csv"))
    options = ["closest", "middle", "farthest"]
    options_map = {
        "closest": "$\mathcal{D}^{closest}_{PA}$",
        "middle": "$\mathcal{D}^{middle}_{PA}$",
        "farthest": "$\mathcal{D}^{farthest}_{PA}$"
    }
    all_runs = sorted(df["run"].unique())
    runs = [r for r in run_order if r in all_runs]
    print(f"Runs to plot: {runs}")
    labels = [run_name_map.get(r, r) if run_name_map else r for r in runs]

    palette = plt.get_cmap("tab10")
    run_colors = {run: palette(i % 10) for i, run in enumerate(runs)}
    positions = np.arange(len(runs)) * (box_width + group_spacing)

    n_panels = len(options) + (1 if add_average else 0)
    fig_width = 3.0 * n_panels
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 4), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, option in zip(axes, options):
        subset = df[df["option"] == option]
        data_by_run = [subset[subset["run"] == run][metric].values for run in runs]
        _plot_group(ax, data_by_run, runs, run_colors, positions, box_width)

        ax.set_title(options_map[option], fontsize=11, fontweight="bold", pad=4)
        ax.set_xlim(positions[0] - box_width, positions[-1] + box_width)
        ax.set_xticks([])
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0.45, 1)
        ax.spines[["top", "right"]].set_visible(False)

    if add_average:
        ax = axes[-1]
        # pool across all options for each run to get an overall "average" panel
        data_by_run = [df[df["run"] == run][metric].values for run in runs]
        _plot_group(ax, data_by_run, runs, run_colors, positions, box_width)

        ax.set_title("Average", fontsize=11, fontweight="normal", style="italic", pad=4)
        ax.set_xlim(positions[0] - box_width, positions[-1] + box_width)
        ax.set_xticks([])
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0.45, 1)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel(metric_map[metric], fontsize=10)

    # Bottom legend: one colored swatch per run (matching the boxplot colors),
    # laid out in several columns so it uses the horizontal space instead of
    # a single tall column.
    n_cols = 4 if len(runs) > 6 else len(runs)
    run_handles = [mpatches.Patch(facecolor=run_colors[r], alpha=0.78, edgecolor="none", label=l)
                   for r, l in zip(runs, labels)]
    mean_handle = plt.scatter([], [], marker="D", s=28, facecolors="white",
                               edgecolors="#222", linewidths=0.8, label="Mean")

    n_rows = int(np.ceil((len(runs) + 1) / n_cols))
    legend_bottom = max(0.02, 0.24 - 0.045 * n_rows)

    fig.legend(
        handles=run_handles + [mean_handle],
        loc="lower center", ncol=n_cols, bbox_to_anchor=(0.5, legend_bottom - 0.1),
        fontsize=8, frameon=False, handlelength=1.2, handleheight=1.2,
        columnspacing=1.4, labelspacing=0.6,
    )

    plt.tight_layout(pad=1.2, rect=[0, legend_bottom, 1, 1])
    out = os.path.join(path, f"{metric}_boxplots.png")
    plt.savefig(out, dpi=350, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot AUC boxplots grouped by run and option.")
    parser.add_argument("--dataset_name", default="GeoPlant", type=str)
    parser.add_argument("--split_type", default="geographical", type=str)
    parser.add_argument("--use_overlapping_species", action="store_true")
    parser.add_argument("--metric", default="avg_auc_species", type=str)
    parser.add_argument("--add_average", action="store_true",
                         help="Add an extra subplot showing per-run averages pooled across all options.")
    args = parser.parse_args()
    dataset_name = args.dataset_name
    split_type = args.split_type
    # spec_name = "intersect" if args.use_overlapping_species else "union"
    # for now we will set the default to intersect as the union will probably not be the one finally used
    spec_name = 'intersect'
    output_dir = f"outputs/split_sweep/{dataset_name}/france_bands/{split_type}/{spec_name}"

    run_order = [
        "po_dme",
        "po_balanced_bce",
        "po_bbce",
        "pa_bce",
        "pa_bbce",
        "po_dme_pa_bipp",
        "po_dme_pa_bce",
        "po_dme_pa_bbce",
        "po_bbce_pa_bbce",
        "po_bbce_pa_bbce_wpocov",
        "po_dme_pa_bbce_wpocov",
        "po_dme_pa_dme",

    ]

    run_name_map = {
        "po_dme":          "PO DeepMaxent",
        "po_bbce":         "PO Balanced BCE",
        "pa_bce":          "PA BCE",
        "pa_bbce":         "PA · Balanced BCE",
        "po_balanced_bce": "PO Balanced BCE",
        "po_dme_pa_bipp":  "PO DeepMaxent + PA BIPP",
        "po_dme_pa_bce":   "PO DeepMaxent + PA BCE",
        "po_dme_pa_dme":   "PO DeepMaxent + PA DeepMaxent",
        "po_dme_pa_bbce":  "PO DeepMaxent + PA Balanced BCE",
        "po_bbce_pa_bbce": "PO Balanced BCE + PA Balanced BCE",
        "po_bbce_pa_bbce_wpocov": "PO Balanced BCE + PA Balanced BCE (w_po_cov)",
        "po_dme_pa_bbce_wpocov": "PO DeepMaxent + PA Balanced BCE (w_po_cov)"

    }

    metric_map = {
        "avg_auc_species": "Average AUC-species",
        "avg_auc_site": "Average AUC-site",
    }

    plot_auc_boxplots(output_dir,
                       run_order=run_order,
                       run_name_map=run_name_map,
                       metric=args.metric,
                    metric_map=metric_map,
                       add_average=args.add_average)