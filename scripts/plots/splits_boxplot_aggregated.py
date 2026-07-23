import argparse
import os

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OPTIONS = ["closest", "middle", "farthest"]
OPTIONS_MAP = {
    "closest": "$\mathcal{D}^{closest}_{PA}$",
    "middle": "$\mathcal{D}^{middle}_{PA}$",
    "farthest": "$\mathcal{D}^{farthest}_{PA}$",
}


# ---------------------------------------------------------------------------
# Shared row-plotting logic: draws the closest/middle/farthest boxplots for
# one run of results onto a given row of axes. Used both by the single-figure
# function and by the grid function (one call per row).
# ---------------------------------------------------------------------------

def _plot_row(
    axes: np.ndarray,
    df: pd.DataFrame,
    runs: list,
    labels: list,
    run_colors: dict,
    box_width: float,
    group_spacing: float,
    row_label: str | None = None,
) -> None:
    positions = np.arange(len(runs)) * (box_width + group_spacing)

    for ax, option in zip(axes, OPTIONS):
        subset = df[df["option"] == option]
        data_by_run = [subset[subset["run"] == run]["avg_auc"].values for run in runs]
        means = [np.mean(d) if len(d) else np.nan for d in data_by_run]
        winner_idx = int(np.nanargmax(means)) if np.any(~np.isnan(means)) else None

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
            if not len(d):
                continue
            ax.plot([x - box_width / 2 + 0.05, x + box_width / 2 - 0.05],
                    [np.median(d), np.median(d)], color="white", lw=1.8, zorder=5)
            ax.scatter(x, mean, marker="D", s=28, facecolors="white",
                       edgecolors="#222", linewidths=0.8, zorder=6)
            if i == winner_idx:
                top = np.max(d)
                ax.text(x, top + 0.02, "★", ha="center", va="bottom", fontsize=9,
                        color="#FFD700", zorder=7,
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="#888")])

        ax.set_title(OPTIONS_MAP[option], fontsize=11, fontweight="bold", pad=4)
        ax.set_xlim(positions[0] - box_width, positions[-1] + box_width)
        ax.set_xticks([])
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0.45, 1)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Average AUC", fontsize=9)

    if row_label is not None:
        mid_ax = axes[len(axes) // 2]
        mid_ax.text(0.5, 1.38, row_label, transform=mid_ax.transAxes,
                    ha="center", va="bottom", fontsize=11, fontweight="bold")


def _make_run_colors(runs: list) -> dict:
    palette = plt.get_cmap("tab10")
    return {run: palette(i % 10) for i, run in enumerate(runs)}


def _add_legend(fig, runs: list, labels: list, run_colors: dict) -> None:
    fig.legend(
        handles=[mpatches.Patch(facecolor=run_colors[r], alpha=0.78, label=l) for r, l in zip(runs, labels)]
            + [plt.scatter([], [], marker="D", s=28, facecolors="white", edgecolors="#222", linewidths=0.8, label="Mean")],
        loc="center left", ncol=1, bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False,
    )


# ---------------------------------------------------------------------------
# Single-run figure (same behaviour as before, now built on top of _plot_row)
# ---------------------------------------------------------------------------

def plot_auc_boxplots(
    path: str,
    run_name_map: dict | None = None,
    run_order: list | None = None,
    box_width: float = 0.6,
    group_spacing: float = 0.05,
):
    df = pd.read_csv(os.path.join(path, "summary_common.csv"))
    all_runs = sorted(df["run"].unique())
    runs = [r for r in (run_order or all_runs) if r in all_runs]
    print(f"Runs to plot: {runs}")
    labels = [run_name_map.get(r, r) if run_name_map else r for r in runs]
    run_colors = _make_run_colors(runs)

    fig, axes = plt.subplots(1, 3, figsize=(9, 4), sharey=True)
    _plot_row(axes, df, runs, labels, run_colors, box_width, group_spacing)
    _add_legend(fig, runs, labels, run_colors)

    plt.tight_layout(pad=1.2)
    out = os.path.join(path, "auc_boxplots.png")
    plt.savefig(out, dpi=350, bbox_inches="tight")
    plt.savefig(os.path.join(path, "auc_boxplots.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(path, "auc_boxplots.svg"), bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Grid figure: one row per (dataset_name, split_type, spec_name) combination,
# each row reusing the exact same closest/middle/farthest plotting as above.
# ---------------------------------------------------------------------------

def plot_auc_boxplots_grid(
    combos: list[dict],
    out_dir: str,
    run_name_map: dict | None = None,
    run_order: list | None = None,
    box_width: float = 0.6,
    group_spacing: float = 0.05,
    row_height: float = 3.2,
):
    """
    combos: list of dicts, each with keys:
        "path"      -> directory containing summary_common.csv
        "row_label" -> short string shown as a title above the row
    """
    dfs, valid_combos = [], []
    for combo in combos:
        csv_path = os.path.join(combo["path"], "summary_common.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping (not found): {csv_path}")
            continue
        dfs.append(pd.read_csv(csv_path))
        valid_combos.append(combo)

    if not dfs:
        raise FileNotFoundError("None of the requested combinations had a summary_common.csv.")

    all_runs = sorted(set().union(*[set(df["run"].unique()) for df in dfs]))
    runs = [r for r in (run_order or all_runs) if r in all_runs]
    labels = [run_name_map.get(r, r) if run_name_map else r for r in runs]
    run_colors = _make_run_colors(runs)

    n_rows = len(dfs)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, row_height * n_rows), sharey=True, squeeze=False)

    for row_idx, (df, combo) in enumerate(zip(dfs, valid_combos)):
        _plot_row(
            axes[row_idx], df, runs, labels, run_colors, box_width, group_spacing,
            row_label=combo.get("row_label"),
        )

    _add_legend(fig, runs, labels, run_colors)

    plt.tight_layout(pad=1.2, h_pad=4.0)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "auc_boxplots_grid.png")
    plt.savefig(out, dpi=350, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "auc_boxplots_grid.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "auc_boxplots_grid.svg"), bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot AUC boxplots grouped by run and option.")
    parser.add_argument("--dataset_name", default="GeoPlant", type=str)
    parser.add_argument("--split_type", default="geographical", type=str)
    parser.add_argument("--use_overlapping_species", action="store_true")
    parser.add_argument("--grid", action="store_true",
                         help="Plot all dataset/split_type/spec combinations stacked as rows in one figure.")
    args = parser.parse_args()

    run_order = [
        "po_dme",
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
        "pa_bce":          "PA BCE",
        "pa_bbce":         "PA · Balanced BCE",
        "po_dme_pa_bipp":  "PO DeepMaxent + PA BIPP",
        "po_dme_pa_bce":   "PO DeepMaxent + PA BCE",
        "po_dme_pa_dme":   "PO DeepMaxent + PA DeepMaxent",
        "po_dme_pa_bbce":  "PO DeepMaxent + PA Balanced BCE",
        "po_bbce_pa_bbce": "PO Balanced BCE + PA Balanced BCE",
        "po_bbce_pa_bbce_wpocov": "PO Balanced BCE + PA Balanced BCE (w_po_cov)",
        "po_dme_pa_bbce_wpocov": "PO DeepMaxent + PA Balanced BCE (w_po_cov)",
    }

    def path_for(dataset_name, split_type, use_overlapping_species):
        spec_name = "intersect" if use_overlapping_species else "union"
        return f"outputs/split_sweep/{dataset_name}/france_gaussian_grid/{split_type}/{spec_name}"

    if not args.grid:
        output_dir = path_for(args.dataset_name, args.split_type, args.use_overlapping_species)
        plot_auc_boxplots(output_dir, run_order=run_order, run_name_map=run_name_map)
    else:
        # edit this list to whichever combinations you want stacked together
        combos_spec = [
            {"dataset_name": args.dataset_name, "split_type": "geographical", "use_overlapping_species": False},
            {"dataset_name": args.dataset_name, "split_type": "geographical", "use_overlapping_species": True},
            {"dataset_name": args.dataset_name, "split_type": "environmental", "use_overlapping_species": False},
            {"dataset_name": args.dataset_name, "split_type": "environmental", "use_overlapping_species": True},
        ]
        combos = [
            {
                "path": path_for(c["dataset_name"], c["split_type"], c["use_overlapping_species"]),
                "row_label": f"{c['split_type']} · {'intersect' if c['use_overlapping_species'] else 'union'}",
            }
            for c in combos_spec
        ]
        output_dir_grid = f"outputs/split_sweep/grid_{args.dataset_name}"
        os.makedirs(output_dir_grid, exist_ok=True)
        plot_auc_boxplots_grid(
            combos,
            out_dir=output_dir_grid,
            run_order=run_order,
            run_name_map=run_name_map,
        )