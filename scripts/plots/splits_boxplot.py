import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import os
import numpy as np
import matplotlib.patheffects as pe


CATEGORY_ORDER = ["PO only", "PA only", "PO + PA"]
CATEGORY_CMAPS = {
    "PO only": "Oranges",
    "PA only": "Purples",
    "PO + PA": "Greens",
}
CATEGORY_SHADE_RANGE = (0.45, 0.85)  # avoid too-light/too-dark ends of the colormap

# --- NEW: texture support -------------------------------------------------
# One hatch pattern per category (set to "" for a category to keep it plain).
# matplotlib hatch chars: / \ | - + x o O . *  (repeat chars to increase density, e.g. "///")
CATEGORY_HATCHES = {
    "PO only": "",
    "PA only": "",
    "PO + PA": "",
}

# If you'd rather vary texture *within* a category (one hatch per run, cycling
# through this list) instead of one hatch per whole category, set this True.
HATCH_BY_RUN_WITHIN_CATEGORY = False
RUN_HATCH_CYCLE = ["", "///", "xxx", "...", "\\\\\\", "+++", "OO", "---"]
# ---------------------------------------------------------------------------


def _infer_category(run: str) -> str:
    """Guess PO-only / PA-only / PO+PA from tokens in the run name."""
    tokens = run.split("_")
    has_po = "po" in tokens
    has_pa = "pa" in tokens
    if has_po and has_pa:
        return "PO + PA"
    elif has_po:
        return "PO only"
    elif has_pa:
        return "PA only"
    return "Other"


def _build_category_map(runs, run_category_map):
    cat_map = {}
    for run in runs:
        if run_category_map and run in run_category_map:
            cat_map[run] = run_category_map[run]
        else:
            cat_map[run] = _infer_category(run)
    return cat_map


def _order_runs_by_category(runs, cat_map):
    """Stable-sort runs into category clusters (preserves relative order within each)."""
    order_index = {cat: i for i, cat in enumerate(CATEGORY_ORDER)}

    def sort_key(item):
        idx, run = item
        cat = cat_map[run]
        return (order_index.get(cat, len(CATEGORY_ORDER)), idx)

    indexed = list(enumerate(runs))
    indexed.sort(key=sort_key)
    return [run for _, run in indexed]


def _assign_category_colors(runs, cat_map):
    """One colormap per category; shade varies across runs within it."""
    runs_by_cat = {}
    for run in runs:
        runs_by_cat.setdefault(cat_map[run], []).append(run)

    run_colors = {}
    for cat, cat_runs in runs_by_cat.items():
        cmap_name = CATEGORY_CMAPS.get(cat, "Greys")
        cmap = plt.get_cmap(cmap_name)
        n = len(cat_runs)
        lo, hi = CATEGORY_SHADE_RANGE
        shades = np.linspace(lo, hi, n) if n > 1 else np.array([hi])
        for run, shade in zip(cat_runs, shades):
            run_colors[run] = cmap(shade)
    return run_colors


def _assign_category_hatches(runs, cat_map, run_hatch_map=None):
    """One hatch pattern per run. Either:
      - a fixed hatch per category (CATEGORY_HATCHES), or
      - a cycling hatch per run *within* a category (HATCH_BY_RUN_WITHIN_CATEGORY=True), or
      - explicit per-run overrides via run_hatch_map={"run_name": "///", ...}
    run_hatch_map (if given) always wins for the runs it names.
    """
    runs_by_cat = {}
    for run in runs:
        runs_by_cat.setdefault(cat_map[run], []).append(run)

    run_hatches = {}
    for cat, cat_runs in runs_by_cat.items():
        if HATCH_BY_RUN_WITHIN_CATEGORY:
            for i, run in enumerate(cat_runs):
                run_hatches[run] = RUN_HATCH_CYCLE[i % len(RUN_HATCH_CYCLE)]
        else:
            hatch = CATEGORY_HATCHES.get(cat, "")
            for run in cat_runs:
                run_hatches[run] = hatch

    if run_hatch_map:
        for run, hatch in run_hatch_map.items():
            if run in run_hatches:
                run_hatches[run] = hatch

    return run_hatches


def _compute_positions(runs, cat_map, box_width, group_spacing, category_gap):
    """Box positions with a normal gap within a cluster and category_gap
    between clusters. Also returns cluster_spans (for background shading)
    and cluster_centers (for the label under each cluster)."""
    positions = []
    x = 0.0
    prev_cat = None
    cluster_positions = {}

    for run in runs:
        cat = cat_map[run]
        if prev_cat is not None and cat != prev_cat:
            x += category_gap
        positions.append(x)
        cluster_positions.setdefault(cat, []).append(x)
        x += box_width + group_spacing
        prev_cat = cat

    pad = group_spacing / 2
    cluster_spans = {
        cat: (min(xs) - box_width / 2 - pad, max(xs) + box_width / 2 + pad)
        for cat, xs in cluster_positions.items()
    }
    cluster_centers = {cat: np.mean(xs) for cat, xs in cluster_positions.items()}
    return np.array(positions), cluster_spans, cluster_centers


def _plot_group(ax, data_by_run, runs, run_colors, run_hatches, positions, box_width):
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
        hatch = run_hatches.get(run, "")
        if hatch:
            patch.set_hatch(hatch)
            # hatch lines take the edgecolor; keep it subtle but visible
            patch.set_edgecolor("#555")
            patch.set_linewidth(0.8)

    for i, (x, d, mean) in enumerate(zip(positions, data_by_run, means)):
        if len(d):
            ax.plot([x - box_width / 2 + 0.05, x + box_width / 2 - 0.05],
                    [np.median(d), np.median(d)], color="white", lw=1.8, zorder=5)
        ax.scatter(x, mean, marker="D", s=28, facecolors="white",
                   edgecolors="#222", linewidths=0.8, zorder=6)
        # if i == winner_idx:
        #     top = np.max(d) if len(d) else mean
        #     ax.text(x, top + 0.02, "★", ha="center", va="bottom", fontsize=9,
        #             color="#FFD700", zorder=7,
        #             path_effects=[pe.withStroke(linewidth=1.5, foreground="#888")])

    return winner_idx


def _shade_category_clusters(ax, cluster_spans, cat_map):
    """Tint the background behind each category's cluster"""
    for cat, (x0, x1) in cluster_spans.items():
        cmap = plt.get_cmap(CATEGORY_CMAPS.get(cat, "Greys"))
        ax.axvspan(x0, x1, facecolor=cmap(0.55), alpha=0.08, zorder=0, linewidth=0)


def _add_category_labels(ax, cluster_centers, show_labels):
    if not show_labels:
        return
    # y in axes-fraction (below the plot), x in data coords
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for cat in CATEGORY_ORDER:
        if cat in cluster_centers:
            cmap = plt.get_cmap(CATEGORY_CMAPS.get(cat, "Greys"))
            ax.text(cluster_centers[cat], -0.02, cat, transform=trans,
                    ha="center", va="top", fontsize=7.5,
                    # style="italic",
                    fontweight="medium", color=cmap(0.85))


def _draw_custom_legend(fig, legend_ax_rect, runs_by_cat, run_colors, run_hatches):
    """One column per source category (entries stacked vertically), plus
    a final column for the Mean marker, styled as a normal one-row entry
    so its spacing matches the rest."""
    lax = fig.add_axes(legend_ax_rect)
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.axis("off")

    present_cats = [c for c in CATEGORY_ORDER if c in runs_by_cat]
    columns = [(cat, runs_by_cat[cat]) for cat in present_cats] + [("Mean", [("mean", "Mean")])]

    # --- tune these to reshape the legend ---
    gap = 0.1           # horizontal gap between columns
    avail_width = 0.6     # total width spent on columns (increase to spread out)
    header_frac = 0.1     # vertical space for the bold category header
    swatch_w, swatch_h_frac = 0.028, 0.4   # swatch size (height as fraction of a row)

    def col_weight(cat, entries):
        if cat == "Mean":
            return 6
        max_len = max((len(lbl) for _, lbl in entries), default=4)
        return max(max_len, len(cat)) + 2  # +2 for the swatch

    weights = [col_weight(cat, entries) for cat, entries in columns]
    col_widths = [avail_width * w / sum(weights) for w in weights]

    max_rows = max(len(entries) for _, entries in columns)
    row_height = (1 - header_frac) / max_rows
    swatch_h = row_height * swatch_h_frac

    x = 0.0
    for (cat, entries), width in zip(columns, col_widths):
        is_mean_col = cat == "Mean"
        header_color = "#333" if is_mean_col else plt.get_cmap(CATEGORY_CMAPS.get(cat, "Greys"))(0.85)
        lax.text(x + 0.02, 1 - header_frac / 2, cat if not is_mean_col else "", ha="left", va="center",
                  fontsize=9, fontweight="bold", color=header_color)

        for k, (run, lbl) in enumerate(entries):
            y = 1 - header_frac - (k + 0.5) * row_height
            sw_x = x + 0.012
            if is_mean_col:
                lax.scatter(sw_x + swatch_w / 2, y, marker="D", s=28, facecolors="white",
                            edgecolors="#222", linewidths=0.8, zorder=3)
            else:
                hatch = run_hatches.get(run, "")
                rect_kwargs = dict(facecolor=run_colors[run], alpha=0.78, edgecolor="none")
                if hatch:
                    rect_kwargs.update(hatch=hatch, edgecolor="#555", linewidth=0.6)
                lax.add_patch(mpatches.Rectangle(
                    (sw_x, y - swatch_h / 2), swatch_w, swatch_h, **rect_kwargs))
            lax.text(sw_x + swatch_w + 0.008, y, lbl, ha="left", va="center", fontsize=7.6)

        if x > 0:
            lax.axvline(x - gap / 2, ymin=0.02, ymax=0.96, color="#ddd", linewidth=0.7)
        x += width + gap


def plot_auc_boxplots(
    path: str,
    show_xlabels: bool = True,
    run_name_map: dict | None = None,
    run_order: list | None = None,
    run_category_map: dict | None = None,   # override auto-detected PO/PA/PO+PA category per run
    run_hatch_map: dict | None = None,      # NEW: override hatch pattern per run, e.g. {"po_dme": "///"}
    box_width: float = 0.6,                 # width of each box
    group_spacing: float = 0.05,            # gap between boxes within a category cluster
    category_gap: float = 0.5,              # extra gap between category clusters
    metric="avg_auc_species",
    metric_map: dict | None = None,
    add_average: bool = False,
):
    # Most layout tweaks live in two places: the args above (box/cluster
    # spacing) and the "tune these" block inside _draw_custom_legend
    # (legend column widths/spacing). legend_height below controls how
    # much vertical room the legend gets.
    #
    # Texture knobs: CATEGORY_HATCHES / HATCH_BY_RUN_WITHIN_CATEGORY / RUN_HATCH_CYCLE
    # at the top of this file, or pass run_hatch_map for one-off overrides.

    df = pd.read_csv(os.path.join(path, "summary_common.csv"))
    options = ["closest", "middle", "farthest"]
    options_map = {
        "closest": "$\mathcal{D}^{closest}_{PA}$",
        "middle": "$\mathcal{D}^{middle}_{PA}$",
        "farthest": "$\mathcal{D}^{farthest}_{PA}$"
    }
    all_runs = sorted(df["run"].unique())
    runs = [r for r in run_order if r in all_runs]

    # --- category assignment + clustering ---
    cat_map = _build_category_map(runs, run_category_map)
    runs = _order_runs_by_category(runs, cat_map)
    print(f"Runs to plot (grouped by source): {runs}")
    print(f"Categories: {[cat_map[r] for r in runs]}")

    labels = [run_name_map.get(r, r) if run_name_map else r for r in runs]
    run_colors = _assign_category_colors(runs, cat_map)
    run_hatches = _assign_category_hatches(runs, cat_map, run_hatch_map)
    positions, cluster_spans, cluster_centers = _compute_positions(
        runs, cat_map, box_width, group_spacing, category_gap
    )

    n_panels = len(options) + (1 if add_average else 0)
    fig_width = 3.0 * n_panels
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 4.3), sharey=True)
    if n_panels == 1:
        axes = [axes]

    def _finish_panel(ax, title, title_kwargs):
        ax.set_title(title, fontsize=11, pad=4, **title_kwargs)
        ax.set_xlim(positions[0] - box_width, positions[-1] + box_width)
        ax.set_xticks([])
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0.45, 1)
        ax.spines[["top", "right"]].set_visible(False)
        _shade_category_clusters(ax, cluster_spans, cat_map)
        _add_category_labels(ax, cluster_centers, show_xlabels)

    for ax, option in zip(axes, options):
        subset = df[df["option"] == option]
        data_by_run = [subset[subset["run"] == run][metric].values for run in runs]
        _plot_group(ax, data_by_run, runs, run_colors, run_hatches, positions, box_width)
        _finish_panel(ax, options_map[option], dict(fontweight="bold"))

    if add_average:
        ax = axes[-1]
        data_by_run = [df[df["run"] == run][metric].values for run in runs]
        _plot_group(ax, data_by_run, runs, run_colors, run_hatches, positions, box_width)
        _finish_panel(ax, "Average", dict(fontweight="normal", style="italic"))

    axes[0].set_ylabel(metric_map[metric], fontsize=10)

    # --- legend: one row per source category, drawn on a dedicated axis ---
    runs_by_cat = {}
    for run, label in zip(runs, labels):
        runs_by_cat.setdefault(cat_map[run], []).append((run, label))

    n_rows = max((len(v) for v in runs_by_cat.values()), default=1)
    legend_height = 0.09 + 0.032 * n_rows
    legend_bottom = 0.02

    plt.tight_layout(pad=1.2, rect=[0, legend_bottom + legend_height, 1, 1])
    _draw_custom_legend(
        fig,
        legend_ax_rect=[0.03, legend_bottom, 0.94, legend_height],
        runs_by_cat=runs_by_cat,
        run_colors=run_colors,
        run_hatches=run_hatches,
    )
    out = os.path.join(path, f"{metric}_boxplots.png")
    plt.savefig(out, dpi=350, bbox_inches="tight")


    out_svg = os.path.join(path, f"{metric}_boxplots.svg")
    # for SVG transparent background
    plt.savefig(out_svg, bbox_inches="tight", facecolor="none")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot AUC boxplots grouped by run and option.")
    parser.add_argument("--dataset_name", default="GeoPlant", type=str)
    parser.add_argument("--split_type", default="geographical", type=str)
    parser.add_argument("--use_overlapping_species", action="store_true")
    parser.add_argument("--metric", default="avg_auc_species", type=str)
    parser.add_argument("--region", default="france", type=str, help="Region to run the split sweep on (default: france).")
    parser.add_argument("--add_average", action="store_true",
                         help="Add an extra subplot showing per-run averages pooled across all options.")
    args = parser.parse_args()
    dataset_name = args.dataset_name
    split_type = args.split_type
    region = args.region
    # spec_name = "intersect" if args.use_overlapping_species else "union"
    # for now we will set the default to intersect as the union will probably not be the one finally used
    spec_name = 'intersect'
    output_dir = f"outputs/split_sweep/{dataset_name}/{region}_bands/{split_type}/{spec_name}"
    run_order = [
        "po_deep_maxent",
        "po_balanced_bce",
        "pa_bce",
        "pa_balanced_bce",
        "po_deep_maxent_pa_deep_maxent",
        "po_balanced_bce_pa_balanced_bce",
        "po_deep_maxent_pa_bce",
        "po_deep_maxent_pa_balanced_bce",
        "po_balanced_bce_pa_balanced_bce_wpocov",
        "po_deep_maxent_pa_balanced_bce_wpocov",
        "po_deep_maxent_pa_balanced_bce_ippp",

    ]

    

    run_name_map = {
        "po_deep_maxent":          "PO DeepMaxent",
        "po_balanced_bce":         "PO Bal. BCE",
        "pa_bce":          "PA BCE",
        "pa_balanced_bce":         "PA Bal. BCE",
        "po_balanced_bce": "PO Balanced BCE",
        "po_deep_maxent_pa_balanced_bce_ippp":  "PO DeepMaxent + PA Bal. IPP",
        "po_deep_maxent_pa_bce":   "PO DeepMaxent + PA BCE",
        "po_deep_maxent_pa_deep_maxent":   "PO DeepMaxent + PA DeepMaxent",
        "po_deep_maxent_pa_balanced_bce":  "PO DeepMaxent + PA Bal. BCE",
        "po_balanced_bce_pa_balanced_bce": "PO Bal. BCE + PA Bal. BCE",
        "po_balanced_bce_pa_balanced_bce_wpocov": "PO Bal. BCE + PA Bal. BCE (w_po_cov)",
        "po_deep_maxent_pa_balanced_bce_wpocov": "PO DeepMaxent + PA Bal. BCE (w_po_cov)"

    }

    # Example: give the "w_po_cov" variants a hatch so they stand out even
    # though they share a color family with their non-wpocov siblings.
    run_hatch_map = {
        # "po_bbce_pa_bbce_wpocov": "///",
        # "po_dme_pa_bbce_wpocov": "///",
    }

    metric_map = {
        "avg_auc_species": "Average AUC-species",
        "avg_auc_site": "Average AUC-site",
    }

    plot_auc_boxplots(output_dir,
                       run_order=run_order,
                       run_name_map=run_name_map,
                       run_hatch_map=run_hatch_map,
                       metric=args.metric,
                       metric_map=metric_map,
                       add_average=args.add_average)