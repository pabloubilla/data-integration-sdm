import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import os
import numpy as np
import matplotlib.patheffects as pe


CATEGORY_ORDER = ["PO only", "PA only", "PO + PA"]

SOURCE_REMAP = {
    "po": "PO only",
    "pa": "PA only",
    "popa": "PO + PA"
}

CATEGORY_NAME_MAP = {
    "PO only": "Loss for PO data\n(single-source)",
    "PA only": "Loss for PA data\n(single-source)",
    "PO + PA": "Loss for PO data & PA data\n(integrated model) ",
}

CATEGORY_CMAPS = {
    "PO only": "Oranges",
    "PA only": "Purples",
    "PO + PA": "Greens",
}


CATEGORY_SHADE_RANGE = (0.2, 0.97)  # avoid too-light/too-dark ends of the colormap

# --- texture support -------------------------------------------------
# One hatch pattern per category (set to "" for a category to keep it plain).
# matplotlib hatch chars: / \ | - + x o O . *  (repeat chars to increase density, e.g. "///")
CATEGORY_HATCHES = {
    "PO only": "",
    "PA only": "",
    "PO + PA": "",
}

# To vary texture *within* a category (one hatch per run, cycling
# through this list) instead of one hatch per whole category, set this True.
HATCH_BY_RUN_WITHIN_CATEGORY = False
RUN_HATCH_CYCLE = ["", "///", "xxx", "...", "\\\\\\", "+++", "OO", "---"]

OPTIONS_MATH_MAP = {
    "closest": "$\mathcal{D}^{closest}_{PA}$",
    "middle": "$\mathcal{D}^{middle}_{PA}$",
    "farthest": "$\mathcal{D}^{farthest}_{PA}$",
}
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
    """Draw one boxplot group (one subplot). Runs with no data at all are
    simply skipped (matplotlib draws nothing for an empty array at that
    position), so a canonical run list can be reused even when a given
    row/panel doesn't have every run."""
    means = [np.mean(d) if len(d) else np.nan for d in data_by_run]
    if np.all(np.isnan(means)):
        winner_idx = None
    else:
        winner_idx = int(np.nanargmax(means))

    bp = ax.boxplot(
        data_by_run, positions=positions, widths=box_width, patch_artist=True,
        medianprops=dict(color="black", linewidth=1),
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

    # For now let's not plot the mean
    # for i, (x, d, mean) in enumerate(zip(positions, data_by_run, means)):
    #     if len(d):
    #         ax.plot([x - box_width / 2 + 0.05, x + box_width / 2 - 0.05],
    #                 [np.median(d), np.median(d)], color="white", lw=1.8, zorder=5)
    #         ax.scatter(x, mean, marker="D", s=28, facecolors="white",
    #                    edgecolors="#222", linewidths=0.8, zorder=6)
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
    """One column per source category (entries stacked vertically)."""
    lax = fig.add_axes(legend_ax_rect)
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.axis("off")

    present_cats = [c for c in CATEGORY_ORDER if c in runs_by_cat]
    columns = [(cat, runs_by_cat[cat]) for cat in present_cats]  # Mean column dropped — unused for now

    # --- tune these to reshape the legend ---
    gap = 0.02             # horizontal gap between columns
    avail_width = 0.4      # total width spent on columns (increase to spread out)
    header_fontsize = 8
    linespacing = 1.2
    swatch_w, swatch_h_frac = 0.028, 0.4   # swatch size (height as fraction of a row)

    def header_lines(cat):
        return CATEGORY_NAME_MAP.get(cat, cat).count("\n") + 1

    def col_weight(cat, entries):
        header_width = max(len(line) for line in CATEGORY_NAME_MAP.get(cat, cat).split("\n"))
        max_len = max((len(lbl) for _, lbl in entries), default=4)
        return max(max_len, header_width) + 2  # +2 for the swatch

    weights = [col_weight(cat, entries) for cat, entries in columns]
    col_widths = [avail_width * w / sum(weights) for w in weights]

    # Convert "N lines of `header_fontsize`pt text" into an axes-fraction
    # height, using the actual physical height of this legend axes — this
    # is what makes header_frac correct regardless of how tall/short the
    # legend ends up being (unlike a fixed axes-fraction-per-line guess).
    axes_height_inches = legend_ax_rect[3] * fig.get_figheight()
    points_per_axes_height = axes_height_inches * 72.0
    max_header_lines = max((header_lines(cat) for cat, _ in columns), default=1)
    header_frac = (header_fontsize * linespacing * max_header_lines) / points_per_axes_height
    header_frac = min(header_frac * 1.15, 0.9)  # small safety margin, clamped

    max_rows = max(len(entries) for _, entries in columns)
    row_height = (1 - header_frac) / max_rows
    swatch_h = row_height * swatch_h_frac

    x = 0.3
    for (cat, entries), width in zip(columns, col_widths):
        header_color = plt.get_cmap(CATEGORY_CMAPS.get(cat, "Greys"))(0.85)
        lax.text(x, 1.0, CATEGORY_NAME_MAP.get(cat, cat), ha="left", va="top",
                  fontsize=header_fontsize, fontweight="bold", color=header_color,
                  linespacing=linespacing)

        for k, (run, lbl) in enumerate(entries):
            y = 1 - header_frac - (k + 0.5) * row_height
            sw_x = x + 0.012
            hatch = run_hatches.get(run, "")
            rect_kwargs = dict(facecolor=run_colors[run], alpha=0.78, edgecolor="none")
            if hatch:
                rect_kwargs.update(hatch=hatch, edgecolor="#555", linewidth=0.6)
            lax.add_patch(mpatches.Rectangle(
                (sw_x, y - swatch_h / 2), swatch_w, swatch_h, **rect_kwargs))
            lax.text(sw_x + swatch_w + 0.008, y, lbl, ha="left", va="center", fontsize=7.6)

        if x > 0.3:
            lax.axvline(x - gap / 2, ymin=0.02, ymax=0.96, color="#ddd", linewidth=0.7)
        x += width + gap

def plot_auc_boxplots(
    path: str,
    show_xlabels: bool = True,
    run_name_map: dict | None = None,
    run_order: list | None = None,
    run_category_map: dict | None = None,   # override auto-detected PO/PA/PO+PA category per run
    run_hatch_map: dict | None = None,      # override hatch pattern per run, e.g. {"po_dme": "///"}
    box_width: float = 0.6,                 # width of each box
    group_spacing: float = 0.05,            # gap between boxes within a category cluster
    category_gap: float = 0.5,              # extra gap between category clusters
    metric="avg_auc_species",
    metric_map: dict | None = None,
    add_average: bool = False,
):
    """Single (path, metric) combination -> one figure with a subplot per
    option (closest/middle/farthest[/average]). Unchanged from before;
    kept for one-off plots. For aggregating several combinations into one
    figure, see plot_auc_boxplots_grid below."""

    df = pd.read_csv(os.path.join(path, "summary_common.csv"))
    options = ["closest", "middle", "farthest"]

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
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 4.3), sharey="row", squeeze=False)
    if n_panels == 1:
        axes = [axes]

    def _finish_panel(ax, title, title_kwargs):
        # ax.set_title(title, fontsize=11, pad=4, **title_kwargs)
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
        _finish_panel(ax, OPTIONS_MATH_MAP[option], dict(fontweight="bold"))

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


# ---------------------------------------------------------------------------
# NEW: aggregate several (path, metric) combinations into one grid figure.
# Each entry in `row_configs` becomes one row of subplots (closest / middle /
# farthest [/ average]); colors, hatches, run ordering, and the legend are
# computed once so every row is visually consistent and directly comparable.
# ---------------------------------------------------------------------------

def _canonical_runs(paths, run_order):
    """Union of runs actually present across all the CSVs involved, kept in
    run_order's ordering. This is what makes colors/positions identical
    across rows even if one split is missing a run or two."""
    present = set()
    for path in set(paths):
        df = pd.read_csv(os.path.join(path, "summary_common.csv"))
        present.update(df["run"].unique())
    return [r for r in run_order if r in present]


def _harmonic_mean(a, b):
    """Row-wise harmonic mean of two columns, NaN-safe. Returns NaN where
    both values are zero/undefined instead of dividing by zero."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        hm = np.where(denom > 0, 2 * a * b / denom, np.nan)
    return hm


def _load_with_derived_metrics(path, derived_metrics):
    """Load summary_common.csv and add any derived columns (e.g. a harmonic
    mean between two existing metric columns) before it's used for plotting.
    derived_metrics: {"new_col_name": ("metric_a", "metric_b")}, harmonic
    mean of metric_a/metric_b is computed row-wise."""
    df = pd.read_csv(os.path.join(path, "summary_common.csv"))
    if derived_metrics:
        for new_col, (metric_a, metric_b) in derived_metrics.items():
            df[new_col] = _harmonic_mean(df[metric_a], df[metric_b])
    return df


def plot_auc_boxplots_grid(
    row_configs: list,                      # [{"path": ..., "metric": ..., "row_title": ...}, ...]
    run_order: list,
    show_xlabels: bool = True,
    run_name_map: dict | None = None,
    run_category_map: dict | None = None,
    run_hatch_map: dict | None = None,
    box_width: float = 0.6,
    group_spacing: float = 0.00,
    category_gap: float = 0.01,
    metric_map: dict | None = None,
    add_average: bool = False,
    options=("closest", "middle", "farthest"),
    out_path: str = "combined_boxplots.png",
    derived_metrics: dict | None = None,
    fig_width: float | None = None,
    fig_height: float | None = None,
):
    """
    Stack several (path, metric) combinations as rows of subplots in one
    figure. Typical use: fix the split_type (one `path`) and put each
    metric in its own row, e.g. avg_auc_species / avg_auc_site / a
    harmonic-mean row derived from both — call this once per split_type
    to get one aggregated figure per split.

    row_configs: list of dicts, each:
        {
          "path": "<dir containing summary_common.csv>",
          "metric": "avg_auc_species",        # column (or derived column) to plot for this row
          "row_title": "Average AUC-species",  # label drawn on the left of the row
        }
    Columns are the same across all rows (closest/middle/farthest[/average]),
    so results line up for direct comparison down each column.

    derived_metrics: optional {"new_col_name": ("metric_a", "metric_b")} —
    computes the row-wise harmonic mean of metric_a/metric_b into a new
    column on load, so it can be referenced as a `metric` in row_configs
    (e.g. {"harmonic_auc": ("avg_auc_species", "avg_auc_site")}).

    fig_width / fig_height: override the auto-computed figure size (useful
    for callers that reuse this function with a very different row/col
    count than the usual 3-metric grid, e.g. a single wide row).
    """
    if not row_configs:
        raise ValueError("row_configs must contain at least one entry")

    paths = [rc["path"] for rc in row_configs]
    runs = _canonical_runs(paths, run_order)

    cat_map = _build_category_map(runs, run_category_map)
    runs = _order_runs_by_category(runs, cat_map)
    print(f"[grid] Runs to plot (grouped by source): {runs}")
    print(f"[grid] Categories: {[cat_map[r] for r in runs]}")

    labels = [run_name_map.get(r, r) if run_name_map else r for r in runs]
    run_colors = _assign_category_colors(runs, cat_map)
    run_hatches = _assign_category_hatches(runs, cat_map, run_hatch_map)
    positions, cluster_spans, cluster_centers = _compute_positions(
        runs, cat_map, box_width, group_spacing, category_gap
    )

    n_cols = len(options) + (1 if add_average else 0)
    n_rows = len(row_configs)
    fig_width = fig_width if fig_width is not None else 3.0 * n_cols + 1.0
    fig_height = fig_height if fig_height is not None else 2.5 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False, sharey="row")

    dfs = {path: _load_with_derived_metrics(path, derived_metrics) for path in set(paths)}

    for row_i, rc in enumerate(row_configs):
        df = dfs[rc["path"]]
        metric = rc["metric"]
        row_axes = axes[row_i]

        for col_i, option in enumerate(options):
            ax = row_axes[col_i]
            subset = df[df["option"] == option]
            data_by_run = [subset[subset["run"] == run][metric].values for run in runs]
            _plot_group(ax, data_by_run, runs, run_colors, run_hatches, positions, box_width)
            if row_i == 0:
                ax.set_title(OPTIONS_MATH_MAP.get(option, option), fontsize=11, pad=4, fontweight="bold")

        if add_average:
            ax = row_axes[-1]
            data_by_run = [df[df["run"] == run][metric].values for run in runs]
            _plot_group(ax, data_by_run, runs, run_colors, run_hatches, positions, box_width)
            if row_i == 0:
                ax.set_title("Average", fontsize=11, pad=4, style="italic")

        for ax in row_axes:
            ax.set_xlim(positions[0] - box_width, positions[-1] + box_width)
            ax.set_xticks([])
            ax.tick_params(axis="x", length=0)
            ax.set_ylim(0.45, 1)
            ax.spines[["top", "right"]].set_visible(False)
            _shade_category_clusters(ax, cluster_spans, cat_map)
            if row_i == n_rows - 1:
                _add_category_labels(ax, cluster_centers, show_xlabels)

        ylabel = metric_map.get(metric, metric) if metric_map else metric
        row_axes[0].set_ylabel(ylabel, fontsize=10)

    # --- shared legend (one set of colors/hatches for the whole grid) ---
    runs_by_cat = {}
    for run, label in zip(runs, labels):
        runs_by_cat.setdefault(cat_map[run], []).append((run, label))

    n_legend_rows = max((len(v) for v in runs_by_cat.values()), default=1)
    legend_height = 0.05 + 0.018 * n_legend_rows
    legend_bottom = 0.01

    plt.tight_layout(pad=1.2, rect=[0.05, legend_bottom + legend_height, 1, 1])

    # # --- row titles, drawn in figure coordinates after layout so they sit
    # # centred on each row regardless of how tight_layout resized things ---
    # plot_top, plot_bottom = 1.0, legend_bottom + legend_height
    # row_span = (plot_top - plot_bottom) / n_rows
    # for row_i, rc in enumerate(row_configs):
    #     row_title = rc.get("row_title")
    #     if not row_title:
    #         continue
    #     y_center = plot_top - (row_i + 0.5) * row_span
    #     fig.text(0.012, y_center, row_title, rotation=90, ha="left", va="center",
    #               fontsize=12, fontweight="bold")

    _draw_custom_legend(
        fig,
        legend_ax_rect=[0.06, legend_bottom, 0.9, legend_height],
        runs_by_cat=runs_by_cat,
        run_colors=run_colors,
        run_hatches=run_hatches,
    )

    plt.savefig(out_path, dpi=350, bbox_inches="tight")
    out_svg = os.path.splitext(out_path)[0] + ".svg"
    plt.savefig(out_svg, bbox_inches="tight", facecolor="none")
    print(f"[grid] Saved → {out_path}")
    plt.show()



def plot_winning_methods_by_distance(
    row_configs: list,                  # [{"path": ..., "metric": ..., "row_title": ...}, ...] — one row per metric
    winner_runs: dict,                  # {"PO only": run_name, "PA only": run_name, "PO + PA": run_name}
    run_name_map: dict | None = None,
    box_width: float = 0.55,
    within_group_gap: float = 0.08,     # gap between the close/mid/far boxes of the same winner
    between_group_gap: float = 0.55,    # gap between the PO / PA / Integrated clusters
    metric_map: dict | None = None,
    options=("closest", "middle", "farthest"),
    single_box_categories=("PO only",),  # categories where distance doesn't actually vary -> one box only
    out_path: str = "winning_methods_by_distance.png",
    derived_metrics: dict | None = None,
):
    """
    Alternative layout for the 'winning method per source' plot: instead of
    one column per distance (closest/middle/farthest), distance becomes
    adjacent boxes *within* each source's cluster, so a given winner's
    behavior across distances reads as one visual group rather than being
    spread across separate panels. Clusters are always ordered PO -> PA ->
    Integrated (matches CATEGORY_ORDER). PO is plotted as a single box by
    default, since distance doesn't meaningfully vary for PO-only runs —
    see single_box_categories to change that.

    Deliberately simpler than the big grid: one solid color per category
    (no shade gradient, no hatches), no background cluster tinting, and a
    plain matplotlib legend instead of the custom multi-column one.
    Distance is never written on the x-axis — it's conveyed purely by a
    box's position within its own cluster (left→right = closest→farthest).
    """
    if not row_configs:
        raise ValueError("row_configs must contain at least one entry")

    present_cats = [c for c in CATEGORY_ORDER if c in winner_runs]
    short_label = {"PO only": "PO", "PA only": "PA", "PO + PA": "Integrated"}
    n_rows = len(row_configs)

    fig_width = 1 * sum(1 if c in single_box_categories else len(options) for c in present_cats) + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, 2.6 * n_rows), squeeze=False, sharex=True)
    axes = axes[:, 0]

    dfs = {rc["path"]: _load_with_derived_metrics(rc["path"], derived_metrics) for rc in row_configs}

    # one solid color per category — no gradient, no hatch
    cat_colors = {cat: plt.get_cmap(CATEGORY_CMAPS.get(cat, "Greys"))(0.75) for cat in present_cats}

    # shared x positions (identical for every row): PO -> PA -> Integrated,
    # each cluster's boxes packed tight (within_group_gap) and clusters
    # separated further apart (between_group_gap)
    # positions_by_cat, tick_positions, tick_labels = {}, [], []
    # x = 0.0
    # for cat in present_cats:
    #     n_boxes = 1 if cat in single_box_categories else len(options)
    #     cat_positions = [x + i * (box_width + within_group_gap) for i in range(n_boxes)]
    #     positions_by_cat[cat] = cat_positions
    #     tick_positions.append(float(np.mean(cat_positions)))
    #     tick_labels.append(short_label.get(cat, cat))
    #     x = cat_positions[-1] + box_width + between_group_gap

    distance_short_label = {"closest": "close", "middle": "mid", "farthest": "far"}

    positions_by_cat, tick_positions, tick_labels = {}, [], []
    x = 0.0
    for cat in present_cats:
        n_boxes = 1 if cat in single_box_categories else len(options)
        cat_positions = [x + i * (box_width + within_group_gap) for i in range(n_boxes)]
        positions_by_cat[cat] = cat_positions

        if cat in single_box_categories:
            tick_positions.append(cat_positions[0])
            tick_labels.append(short_label.get(cat, cat))
        else:
            for pos, opt in zip(cat_positions, options):
                tick_positions.append(pos)
                tick_labels.append(distance_short_label.get(opt, opt))

        x = cat_positions[-1] + box_width + between_group_gap

    xmin = positions_by_cat[present_cats[0]][0] - box_width
    xmax = positions_by_cat[present_cats[-1]][-1] + box_width
    
    ymin = 0.6





    for row_i, rc in enumerate(row_configs):
        ax = axes[row_i]
        df = dfs[rc["path"]]
        metric = rc["metric"]

        for cat in present_cats:
            run = winner_runs[cat]
            cat_positions = positions_by_cat[cat]
            if cat in single_box_categories:
                # only one representative option, not all 3 pooled together
                # — avoids artificially tripling the sample size for a
                # value that doesn't actually depend on distance
                data = [df[(df["run"] == run) & (df["option"] == options[0])][metric].values]
            else:
                data = [df[(df["run"] == run) & (df["option"] == opt)][metric].values for opt in options]

            bp = ax.boxplot(
                data, positions=cat_positions, 
                widths=box_width, patch_artist=True,
                medianprops=dict(color="black", linewidth=1),
                whiskerprops=dict(linewidth=0.9, color="#555"),
                capprops=dict(linewidth=0.9, color="#555"),
                flierprops=dict(marker=".", markersize=3, alpha=0.4, color="#888"),
                boxprops=dict(linewidth=0.8), manage_ticks=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(cat_colors[cat])
                patch.set_alpha(0.85)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ylabel = metric_map.get(metric, metric) if metric_map else metric
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks([])  # cleared on every row; the bottom row gets real ticks below

    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels(tick_labels, fontsize=9)
    axes[-1].tick_params(axis="x", length=0)

    # simple legend: one patch per category, labelled with the actual
    # winning run so it doubles as "which run is this color"
    legend_handles = [
        mpatches.Patch(
            facecolor=cat_colors[cat], alpha=0.85,
            label=f"{short_label.get(cat, cat)}: "
                  f"{run_name_map.get(winner_runs[cat], winner_runs[cat]) if run_name_map else winner_runs[cat]}",
        )
        for cat in present_cats
    ]

    # add horizontal grid lines behind the boxes, for easier visual comparison across rows
    for ax in axes:
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
    
    # add ticks also every 0.05 but only show every 0.1 the number
    yticks = np.arange(ymin, 1.01, 0.05)
    for ax in axes:
        ax.set_yticks(yticks)
        ax.set_yticklabels(
            [f"{y:.2f}" if np.isclose(y * 10, np.round(y * 10)) else "" for y in yticks],
            fontsize=8)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(present_cats),
               bbox_to_anchor=(0.5, 0.0), fontsize=8.5, frameon=False)

    plt.savefig(out_path, dpi=350, bbox_inches="tight")
    out_svg = os.path.splitext(out_path)[0] + ".svg"
    plt.savefig(out_svg, bbox_inches="tight", facecolor="none")
    print(f"[winners-by-distance] Saved → {out_path}")
    plt.show()


def plot_winning_methods_by_distance_v2_old(
    row_configs: list,                  # [{"path": ..., "metric": ..., "row_title": ...}, ...] — one row per metric
    winner_runs: dict,                  # {"PO only": run_name, "PA only": run_name, "PO + PA": run_name}
    run_name_map: dict | None = None,
    box_width: float = 0.5,
    within_group_gap: float = 0.0,     # gap between PA/Integrated boxes sitting at the same distance
    between_group_gap: float = 0.2,    # gap between close/mid/far clusters, and between avg-panel boxes
    metric_map: dict | None = None,
    options=("closest", "middle", "farthest"),
    single_box_categories=("PO only",),  # categories left OUT of the by-distance panel (no distance variation)
    ymin: float = 0.55,
    out_path: str = "winning_methods_by_distance.png",
    derived_metrics: dict | None = None,
):
    """
    Two panels per row (one row per metric):
      - LEFT ('by distance'): x ticks are close/mid/far; at each tick, PA
        and Integrated are drawn side by side so those two — the ones that
        actually vary by distance — are easy to compare directly.
      - RIGHT ('averaged'): one box per category (PO, PA, Integrated),
        each pooling every distance's rows together, so all three sources
        — PO included — can be compared on equal footing overall.

    Deliberately simple styling: solid color per category (no gradient, no
    hatch), a plain legend, and horizontal reference gridlines with every
    other y-tick labeled (every 0.1) for easier row-to-row comparison.
    """
    if not row_configs:
        raise ValueError("row_configs must contain at least one entry")

    present_cats = [c for c in CATEGORY_ORDER if c in winner_runs]
    dist_cats = [c for c in present_cats if c not in single_box_categories]
    short_label = {"PO only": "PO", "PA only": "PA", "PO + PA": "PO & PA"}
    distance_short_label = {"closest": "close", "middle": "medium", "farthest": "far"}
    n_rows = len(row_configs)

    dfs = {rc["path"]: _load_with_derived_metrics(rc["path"], derived_metrics) for rc in row_configs}

    # one solid color per category — no gradient, no hatch
    cat_colors = {cat: plt.get_cmap(CATEGORY_CMAPS.get(cat, "Greys"))(0.75) for cat in present_cats}

    # --- LEFT panel positions: one close/mid/far cluster, each holding a
    # box per dist_cats (PA, Integrated), packed tight ---
    dist_positions_by_cat = {cat: [] for cat in dist_cats}
    dist_tick_positions, dist_tick_labels = [], []
    x = 0.0
    for opt in options:
        cluster_positions = [x + i * (box_width + within_group_gap) for i in range(len(dist_cats))]
        for cat, pos in zip(dist_cats, cluster_positions):
            dist_positions_by_cat[cat].append(pos)
        dist_tick_positions.append(float(np.mean(cluster_positions)))
        dist_tick_labels.append(distance_short_label.get(opt, opt))
        x = cluster_positions[-1] + box_width + between_group_gap
    dist_xmin = dist_positions_by_cat[dist_cats[0]][0] - box_width
    dist_xmax = dist_positions_by_cat[dist_cats[-1]][-1] + box_width

    # --- RIGHT panel positions: one box per category, PO included here ---
    avg_x = np.arange(len(present_cats)) * (box_width + within_group_gap)
    avg_positions = dict(zip(present_cats, avg_x))
    avg_tick_positions = list(avg_positions.values())
    avg_tick_labels = ['', 'Average', '']
    avg_xmin = avg_tick_positions[0] - box_width
    avg_xmax = avg_tick_positions[-1] + box_width

    # column widths proportional to how many boxes each panel actually draws
    # dist_units = len(options) * len(dist_cats) + (len(options) - 1) * 1.5
    # avg_units = len(present_cats) * 1.5
    # fig, axes = plt.subplots(
    #     n_rows, 2, figsize=(0.4 * (dist_units + avg_units) + 1, 2.6 * n_rows),
    #     squeeze=False, sharex="col", sharey="row",
    #     gridspec_kw={"width_ratios": [dist_units, avg_units], "wspace": 0.08},
    # )
    dist_span = dist_xmax - dist_xmin
    avg_span = avg_xmax - avg_xmin
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(0.9 * (dist_span + avg_span) + 1, 2.6 * n_rows),
        squeeze=False, sharex="col", sharey="row",
        gridspec_kw={"width_ratios": [dist_span, avg_span], "wspace": 0.08},
    )

    def _draw_box(ax, data, position, color):
        bp = ax.boxplot(
            [data], positions=[position], widths=box_width, patch_artist=True,
            medianprops=dict(color="black", linewidth=1),
            whiskerprops=dict(linewidth=0.9, color="#555"),
            capprops=dict(linewidth=0.9, color="#555"),
            flierprops=dict(marker=".", markersize=3, alpha=0.4, color="#888"),
            boxprops=dict(linewidth=0.8), manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

    for row_i, rc in enumerate(row_configs):
        ax_dist, ax_avg = axes[row_i]
        df = dfs[rc["path"]]

        metric = rc["metric"]

        # LEFT: PA / Integrated, split by distance
        for cat in dist_cats:
            run = winner_runs[cat]
            for pos, opt in zip(dist_positions_by_cat[cat], options):
                data = df[(df["run"] == run) & (df["option"] == opt)][metric].values
                _draw_box(ax_dist, data, pos, cat_colors[cat])

        # RIGHT: PO / PA / Integrated, each pooled across every distance
        for cat in present_cats:
            run = winner_runs[cat]
            data = df[df["run"] == run].groupby("test_number")[metric].mean().values  # average across distances
            _draw_box(ax_avg, data, avg_positions[cat], cat_colors[cat])

        ax_dist.set_xlim(dist_xmin, dist_xmax)
        ax_avg.set_xlim(avg_xmin, avg_xmax)
        for ax in (ax_dist, ax_avg):
            ax.set_ylim(ymin, 1)
            ax.spines[["top", "right"]].set_visible(False)
        # ax_avg.spines["left"].set_visible(False)
        # ax_avg.tick_params(axis="y", left=False, labelleft=False)

        ylabel = metric_map.get(metric, metric) if metric_map else metric
        ax_dist.set_ylabel(ylabel, fontsize=10)

    axes[-1, 0].set_xticks(dist_tick_positions)
    axes[-1, 0].set_xticklabels(dist_tick_labels, fontsize=9)
    axes[-1, 0].tick_params(axis="x", length=0)
    axes[-1, 1].set_xticks(avg_tick_positions)
    axes[-1, 1].set_xticklabels(avg_tick_labels, fontsize=9)
    axes[-1, 1].tick_params(axis="x", length=0)

    # simple legend: one patch per category, labelled with the actual
    # winning run so it doubles as "which run is this color"
    legend_handles = [
        mpatches.Patch(
            facecolor=cat_colors[cat], alpha=0.85,
            label=f"{short_label.get(cat, cat)}: "
                  f"{run_name_map.get(winner_runs[cat], winner_runs[cat]) if run_name_map else winner_runs[cat]}",
        )
        for cat in present_cats
    ]

    # horizontal reference gridlines behind the boxes, both panels
    for ax in axes.ravel():
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)

    # y-ticks every 0.05, numeric label only every 0.1 (float-safe check —
    # `% 0.1 == 0` fails for arange floats due to precision, see np.isclose)
    yticks = np.arange(ymin, 1.01, 0.05)
    for ax in axes[:, 0]:
        ax.set_yticks(yticks)
        ax.set_yticklabels(
            [f"{y:.2f}" if np.isclose(y * 10, np.round(y * 10)) else "" for y in yticks],
            fontsize=8,
        )
    for ax in axes[:, 1]:
        ax.set_yticks(yticks)  # keeps gridlines aligned; labels stay hidden (labelleft=False above)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(present_cats),
               bbox_to_anchor=(0.5, 0.0), fontsize=8.5, frameon=False)

    plt.savefig(out_path, dpi=350, bbox_inches="tight")
    out_svg = os.path.splitext(out_path)[0] + ".svg"
    plt.savefig(out_svg, bbox_inches="tight", facecolor="none")
    print(f"[winners-by-distance] Saved → {out_path}")
    plt.show()

def plot_winning_methods_by_distance_v2(
    row_configs: list,                  # [{"path": ..., "metric": ..., "row_title": ...}, ...] — one row per metric
    winner_runs: dict,                  # {"PO only": run_name, "PA only": run_name, "PO + PA": run_name}
    run_name_map: dict | None = None,
    box_width: float = 0.5,
    within_group_gap: float = 0.0,     # gap between PA/Integrated boxes sitting at the same distance
    between_group_gap: float = 0.2,    # gap between close/mid/far clusters, and between avg-panel boxes
    metric_map: dict | None = None,
    options=("closest", "middle", "farthest"),
    single_box_categories=("PO only",),  # categories left OUT of the by-distance panel (no distance variation)
    ymin: float = 0.55,
    out_path: str = "winning_methods_by_distance.png",
    derived_metrics: dict | None = None,
):
    """
    Three panels per row (one row per metric):
      - LEFT ('by distance'): x ticks are close/mid/far; at each tick, PA
        and Integrated are drawn side by side so those two — the ones that
        actually vary by distance — are easy to compare directly.
      - MIDDLE ('averaged'): one box per non-single-box category (PA,
        Integrated), each pooling every distance's rows together.
      - RIGHT ('PO'): a single box per single-box category (PO only),
        which has no distance variation to begin with.

    Deliberately simple styling: solid color per category (no gradient, no
    hatch), a plain legend, and horizontal reference gridlines with every
    other y-tick labeled (every 0.1) for easier row-to-row comparison.
    """
    if not row_configs:
        raise ValueError("row_configs must contain at least one entry")

    present_cats = [c for c in CATEGORY_ORDER if c in winner_runs]
    dist_cats = [c for c in present_cats if c not in single_box_categories]
    po_cats = [c for c in present_cats if c in single_box_categories]
    short_label = {"PO only": "PO", "PA only": "PA", "PO + PA": "PO & PA"}
    distance_short_label = {"closest": "close", "middle": "medium", "farthest": "far"}
    n_rows = len(row_configs)

    dfs = {rc["path"]: _load_with_derived_metrics(rc["path"], derived_metrics) for rc in row_configs}

    # one solid color per category — no gradient, no hatch
    cat_colors = {cat: plt.get_cmap(CATEGORY_CMAPS.get(cat, "Greys"))(0.75) for cat in present_cats}

    # --- LEFT panel positions: one close/mid/far cluster, each holding a
    # box per dist_cats (PA, Integrated), packed tight ---
    dist_positions_by_cat = {cat: [] for cat in dist_cats}
    dist_tick_positions, dist_tick_labels = [], []
    x = 0.0
    for opt in options:
        cluster_positions = [x + i * (box_width + within_group_gap) for i in range(len(dist_cats))]
        for cat, pos in zip(dist_cats, cluster_positions):
            dist_positions_by_cat[cat].append(pos)
        dist_tick_positions.append(float(np.mean(cluster_positions)))
        dist_tick_labels.append(distance_short_label.get(opt, opt))
        x = cluster_positions[-1] + box_width + between_group_gap
    dist_xmin = dist_positions_by_cat[dist_cats[0]][0] - box_width
    dist_xmax = dist_positions_by_cat[dist_cats[-1]][-1] + box_width

    # --- MIDDLE panel positions: one box per dist_cat (PA, Integrated), pooled ---
    avg_x = np.arange(len(dist_cats)) * (box_width + within_group_gap)
    avg_positions = dict(zip(dist_cats, avg_x))
    avg_box_positions = list(avg_positions.values())
    avg_xmin = avg_box_positions[0] - box_width
    avg_xmax = avg_box_positions[-1] + box_width
    # single centered tick reading "Average" instead of one label per box
    avg_tick_positions = [float(np.mean(avg_box_positions))]
    avg_tick_labels = ["Avg. over distance"]

    # --- RIGHT panel positions: one box per single-box category (PO only) ---
    po_x = np.arange(len(po_cats)) * (box_width + within_group_gap)
    po_positions = dict(zip(po_cats, po_x))
    po_tick_positions = list(po_positions.values())
    # po_tick_labels = [short_label.get(cat, cat) for cat in po_cats]
    po_xmin = po_tick_positions[0] - box_width
    po_xmax = po_tick_positions[-1] + box_width

    # column widths proportional to how many boxes each panel actually draws
    dist_span = dist_xmax - dist_xmin
    avg_span = avg_xmax - avg_xmin
    po_span = po_xmax - po_xmin
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(0.9 * (dist_span + avg_span + po_span) + 1, 2.6 * n_rows),
        squeeze=False, sharex="col", sharey="row",
        gridspec_kw={"width_ratios": [dist_span, avg_span, po_span], "wspace": 0.08},
    )

    def _draw_box(ax, data, position, color):
        bp = ax.boxplot(
            [data], positions=[position], widths=box_width, patch_artist=True,
            medianprops=dict(color="black", linewidth=1),
            whiskerprops=dict(linewidth=0.9, color="#555"),
            capprops=dict(linewidth=0.9, color="#555"),
            flierprops=dict(marker=".", markersize=3, alpha=0.4, color="#888"),
            boxprops=dict(linewidth=0.8), manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

    for row_i, rc in enumerate(row_configs):
        ax_dist, ax_avg, ax_po = axes[row_i]
        df = dfs[rc["path"]]

        metric = rc["metric"]

        # LEFT: PA / Integrated, split by distance
        for cat in dist_cats:
            run = winner_runs[cat]
            for pos, opt in zip(dist_positions_by_cat[cat], options):
                data = df[(df["run"] == run) & (df["option"] == opt)][metric].values
                _draw_box(ax_dist, data, pos, cat_colors[cat])

        # MIDDLE: PA / Integrated, each pooled across every distance
        for cat in dist_cats:
            run = winner_runs[cat]
            data = df[df["run"] == run].groupby("test_number")[metric].mean().values  # average across distances
            _draw_box(ax_avg, data, avg_positions[cat], cat_colors[cat])

        # RIGHT: PO only, pooled (no distance variation to begin with)
        for cat in po_cats:
            run = winner_runs[cat]
            data = df[df["run"] == run].groupby("test_number")[metric].mean().values
            _draw_box(ax_po, data, po_positions[cat], cat_colors[cat])

        ax_dist.set_xlim(dist_xmin, dist_xmax)
        ax_avg.set_xlim(avg_xmin, avg_xmax)
        ax_po.set_xlim(po_xmin, po_xmax)
        for ax in (ax_dist, ax_avg, ax_po):
            ax.set_ylim(ymin, 1)
            ax.spines[["top", "right"]].set_visible(False)

        ylabel = metric_map.get(metric, metric) if metric_map else metric
        ax_dist.set_ylabel(ylabel, fontsize=10)

    axes[-1, 0].set_xticks(dist_tick_positions)
    axes[-1, 0].set_xticklabels(dist_tick_labels, fontsize=9)
    axes[-1, 0].tick_params(axis="x", length=0)
    axes[-1, 1].set_xticks(avg_tick_positions)
    axes[-1, 1].set_xticklabels(avg_tick_labels, fontsize=9)
    axes[-1, 1].tick_params(axis="x", length=0)
    axes[-1, 2].set_xticks(po_tick_positions)
    axes[-1, 2].set_xticklabels([""], fontsize=9)
    axes[-1, 2].tick_params(axis="x", length=0)

    # simple legend: one patch per category, labelled with the actual
    # winning run so it doubles as "which run is this color"
    legend_handles = [
        mpatches.Patch(
            facecolor=cat_colors[cat], alpha=0.85,
            label=f"{short_label.get(cat, cat)}: "
                  f"{run_name_map.get(winner_runs[cat], winner_runs[cat]) if run_name_map else winner_runs[cat]}",
        )
        for cat in present_cats
    ]

    # horizontal reference gridlines behind the boxes, all panels
    for ax in axes.ravel():
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)

    # y-ticks every 0.05, numeric label only every 0.1 (float-safe check —
    # `% 0.1 == 0` fails for arange floats due to precision, see np.isclose)
    yticks = np.arange(ymin, 1.01, 0.05)
    for ax in axes[:, 0]:
        ax.set_yticks(yticks)
        ax.set_yticklabels(
            [f"{y:.2f}" if np.isclose(y * 10, np.round(y * 10)) else "" for y in yticks],
            fontsize=8,
        )
    for ax in axes[:, 1:].ravel():
        ax.set_yticks(yticks)  # keeps gridlines aligned; labels stay hidden

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(present_cats),
               bbox_to_anchor=(0.5, 0.0), fontsize=8.5, frameon=False)

    plt.savefig(out_path, dpi=350, bbox_inches="tight")
    out_svg = os.path.splitext(out_path)[0] + ".svg"
    plt.savefig(out_svg, bbox_inches="tight", facecolor="none")
    print(f"[winners-by-distance] Saved → {out_path}")
    plt.show()


def build_method_comparison_table(
    split_paths: dict,                 # e.g. {"Environmental": path_env, "Geographical": path_geo}, in column order
    metrics: list,                      # e.g. ["avg_auc_species", "avg_auc_site"]; derived_metrics appended after
    run_order: list,
    run_name_map: dict | None = None,
    run_category_map: dict | None = None,
    metric_map: dict | None = None,
    derived_metrics: dict | None = None,   # {"harmonic_auc": ("avg_auc_species", "avg_auc_site")}
    include_std: bool = False,
    bold_best: bool = True,
    higher_is_better: bool = True,
    decimals: int = 3,
    small: bool = True,          # wraps the tabular in \small for a more compact table
    tabcolsep: str = "4pt",      # tighter horizontal cell padding (LaTeX default is 6pt)
    caption: str | None = None,
    label: str | None = None,
    out_path: str = "method_comparison_table.tex",
) -> str:
    r"""
    Nice table to compare methods averaged over distance
    """
    if not split_paths:
        raise ValueError("split_paths must contain at least one entry")
    
    header_order = ['Geographical', 'Environmental']

    all_metrics = list(metrics) + (list(derived_metrics.keys()) if derived_metrics else [])
    metric_labels = [metric_map.get(m, m) if metric_map else m for m in all_metrics]

    # canonical run list/order, shared across every split so rows line up
    runs = _canonical_runs(list(split_paths.values()), run_order)
    cat_map = _build_category_map(runs, run_category_map)
    runs = _order_runs_by_category(runs, cat_map)
    run_labels = {r: (run_name_map.get(r, r) if run_name_map else r) for r in runs}

    runs_by_cat: dict[str, list[str]] = {}
    for run in runs:
        runs_by_cat.setdefault(cat_map[run], []).append(run)
    present_cats = [c for c in CATEGORY_ORDER if c in runs_by_cat]
    short_cat_label = {"PO only": "PO", "PA only": "PA", "PO + PA": "PO \\& PA"}

    dfs = {split_label: _load_with_derived_metrics(path, derived_metrics)
           for split_label, path in split_paths.items()}

    # --- compute every cell (mean, std) up front, so bold-best can compare
    # within a column before any text formatting happens ---
    values: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    for split_label, df in dfs.items():
        values[split_label] = {}
        for metric in all_metrics:
            col = {}
            for run in runs:
                data = df[df["run"] == run][metric].values
                col[run] = (float(np.mean(data)), float(np.std(data))) if len(data) else (float("nan"), float("nan"))
            values[split_label][metric] = col

    n_data_cols = len(split_paths) * len(all_metrics)
    col_format = "ll" + "c" * n_data_cols  # Source + Method label columns

    lines = [r"\begin{table}[t]", r"\centering"]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    if small:
        lines.append(r"\small")
    lines.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}")
    lines.append(rf"\begin{{tabular}}{{{col_format}}}")
    lines.append(r"\toprule")

    # header row 1: split-type group headers (cols 1-2 are Source/Method)

    header1 = ["", ""] + [rf"\multicolumn{{{len(all_metrics)}}}{{c}}{{{split_label}}}" for split_label in header_order]
    lines.append(" & ".join(header1) + r" \\")

    # cmidrules under each split-type group
    cmidrules, start = [], 3
    for _ in split_paths:
        end = start + len(all_metrics) - 1
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append(" ".join(cmidrules))

    # header row 2: metric names, repeated per split-type group
    header2 = ["Source", "Loss(es)"]
    for _ in split_paths:
        header2.extend(metric_labels)
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    # --- body ---
    for cat_i, cat in enumerate(present_cats):
        cat_runs = runs_by_cat[cat]
        n = len(cat_runs)
        for i, run in enumerate(cat_runs):
            source_cell = rf"\multirow{{{n}}}{{*}}{{{short_cat_label.get(cat, cat)}}}" if i == 0 else ""
            row_cells = [source_cell, run_labels[run]]

            for split_label in header_order:
                for metric in all_metrics:
                    col_values = values[split_label][metric]
                    mean_v, std_v = col_values[run]
                    is_best = False
                    if bold_best and not np.isnan(mean_v):
                        finite_means = [m for m, _ in col_values.values() if not np.isnan(m)]
                        best_v = max(finite_means) if higher_is_better else min(finite_means)
                        is_best = np.isclose(mean_v, best_v)

                    if np.isnan(mean_v):
                        text = "--"
                    elif include_std:
                        text = rf"{mean_v:.{decimals}f} $\pm$ {std_v:.{decimals}f}"
                    else:
                        text = f"{mean_v:.{decimals}f}"
                    if is_best:
                        text = rf"\textbf{{{text}}}"
                    row_cells.append(text)

            lines.append(" & ".join(row_cells) + r" \\")
        if cat_i < len(present_cats) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    latex_source = "\n".join(lines)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(latex_source)
    print(f"[table] Saved → {out_path}")

    return latex_source

def plot_method_comparison_combined(
    path: str,
    metric: str,
    run_order: list,
    run_name_map: dict | None = None,
    run_category_map: dict | None = None,
    run_hatch_map: dict | None = None,
    box_width: float = 0.6,
    group_spacing: float = 0.1,
    category_gap: float = 0.6,
    metric_map: dict | None = None,
    out_path: str = "method_comparison_boxplots.png",
):
    """
    All runs together in a single panel, clustered by source (same
    background shading + legend as the big grid), using the fully pooled
    'average' metric across all options — i.e. exactly the 'Average'
    column of plot_auc_boxplots_grid, rendered standalone as a one-row,
    one-column grid (options=() so no per-distance boxes are drawn).
    """
    row_title = metric_map.get(metric, metric) if metric_map else metric

    # Figure width sized to the actual number of runs (rather than the
    # grid's usual "3 per option column" assumption), so a single wide
    # panel with many runs doesn't come out cramped.
    runs = _canonical_runs([path], run_order)
    fig_width = max(0.5 * len(runs) + 2.0, 4.0)

    plot_auc_boxplots_grid(
        [{"path": path, "metric": metric, "row_title": row_title}],
        run_order=run_order,
        run_name_map=run_name_map,
        run_category_map=run_category_map,
        run_hatch_map=run_hatch_map,
        box_width=box_width,
        group_spacing=group_spacing,
        category_gap=category_gap,
        metric_map=metric_map,
        add_average=True,
        options=(),                 # no option columns — pooled average only
        out_path=out_path,
        fig_width=fig_width,
        fig_height=4.5,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot AUC boxplots grouped by run and option.")
    parser.add_argument("--dataset_name", default="GeoPlant", type=str)
    parser.add_argument("--split_types", default="geographical,environmental", type=str,
                         help="Comma-separated split types to aggregate as rows.")
    parser.add_argument("--metrics", default="avg_auc_species,avg_auc_site", type=str,
                         help="Comma-separated metrics; one aggregated figure is produced per metric.")
    parser.add_argument("--use_overlapping_species", action="store_true")
    parser.add_argument("--region", default="france", type=str, help="Region to run the split sweep on (default: france).")
    parser.add_argument("--add_average", action="store_true",
                         help="Add an extra column showing per-run averages pooled across all options.")
    parser.add_argument("--single", action="store_true",
                         help="Fall back to the original single (split_type, metric) plot instead of the grid.")
    parser.add_argument("--plot_type", default="all", choices=["grid", "winners", "comparison", "all"],
                         help="Which aggregated figure(s) to produce (ignored if --single). "
                              "'grid' = the full big plot (rows=metrics), 'winners' = one box per "
                              "source using only the given --winner_runs, 'comparison' = one subplot "
                              "per source comparing all its methods, 'all' = produce every one.")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    region = args.region
    split_types = [s.strip() for s in args.split_types.split(",") if s.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    # spec_name = "intersect" if args.use_overlapping_species else "union"
    # for now we will set the default to intersect as the union will probably not be the one finally used
    spec_name = 'intersect'

    def output_dir_for(split_type):
        return f"outputs/split_sweep/{dataset_name}/{region}_bands/{split_type}/{spec_name}"

    run_order = [
        "po_deep_maxent",
        "po_balanced_bce",
        "po_bce",
        "pa_bce",
        "pa_balanced_bce",
        "po_deep_maxent_pa_deep_maxent",
        "po_balanced_bce_pa_balanced_bce",
        "po_bce_pa_bce",
        "po_deep_maxent_pa_bce",
        "po_deep_maxent_pa_bce_ippp",
        "po_deep_maxent_pa_balanced_bce",
        "po_deep_maxent_pa_balanced_bce_ippp",
        # "po_balanced_bce_pa_balanced_bce_wpocov",
        # "po_deep_maxent_pa_balanced_bce_wpocov",
    ]

    # run_order = [
    #     "po_balanced_bce",
    #     "pa_balanced_bce",
    #     "po_deep_maxent_pa_balanced_bce_ippp",
    # ]


    METHOD_LABELS = {
        "deep_maxent": "DeepMaxent",
        "balanced_bce": "Bal. BCE",
        "bce": "BCE",
    }

    SUFFIX_LABELS = {
        "ippp": "IPP",
        "wpocov": "w/ PO cov",
    }

    def make_run_name(run_key: str) -> str:
        # peel off a known suffix, if present
        suffix = None
        for suf in SUFFIX_LABELS:
            if run_key.endswith(f"_{suf}"):
                suffix = suf
                run_key = run_key[: -(len(suf) + 1)]
                break

        if "_pa_" in run_key:
            po_part, pa_method = run_key.split("_pa_", 1)
            po_method = po_part.removeprefix("po_")

            po_label = METHOD_LABELS[po_method]
            pa_label = METHOD_LABELS[pa_method]

            if po_method == pa_method:
                label = f"{po_label} (both"
                if suffix:
                    label += f", {SUFFIX_LABELS[suffix]}"
                return label + ")"

            suffix_label = SUFFIX_LABELS[suffix] if suffix else "Sigmoid"
            return f"{po_label} + {pa_label} ({suffix_label})"

        # PO-only or PA-only run
        method = run_key.removeprefix("po_").removeprefix("pa_")
        return METHOD_LABELS[method]


    run_name_map = {k: make_run_name(k) for k in run_order}

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

    # winner_runs = {
    #     "PO only": "po_balanced_bce",
    #     "PA only": "pa_balanced_bce",
    #     "PO + PA": "po_deep_maxent_pa_balanced_bce_ippp",
    # } 

    # Harmonic-mean row: combines the two AUC metrics into a single score
    # that penalizes runs which do well on one but poorly on the other.
    HARMONIC_COL = "harmonic_auc_species_site"
    derived_metrics = {HARMONIC_COL: ("avg_auc_species", "avg_auc_site")}
    metric_map[HARMONIC_COL] = "Harmonic mean AUC\n(species, site)"


    metric_map_short = {
        "avg_auc_species": "AUC\nspecies",
        "avg_auc_site": "AUC\nsite",
        HARMONIC_COL: "H(AUC)",
    }

    winner_runs = {}  # will be filled in below
    # determine best run per split_type
    for split_type in split_types:
        print(f"Finding best runs for split type: {split_type}")
        winner_runs[split_type] = {}
        df = pd.read_csv(os.path.join(output_dir_for(split_type), "summary_common.csv"))
        df['source'] = df['source'].map(SOURCE_REMAP)
        df[HARMONIC_COL] = _harmonic_mean(df["avg_auc_species"], df["avg_auc_site"])
        for source in ["PO only", "PA only", "PO + PA"]:
            source_df = df[df['source'] == source]
            # average the harmonic mean across all options for each run
            source_df = source_df.groupby('run')[HARMONIC_COL].mean().reset_index()
            best_run = source_df.loc[source_df[HARMONIC_COL].idxmax()]['run']
            winner_runs[split_type][source] = best_run
    print(f"Winner runs by source: {winner_runs}")


    if args.single:
        # Original behaviour: one (split_type, metric) combo per call.
        for split_type in split_types:
            for metric in metrics:
                plot_auc_boxplots(
                    output_dir_for(split_type),
                    run_order=run_order,
                    run_name_map=run_name_map,
                    run_hatch_map=run_hatch_map,
                    metric=metric,
                    metric_map=metric_map,
                    add_average=args.add_average,
                )
    else:
        row_metrics = metrics + [HARMONIC_COL]

        if args.plot_type in ("grid", "all"):
            # Aggregated big plot: one figure per split_type, with a row
            # per metric (the two AUC metrics plus their harmonic mean).
            for split_type in split_types:
                path = output_dir_for(split_type)
                row_configs = [
                    {
                        "path": path,
                        "metric": metric,
                        "row_title": metric_map.get(metric, metric).replace("\n", " "),
                    }
                    for metric in row_metrics
                ]
                
                out_path = f"{output_dir_for(split_type)}/plots/combined_metrics_boxplots.png"
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                plot_auc_boxplots_grid(
                    row_configs,
                    run_order=run_order,
                    run_name_map=run_name_map,
                    run_hatch_map=run_hatch_map,
                    metric_map=metric_map,
                    add_average=args.add_average,
                    out_path=out_path,
                    derived_metrics=derived_metrics,
                )

        if args.plot_type in ("winners", "all"):
            # Grid version of the 'best of each source' plot: rows = the
            # given metrics + harmonic mean, columns = options + average,
            # restricted to only the winning run per category.
            for split_type in split_types:
                path = output_dir_for(split_type)
                row_configs = [
                    {
                        "path": path,
                        "metric": metric,
                        "row_title": metric_map.get(metric, metric).replace("\n", " "),
                    }
                    for metric in row_metrics
                ]
                out_path = (f"{output_dir_for(split_type)}/plots/winning_methods_grid_{split_type}.png")    
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                plot_winning_methods_by_distance_v2(row_configs, winner_runs[split_type], run_name_map=run_name_map,
                                  metric_map=metric_map, derived_metrics=derived_metrics,
                                  out_path=out_path)

        if args.plot_type in ("comparison", "all"):
            # All methods together in one panel, clustered by source,
            # fully pooled/averaged (no per-option split).
            # for split_type in split_types:
            #     path = output_dir_for(split_type)
            #     for metric in metrics:
            #         out_path = (f"outputs/split_sweep/{dataset_name}/{region}_bands/{split_type}/"
            #                     f"method_comparison_{metric}_boxplots.png")
            #         os.makedirs(os.path.dirname(out_path), exist_ok=True)
            #         plot_method_comparison_combined(
            #             path,
            #             metric=metric,
            #             run_order=run_order,
            #             run_name_map=run_name_map,
            #             run_hatch_map=run_hatch_map,
            #             metric_map=metric_map,
            #             out_path=out_path,
            #         )
            tex = build_method_comparison_table(
                split_paths={"Environmental": output_dir_for("environmental"), "Geographical": output_dir_for("geographical")},
                metrics=["avg_auc_species", "avg_auc_site"],
                run_order=run_order,
                run_name_map=run_name_map,
                metric_map=metric_map_short,
                derived_metrics=derived_metrics,   # your existing harmonic-mean dict
                include_std=False,
                bold_best=True,
                caption="Comparison of winning losses across splits.",
                label="tab:method_comparison",
                out_path="outputs/tables/method_comparison.tex",
            )