"""
Builds a paper figure comparing example geographical and environmental
splits, side by side, with the PO dataset shown as a shared background
layer across every panel.

Reads splits already generated and saved by scripts/generate_splits.py
(via save_split_specs), so it never recomputes clustering or anchors —
it only loads and plots.

Usage:
    python scripts/plot_paper_figure.py \
        --geo_split_dir outputs/splits/GeoPlant/france_bands/geographical \
        --env_split_dir outputs/splits/GeoPlant/france_bands/environmental \
        --data_path data/processed/GeoPlant/france \
        --n_splits 3 \
        --output outputs/figures/splits_map.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


OPTIONS_ORDER = ("test", "closest", "middle", "farthest")

# LaTeX-ish labels for panel titles, using \mathcal to match paper notation
OPTION_LABEL = {
    "test": r"$\mathcal{D}_{\mathrm{test}}^{(k)}$",
    "closest": r"$\mathcal{D}_{\mathrm{PA}}^{\mathrm{close},(k)}$",
    "middle": r"$\mathcal{D}_{\mathrm{PA}}^{\mathrm{medium},(k)}$",
    "farthest": r"$\mathcal{D}_{\mathrm{PA}}^{\mathrm{far},(k)}$",
}

COLOR_MAP = {
    "test": "#5DA4AF",
    "closest": "#9A2094",
    "middle": "#9A2094",
    "farthest": "#9A2094",
    "po": "#CC8B28",
}

LEGEND_LABEL_TEST = r"$\mathcal{D}_{\mathrm{PA}}^{\mathrm{test}}$"
LEGEND_LABEL_TRAIN = r"$\mathcal{D}_{\mathrm{PA}}^{\mathrm{train}}$"
LEGEND_LABEL_PO = r"$\mathcal{D}_{\mathrm{PO}}$"


def load_saved_specs(split_dir: str | Path) -> list[dict]:
    """
    Reads splits.csv + per-split npz files saved by save_split_specs, and
    returns a flat list of dicts (one per split) with everything needed
    for plotting: option, test_number, distance, train_idx, test_idx.
    """
    split_dir = Path(split_dir)
    table = pd.read_csv(split_dir / "splits.csv")

    specs = []
    for _, row in table.iterrows():
        arr = np.load(split_dir / row["split_file"])
        specs.append({
            "option": row["option"],
            "test_number": int(row["test_number"]),
            "distance": float(row["distance"]),
            "train_idx": arr["train_idx"],
            "test_idx": arr["test_idx"],
            "anchor_x": float(row["anchor_x"]),
            "anchor_y": float(row["anchor_y"]),
        })
    return specs


def pick_example_anchors(specs: list[dict], n_examples: int) -> list[int]:
    """Picks the first n_examples anchors (by test_number) that have all
    three plain options (closest/middle/farthest) available."""
    by_anchor: dict[int, set[str]] = {}
    for s in specs:
        if s["option"] in ("closest", "middle", "farthest"):
            by_anchor.setdefault(s["test_number"], set()).add(s["option"])

    complete = sorted(k for k, opts in by_anchor.items() if len(opts) == 3)
    return complete[:n_examples]


def _new_axis(fig, gs_slot, use_cartopy: bool, projection=None):
    if use_cartopy:
        return fig.add_subplot(gs_slot, projection=projection)
    return fig.add_subplot(gs_slot)


def _style_axis(ax, x_lim, y_lim, use_cartopy: bool):
    if use_cartopy:
        # data stays in lon/lat (PlateCarree) regardless of the axes' own
        # display projection — this just defines the visible window
        ax.set_extent([*x_lim, *y_lim], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor="#dceefb", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#f7f5f0", zorder=0)
        ax.add_feature(cfeature.BORDERS, edgecolor="#999999", linewidth=0.6, zorder=0.5)
        ax.add_feature(cfeature.COASTLINE, edgecolor="#777777", linewidth=0.6, zorder=0.5)
    else:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        # raw lon/lat isn't Cartesian: 1° longitude covers cos(lat) as much
        # ground distance as 1° latitude, so a plain "equal" aspect stretches
        # the map horizontally at mid-latitudes (e.g. France, ~46-47°N).
        # This is the fallback-only fix; the cartopy path handles it via
        # the LambertConformal display projection instead.
        mean_lat = (y_lim[0] + y_lim[1]) / 2
        ax.set_aspect(1 / np.cos(np.deg2rad(mean_lat)))
        ax.set_facecolor("#f7f5f0")
    ax.set_xticks([])
    ax.set_yticks([])


BAND_OPTIONS = ("closest", "middle", "farthest")

BAND_LABEL = {
    "closest": "close",
    "middle": "medium",
    "farthest": "far",
}


def _group_by_anchor(specs: list[dict]) -> dict[int, dict[str, dict]]:
    grouped: dict[int, dict[str, dict]] = {}
    for s in specs:
        if s["option"] in BAND_OPTIONS:
            grouped.setdefault(s["test_number"], {})[s["option"]] = s
    return grouped


def plot_paper_splits_figure(
    X_pa: np.ndarray,
    X_po: np.ndarray,
    geo_specs: list[dict],
    env_specs: list[dict],
    n_splits: int = 1,
    x_lim: tuple[float, float] = (-5, 8.5),
    y_lim: tuple[float, float] = (42, 51.5),
    output_path: str | Path = "paper_splits_figure.png",
    po_sample_size: int | None = None,
    rng_seed: int = 0,
    po_gap_ratio: float = 0.6,
    legend_y: float = 0.1,
    legend_gap = 0.05
):
    """
    Builds a figure with one dedicated top panel showing D_PO alone (a
    single, constant dataset, shown once), followed by a grid with 6
    columns — (Geographical: close, medium, far) then
    (Environmental: close, medium, far) — and one row per split example
    (n_splits controls how many). Every panel overlays both the train band
    and the fixed test set as a single, randomly-interleaved scatter, so
    their overlap is directly visible rather than one systematically
    covering the other.

    X_pa / X_po: (N, 2) arrays of plotting coordinates in lon/lat degrees,
    aligned with the indices stored in geo_specs / env_specs. All axes are
    rendered in a proper projected CRS (LambertConformal, centered on
    x_lim/y_lim) rather than raw PlateCarree, so France doesn't get
    stretched horizontally by meridian convergence at this latitude.

    po_gap_ratio: controls the vertical gap between the PO panel and the
    PA grid below it, as a fraction of one data row's height. Larger value
    = more space between them.
    legend_y: figure-fraction y-position of the legend (0 = bottom edge of
    the figure, 1 = top). The bottom margin is automatically widened to
    fit, so increasing this both moves the legend up and adds more space
    below the PA grid.
    """
    use_cartopy = HAS_CARTOPY
    if not use_cartopy:
        print("cartopy not available — falling back to plain scatter (no basemap).")

    # Display projection for all cartopy axes: data is supplied in lon/lat
    # (transform=PlateCarree() below) but rendered in a projected CRS
    # centered on the region, which is what actually fixes the "too wide"
    # distortion — PlateCarree as a *display* projection has no correction
    # for meridian convergence at non-equatorial latitudes.
    display_proj = None
    if use_cartopy:
        central_lon = float(np.mean(x_lim))
        central_lat = float(np.mean(y_lim))
        display_proj = ccrs.LambertConformal(central_longitude=central_lon,
                                              central_latitude=central_lat)

    po_idx = np.arange(len(X_po))
    if po_sample_size is not None and len(po_idx) > po_sample_size:
        rng = np.random.default_rng(rng_seed)
        po_idx = rng.choice(po_idx, size=po_sample_size, replace=False)
    X_po_plot = X_po[po_idx]

    geo_by_anchor = _group_by_anchor(geo_specs)
    env_by_anchor = _group_by_anchor(env_specs)
    geo_anchors = pick_example_anchors(geo_specs, n_splits)
    env_anchors = pick_example_anchors(env_specs, n_splits)

    n_rows_data = max(len(geo_anchors), len(env_anchors))
    if n_rows_data < n_splits:
        print(f"Only {n_rows_data} complete anchors available (requested n_splits={n_splits}).")

    col_defs = [("Geographical", geo_by_anchor, geo_anchors, opt) for opt in BAND_OPTIONS]
    col_defs += [("Environmental", env_by_anchor, env_anchors, opt) for opt in BAND_OPTIONS]

    # narrow spacer column between the two 3-column blocks so they read as
    # visually distinct groups; grid_cols maps col_defs index -> gridspec column
    grid_cols = [0, 1, 2, 4, 5, 6]
    width_ratios = [1, 1, 1, 0.2, 1, 1, 1]
    ncols_grid = len(width_ratios)

    # spacer ROW between the PO panel and the PA grid, same trick as the
    # column spacer above: no axis is placed there, it just reserves
    # vertical space, controlled directly by po_gap_ratio
    height_ratios = [1.1, po_gap_ratio] + [1] * n_rows_data
    nrows_grid = len(height_ratios)

    fig = plt.figure(figsize=(2.1 * len(col_defs) + 1, 2.2 * (n_rows_data + 1)))
    gs = fig.add_gridspec(nrows_grid, ncols_grid, wspace=0.05, hspace=0.01,
                           width_ratios=width_ratios,
                           height_ratios=height_ratios)

    # ── top panel: PO shown once, spanning the full width ──────────────
    po_ax = _new_axis(fig, gs[0, :], use_cartopy, projection=display_proj)
    _style_axis(po_ax, x_lim, y_lim, use_cartopy)
    po_kw = {"transform": ccrs.PlateCarree()} if use_cartopy else {}
    po_ax.scatter(X_po_plot[:, 0], X_po_plot[:, 1],
                  c=COLOR_MAP["po"], s=.5, alpha=0.3, linewidths=0, zorder=1, **po_kw)
    po_ax.set_title(LEGEND_LABEL_PO + " (kept constant across all splits)", fontsize=11)

    # ── grid: rows = split examples, columns = (block, band) pairs ─────
    # row offset of 2 accounts for the PO row (index 0) and the spacer
    # row (index 1) inserted above
    top_row_axes: dict[int, plt.Axes] = {}
    for row in range(n_rows_data):
        for col, (block_name, by_anchor, anchors, option) in enumerate(col_defs):
            grid_col = grid_cols[col]
            ax = _new_axis(fig, gs[row + 2, grid_col], use_cartopy, projection=display_proj)
            _style_axis(ax, x_lim, y_lim, use_cartopy)
            ax_kw = {"transform": ccrs.PlateCarree()} if use_cartopy else {}

            if row < len(anchors):
                anchor_ix = anchors[row]
                option_specs = by_anchor[anchor_ix]
                test_idx = option_specs["closest"]["test_idx"]  # shared across options
                spec = option_specs.get(option)
                train_idx = spec["train_idx"] if spec is not None else np.array([], dtype=np.int64)

                # combine train + test into ONE scatter call, drawn in
                # random order, so overlap is genuinely interleaved rather
                # than one color systematically covering the other
                combined_idx = np.concatenate([train_idx, test_idx])
                combined_colors = np.array(
                    [COLOR_MAP[option]] * len(train_idx) + [COLOR_MAP["test"]] * len(test_idx)
                )
                perm_rng = np.random.default_rng(rng_seed + row * 100 + col)
                perm = perm_rng.permutation(len(combined_idx))

                ax.scatter(X_pa[combined_idx[perm], 0], X_pa[combined_idx[perm], 1],
                           c=combined_colors[perm], s=8, linewidths=0, zorder=2, **ax_kw)

            if row == 0:
                ax.set_title(BAND_LABEL[option], fontsize=10)
                top_row_axes[col] = ax
            if col == 0:
                ax.text(-0.05, 0.5, f"$k={row+1}$", transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=11)

    # shared legend — bbox_to_anchor gives direct control over its
    # vertical position, independent of tight_layout's own margins
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MAP["po"],
               markersize=10, label=LEGEND_LABEL_PO),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MAP["test"],
               markersize=10, label=LEGEND_LABEL_TEST),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MAP["closest"],
               markersize=10, label=LEGEND_LABEL_TRAIN),
    ]
    fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, legend_y),
               ncol=3, fontsize=9, frameon=True)

    # plt.tight_layout(rect=[0.03, legend_y, 1, 0.98])
    fig.subplots_adjust(left=0.03, right=1.0, top=0.98, bottom=legend_y + legend_gap)

    # group-level super-titles ("Geographical" / "Environmental"), spanning
    # each 3-column block; computed from the top data row's axis positions
    # so they line up correctly regardless of figure size / column widths
    for block_name, cols in (("Geographical", [0, 1, 2]), ("Environmental", [3, 4, 5])):
        lefts = [top_row_axes[c].get_position().x0 for c in cols]
        rights = [top_row_axes[c].get_position().x1 for c in cols]
        top = max(top_row_axes[c].get_position().y1 for c in cols)
        mid_x = (min(lefts) + max(rights)) / 2
        fig.text(mid_x, top + 0.03, block_name, ha="center", va="bottom",
                  fontsize=13, fontweight="bold")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {output_path}")


def main():
    from isdm.load_data import load_geoplant_processed  # local import: only needed for the CLI entry point

    parser = argparse.ArgumentParser()
    parser.add_argument("--geo_split_dir", type=Path,
                         default=Path("outputs/splits/GeoPlant/france_bands/geographical"))
    parser.add_argument("--env_split_dir", type=Path,
                         default=Path("outputs/splits/GeoPlant/france_bands/environmental"))
    parser.add_argument("--data_path", type=str, default="data/processed/GeoPlant/france")
    parser.add_argument("--n_splits", type=int, default=3,
                         help="Number of split examples (anchors) to show, one per row.")
    parser.add_argument("--output", type=Path, default=Path("outputs/figures/splits_maps.png"))
    parser.add_argument("--po_gap_ratio", type=float, default=0.15,
                         help="Vertical gap between the PO panel and the PA grid.")
    parser.add_argument("--legend_y", type=float, default=0.02,
                         help="Figure-fraction y-position of the legend.")
    args = parser.parse_args()

    data = load_geoplant_processed(args.data_path, add_coordinates=True)

    covs_plot = [c for c in ["lon", "lat", "x", "y"] if c in data.X_pa_train.columns][:2]
    X_pa = data.X_pa_train.reset_index(drop=True)[covs_plot].to_numpy()
    X_po = data.X_po.reset_index(drop=True)[covs_plot].to_numpy()

    geo_specs = load_saved_specs(args.geo_split_dir)
    env_specs = load_saved_specs(args.env_split_dir)

    plot_paper_splits_figure(
        X_pa=X_pa,
        X_po=X_po,
        geo_specs=geo_specs,
        env_specs=env_specs,
        n_splits=args.n_splits,
        output_path=args.output,
        po_gap_ratio=args.po_gap_ratio,
        legend_y=args.legend_y,
    )


if __name__ == "__main__":
    main()