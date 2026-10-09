'''
Images that take part of Figure 2.
The final figure is produced in Inkscape (cool tool!)
'''

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from isdm.splits import partition_sweep_ranges_v2_indices
from isdm.splits_bands import partition_sweep_bands


# ── Style config ─────────────────────────────────────────────────────────────

STYLE = {
    # Colors
    "color_pa":       "#A131BA",
    "color_po":       "#CA7418",
    "color_train":    "#EB64E0",
    "color_test":     "#6DB1D9",
    "color_bg":       "#D1D5DBA0",
    "color_border":   "#111827",
    "color_grid":     "#6B7280",
    "color_edge":     "#1F2937",  # outline around markers

    # Marker sizes  (matplotlib `s`, i.e. pt²)
    "size_pa":        140,
    "size_po":        45,
    "size_bg":        140,
    "size_train":     140,
    "size_test":      140,

    # Outline width (0 = no outline)
    "lw_pa":          2,
    "lw_po":          1,
    "lw_bg":          0,
    "lw_train":       2,
    "lw_test":        2,

    # Marker shapes  (matplotlib marker codes)
    "marker_pa":      "s",
    "marker_po":      "o",
    "marker_bg":      "s",
    "marker_train":   "s",
    "marker_test":    "s",

    # Alpha
    "alpha_pa":       0.85,
    "alpha_po":       0.55,
    "alpha_bg":       0.5,
    "alpha_train":    0.85,
    "alpha_test":     0.85,

    # Symbol drawn inside each test square (set to None to disable)
    "mark_test":      "D",       # any matplotlib marker: "x", "+", ".", "_", "|", /
    "mark_color":     "darkblue",
    "mark_size":      30,         # pt², keep below the square size
    "mark_lw":        2,

    # Figure
    "fig_size":       (8, 8),
    "border_lw":      3,
    "grid_lw":        1.2,
}

# Test numbers to plot (add as many as you like)
TEST_NUMBERS = [0, 1, 2]

SQUARE_SIZE = 4 / 24  # 4 lattice steps per grid cell (map width 4 / N_BINS / 4) # PA lattice spacing (data units)
N_BINS = 6        # grid bins per axis (method clusters + spatial block split)
MAP_LIMITS = (-2.0, 2.0)
N_EMPTY_CELLS = 3   # grid cells emptied on purpose
SHOW_GRID = True   # draw the grid on PA plots

# ─────────────────────────────────────────────────────────────────────────────


def clean_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(STYLE["border_lw"])
        spine.set_color(STYLE["color_border"])


def save_fig(fig, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".svg",):
        fig.savefig(out_base.with_suffix(ext), dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def grid_edges(n_bins: int = N_BINS):
    # equals make_grid_clusters' edges because two hidden corner points pin the data range to MAP_LIMITS
    e = np.linspace(*MAP_LIMITS, n_bins + 1)
    return [e, e]


def draw_grid(ax, n_bins: int = N_BINS):
    ex, ey = grid_edges(n_bins)
    kw = dict(colors=STYLE["color_grid"], linewidths=STYLE["grid_lw"], zorder=0)
    ax.vlines(ex, ey[0], ey[-1], **kw)
    ax.hlines(ey, ex[0], ex[-1], **kw)


def draw_points(ax, xy: np.ndarray, kind: str, mark: str | None = None):
    """Fill, optional outline, optional symbol centered in each marker."""
    common = dict(s=STYLE[f"size_{kind}"], marker=STYLE[f"marker_{kind}"])
    ax.scatter(xy[:, 0], xy[:, 1], c=STYLE[f"color_{kind}"], alpha=STYLE[f"alpha_{kind}"],
               linewidths=0, **common)
    if STYLE[f"lw_{kind}"]:
        ax.scatter(xy[:, 0], xy[:, 1], facecolors="none", edgecolors=STYLE["color_edge"],
                   linewidths=STYLE[f"lw_{kind}"], **common)
    if mark:
        ax.scatter(xy[:, 0], xy[:, 1], s=STYLE["mark_size"], marker=mark,
                   c=STYLE["mark_color"], linewidths=STYLE["mark_lw"])


def simulate_fake_map(seed: int = 12):
    rng = np.random.default_rng(seed)

    map_limits = MAP_LIMITS

    square_size = SQUARE_SIZE
    pa = rng.uniform(low=-1.8, high=1.8, size=(230, 2))
    pa += rng.normal(loc=0.0, scale=0.3, size=pa.shape)
    pa = np.clip(pa, *map_limits)
    lo = map_limits[0]
    pa = np.unique(lo + np.round((pa - lo) / square_size) * square_size, axis=0)

    # drop points sitting on any grid line, border included (ambiguous cell, square crosses the line)
    ex, ey = grid_edges()
    near_x = np.abs(pa[:, [0]] - ex).min(axis=1) < square_size / 2
    near_y = np.abs(pa[:, [1]] - ey).min(axis=1) < square_size / 2
    pa = pa[~(near_x | near_y)]

    # empty a few random cells
    cell = np.searchsorted(ex[1:-1], pa[:, 0]) * N_BINS + np.searchsorted(ey[1:-1], pa[:, 1])
    pa = pa[~np.isin(cell, rng.choice(N_BINS ** 2, size=N_EMPTY_CELLS, replace=False))]

    po_centers = np.array([
        [ 0.8,  0.7],
        [ 0.3, -0.6],
        [-0.7,  0.5],
        [ 1.0, -1.0],
        [-0.4,  1.1],
        [-0.9, -1.1]
    ])
    po_sizes   = [200, 150, 170, 120, 130, 35]
    po_spreads = [0.33, 0.28, 0.30, 0.42, 0.28, 0.12]

    po_parts = []
    for c, sz, sp in zip(po_centers, po_sizes, po_spreads):
        pts = rng.normal(loc=c, scale=sp, size=(sz, 2))
        po_parts.append(pts)

    uniform_pts = rng.uniform(low=-2.0, high=2.0, size=(110, 2))
    po = np.vstack(po_parts + [uniform_pts])
    po = po[np.all((po >= map_limits[0]) & (po <= map_limits[1]), axis=1)]  # drop, not clip (avoids lines on the border)

    return pa, po


def as_dataframe(xy: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]})


def get_splits_for_test_numbers(
    pa_xy: np.ndarray,
    test_numbers: list[int],
    seed: int = 42,
):
    """Return (all_specs, {test_number: {"closest": spec, "middle": spec, "farthest": spec}}).
    all_specs is kept so we can reconstruct cluster labels for the random split.
    """
    # two hidden corner points pin the method's grid to MAP_LIMITS (equal cells, nothing on the border)
    n = len(pa_xy)
    corners = np.array([[MAP_LIMITS[0]] * 2, [MAP_LIMITS[1]] * 2])
    X_pa = as_dataframe(np.vstack([pa_xy, corners]))
    y_pa = [[0] for i in range(len(X_pa))]  # dummy species labels

    all_specs = partition_sweep_bands(
        X_pa=X_pa,
        y_pa=y_pa,
        covs_cluster=["x", "y"],
        covs_distance=["x", "y"],
        n_bins_per_axis=N_BINS,
        # select_subset=15,
        test_proportion=0.25,
        distance_metric="euclidean",
        seed=seed,
        options=("closest", "middle", "farthest"),
        n_anchors=len(test_numbers),
        reserve_validation=False
    )

    for s in all_specs:  # remove the corner points again
        s.train_idx = s.train_idx[s.train_idx < n]
        s.test_idx = s.test_idx[s.test_idx < n]

    result = {}
    for tn in test_numbers:
        group = [s for s in all_specs if s.test_number == tn]
        if len(group) != 3:
            raise RuntimeError(
                f"Expected 3 specs for test_number={tn}, got {len(group)}."
            )
        result[tn] = {s.option: s for s in group}

    return all_specs, result


def reconstruct_cluster_labels(all_specs: list, n_points: int) -> np.ndarray:
    """
    Reconstruct a cluster-label array (length n_points) from the full spec list.

    Each spec has a single test_cluster; its test_idx are exactly the points
    belonging to that cluster. We iterate over all unique test clusters and
    stamp their indices. Points that never appear as test_idx in any spec are
    assigned label -1 (they only ever appear as train, i.e. they belong to
    clusters that were never selected as test — mark separately if needed).
    """
    labels = np.full(n_points, -1, dtype=int)

    # One spec per unique test_cluster is enough; duplicates (same cluster,
    # different option) share identical test_idx so overwriting is harmless.
    seen = set()
    for spec in all_specs:
        c = spec.test_cluster
        if c not in seen:
            labels[spec.test_idx] = c
            seen.add(c)

    return labels


def make_random_cluster_split(
    all_specs: list,
    n_points: int,
    test_fraction: float = 1 / 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reuse the exact cluster structure from partition_sweep_ranges_v2_indices,
    then randomly assign clusters to train/test.
    Points with label -1 (clusters never used as test) are always train.
    """
    labels = reconstruct_cluster_labels(all_specs, n_points)

    known_clusters = np.array(sorted(c for c in np.unique(labels) if c != -1))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(known_clusters))
    shuffled = known_clusters[perm]

    n_test = max(1, round(len(shuffled) * test_fraction))
    test_clusters  = set(shuffled[:n_test].tolist())

    test_idx  = np.where(np.isin(labels, list(test_clusters)))[0]
    train_idx = np.where(~np.isin(labels, list(test_clusters)))[0]  # includes label -1

    return train_idx, test_idx


def make_random_block_split(
    xy: np.ndarray,
    test_fraction: float = 0.25,
    n_bins: int = N_BINS,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Classic spatial block CV: grid the map, send a random 25% of non-empty blocks to test."""
    ex, ey = grid_edges(n_bins)
    bx = np.searchsorted(ex[1:-1], xy[:, 0], side="right")
    by = np.searchsorted(ey[1:-1], xy[:, 1], side="right")
    block = bx * n_bins + by

    blocks = np.unique(block)
    rng = np.random.default_rng(seed)
    test_blocks = rng.choice(blocks, size=round(test_fraction * len(blocks)), replace=False)

    is_test = np.isin(block, test_blocks)
    return np.where(~is_test)[0], np.where(is_test)[0]


def plot_distribution(xy, kind: str, out_base: Path, grid: bool = False):
    fig, ax = plt.subplots(figsize=STYLE["fig_size"])
    if grid:
        draw_grid(ax)
    draw_points(ax, xy, kind)
    clean_ax(ax)
    save_fig(fig, out_base)


def plot_overlap(pa, po, out_base: Path):
    fig, ax = plt.subplots(figsize=STYLE["fig_size"])
    draw_points(ax, po, "po")
    draw_points(ax, pa, "pa")
    clean_ax(ax)
    save_fig(fig, out_base)


def plot_split_from_spec(xy, spec, out_base: Path, grid: bool = False):
    fig, ax = plt.subplots(figsize=STYLE["fig_size"])
    if grid:
        draw_grid(ax)

    draw_points(ax, xy, "bg")
    draw_points(ax, xy[spec.train_idx], "train")
    draw_points(ax, xy[spec.test_idx], "test", mark=STYLE["mark_test"])

    clean_ax(ax)
    save_fig(fig, out_base)


def plot_random_cluster_split(
    xy: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    out_base: Path,
    grid: bool = False,
):
    """Same colors/style as split plots but no background layer —
    every point is either train or test."""
    fig, ax = plt.subplots(figsize=STYLE["fig_size"])
    if grid:
        draw_grid(ax)

    draw_points(ax, xy[train_idx], "train")
    draw_points(ax, xy[test_idx], "test", mark=STYLE["mark_test"])

    clean_ax(ax)
    save_fig(fig, out_base)


def main():
    out_dir = Path("outputs/splits_diagram")

    pa, po = simulate_fake_map(seed=5)

    # Distribution plots
    plot_distribution(pa, "pa", out_base=out_dir / "pa_distribution_simulated", grid=SHOW_GRID)
    plot_distribution(po, "po", out_base=out_dir / "po_distribution_simulated")
    plot_overlap(pa, po, out_base=out_dir / "pa_po_overlap_simulated")

    # Spatial split plots
    all_specs, splits_by_tn = get_splits_for_test_numbers(
        pa_xy=pa, test_numbers=TEST_NUMBERS, seed=42,
    )

    for tn, options in splits_by_tn.items():
        for option, spec in options.items():
            plot_split_from_spec(
                pa,
                spec=spec,
                out_base=out_dir / f"pa_split_test{tn}_{option}_simulated",
                grid=SHOW_GRID,
            )

    # Random cluster split — same clusters, random assignment
    train_idx, test_idx = make_random_cluster_split(
        all_specs=all_specs,
        n_points=len(pa),
        test_fraction=1 / 3,
        seed=4,
    )
    plot_random_cluster_split(
        pa, train_idx, test_idx,
        out_base=out_dir / "pa_split_random_cluster_simulated",
        grid=SHOW_GRID,
    )

    # Standard spatial block split (no method) — random 25% of grid blocks to test
    train_idx, test_idx = make_random_block_split(pa, test_fraction=0.25, seed=4)
    plot_random_cluster_split(
        pa, train_idx, test_idx,
        out_base=out_dir / "pa_split_random_block_simulated",
        grid=SHOW_GRID,
    )

    print(f"Saved simulated schematic figures to: {out_dir}")


if __name__ == "__main__":
    main()