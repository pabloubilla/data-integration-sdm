# scripts/plots/make_simulated_split_schematic.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from isdm.splits import partition_sweep_ranges_v2_indices


# ── Style config ─────────────────────────────────────────────────────────────

STYLE = {
    # Colors
    "color_pa":       "#42B312",
    "color_po":       "#D82CA4",
    "color_train":    "#EC3D26",
    "color_test":     "#231AC6",
    "color_bg":       "#D1D5DBA0",
    "color_border":   "#111827",

    # Marker sizes  (matplotlib `s`, i.e. pt²)
    "size_pa":        14,
    "size_po":        6,
    "size_bg":        14,
    "size_train":     14,
    "size_test":      14,

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

    # Figure
    "fig_size":       (4, 4),
    "border_lw":      1.4,
}

# Test numbers to plot (add as many as you like)
TEST_NUMBERS = [0, 1]

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


def simulate_fake_map(seed: int = 42):
    rng = np.random.default_rng(seed)

    map_limits = (-2.0, 2.0)

    pa = rng.uniform(low=-1.8, high=1.8, size=(400, 2))
    pa += rng.normal(loc=0.0, scale=0.3, size=pa.shape)
    pa = np.clip(pa, *map_limits)

    po_centers = np.array([
        [ 0.8,  0.7],
        [ 0.3, -0.6],
        [-0.7,  0.5],
        [ 1.0, -1.0],
        [-0.4,  1.1],
    ])
    po_sizes   = [600, 450, 500, 350, 400]
    po_spreads = [0.30, 0.25, 0.28, 0.22, 0.26]

    po_parts = []
    for c, sz, sp in zip(po_centers, po_sizes, po_spreads):
        pts = rng.normal(loc=c, scale=sp, size=(sz, 2))
        po_parts.append(pts)

    uniform_pts = rng.uniform(low=-1.8, high=1.8, size=(300, 2))
    po = np.vstack(po_parts + [uniform_pts])
    po = np.clip(po, *map_limits)

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
    X_pa = as_dataframe(pa_xy)

    all_specs = partition_sweep_ranges_v2_indices(
        X_pa=X_pa,
        covs_cluster=["x", "y"],
        covs_distance=["x", "y"],
        k_clusters=15,
        select_subset=15,
        train_proportion=0.33,
        distance_metric="euclidean",
        seed=seed,
        options=("closest", "middle", "farthest"),
    )

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


def plot_distribution(xy, color, size, marker, alpha, out_base: Path):
    fig, ax = plt.subplots(figsize=STYLE["fig_size"])
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=color, marker=marker, alpha=alpha, linewidths=0)
    clean_ax(ax)
    save_fig(fig, out_base)


def plot_split_from_spec(xy, spec, out_base: Path):
    train_idx = spec.train_idx
    test_idx  = spec.test_idx

    fig, ax = plt.subplots(figsize=STYLE["fig_size"])

    ax.scatter(
        xy[:, 0], xy[:, 1],
        s=STYLE["size_bg"], c=STYLE["color_bg"],
        marker=STYLE["marker_bg"], alpha=STYLE["alpha_bg"], linewidths=0,
    )
    ax.scatter(
        xy[train_idx, 0], xy[train_idx, 1],
        s=STYLE["size_train"], c=STYLE["color_train"],
        marker=STYLE["marker_train"], alpha=STYLE["alpha_train"], linewidths=0,
    )
    ax.scatter(
        xy[test_idx, 0], xy[test_idx, 1],
        s=STYLE["size_test"], c=STYLE["color_test"],
        marker=STYLE["marker_test"], alpha=STYLE["alpha_test"], linewidths=0,
    )

    clean_ax(ax)
    save_fig(fig, out_base)


def plot_random_cluster_split(
    xy: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    out_base: Path,
):
    """Same colors/style as split plots but no background layer —
    every point is either train or test."""
    fig, ax = plt.subplots(figsize=STYLE["fig_size"])

    ax.scatter(
        xy[train_idx, 0], xy[train_idx, 1],
        s=STYLE["size_train"], c=STYLE["color_train"],
        marker=STYLE["marker_train"], alpha=STYLE["alpha_train"], linewidths=0,
    )
    ax.scatter(
        xy[test_idx, 0], xy[test_idx, 1],
        s=STYLE["size_test"], c=STYLE["color_test"],
        marker=STYLE["marker_test"], alpha=STYLE["alpha_test"], linewidths=0,
    )

    clean_ax(ax)
    save_fig(fig, out_base)


def main():
    out_dir = Path("outputs/splits_diagram_v2")

    pa, po = simulate_fake_map(seed=42)

    # Distribution plots
    plot_distribution(
        pa,
        color=STYLE["color_pa"], size=STYLE["size_pa"],
        marker=STYLE["marker_pa"], alpha=STYLE["alpha_pa"],
        out_base=out_dir / "pa_distribution_simulated",
    )
    plot_distribution(
        po,
        color=STYLE["color_po"], size=STYLE["size_po"],
        marker=STYLE["marker_po"], alpha=STYLE["alpha_po"],
        out_base=out_dir / "po_distribution_simulated",
    )

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
    )

    print(f"Saved simulated schematic figures to: {out_dir}")


if __name__ == "__main__":
    main()
