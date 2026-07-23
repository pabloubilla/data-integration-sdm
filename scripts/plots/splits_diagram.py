# scripts/plots/make_simulated_split_schematic.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from isdm.splits import partition_sweep_ranges_v2_indices


POINT_PA = "#3B82F6"       # blue
POINT_PO = "#10B981"       # green
POINT_TRAIN = "#2563EB"    # darker blue
POINT_TEST = "#EF4444"     # red
POINT_BG = "#D1D5DB"       # gray
BORDER = "#111827"


def clean_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal", adjustable="box")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)
        spine.set_color(BORDER)


def save_fig(fig, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def simulate_fake_map(seed: int = 42):
    rng = np.random.default_rng(seed)

    centers = np.array([
        [-1.2,  0.6],
        [-0.2,  1.0],
        [ 0.8,  0.5],
        [-0.7, -0.4],
        [ 0.4, -0.7],
        [ 1.1, -0.2],
    ])

    pa_parts = []
    for c in centers:
        pts = rng.normal(loc=c, scale=[0.22, 0.18], size=(130, 2))
        pa_parts.append(pts)
    pa = np.vstack(pa_parts)

    po_centers = np.array([
        [-0.2, 1.0],
        [0.8, 0.5],
        [1.1, -0.2],
    ])

    po_parts = []
    for c in po_centers:
        pts = rng.normal(loc=c, scale=[0.28, 0.22], size=(350, 2))
        po_parts.append(pts)

    noise = rng.normal(loc=[0.0, 0.0], scale=[0.9, 0.75], size=(250, 2))
    po = np.vstack(po_parts + [noise])

    return pa, po


def as_dataframe(xy: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
        }
    )


def get_one_test_cluster_splits(
    pa_xy: np.ndarray,
    test_number: int = 0,
    seed: int = 42,
):
    """
    Uses src.splits.partition_sweep_ranges_v2_indices directly.

    We request several test clusters, then select one test_number so that
    closest/middle/farthest all use the same test cluster.
    """
    X_pa = as_dataframe(pa_xy)

    specs = partition_sweep_ranges_v2_indices(
        X_pa=X_pa,
        covs_cluster=["x", "y"],
        covs_distance=["x", "y"],
        k_clusters=10,
        select_subset=5,
        train_proportion=0.35,
        distance_metric="euclidean",
        seed=seed,
        options=("closest", "middle", "farthest"),
    )

    specs_same_test = [s for s in specs if s.test_number == test_number]

    if len(specs_same_test) != 3:
        raise RuntimeError(
            f"Expected 3 specs for test_number={test_number}, "
            f"got {len(specs_same_test)}."
        )

    by_option = {s.option: s for s in specs_same_test}

    return {
        "closest": by_option["closest"],
        "middle": by_option["middle"],
        "farthest": by_option["farthest"],
    }


def plot_distribution(xy, color, out_base: Path):
    fig, ax = plt.subplots(figsize=(4, 4))

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=8,
        c=color,
        alpha=0.78,
        linewidths=0,
    )

    clean_ax(ax)
    save_fig(fig, out_base)


def plot_split_from_spec(xy, spec, out_base: Path):
    train_idx = spec.train_idx
    test_idx = spec.test_idx

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=6,
        c=POINT_BG,
        alpha=0.45,
        linewidths=0,
    )

    ax.scatter(
        xy[train_idx, 0],
        xy[train_idx, 1],
        s=9,
        c=POINT_TRAIN,
        alpha=0.85,
        linewidths=0,
    )

    ax.scatter(
        xy[test_idx, 0],
        xy[test_idx, 1],
        s=16,
        c=POINT_TEST,
        alpha=0.95,
        linewidths=0,
    )

    clean_ax(ax)
    save_fig(fig, out_base)


def main():
    out_dir = Path("outputs/splits_diagram_v2")

    pa, po = simulate_fake_map(seed=42)

    plot_distribution(
        pa,
        color=POINT_PA,
        out_base=out_dir / "pa_distribution_simulated",
    )

    plot_distribution(
        po,
        color=POINT_PO,
        out_base=out_dir / "po_distribution_simulated",
    )

    split_specs = get_one_test_cluster_splits(
        pa_xy=pa,
        test_number=0,
        seed=42,
    )

    for option, spec in split_specs.items():
        plot_split_from_spec(
            pa,
            spec=spec,
            out_base=out_dir / f"pa_split_{option}_simulated",
        )

    print(f"Saved simulated schematic figures to: {out_dir}")


if __name__ == "__main__":
    main()