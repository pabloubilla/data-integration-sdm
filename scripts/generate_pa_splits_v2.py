from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from isdm.load_data import load_geoplant_processed
from isdm.splits_bands import (
    SplitSpec,
    partition_sweep_bands,
    save_split_specs,
)

from sklearn.preprocessing import StandardScaler

import argparse

import cartopy.crs as ccrs
import cartopy.feature as cfeature



def plot_splits_overview(
    X_pa,
    specs,
    covs_plot: list[str],
    output_dir: Path,
    nrows: int = 8,
    region: str = "france"
):
    """
    One panel per anchor row, one column per option (closest/middle/farthest).
    Only plain (non-validation) specs are used here — validation splits are
    plotted separately by plot_validation_overview.
    """
    X_mat = X_pa[covs_plot].to_numpy()
    all_idx = np.arange(len(X_pa))

    if region == "france":
        x_lim = (-5, 8.5)
        y_lim = (42, 51.5)
    elif region == "full":
        x_lim = (-20, 40)
        y_lim = (30, 70)
    elif region == "denmark":
        x_lim = (7, 13)
        y_lim = (54, 58)
    proj = ccrs.PlateCarree()

    options_order = ("closest", "middle", "farthest")

    # NEW: restrict to plain options only, so anchor 0's extra _val specs
    # (same test_number, different option strings) don't get mixed in
    anchor_groups: dict[int, dict[str, SplitSpec]] = {}
    for spec in specs:
        if spec.option in options_order:
            anchor_groups.setdefault(spec.test_number, {})[spec.option] = spec

    color_map = {
            "test": "#E7651F",
            "closest": "#1E88E5",
            "middle": "#1E88E5",
            "farthest": "#1E88E5"
            }
    cx, cy = 0, 1

    if nrows == -1:
        nrows = len(anchor_groups)
    ncols = len(options_order)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows),
        subplot_kw={"projection": proj},
    )
    axes = np.atleast_2d(axes)

    anchor_items = sorted(anchor_groups.items())[:nrows]

    for row, (anchor_ix, option_specs) in enumerate(anchor_items):
        any_spec = next(iter(option_specs.values()))
        test_idx = any_spec.test_idx

        all_train_idx = np.concatenate([s.train_idx for s in option_specs.values()])
        unused_idx = np.setdiff1d(
            np.setdiff1d(all_idx, test_idx),
            all_train_idx,
        )

        for col, option in enumerate(options_order):
            ax = axes[row, col]

            ax.set_extent([*x_lim, *y_lim], crs=proj)
            ax.add_feature(cfeature.OCEAN, facecolor="#dceefb", zorder=0)
            ax.add_feature(cfeature.LAND, facecolor="#f7f5f0", zorder=0)
            ax.add_feature(cfeature.BORDERS, edgecolor="#999999", linewidth=0.6, zorder=0.5)
            ax.add_feature(cfeature.COASTLINE, edgecolor="#777777", linewidth=0.6, zorder=0.5)
            ax.add_feature(cfeature.LAKES, facecolor="#dceefb", edgecolor="#999999", linewidth=0.3, zorder=0.5)

            spec = option_specs.get(option)
            train_idx = spec.train_idx if spec is not None else np.array([], dtype=np.int64)

            if len(unused_idx) > 0:
                ax.scatter(X_mat[unused_idx, cx], X_mat[unused_idx, cy],
                           c="#bbbbbb", s=5, zorder=1, linewidths=0, transform=proj)
            if len(train_idx) > 0:
                ax.scatter(X_mat[train_idx, cx], X_mat[train_idx, cy],
                           c=color_map[option], s=10, zorder=2, linewidths=0, transform=proj)
            if len(test_idx) > 0:
                ax.scatter(X_mat[test_idx, cx], X_mat[test_idx, cy],
                           c=color_map["test"], s=10, zorder=3, linewidths=0, transform=proj)

            title = (
                f"anchor {anchor_ix} | {option}\n"
                f"dist={spec.distance:.3f} train={spec.train_size} test={spec.test_size}"
                if spec else f"anchor {anchor_ix} | {option}\n(missing)"
            )
            ax.set_title(title, fontsize=8)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#cccccc", alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": 6}
            gl.ylabel_style = {"size": 6}

            anchor_x = any_spec.anchor[0]
            anchor_y = any_spec.anchor[1]
            ax.scatter(anchor_x, anchor_y, c="black", marker="X", s=100, zorder=4, transform=proj)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out_path = output_dir / "splits_overview.png"
    print(f"Saving plot to {out_path}...")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")


def plot_validation_overview(
    X_pa,
    specs,
    covs_plot: list[str],
    output_dir: Path,
    region: str = "france"
):
    """
    Same panel layout as plot_splits_overview, but for the validation splits
    only (option in {closest_val, middle_val, farthest_val}). There is only
    one anchor with validation splits (the designated validation anchor), so
    this is always a single row.
    """
    X_mat = X_pa[covs_plot].to_numpy()
    all_idx = np.arange(len(X_pa))

    if region == "france":  
        x_lim = (-5, 8.5)
        y_lim = (42, 51.5)
    elif region == "full":
        x_lim = (-20, 40)
        y_lim = (30, 70)
    elif region == "denmark":
        x_lim = (7, 13)
        y_lim = (54, 58)
    proj = ccrs.PlateCarree()

    val_options_order = ("closest_val", "middle_val", "farthest_val")

    anchor_groups: dict[int, dict[str, SplitSpec]] = {}
    for spec in specs:
        if spec.option in val_options_order:
            anchor_groups.setdefault(spec.test_number, {})[spec.option] = spec

    if not anchor_groups:
        print("No validation specs found — skipping validation plot.")
        return

    color_map = {
            "validation": "#8E44AD",
            "closest_val": "#1E88E5",
            "middle_val": "#1E88E5",
            "farthest_val": "#1E88E5",
            }
    cx, cy = 0, 1

    ncols = len(val_options_order)
    anchor_items = sorted(anchor_groups.items())
    nrows = len(anchor_items)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows),
        subplot_kw={"projection": proj},
    )
    axes = np.atleast_2d(axes)

    for row, (anchor_ix, option_specs) in enumerate(anchor_items):
        any_spec = next(iter(option_specs.values()))
        val_idx = any_spec.test_idx  # validation points are stored as test_idx for _val specs

        all_train_idx = np.concatenate([s.train_idx for s in option_specs.values()])
        unused_idx = np.setdiff1d(
            np.setdiff1d(all_idx, val_idx),
            all_train_idx,
        )

        for col, option in enumerate(val_options_order):
            ax = axes[row, col]

            ax.set_extent([*x_lim, *y_lim], crs=proj)
            ax.add_feature(cfeature.OCEAN, facecolor="#dceefb", zorder=0)
            ax.add_feature(cfeature.LAND, facecolor="#f7f5f0", zorder=0)
            ax.add_feature(cfeature.BORDERS, edgecolor="#999999", linewidth=0.6, zorder=0.5)
            ax.add_feature(cfeature.COASTLINE, edgecolor="#777777", linewidth=0.6, zorder=0.5)
            ax.add_feature(cfeature.LAKES, facecolor="#dceefb", edgecolor="#999999", linewidth=0.3, zorder=0.5)

            spec = option_specs.get(option)
            train_idx = spec.train_idx if spec is not None else np.array([], dtype=np.int64)

            if len(unused_idx) > 0:
                ax.scatter(X_mat[unused_idx, cx], X_mat[unused_idx, cy],
                           c="#bbbbbb", s=5, zorder=1, linewidths=0, transform=proj)
            if len(train_idx) > 0:
                ax.scatter(X_mat[train_idx, cx], X_mat[train_idx, cy],
                           c=color_map[option], s=10, zorder=2, linewidths=0, transform=proj)
            if len(val_idx) > 0:
                ax.scatter(X_mat[val_idx, cx], X_mat[val_idx, cy],
                           c=color_map["validation"], s=10, zorder=3, linewidths=0, transform=proj)

            title = (
                f"anchor {anchor_ix} | {option}\n"
                f"dist={spec.distance:.3f} train={spec.train_size} val={spec.test_size}"
                if spec else f"anchor {anchor_ix} | {option}\n(missing)"
            )
            ax.set_title(title, fontsize=8)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#cccccc", alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": 6}
            gl.ylabel_style = {"size": 6}

            anchor_x = any_spec.anchor[0]
            anchor_y = any_spec.anchor[1]
            ax.scatter(anchor_x, anchor_y, c="black", marker="X", s=100, zorder=4, transform=proj)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out_path = output_dir / "splits_overview_validation.png"
    print(f"Saving plot to {out_path}...")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")


def main(dataset_name: str = "GeoPlant",
         region: str = "france",
         split_type: str = "geographical",
         n_anchors: int = 10,
         without_validation: bool = False):

    k_clusters = 200
    data_path = f"data/processed/{dataset_name}/{region}"
    # output_dir = Path(f"outputs/splits/{dataset_name}/{region}_gaussian_k{k_clusters}")
    output_dir = Path(f"outputs/splits/{dataset_name}/{region}_bands/{split_type}")


    data = load_geoplant_processed(data_path, add_coordinates=True)

    X_pa = data.X_pa_train
    y_pa = data.y_pa_train
    covariates = data.covariates

    # covs_cluster = [c for c in ["x", "y", "lon", "lat"] if c in X_pa.columns]
    # if not covs_cluster:
    #     covs_cluster = covariates

    # covs_distance = covariates

    covs_cluster = ['lon', 'lat']

    if split_type == 'geographical':

        covs_distance = ['lon', 'lat']
        distance_metric = 'euclidean'

    if split_type == 'environmental':
        covs_distance = [c for c in covariates if c not in covs_cluster]
        distance_metric = 'mahalanobis'
    # covs_distance = [c for c in covariates if c not in covs_cluster]

    print("covs_cluster:", covs_cluster)
    print("covs_distance:", covs_distance)

    
    bandwidth_scale = .2
    test_proportion = 0.25
    train_proportion_of_clusters = 0.33
    seed = 42

    specs = partition_sweep_bands(
        X_pa=X_pa,
        y_pa=y_pa,
        covs_cluster=covs_cluster,
        covs_distance=covs_distance,
        n_anchors=n_anchors,
        n_bins_per_axis=80,
        # bandwidth_scale=bandwidth_scale,
        test_proportion=test_proportion,
        # train_proportion_of_clusters=train_proportion_of_clusters,
        distance_metric=distance_metric,
        seed=seed,
        options=("closest", "middle", "farthest"),
        reserve_validation = not without_validation
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_split_specs(specs, output_dir)

    # plot — use spatial coords if available, else first 2 distance covariates
    covs_plot = [c for c in ["x", "y", "lon", "lat"] if c in X_pa.columns]
    if len(covs_plot) < 2:
        covs_plot = covs_distance[:2]

    # recompute labels for the plot (same seed → identical to what partition_sweep_gaussian used)
    # labels, _ = make_grid_clusters(X_pa, covariates=covs_cluster)

    plot_splits_overview(
        X_pa=X_pa.reset_index(drop=True),
        specs=specs,
        covs_plot=covs_plot,
        output_dir=output_dir,
        region=region,
        nrows=n_anchors
)
    
    plot_validation_overview(
        X_pa=X_pa.reset_index(drop=True),
        specs=specs,
        covs_plot=covs_plot,
        output_dir=output_dir,
        region=region
    )

    metadata = {
        "data_path": data_path,
        "method": "partition_sweep_bands",
        "region": data.metadata.get("region"),
        "vocab_mode": data.metadata.get("vocab_mode"),
        "covs_cluster": covs_cluster,
        "covs_distance": covs_distance,
        "k_clusters": k_clusters,
        "n_anchors": n_anchors,
        "bandwidth_scale": bandwidth_scale,
        "test_proportion": test_proportion,
        "train_proportion_of_clusters": train_proportion_of_clusters,
        "distance_metric": distance_metric,
        "split_type": split_type,
        "seed": seed,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved split metadata.")


if __name__ == "__main__":

    # grab output dir from command line if provided, else use default

    parser = argparse.ArgumentParser(description="Generate Gaussian splits for GeoPlant France.")
    parser.add_argument("--dataset_name", type=str, default="GeoPlant", help="Name of the dataset (used in output dir).")
    parser.add_argument("--region", type=str, default="france", help="Region name (used in output dir).")
    parser.add_argument("--split_type", type=str, default="geographical", help="Name of the split (used in output dir).")
    parser.add_argument("--n_anchors", type=int, default=3, help="Number of anchors to use for partitioning.")
    parser.add_argument("--without_validation", action="store_true", help="If set, do not generate validation splits.")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    split_type = args.split_type


    main(dataset_name=dataset_name, region=args.region, split_type=split_type, n_anchors=args.n_anchors, without_validation=args.without_validation)