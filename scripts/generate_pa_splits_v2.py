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
):
    """
    One panel per anchor row, one column per option (closest/middle/farthest).
    Cluster positions inferred directly from spec indices — no label recomputation.
    Panels are drawn on a cartopy PlateCarree basemap (coastline, borders,
    land/ocean shading) so points sit on a recognizable map of France/Europe.
    """
    X_mat = X_pa[covs_plot].to_numpy()
    all_idx = np.arange(len(X_pa))
 
    x_lim = (-5, 11)
    y_lim = (39, 52)
    proj = ccrs.PlateCarree()
 
    anchor_groups: dict[int, dict[str, SplitSpec]] = {}
    for spec in specs:
        anchor_groups.setdefault(spec.test_number, {})[spec.option] = spec
 
    options_order = ("closest", "middle", "farthest")
    option_colors = {"closest": "#1E88E5", "middle": "#FFA007", "farthest": "#24e18c"}
    cx, cy = 0, 1
 
    nrows = len(anchor_groups)
    ncols = len(options_order)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows),
        subplot_kw={"projection": proj},
    )
    axes = np.atleast_2d(axes)
 
    for row, (anchor_ix, option_specs) in enumerate(sorted(anchor_groups.items())):
        any_spec = next(iter(option_specs.values()))
        test_idx = any_spec.test_idx
 
        # union of all train indices across options for this anchor
        all_train_idx = np.concatenate([s.train_idx for s in option_specs.values()])
        unused_idx = np.setdiff1d(
            np.setdiff1d(all_idx, test_idx),
            all_train_idx,
        )
 
        for col, option in enumerate(options_order):
            ax = axes[row, col]
 
            # basemap: ocean/land shading, coastline, and country borders,
            # drawn first (low zorder) so the scatter points sit on top
            ax.set_extent([*x_lim, *y_lim], crs=proj)
            ax.add_feature(cfeature.OCEAN, facecolor="#dceefb", zorder=0)
            ax.add_feature(cfeature.LAND, facecolor="#f7f5f0", zorder=0)
            ax.add_feature(cfeature.BORDERS, edgecolor="#999999", linewidth=0.6, zorder=0.5)
            ax.add_feature(cfeature.COASTLINE, edgecolor="#777777", linewidth=0.6, zorder=0.5)
            ax.add_feature(cfeature.LAKES, facecolor="#dceefb", edgecolor="#999999", linewidth=0.3, zorder=0.5)
 
            spec = option_specs.get(option)
            train_idx = spec.train_idx if spec is not None else np.array([], dtype=np.int64)
 
            # plot each group as a single scatter call — much faster than per-cluster loop
            if len(unused_idx) > 0:
                ax.scatter(X_mat[unused_idx, cx], X_mat[unused_idx, cy],
                           c="#bbbbbb", s=5, zorder=1, linewidths=0, transform=proj)
            if len(train_idx) > 0:
                ax.scatter(X_mat[train_idx, cx], X_mat[train_idx, cy],
                           c=option_colors[option], s=10, zorder=2, linewidths=0, transform=proj)
            if len(test_idx) > 0:
                ax.scatter(X_mat[test_idx, cx], X_mat[test_idx, cy],
                           c="#D81B60", s=10, zorder=3, linewidths=0, transform=proj)
 
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
            
            # plot anchor
            anchor_x = any_spec.anchor[0]
            anchor_y = any_spec.anchor[1]
            ax.scatter(anchor_x, anchor_y, c="black", marker="X", s=100, zorder=4, transform=proj)

        print('The anchor is:')
        print(anchor_x, anchor_y)
 


    legend_handles = [
        mpatches.Patch(color="#e74c3c", label="test"),
        mpatches.Patch(color=option_colors["closest"], label="train closest"),
        mpatches.Patch(color=option_colors["middle"], label="train middle"),
        mpatches.Patch(color=option_colors["farthest"], label="train farthest"),
        mpatches.Patch(color="#bbbbbb", label="unused"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="black",
            markeredgecolor="black", markersize=10, linestyle="none",
            label="anchor"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
 
    out_path = output_dir / "splits_overview.png"
    print(f"Saving plot to {out_path}...")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")


def main(dataset_name: str = "GeoPlant",
         split_type: str = "geographical"):

    k_clusters = 200
    data_path = f"data/processed/{dataset_name}/france"
    # output_dir = Path(f"outputs/splits/{dataset_name}/france_gaussian_k{k_clusters}")
    output_dir = Path(f"outputs/splits/{dataset_name}/france_bands/{split_type}")


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

    
    n_anchors = 8
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
        options=("closest", "middle", "farthest")
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
    parser.add_argument("--split_type", type=str, default="geographical", help="Name of the split (used in output dir).")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    split_type = args.split_type


    main(dataset_name=dataset_name, split_type=split_type)