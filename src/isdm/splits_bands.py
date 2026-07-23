from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from isdm.utils import get_overlapping_species_subset


SplitOption = Literal["closest", "middle", "farthest"]
DistanceMetric = Literal["euclidean", "mahalanobis"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SplitSpec:
    split_id: str
    option: str
    test_cluster: int
    distance: float
    train_size: int
    test_size: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_number: int
    species_list: list[int] = None
    anchor: list[float] = None


# ---------------------------------------------------------------------------
# Grid clustering
# ---------------------------------------------------------------------------

def make_grid_clusters(
    X: pd.DataFrame,
    covariates: list[str],
    n_bins_per_axis: int = 50,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Partition points into a regular grid. Each non-empty cell becomes a cluster.

    Returns:
        labels    : int array [n_points], contiguous cell index per point
        grid_info : DataFrame with [label, <covariate midpoints>, size]
    """
    X_mat = X[covariates].to_numpy(dtype=np.float64)
    n, d = X_mat.shape

    bin_indices = np.zeros((n, d), dtype=np.int64)
    bin_edges = []

    for j in range(d):
        col = X_mat[:, j]
        edges = np.linspace(col.min(), col.max(), n_bins_per_axis + 1)
        bin_indices[:, j] = np.searchsorted(edges[1:-1], col, side="right")
        bin_edges.append(edges)

    shape = np.array([n_bins_per_axis] * d, dtype=np.int64)
    flat_labels = np.ravel_multi_index(bin_indices.T, shape)
    unique_cells, labels = np.unique(flat_labels, return_inverse=True)

    rows = []
    for cell_flat in unique_cells:
        multi_idx = np.unravel_index(cell_flat, shape)
        row = {
            covariates[j]: 0.5 * (bin_edges[j][multi_idx[j]] + bin_edges[j][multi_idx[j] + 1])
            for j in range(d)
        }
        rows.append(row)

    grid_info = pd.DataFrame(rows)
    grid_info.insert(0, "label", np.arange(len(unique_cells)))
    grid_info["size"] = np.array([np.sum(labels == c) for c in grid_info["label"].values])

    n_total = n_bins_per_axis ** d
    print(f"Grid: {n_bins_per_axis}^{d} = {n_total} cells, {len(unique_cells)} non-empty, {n_total - len(unique_cells)} empty.")

    return labels, grid_info


# ---------------------------------------------------------------------------
# Distance matrix
# ---------------------------------------------------------------------------

def compute_distance_matrix(
    X: pd.DataFrame,
    covariates: list[str],
    metric: DistanceMetric = "mahalanobis",
) -> np.ndarray:
    X_mat = X[covariates].to_numpy(dtype=np.float64)

    if metric == "euclidean":
        return pairwise_distances(X_mat, metric="euclidean")

    cov = np.cov(X_mat.T)
    VI = np.linalg.pinv(cov)
    return pairwise_distances(X_mat, metric="mahalanobis", VI=VI)


def partition_distance(test_idx: np.ndarray, train_idx: np.ndarray, D: np.ndarray) -> float:
    if len(test_idx) == 0 or len(train_idx) == 0:
        return np.nan
    return float(np.mean(np.min(D[np.ix_(test_idx, train_idx)], axis=1)))


def compute_cluster_centroids(
    X: pd.DataFrame,
    labels: np.ndarray,
    covariates: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (cluster_ids, centroids), centroids[i] <-> cluster_ids[i]."""
    cluster_ids = np.unique(labels)
    X_mat = X[covariates].to_numpy(dtype=np.float64)
    centroids = np.vstack([X_mat[labels == c].mean(axis=0) for c in cluster_ids])
    return cluster_ids, centroids


# ---------------------------------------------------------------------------
# Band assignment: sort clusters by distance to anchor, cut into equal-quota
# contiguous chunks. Near zone is interleaved by rank parity into test/closest;
# middle and farthest are the next two contiguous slices outward.
#
#   sorted by distance: [ -------- near zone -------- ][ middle ][ farthest ]
#                         test, close, test, close, ...
# ---------------------------------------------------------------------------

def assign_bands_by_distance(
    cluster_ids: np.ndarray,
    cluster_sizes: np.ndarray,
    centroids: np.ndarray,
    anchor: np.ndarray,
    test_proportion: float,
    total_n: int,
) -> dict[str, np.ndarray]:
    """
    Rank clusters by distance to anchor (in the same covariate space used for
    `centroids`). Fill quotas outward from the anchor:
      - near zone (quota_test + quota_train_band points) is split by rank
        parity into test / closest
      - next quota_train_band points -> middle
      - remainder -> farthest
    quota_train_band is set so |closest| ~= |middle| ~= |farthest|.
    """
    dists = np.linalg.norm(centroids - anchor, axis=1)
    order = np.argsort(dists)
    sorted_ids = cluster_ids[order]
    sorted_sizes = cluster_sizes[order]

    quota_test = int(round(test_proportion * total_n))
    quota_train_band = int(round((total_n - quota_test) / 3))  # closest = middle = farthest
    quota_near = quota_test + quota_train_band

    bands: dict[str, list] = {"test": [], "closest": [], "middle": [], "farthest": []}

    i, cum = 0, 0
    near_ids, near_sizes = [], []
    while i < len(sorted_ids) and cum < quota_near:
        near_ids.append(sorted_ids[i])
        near_sizes.append(sorted_sizes[i])
        cum += sorted_sizes[i]
        i += 1

    for k, c in enumerate(near_ids):
        bands["test" if k % 2 == 0 else "closest"].append(c)

    cum = 0
    while i < len(sorted_ids) and cum < quota_train_band:
        bands["middle"].append(sorted_ids[i])
        cum += sorted_sizes[i]
        i += 1

    bands["farthest"].extend(sorted_ids[i:])

    return {k: np.array(v) for k, v in bands.items()}


def plot_band_assignment(
    centroids: np.ndarray,
    bands: dict[str, np.ndarray],
    anchor: np.ndarray,
    out_path: str = "cluster_assignment.png",
) -> None:
    import matplotlib.pyplot as plt

    colors = {"test": "red", "closest": "blue", "middle": "green", "farthest": "orange"}
    plt.figure(figsize=(8, 6))
    for name, ids in bands.items():
        if len(ids):
            plt.scatter(centroids[ids, 0], centroids[ids, 1], c=colors[name], label=name, s=10)
    plt.scatter(anchor[0], anchor[1], c="black", marker="X", s=100, label="anchor")
    plt.legend()
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def partition_sweep_bands(
    X_pa: pd.DataFrame,
    y_pa: pd.DataFrame,
    covs_cluster: list[str],
    covs_distance: list[str],
    n_bins_per_axis: int = 30,
    n_anchors: int = 20,
    test_proportion: float = 0.25,
    distance_metric: DistanceMetric = "mahalanobis",
    seed: int = 42,
    options: tuple[SplitOption, ...] = ("closest", "middle", "farthest"),
    plot: bool = False,
) -> list[SplitSpec]:
    rng = np.random.default_rng(seed)

    X = X_pa.reset_index(drop=True).copy()

    n = len(X)
    if n == 0:
        raise ValueError("X_pa is empty.")

    # standardize distance covariates
    scaler = StandardScaler()
    scaler.fit(X[covs_distance])
    X[covs_distance] = scaler.transform(X[covs_distance])

    D = compute_distance_matrix(X, covariates=covs_distance, metric=distance_metric)

    labels, grid_info = make_grid_clusters(X, covariates=covs_cluster, n_bins_per_axis=n_bins_per_axis)
    print(f"Assigned {n} points to {len(grid_info)} grid cells.")

    cluster_ids, centroids = compute_cluster_centroids(X, labels, covariates=covs_distance)
    cluster_sizes = np.array([grid_info.loc[grid_info["label"] == c, "size"].values[0] for c in cluster_ids])

    specs = []
    split_counter = 0

    # anchors: k-means centers over cluster centroids (multiple anchors -> loop)
    km = KMeans(n_clusters=n_anchors, n_init=10, random_state=rng.integers(0, 2**32))
    km.fit(centroids)
    anchors = km.cluster_centers_

    for anchor_ix in tqdm.tqdm(range(n_anchors), desc="Anchors"):
        anchor = anchors[anchor_ix]

        bands = assign_bands_by_distance(
            cluster_ids=cluster_ids,
            cluster_sizes=cluster_sizes,
            centroids=centroids,
            anchor=anchor,
            test_proportion=test_proportion,
            total_n=n,
        )

        if plot:
            plot_band_assignment(centroids, bands, anchor, out_path=f"cluster_assignment_{anchor_ix:03d}.png")

        test_idx = np.where(np.isin(labels, bands["test"]))[0]

        species_list_dict = {"test": sorted(set(s for i in test_idx for s in y_pa[i]))}
        print(f"Split {split_counter:03d} test: {len(species_list_dict['test'])} unique species")

        for option in options:
            train_idx = np.where(np.isin(labels, bands[option]))[0]
            species_list_dict[option] = sorted(set(s for i in train_idx for s in y_pa[i]))
            print(f"Split {split_counter:03d} option {option}: {len(species_list_dict[option])} unique species")

        overlapping_species = set(species_list_dict["test"])
        for option in options:
            overlapping_species &= set(species_list_dict[option])
        overlapping_species = sorted(overlapping_species)
        print(f"Split {split_counter:03d} overlapping species across all sets: {len(overlapping_species)}")

        for option in options:
            train_idx = np.where(np.isin(labels, bands[option]))[0]
            
            # save anchor point to plot later
            anchor_original_scale = scaler.inverse_transform(anchor.reshape(1, -1))[0]

            specs.append(SplitSpec(
                split_id=f"split_{split_counter:03d}_{option}_anchor_{anchor_ix:03d}",
                option=option,
                test_cluster=anchor_ix,
                distance=partition_distance(test_idx, train_idx, D),
                train_size=int(len(train_idx)),
                test_size=int(len(test_idx)),
                train_idx=train_idx.astype(np.int64),
                test_idx=test_idx.astype(np.int64),
                test_number=anchor_ix,
                species_list=overlapping_species,
                anchor=anchor_original_scale
            ))
            split_counter += 1

    if not specs:
        raise RuntimeError("No valid splits generated. Try increasing n_anchors or adjusting quotas.")

    specs = sorted(specs, key=lambda s: s.distance)
    print(f"Generated {len(specs)} specs. Distance range: {specs[0].distance:.4f} -> {specs[-1].distance:.4f}")
    return specs


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_split_specs(specs: list[SplitSpec], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in specs:
        split_file = output_dir / f"{spec.split_id}.npz"
        np.savez(split_file, train_idx=spec.train_idx, test_idx=spec.test_idx, species_list=np.array(spec.species_list))
        rows.append({
            "split_id": spec.split_id,
            "split_file": split_file.name,
            "option": spec.option,
            "test_cluster": spec.test_cluster,
            "distance": spec.distance,
            "train_size": spec.train_size,
            "test_size": spec.test_size,
            "test_number": spec.test_number,
            "anchor_x": spec.anchor[0],
            "anchor_y": spec.anchor[1]
        })

    pd.DataFrame(rows).to_csv(output_dir / "splits.csv", index=False)
    print(f"Saved {len(specs)} split specs to {output_dir}")


def load_split_specs(split_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(split_dir) / "splits.csv")


def load_split_indices(split_dir: str | Path, split_file: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(Path(split_dir) / split_file)
    return arr["train_idx"], arr["test_idx"]