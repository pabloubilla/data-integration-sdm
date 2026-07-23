from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import tqdm

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import torch
import wandb
from torch.utils.data import DataLoader

from isdm.utils import get_overlapping_species_subset

try:
    from k_means_constrained import KMeansConstrained
except ImportError:
    KMeansConstrained = None

from sklearn.cluster import KMeans



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


# ---------------------------------------------------------------------------
# Gaussian density in centroid space
# ---------------------------------------------------------------------------

def compute_cluster_centroids(
    X: pd.DataFrame,
    labels: np.ndarray,
    covariates: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (cluster_ids, centroids), centroids[i] ↔ cluster_ids[i]."""
    cluster_ids = np.unique(labels)
    X_mat = X[covariates].to_numpy(dtype=np.float64)
    centroids = np.vstack([X_mat[labels == c].mean(axis=0) for c in cluster_ids])
    return cluster_ids, centroids


def compute_gaussian_density(
    centroids: np.ndarray,
    anchor: np.ndarray,
    bandwidth_scale: float,
) -> np.ndarray:
    """
    Isotropic Gaussian centered at anchor, covariance = cov(centroids) * bandwidth_scale^2.
    Returns densities normalized to [0, 1].
    """
    cov = np.cov(centroids.T) * (bandwidth_scale ** 2)
    cov += np.eye(len(cov)) * 1e-6  # regularize

    densities = multivariate_normal.pdf(centroids, mean=anchor, cov=cov)
    max_d = densities.max()
    return densities / max_d if max_d > 0 else densities


# ---------------------------------------------------------------------------
# Cluster assignment by point-count quota (shared logic for test and terciles)
# ---------------------------------------------------------------------------

def sample_clusters_by_quota(
    cluster_ids: np.ndarray,
    cluster_sizes: np.ndarray,
    densities: np.ndarray,
    target_n_points: int,
    rng: np.random.Generator,
    descending: bool = True,
) -> np.ndarray:
    """
    Sequentially draw clusters without replacement, weighted by density,
    until target_n_points is reached. Each draw reweights over remaining clusters.

    descending=True  → high density first (test assignment)
    descending=False → used internally; caller controls order via densities
    """
    remaining_ids = list(cluster_ids)
    remaining_sizes = list(cluster_sizes)
    remaining_densities = list(densities if not descending else densities)

    selected = []
    count = 0

    while count < target_n_points and remaining_ids:
        w = np.array(remaining_densities)
        if descending:
            pass  # high density = high probability, draw normally
        w = w / w.sum()

        idx = int(rng.choice(len(remaining_ids), p=w))
        selected.append(remaining_ids[idx])
        count += remaining_sizes[idx]

        remaining_ids.pop(idx)
        remaining_sizes.pop(idx)
        remaining_densities.pop(idx)

    return np.array(selected)


def assign_test_and_train_clusters(
    cluster_ids: np.ndarray,
    cluster_sizes: np.ndarray,
    centroids: np.ndarray,
    anchor: np.ndarray,
    bandwidth_scale: float,
    test_proportion: float,
    n_train_sets: int,
    total_n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw clusters into test and train sets by sampling points from a Gaussian centered at anchor in centroid space.
    """
    
    current_test = 0
    quota_test = int(round(test_proportion * total_n))
    current_train = 0
    quota_close = int(round(0.33 * (total_n - quota_test)))  
    quota_middle = quota_close + int(round(0.33 * (total_n - quota_test)))
    quota_farthest = total_n - quota_test

    # print quotas


    selected_for_test = set()
    selected_for_closest = set()
    selected_for_middle = set()
    selected_for_farthest = set()

    left_clusters_ids = cluster_ids.tolist().copy()

    counter = 0
    for _ in tqdm.tqdm(range(cluster_ids.shape[0]), desc="Assigning clusters", disable=True):
        # sample_point = np.random.multivariate_normal(mean=anchor, cov=np.cov(centroids.T) * (bandwidth_scale ** 2))
        sample_point = rng.multivariate_normal(mean=anchor, cov=np.cov(centroids.T) * (bandwidth_scale ** 2))
        # closest centroid
        dists = np.linalg.norm(centroids - sample_point, axis=1)
        closest_idx = np.argmin(dists[left_clusters_ids])
        if current_test < quota_test and counter % 2 == 0:
            selected_for_test.add(left_clusters_ids[closest_idx])
            current_test += cluster_sizes[left_clusters_ids[closest_idx]]
        elif (current_train < quota_close and counter % 2 == 1) or (current_test >= quota_test and current_train < quota_close):
            selected_for_closest.add(left_clusters_ids[closest_idx])
            current_train += cluster_sizes[left_clusters_ids[closest_idx]]
            # print(f"Current closest train count: {current_train}/{quota_close}" )
        elif (current_train >= quota_close and current_train < quota_middle):
            selected_for_middle.add(left_clusters_ids[closest_idx])
            current_train += cluster_sizes[left_clusters_ids[closest_idx]]
        else: 
            selected_for_farthest.add(left_clusters_ids[closest_idx])
            current_train += cluster_sizes[left_clusters_ids[closest_idx]]

        left_clusters_ids.pop(closest_idx)
        counter += 1

    # swap randomly some samples between closest, middle and farthest
    # fraction_to_swap = 0.1

    # def swap_between(set_a: set, set_b: set, frac: float, rng: np.random.Generator):
    #     """Swap a fraction of elements between two sets in-place."""
    #     n = max(1, int(round(frac * min(len(set_a), len(set_b)))))
    #     from_a = set(rng.choice(list(set_a), size=n, replace=False))
    #     from_b = set(rng.choice(list(set_b), size=n, replace=False))
    #     set_a -= from_a; set_a |= from_b
    #     set_b -= from_b; set_b |= from_a

    # swap_between(selected_for_closest, selected_for_middle,   fraction_to_swap, rng)
    # swap_between(selected_for_middle,  selected_for_farthest, fraction_to_swap, rng)


    

    # quick plot with centroids
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    # plt.scatter(centroids[:, 0], centroids[:, 1], c='lightgray', label='All Clusters')
    plt.scatter(centroids[list(selected_for_test), 0], centroids[list(selected_for_test), 1], c='red', label='Test Clusters')
    plt.scatter(centroids[list(selected_for_closest), 0], centroids[list(selected_for_closest), 1], c='blue', label='Closest Train Clusters')
    plt.scatter(centroids[list(selected_for_middle), 0], centroids[list(selected_for_middle), 1], c='green', label='Middle Train Clusters')
    plt.scatter(centroids[list(selected_for_farthest), 0], centroids[list(selected_for_farthest), 1], c='orange', label='Farthest Train Clusters')
    # print sizes
    plt.scatter(anchor[0], anchor[1], c='black', marker='X', s=100, label='Anchor')
    plt.legend()
    plt.savefig("cluster_assignment.png")
    plt.close()

    return {
        "test": np.array(list(selected_for_test)),
        "closest": np.array(list(selected_for_closest)),
        "middle": np.array(list(selected_for_middle)),
        "farthest": np.array(list(selected_for_farthest)),
    }



        


# def assign_train_terciles(
#     cluster_ids: np.ndarray,  # ascending density order (farthest first)
#     cluster_sizes: np.ndarray,
#     densities: np.ndarray,
#     total_n: int,
#     rng: np.random.Generator,
# ) -> dict[str, np.ndarray]:
#     """
#     Assign remaining (non-test) clusters to farthest/middle/closest terciles,
#     each targeting ~1/3 of total_n points. Uses the same quota-based sequential
#     sampling as test assignment, applied to each tercile in order.

#     Order: farthest (low density) → middle → closest (high density).
#     Inverts densities for farthest so low-density clusters are drawn first.
#     """
#     target = int(round(total_n / 3))
#     result = {}
#     remaining_ids = list(cluster_ids)
#     remaining_sizes = list(cluster_sizes)
#     remaining_densities = list(densities)

#     for option in ["farthest", "middle", "closest"]:
#         if not remaining_ids:
#             result[option] = np.array([], dtype=np.int64)
#             continue

#         # farthest: invert densities so low-density clusters drawn first
#         # middle and closest: use densities as-is (middle draws from what's left after farthest)
#         if option == "farthest":
#             w = 1.0 - np.array(remaining_densities)
#             w = np.clip(w, 1e-9, None)
#         else:
#             w = np.array(remaining_densities)
#             w = np.clip(w, 1e-9, None)

#         selected = []
#         count = 0
#         rem_ids = list(remaining_ids)
#         rem_sizes = list(remaining_sizes)
#         rem_w = list(w)

#         while count < target and rem_ids:
#             probs = np.array(rem_w) / np.array(rem_w).sum()
#             idx = int(rng.choice(len(rem_ids), p=probs))
#             selected.append(rem_ids[idx])
#             count += rem_sizes[idx]
#             rem_ids.pop(idx)
#             rem_sizes.pop(idx)
#             rem_w.pop(idx)

#         result[option] = np.array(selected)

#         # remove selected from remaining pool for next tercile
#         selected_set = set(selected)
#         keep = [i for i, c in enumerate(remaining_ids) if c not in selected_set]
#         remaining_ids = [remaining_ids[i] for i in keep]
#         remaining_sizes = [remaining_sizes[i] for i in keep]
#         remaining_densities = [remaining_densities[i] for i in keep]

#     return result

def assign_train_terciles(
    cluster_ids: np.ndarray,
    cluster_sizes: np.ndarray,
    densities: np.ndarray,  # Gaussian density at each train centroid
    total_n: int,
) -> dict[str, np.ndarray]:
    """
    Sort train clusters by Gaussian density descending (highest = closest to anchor).
    Fill closest → middle → farthest sequentially by point-count quota.
    This gives concentric rings: closest hugs the test region, farthest is the periphery.
    """
    order = np.argsort(densities)[::-1]  # descending: high density first
    sorted_ids   = cluster_ids[order]
    sorted_sizes = cluster_sizes[order]

    target = int(round(total_n / 3))
    groups = {"closest": [], "middle": [], "farthest": []}

    current_option = iter(["closest", "middle", "farthest"])
    current = next(current_option)
    count = 0

    for c, size in zip(sorted_ids, sorted_sizes):
        if count >= target:
            try:
                current = next(current_option)
                count = 0
            except StopIteration:
                pass  # last bucket absorbs remainder
        groups[current].append(c)
        count += size

    return {k: np.array(v) for k, v in groups.items()}

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def partition_sweep_gaussian(
    X_pa: pd.DataFrame,
    y_pa: pd.DataFrame,
    covs_cluster: list[str],
    covs_distance: list[str],
    n_bins_per_axis: int = 30,
    n_anchors: int = 20,
    bandwidth_scale: float = 1.0,
    test_proportion: float = 0.25,
    distance_metric: DistanceMetric = "mahalanobis",
    seed: int = 42,
    options: tuple[SplitOption, ...] = ("closest", "middle", "farthest"),
) -> list[SplitSpec]:
    rng = np.random.default_rng(seed)

    X = X_pa.reset_index(drop=True).copy()


    n = len(X)
    if n == 0:
        raise ValueError("X_pa is empty.")

    # standardize distance covariates
    X[covs_distance] = StandardScaler().fit_transform(X[covs_distance])

    D = compute_distance_matrix(X, covariates=covs_distance, metric=distance_metric)

    labels, grid_info = make_grid_clusters(X, covariates=covs_cluster, n_bins_per_axis=n_bins_per_axis)
    print(f"Assigned {n} points to {len(grid_info)} grid cells.")

    cluster_ids, centroids = compute_cluster_centroids(X, labels, covariates=covs_distance)
    cluster_sizes = np.array([grid_info.loc[grid_info["label"] == c, "size"].values[0] for c in cluster_ids])

    all_idx = np.arange(n)
    specs = []
    split_counter = 0

    def kmeans_anchors(centroids, n_anchors, rng):
        km = KMeans(n_clusters=n_anchors, n_init=10, random_state=rng.integers(0, 2**32))
        km.fit(centroids)
        return km.cluster_centers_
    
    anchors = kmeans_anchors(centroids, n_anchors, rng)

    for anchor_ix in range(n_anchors):
        anchor = anchors[anchor_ix]

        dict_cluster_assignments = assign_test_and_train_clusters(
            cluster_ids=cluster_ids,
            cluster_sizes=cluster_sizes,
            centroids=centroids,
            anchor=anchor,
            bandwidth_scale=bandwidth_scale,
            test_proportion=test_proportion,
            n_train_sets=len(options),
            total_n=n,
            rng=rng,
        )

        test_clusters = dict_cluster_assignments["test"]
        test_idx = np.isin(labels, test_clusters)
        test_idx = np.where(test_idx)[0]

        species_list_dict = {}

        species_list_dict['test'] = sorted(set(s for obs in (y_pa[i] for i in test_idx) for s in obs))
        print(f"Split {split_counter:03d} test: {len(species_list_dict['test'])} unique species, first 10: {species_list_dict['test'][:10]}")

        for option in options:
            train_clusters = dict_cluster_assignments[option]
            train_idx = np.isin(labels, train_clusters)
            train_idx = np.where(train_idx)[0]

            # get species list for this split
            y_train_split = [y_pa[i] for i in train_idx]

            # it is a list of lists, find unique species across all observations
            species_list = sorted(set(s for obs in y_train_split for s in obs))
            species_list_dict[option] = species_list
            
            # print first 10 species for this split
            print(f"Split {split_counter:03d} option {option}: {len(species_list)} unique species, first 10: {species_list[:10]}")

        # intersect all the species lists to find the overlapping species
        overlapping_species = set(species_list_dict['test'])
        for option in options:
            overlapping_species &= set(species_list_dict[option])
        overlapping_species = sorted(overlapping_species)
        print(f"Split {split_counter:03d} overlapping species across all sets: {len(overlapping_species)}, first 10: {overlapping_species[:10]}")


        for option in options:
            train_clusters = dict_cluster_assignments[option]
            train_idx = np.isin(labels, train_clusters)
            train_idx = np.where(train_idx)[0]

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
            ))
            split_counter += 1

    if not specs:
        raise RuntimeError("No valid splits generated. Try increasing bandwidth_scale or n_anchors.")

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
        })

    pd.DataFrame(rows).to_csv(output_dir / "splits.csv", index=False)
    print(f"Saved {len(specs)} split specs to {output_dir}")


def load_split_specs(split_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(split_dir) / "splits.csv")


def load_split_indices(split_dir: str | Path, split_file: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(Path(split_dir) / split_file)
    return arr["train_idx"], arr["test_idx"]