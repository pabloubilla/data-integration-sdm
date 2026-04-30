from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

try:
    from k_means_constrained import KMeansConstrained
except ImportError:
    KMeansConstrained = None


SplitOption = Literal["closest", "middle", "farthest"]
DistanceMetric = Literal["euclidean", "mahalanobis"]


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


def partition_distance(test_idx: np.ndarray, train_idx: np.ndarray, D: np.ndarray) -> float:
    """
    Mean distance from each test point to its nearest train point.
    Higher means more spatial/environmental separation.
    """
    if len(test_idx) == 0 or len(train_idx) == 0:
        return np.nan

    min_dists = np.min(D[np.ix_(test_idx, train_idx)], axis=1)
    return float(np.mean(min_dists))


def _drop_zero_std_covariates(X: pd.DataFrame, covariates: list[str], eps: float = 1e-6) -> list[str]:
    stds = X[covariates].std(axis=0)
    keep = [c for c in covariates if stds[c] >= eps]

    dropped = [c for c in covariates if c not in keep]
    if dropped:
        print(f"Dropping zero-std distance covariates: {dropped}")

    return keep


def compute_distance_matrix(
    X: pd.DataFrame,
    covariates: list[str],
    metric: DistanceMetric = "mahalanobis",
) -> tuple[np.ndarray, list[str]]:
    """
    Computes pointwise distance matrix using selected covariates.
    """
    covariates = _drop_zero_std_covariates(X, covariates)
    X_mat = X[covariates].to_numpy(dtype=np.float64)

    if metric == "euclidean":
        D = pairwise_distances(X_mat, metric="euclidean")

    elif metric == "mahalanobis":
        cov = np.cov(X_mat.T)
        VI = np.linalg.pinv(cov)
        D = pairwise_distances(X_mat, metric="mahalanobis", VI=VI)

    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    return D, covariates


def compute_centroid_to_point_distances(
    X: pd.DataFrame,
    labels: np.ndarray,
    covariates: list[str],
    metric: DistanceMetric = "mahalanobis",
) -> np.ndarray:
    """
    Returns distance from each cluster centroid to each point.
    Shape: [n_clusters, n_points]
    """
    X_mat = X[covariates].to_numpy(dtype=np.float64)
    cluster_ids = np.unique(labels)

    centroids = np.vstack([
        X_mat[labels == k].mean(axis=0)
        for k in cluster_ids
    ])

    if metric == "euclidean":
        return pairwise_distances(centroids, X_mat, metric="euclidean")

    if metric == "mahalanobis":
        cov = np.cov(X_mat.T)
        VI = np.linalg.pinv(cov)
        return pairwise_distances(centroids, X_mat, metric="mahalanobis", VI=VI)

    raise ValueError(f"Unknown distance metric: {metric}")


def make_size_constrained_clusters(
    X: pd.DataFrame,
    covariates: list[str],
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    """
    Size-constrained KMeans labels.
    """
    if KMeansConstrained is None:
        raise ImportError(
            "k_means_constrained is not installed. "
            "Install it or replace with sklearn KMeans."
        )

    n = len(X)
    n_clusters = min(n_clusters, max(1, n // 10))

    avg = n / n_clusters
    size_min = int(np.floor(avg))
    size_max = int(np.ceil(avg))

    print(
        f"Generating {n_clusters} constrained clusters "
        f"with sizes in [{size_min}, {size_max}]"
    )

    km = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=size_max,
        random_state=seed,
        n_init=10,
    )

    labels = km.fit_predict(X[covariates].to_numpy(dtype=np.float64))
    return labels


def select_ranked_points(
    distances: np.ndarray,
    option: SplitOption,
    n_select: int,
) -> np.ndarray:
    """
    Select closest, middle, or farthest points according to distances.
    """
    order = np.argsort(distances)

    if option == "closest":
        return order[:n_select]

    if option == "middle":
        start = max(0, (len(order) - n_select) // 2)
        return order[start:start + n_select]

    if option == "farthest":
        return order[::-1][:n_select]

    raise ValueError(f"Unknown split option: {option}")


def partition_sweep_ranges_v2_indices(
    X_pa: pd.DataFrame,
    covs_cluster: list[str],
    covs_distance: list[str],
    k_clusters: int = 50,
    select_subset: int = 20,
    train_proportion: float = 0.4,
    distance_metric: DistanceMetric = "mahalanobis",
    seed: int = 42,
    options: tuple[SplitOption, ...] = ("closest", "middle", "farthest"),
) -> list[SplitSpec]:
    """
    Clean version of partition_sweep_ranges_v2.

    Produces split indices only:
      - test set = one constrained cluster
      - train set = closest/middle/farthest points relative to that test cluster centroid
    """
    rng = np.random.default_rng(seed)

    X = X_pa.reset_index(drop=True).copy()
    n = len(X)

    if n == 0:
        raise ValueError("X_pa is empty.")

    D, covs_distance_used = compute_distance_matrix(
        X,
        covariates=covs_distance,
        metric=distance_metric,
    )

    labels = make_size_constrained_clusters(
        X=X,
        covariates=covs_cluster,
        n_clusters=k_clusters,
        seed=seed,
    )

    cluster_ids = np.unique(labels)
    if select_subset > len(cluster_ids):
        select_subset = len(cluster_ids)

    selected_clusters = rng.choice(
        cluster_ids,
        size=select_subset,
        replace=False,
    )

    centroid_to_point_D = compute_centroid_to_point_distances(
        X,
        labels=labels,
        covariates=covs_distance_used,
        metric=distance_metric,
    )

    cluster_id_to_pos = {cluster_id: pos for pos, cluster_id in enumerate(cluster_ids)}

    n_train = int(round(n * train_proportion))
    all_idx = np.arange(n)

    specs = []
    split_counter = 0

    # for test_cluster in selected_clusters:
    for ix, test_cluster in enumerate(selected_clusters):
        test_idx = all_idx[labels == test_cluster]
        test_pos = cluster_id_to_pos[test_cluster]

        dists_to_test_centroid = centroid_to_point_D[test_pos].copy()

        # exclude test cluster points from train candidates
        dists_to_test_centroid[test_idx] = np.nan

        candidate_idx = all_idx[~np.isnan(dists_to_test_centroid)]
        candidate_dists = dists_to_test_centroid[candidate_idx]

        for option in options:
            selected_pos = select_ranked_points(
                candidate_dists,
                option=option,
                n_select=min(n_train, len(candidate_idx)),
            )

            train_idx = candidate_idx[selected_pos]

            distance = partition_distance(test_idx, train_idx, D)

            split_id = f"split_{split_counter:03d}_{option}_cluster_{int(test_cluster)}"

            specs.append(
                SplitSpec(
                    split_id=split_id,
                    option=option,
                    test_cluster=int(test_cluster),
                    distance=distance,
                    train_size=int(len(train_idx)),
                    test_size=int(len(test_idx)),
                    train_idx=train_idx.astype(np.int64),
                    test_idx=test_idx.astype(np.int64),
                    test_number=ix,
                )
            )

            split_counter += 1

    specs = sorted(specs, key=lambda s: s.distance)

    print(f"Generated {len(specs)} split specs.")
    print(
        f"Distance range: "
        f"{specs[0].distance:.4f} -> {specs[-1].distance:.4f}"
    )

    return specs


def save_split_specs(specs: list[SplitSpec], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for i, spec in enumerate(specs):
        split_file = output_dir / f"{spec.split_id}.npz"

        np.savez(
            split_file,
            train_idx=spec.train_idx,
            test_idx=spec.test_idx,
        )

        rows.append(
            {
                "split_id": spec.split_id,
                "split_file": split_file.name,
                "option": spec.option,
                "test_cluster": spec.test_cluster,
                "distance": spec.distance,
                "train_size": spec.train_size,
                "test_size": spec.test_size,
                "test_number": spec.test_number,

            }
        )

    pd.DataFrame(rows).to_csv(output_dir / "splits.csv", index=False)

    print(f"Saved split specs to {output_dir}")


def load_split_specs(split_dir: str | Path) -> pd.DataFrame:
    split_dir = Path(split_dir)
    return pd.read_csv(split_dir / "splits.csv")


def load_split_indices(split_dir: str | Path, split_file: str) -> tuple[np.ndarray, np.ndarray]:
    split_dir = Path(split_dir)
    arr = np.load(split_dir / split_file)
    return arr["train_idx"], arr["test_idx"]