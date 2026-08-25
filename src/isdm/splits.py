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


from pathlib import Path
from functools import partial
import json
import pickle

import numpy as np
import pandas as pd
import torch
import wandb

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch import nn

from isdm.load_data import load_geoplant_processed
from isdm.datasets import MultiLabelDataset, collate_multilabel
from isdm.models import MLP
from isdm.losses import  BalancedBCELoss, DeepMaxEntLoss, BernoulliFromLogRateLoss, IntegratedLoss, LOSS_REGISTRY
from isdm.train import train_single_source, train_double_source
from isdm.evaluation import predict_logits, per_species_auc_sparse, per_site_auc_sparse, LogitsStore
from isdm.utils import get_overlapping_species_subset, set_all_seeds, get_device, filter_and_remap

from typing import Optional



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
    size_range:int = 10 # defines a range to make the problem less tight (Kmeans)
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
    size_min = int(np.floor(avg)) - size_range
    size_max = int(np.ceil(avg)) + size_range

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


def load_split(split_dir: str | Path, split_file: str):
    split_dir = Path(split_dir)
    arr = np.load(split_dir / split_file)
    print(arr)
    return arr #arr["train_idx"], arr["test_idx"]


def subset_list(xs, idx):
    return [xs[int(i)] for i in idx]


def make_loader(X, y_lists, num_classes: int, batch_size: int, shuffle: bool):
    ds = MultiLabelDataset(X, y_lists)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(collate_multilabel, num_classes=num_classes),
    )

def run_one_split_po_or_pa_tunable(*, loss_name: str | None = None, **kwargs):
    criterion = LOSS_REGISTRY[loss_name]() if loss_name is not None else None
    return run_one_split_po_or_pa(criterion=criterion, **kwargs)
def run_one_split_po_or_pa(
    *,
    split_row,
    split_dir: Path,
    data,
    output_dir: Path,
    project_name: str,
    seed: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    hidden_layers: int,
    source: Literal["po", "pa"] = "pa",
    criterion=None,
    use_overlapping_species: bool = True,
    validate_during_training: bool = False,
    val_patience: Optional[int] = None,
):
    """
    Trains a single-source model (PO-only or PA-only), restricted to this
    split's overlapping species, and evaluates on this split's PA test set.

    source="pa": trains and evaluates on the split's own PA train/test.
    source="po": trains on the FULL PO dataset (restricted to this split's
    overlapping species), evaluates on this split's PA test set — there is
    no separate PO test set; this lets PO-only be compared against PA-only
    and POPA on the exact same test points.
    """
    split = load_split(split_dir, split_row["split_file"])
    overlapping_species_list = split["species_list"]
    species = overlapping_species_list if use_overlapping_species else data.species
    num_classes = len(species)

    covariates = data.covariates
    device = get_device()

    if source == "pa":
        train_idx, test_idx = split["train_idx"], split["test_idx"]
        X_all = data.X_pa_train.reset_index(drop=True)
        y_all = data.y_pa_train

        X_train_df = X_all.iloc[train_idx].copy()
        X_test_df = X_all.iloc[test_idx].copy()
        y_train = subset_list(y_all, train_idx)
        y_test = subset_list(y_all, test_idx)

        if criterion is None:
            criterion = BalancedBCELoss()

    elif source == "po":
        test_idx = split["test_idx"]
        X_all = data.X_pa_train.reset_index(drop=True)
        y_all = data.y_pa_train

        X_train_df = data.X_po.reset_index(drop=True)
        y_train = data.y_po
        X_test_df = X_all.iloc[test_idx].copy()
        y_test = subset_list(y_all, test_idx)

        if criterion is None:
            criterion = DeepMaxEntLoss()

    else:
        raise ValueError(f"Unknown source: {source!r}")

    if use_overlapping_species:
        y_train = filter_and_remap(overlapping_species_list, y_train)
        y_test = filter_and_remap(overlapping_species_list, y_test)

    scaler = StandardScaler().fit(X_train_df[covariates])
    X_train = scaler.transform(X_train_df[covariates]).astype(np.float32)
    X_test = scaler.transform(X_test_df[covariates]).astype(np.float32)

    train_loader = make_loader(
        X_train,
        y_train,
        num_classes=num_classes,
        batch_size=batch_size,
        shuffle=True,
    )

    model = MLP(
        input_size=len(covariates),
        output_size=num_classes,
        hidden_size=hidden_dim,
        hidden_layers=hidden_layers,
    )

    run = wandb.init(
        project=project_name,
        name=f"{source}_split_{split_row['split_id']}",
        reinit=True,
        config={
            "experiment_name": f"{source}_split_sweep",
            "split_id": split_row["split_id"],
            "split_option": split_row["option"],
            "split_distance": float(split_row["distance"]),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "source_name": source,
            "model_name": "mlp_multilabel",
            "loss_name": criterion.__class__.__name__,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "num_species": num_classes,
            "num_covariates": len(covariates),
            "seed": seed,
        },
    )

    val_fn = None
    if validate_during_training:
        def val_fn(model):
            logits = predict_logits(model, X_test, device=device)
            aucs = per_site_auc_sparse(logits=logits, y_lists=y_test, num_classes=num_classes)
            return float(np.nanmean(list(aucs.values())))

    history = train_single_source(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=None,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        wandb_run=run,
        val_fn=val_fn,
        val_mode="max",
        patience=val_patience,
        restore_best_weights=True,
    )

    logits = predict_logits(model, X_test, device=device)
    aucs_species = per_species_auc_sparse(
        logits=logits,
        y_lists=y_test,
        num_classes=num_classes,
    )
    valid_species = int(sum(~np.isnan(list(aucs_species.values()))))
    avg_auc_species = float(np.nanmean(list(aucs_species.values())))

    wandb.log({
        "test/avg_auc_species": avg_auc_species,
        "test/valid_species_auc": valid_species,
    })
    print(
        f"{source.upper()} | {split_row['split_id']} | "
        f"distance={split_row['distance']:.4f} | "
        f"avg_auc_species={avg_auc_species:.4f}"
    )

    aucs_site = per_site_auc_sparse(
        logits=logits,
        y_lists=y_test,
        num_classes=num_classes,
    )
    print(f"Number of sites with valid AUC: {sum(~np.isnan(list(aucs_site.values())))} / {len(y_test)}")
    avg_auc_site = float(np.nanmean(list(aucs_site.values())))

    wandb.log({"test/avg_auc_site": avg_auc_site})
    print(
        f"{source.upper()} | {split_row['split_id']} | "
        f"distance={split_row['distance']:.4f} | "
        f"avg_auc_site={avg_auc_site:.4f}"
    )

    exp_dir = output_dir / f"{source}_split_sweep" / split_row["split_id"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), exp_dir / "model.pt")

    with open(exp_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "avg_auc_species": avg_auc_species,
                "avg_auc_site": avg_auc_site,
                "valid_species_auc": valid_species,
                "split_distance": float(split_row["distance"]),
                "aucs": {
                    str(k): None if np.isnan(v) else float(v)
                    for k, v in aucs_species.items()
                },
            },
            f,
            indent=2,
        )

    run.finish()

    return {
        "source": source,
        "split_id": split_row["split_id"],
        "option": split_row["option"],
        "distance": float(split_row["distance"]),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "test_number": int(split_row["test_number"]),
        "avg_auc_species": avg_auc_species,
        "avg_auc_site": avg_auc_site,
        "valid_species_auc": valid_species,
        "logits": None,
        "y_test": y_test,
        "best_epoch": history.get("best_epoch", epochs),
    }

def run_one_split_popa_tunable(*, loss_po_name: str, loss_pa_name: str = "balanced_bce", **kwargs):
    """
    Thin wrapper around run_one_split_popa that resolves loss names
    (from LOSS_REGISTRY) into instantiated loss objects, so loss choice
    can be swept as a plain string param in the grid search.
    """
    criterion_po = LOSS_REGISTRY[loss_po_name]()
    criterion_pa = LOSS_REGISTRY[loss_pa_name]()
    return run_one_split_popa(
        criterion_po=criterion_po,
        criterion_pa=criterion_pa,
        **kwargs,
    )
def run_one_split_popa(
    *,
    split_row,
    split_dir: Path,
    data,
    output_dir: Path,
    project_name: str,
    seed: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    hidden_layers: int,
    w_po: float = .5,
    w_pa: float = .5,
    criterion_po = DeepMaxEntLoss(),
    criterion_pa = BalancedBCELoss(),
    concat_sources: bool = False,
    return_logits: bool = False,
    add_po_cov: bool = False,
    use_overlapping_species: bool = True,
    # NEW:
    validate_during_training: bool = False,
    val_patience: Optional[int] = None,
):
    split = load_split(split_dir, split_row["split_file"])

    train_idx, test_idx = split["train_idx"], split["test_idx"]
    overlapping_species_list = split['species_list']
    if use_overlapping_species:
        species = overlapping_species_list
    else:
        species = data.species

    covariates = data.covariates
    num_classes = len(species)


    device = get_device()

    # -------------------------
    # PO source: all PO
    # -------------------------
    X_po_df = data.X_po.reset_index(drop=True)
    y_po = data.y_po

    if use_overlapping_species:
        y_po = filter_and_remap(overlapping_species_list, y_po)

    # -------------------------
    # PA source: split-specific PA train/test
    # -------------------------
    X_pa_all = data.X_pa_train.reset_index(drop=True)
    y_pa_all = data.y_pa_train

    X_pa_train_df = X_pa_all.iloc[train_idx].copy()
    X_pa_test_df = X_pa_all.iloc[test_idx].copy()

    y_pa_train = subset_list(y_pa_all, train_idx)
    y_pa_test = subset_list(y_pa_all, test_idx)

    if use_overlapping_species:
        y_pa_train = filter_and_remap(overlapping_species_list, y_pa_train)
        y_pa_test = filter_and_remap(overlapping_species_list, y_pa_test)




    if add_po_cov:
        X_po_df['PO'] = 1
        X_pa_train_df['PO'] = 0
        X_pa_test_df['PO'] = 0
        covariates = covariates + ['PO']

    # -------------------------
    # Fit scaler on PO + PA train
    # -------------------------
    scaler = StandardScaler().fit(
        pd.concat(
            [
                X_po_df[covariates],
                X_pa_train_df[covariates],
            ],
            axis=0,
        )
    )

    X_po = scaler.transform(X_po_df[covariates]).astype(np.float32)
    X_pa_train = scaler.transform(X_pa_train_df[covariates]).astype(np.float32)
    X_pa_test = scaler.transform(X_pa_test_df[covariates]).astype(np.float32)

    po_loader = make_loader(
        X_po,
        y_po,
        num_classes=num_classes,
        batch_size=batch_size,
        shuffle=True,
    )

    pa_loader = make_loader(
        X_pa_train,
        y_pa_train,
        num_classes=num_classes,
        batch_size=batch_size,
        shuffle=True,
    )

    model = MLP(
        input_size=len(covariates),
        output_size=num_classes,
        hidden_size=hidden_dim,
        hidden_layers=hidden_layers,
    )

    criterion = IntegratedLoss(
        source1_loss=criterion_po,
        source2_loss=criterion_pa,
        source1_weight=w_po,
        source2_weight=w_pa,
        bias_weight=0.0,
        clamp_source1=(-10, 10),
        clamp_source2=(-10, 10),
        concat_sources=concat_sources,
    )

    run = wandb.init(
        project=project_name,
        name=f"popa_split_{split_row['split_id']}",
        reinit=True,
        config={
            "experiment_name": "popa_split_sweep",
            "split_id": split_row["split_id"],
            "split_option": split_row["option"],
            "split_distance": float(split_row["distance"]),
            "test_number": int(split_row["test_number"]),
            "source1_name": "po",
            "source2_name": "pa",
            "model_name": "mlp_multilabel",
            "loss_name": "integrated_deepmaxent_bernoulli_rate",
            "po_loss": "deepmaxent",
            "pa_loss": "bernoulli_from_log_rate",
            "w_po": w_po,
            "w_pa": w_pa,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "num_species": num_classes,
            "num_covariates": len(covariates),
            "seed": seed,
        },
    )

    val_fn = None
    if validate_during_training:
        def val_fn(model):
            logits = predict_logits(model, X_pa_test, device=device)
            aucs = per_site_auc_sparse(logits=logits, y_lists=y_pa_test, num_classes=num_classes)
            return float(np.nanmean(list(aucs.values())))

    history = train_double_source(
        model=model,
        criterion=criterion,
        source1_loader=po_loader,
        source2_loader=pa_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        wandb_run=run,
        source1_name="po",
        source2_name="pa",
        cycle_shorter_loader=True,
        max_steps_per_epoch=max(len(po_loader), len(pa_loader)),  # one epoch = one pass through longer dataset
        ## for validation:
        val_fn=val_fn,
        val_mode="max",
        patience=val_patience,
        restore_best_weights=True,
    )

    logits = predict_logits(model, X_pa_test, device=device)

    aucs_species = per_species_auc_sparse(
        logits=logits,
        y_lists=y_pa_test,
        num_classes=num_classes,
    )

    valid_species = int(sum(~np.isnan(list(aucs_species.values()))))
    avg_auc_species = float(np.nanmean(list(aucs_species.values())))

    wandb.log(
        {
            "test/avg_auc_species": avg_auc_species,
            "test/valid_species_auc": valid_species,
        }
    )

    aucs_site = per_site_auc_sparse(
        logits=logits,
        y_lists=y_pa_test,
        num_classes=num_classes,
    )
    # report how many sites are not nan
    print(f"Number of sites with valid AUC: {sum(~np.isnan(list(aucs_site.values())))} / {len(y_pa_test)}")
    avg_auc_site = float(np.nanmean(list(aucs_site.values())))

    wandb.log({"test/avg_auc_site": avg_auc_site})

    print(
        f"PO+PA | {split_row['split_id']} | "
        f"distance={split_row['distance']:.4f} | "
        f"avg_auc_site={avg_auc_site:.4f} | "
        f"avg_auc_species={avg_auc_species:.4f} | "
        f"valid_species={valid_species}/{num_classes}"
    )

    exp_dir = output_dir / "popa_split_sweep" / split_row["split_id"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), exp_dir / "model.pt")

    with open(exp_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "avg_auc_species": avg_auc_species,
                "valid_species_auc": valid_species,
                "split_distance": float(split_row["distance"]),
                "aucs": {
                    str(k): None if np.isnan(v) else float(v)
                    for k, v in aucs_species.items()
                },
            },
            f,
            indent=2,
        )

    run.finish()

    return {
        "source": "popa",
        "split_id": split_row["split_id"],
        "option": split_row["option"],
        "distance": float(split_row["distance"]),
        "train_size": int(split_row["train_size"]),
        "test_size": int(split_row["test_size"]),
        "test_number": int(split_row["test_number"]),
        "avg_auc_species": avg_auc_species,
        "avg_auc_site": avg_auc_site,
        "valid_species_auc": valid_species,
        "logits": logits if return_logits else None,
        "y_test": y_pa_test,
        "best_epoch": history.get("best_epoch", epochs)
    }




# TODO: For results with PO-only model check the intersection part
def train_po_for_splits(
    *,
    data,
    output_dir: Path,
    project_name: str,
    seed: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    hidden_layers: int,
    criterion = DeepMaxEntLoss(),
    use_overlapping_species: bool = True
):
    print("\n=== Training one PO-only model on all PO data ===")

    covariates = data.covariates
    num_classes = len(data.species)
    device = get_device()

    X_po_df = data.X_po.reset_index(drop=True)
    y_po = data.y_po

    scaler = StandardScaler().fit(X_po_df[covariates])
    X_po = scaler.transform(X_po_df[covariates]).astype(np.float32)

    train_loader = make_loader(
        X_po,
        y_po,
        num_classes=num_classes,
        batch_size=batch_size,
        shuffle=True,
    )

    model = MLP(
        input_size=len(covariates),
        output_size=num_classes,
        hidden_size=hidden_dim,
        hidden_layers=hidden_layers,
    )


    run = wandb.init(
        project=project_name,
        name="po_only_all_po",
        reinit=True,
        config={
            "experiment_name": "po_only_all_po_split_eval",
            "source_name": "po",
            "model_name": "mlp_multilabel",
            "loss_name": "",
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "num_species": num_classes,
            "num_covariates": len(covariates),
            "seed": seed,
        },
    )

    history = train_single_source(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=None,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        wandb_run=run,
    )

    exp_dir = output_dir / "po_only_all_po"
    exp_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), exp_dir / "model.pt")

    with open(exp_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    run.finish()

    return model, scaler

def evaluate_model_on_all_pa_split_tests(
    *,
    model,
    scaler,
    split_table: pd.DataFrame,
    split_dir: Path,
    data,
    output_dir: Path,
    project_name: str,
    source_name: str,
    use_overlapping_species: bool = True,
) -> tuple[pd.DataFrame, LogitsStore]:
    """
    Returns:
        summary  — lightweight DataFrame of scalar metrics (CSV-ready)
        store    — LogitsStore with logits + y_lists (save separately as .npz)
    """
    covariates = data.covariates
    num_classes = len(data.species)
    device = get_device()

    model.to(device)
    model.eval()

    X_all = data.X_pa_train.reset_index(drop=True)
    y_all = data.y_pa_train

    results = []
    store = LogitsStore()

    # run = wandb.init(
    #     project=project_name,
    #     name=f"{source_name}_eval_all_pa_splits",
    #     reinit=True,
    #     config={
    #         "experiment_name": f"{source_name}_eval_all_pa_splits",
    #         "source_name": source_name,
    #         "model_name": model.__class__.__name__,
    #         "num_species": num_classes,
    #         "num_covariates": len(covariates),
    #     },
    # )

    for _, split_row in split_table.iterrows():
        # TODO adapt to new split
        split = load_split(split_dir, split_row["split_file"])

        train_idx, test_idx = split["train_idx"], split["test_idx"]
        overlapping_species_list = split['species_list']


        X_test_df = X_all.iloc[test_idx].copy()
        y_test = subset_list(y_all, test_idx)
        # for y_test only keep species in overlapping_species_list (without remapping)
        # this might not be the best way to do it, it still trains a big model for PO
        if use_overlapping_species: y_test = [[s for s in obs if s in overlapping_species_list] for obs in y_test]

        X_test = scaler.transform(X_test_df[covariates]).astype(np.float32)

        logits = predict_logits(model, X_test, device=device)
        aucs_species = per_species_auc_sparse(logits=logits, y_lists=y_test, num_classes=num_classes)

        valid_species = int(sum(~np.isnan(list(aucs_species.values()))))
        avg_auc_species = float(np.nanmean(list(aucs_species.values())))

        aucs_site = per_site_auc_sparse(logits=logits, y_lists=y_test, num_classes=num_classes)
        avg_auc_site = float(np.nanmean(list(aucs_site.values())))

        # scalars only in results
        results.append({
            "source": source_name,
            "split_id": split_row["split_id"],
            "option": split_row["option"],
            "distance": float(split_row["distance"]),
            "train_size": int(split_row["train_size"]),
            "test_size": int(split_row["test_size"]),
            "test_number": int(split_row["test_number"]),
            "avg_auc_species": avg_auc_species,
            "avg_auc_site": avg_auc_site,

            "valid_species_auc": valid_species,
        })

        # arrays go directly into the store
        store.add(
            split_id=split_row["split_id"],
            option=split_row["option"],
            distance=float(split_row["distance"]),
            test_number=int(split_row["test_number"]),
            logits=logits,
            y_lists=y_test,
        )

        

    # run.finish()

    summary = pd.DataFrame(results)
    summary.to_csv(output_dir / f"summary_{source_name}.csv", index=False)

    return summary, store



