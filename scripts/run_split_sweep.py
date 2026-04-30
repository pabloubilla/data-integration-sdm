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

from isdm.load_data import load_geoplant_processed
from isdm.datasets import MultiLabelDataset, collate_multilabel
from isdm.models import MLP, BalancedBCELoss
from isdm.train import train_single_source
from isdm.evaluation import predict_logits, per_species_auc_sparse
from isdm.splits import load_split_specs, load_split_indices
from isdm.utils import set_all_seeds


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


def run_one_split(
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

):
    train_idx, test_idx = load_split_indices(split_dir, split_row["split_file"])

    X_all = data.X_pa_train.reset_index(drop=True)
    y_all = data.y_pa_train

    covariates = data.covariates
    num_classes = len(data.species)

    X_train_df = X_all.iloc[train_idx].copy()
    X_test_df = X_all.iloc[test_idx].copy()

    y_train = subset_list(y_all, train_idx)
    y_test = subset_list(y_all, test_idx)

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

    # no separate val here; this is split-sweep evaluation
    val_loader = None

    # check device including mps
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = MLP(
        input_size=len(covariates),
        output_size=num_classes,
        hidden_size=hidden_dim,
        hidden_layers=hidden_layers,
    )

    run = wandb.init(
        project=project_name,
        name=f"pa_split_{split_row['split_id']}",
        reinit=True,
        config={
            "experiment_name": "pa_split_sweep",
            "split_id": split_row["split_id"],
            "split_option": split_row["option"],
            "split_distance": float(split_row["distance"]),
            "train_size": int(split_row["train_size"]),
            "test_size": int(split_row["test_size"]),
            "source_name": "pa",
            "model_name": "mlp_multilabel",
            "loss_name": "balanced_bce",
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
        criterion=BalancedBCELoss(),
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        wandb_run=run,
    )

    logits = predict_logits(model, X_test, device=device)
    aucs = per_species_auc_sparse(
        logits=logits,
        y_lists=y_test,
        num_classes=num_classes,
    )
    # report how many species are not nan
    print(f"Number of species with valid AUC: {sum(~np.isnan(list(aucs.values())))} / {num_classes}")
    avg_auc = float(np.nanmean(list(aucs.values())))

    wandb.log({"test/avg_auc": avg_auc})
    print(
        f"{split_row['split_id']} | "
        f"distance={split_row['distance']:.4f} | "
        f"avg_auc={avg_auc:.4f}"
    )

    exp_dir = output_dir / split_row["split_id"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), exp_dir / "model.pt")

    with open(exp_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "avg_auc": avg_auc,
                "split_distance": float(split_row["distance"]),
                "aucs": {
                    str(k): None if np.isnan(v) else float(v)
                    for k, v in aucs.items()
                },
            },
            f,
            indent=2,
        )

    run.finish()

    return {
        "split_id": split_row["split_id"],
        "option": split_row["option"],
        "distance": float(split_row["distance"]),
        "train_size": int(split_row["train_size"]),
        "test_size": int(split_row["test_size"]),
        "avg_auc": avg_auc,
        "test_number": int(split_row["test_number"]),
    }


def main():
    seed = 42
    set_all_seeds(seed)

    data_path = "data/processed/GeoPlant/france"
    split_dir = Path("outputs/splits/france")
    output_dir = Path("outputs/split_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    project_name = "sdm-integration"

    batch_size = 500
    epochs = 10
    lr = 1e-3
    weight_decay = 3e-4
    hidden_dim = 250
    hidden_layers = 2

    data = load_geoplant_processed(data_path)
    split_table = load_split_specs(split_dir)

    results = []

    # print unique test numbers
    test_numbers = split_table["test_number"].unique()
    # subsample 5 test numbers for quick testing
    test_numbers = np.random.choice(test_numbers, size=5, replace=False)
    split_table = split_table[split_table["test_number"].isin(test_numbers)]

    for _, row in split_table.iterrows():
        result = run_one_split(
            split_row=row,
            split_dir=split_dir,
            data=data,
            output_dir=output_dir,
            project_name=project_name,
            seed=seed,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        )
        results.append(result)

    summary = pd.DataFrame(results)
    summary.to_csv(output_dir / "summary.csv", index=False)

    print("\n=== Split sweep summary ===")
    print(summary.sort_values("distance"))


if __name__ == "__main__":
    main()