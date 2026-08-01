# scripts/run_single_source_baselines.py

from pathlib import Path
import json
import pickle
import time

import numpy as np
import pandas as pd
import torch
import wandb

from functools import partial

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch import nn

from isdm.load_data import load_geoplant_processed
from isdm.datasets import MultiLabelDataset, collate_multilabel
from isdm.models import MLP
from isdm.losses import DeepMaxEntLoss, BalancedBCELoss, IntegratedLoss, BernoulliFromLogRateLoss
# import BCE Loss
from torch.nn import BCEWithLogitsLoss
from isdm.train import train_single_source, train_double_source
from isdm.train import train_single_source
from isdm.evaluation import predict_logits, per_species_auc_sparse, per_site_auc_sparse
from isdm.utils import set_all_seeds


def split_train_val(X, y_lists, val_fraction: float, seed: int):
    idx = np.arange(len(X))
    idx_train, idx_val = train_test_split(
        idx,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
    )

    X_train = X.iloc[idx_train].copy()
    X_val = X.iloc[idx_val].copy()

    y_train = [y_lists[i] for i in idx_train]
    y_val = [y_lists[i] for i in idx_val]

    return X_train, y_train, X_val, y_val


def make_loader(X, y_lists, num_classes: int, batch_size: int, shuffle: bool):
    dataset = MultiLabelDataset(X, y_lists)

    collate_fn = partial(collate_multilabel, num_classes=num_classes)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=10,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )


def run_single_source_experiment(
    *,
    source_name: str,
    X_train_df: pd.DataFrame,
    y_train_lists,
    X_test_df: pd.DataFrame,
    y_test_lists,
    covariates,
    num_classes: int,
    output_dir: Path,
    seed: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    hidden_layers: int,
    criterion,
    criterion_name: str,
    project_name: str,
):
    print(f"\n=== Running {source_name.upper()}-only experiment ===")

    X_tr_df, y_tr, X_val_df, y_val = split_train_val(
        X_train_df,
        y_train_lists,
        val_fraction=0.2,
        seed=seed,
    )

    scaler = StandardScaler().fit(X_tr_df[covariates])

    X_tr = scaler.transform(X_tr_df[covariates]).astype(np.float32)
    X_val = scaler.transform(X_val_df[covariates]).astype(np.float32)
    X_test = scaler.transform(X_test_df[covariates]).astype(np.float32)

    train_loader = make_loader(
        X_tr,
        y_tr,
        num_classes=num_classes,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = make_loader(
        X_val,
        y_val,
        num_classes=num_classes,
        batch_size=batch_size,
        shuffle=False,
    )

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # check for cuda or mps
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
        name=f"{source_name}_only_{criterion_name}",
        reinit=True,
        config={
            "experiment_name": f"{source_name}_only",
            "source_name": source_name,
            "model_name": "mlp_multilabel",
            "loss_name": criterion_name,
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
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        wandb_run=run,
    )

    logits_test = predict_logits(model, X_test, device=device)
    species_auc = per_species_auc_sparse(
        logits=logits_test,
        y_lists=y_test_lists,
        num_classes=num_classes,
    )
    avg_species_auc = float(np.nanmean(list(species_auc.values())))

    sites_auc = per_site_auc_sparse(
        logits=logits_test,
        y_lists=y_test_lists,
        num_classes=num_classes,
    )
    avg_sites_auc = float(np.nanmean(list(sites_auc.values())))

    wandb.log({"test/avg_species_auc": avg_species_auc, "test/avg_sites_auc": avg_sites_auc})
    print(f"[{source_name.upper()}]\ttest avg AUC: {avg_species_auc:.4f}\ttest avg site AUC: {avg_sites_auc:.4f}")

    exp_dir = output_dir / f"{source_name}_only"
    exp_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), exp_dir / "model.pt")

    with open(exp_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "avg_auc": avg_species_auc,
                "aucs": {str(k): None if np.isnan(v) else float(v) for k, v in species_auc.items()},
            },
            f,
            indent=2,
        )

    run.finish()

    return {
        "source": source_name,
        "avg_species_auc": avg_species_auc,
        "avg_sites_auc": avg_sites_auc,
        "model_path": str(exp_dir / "model.pt"),
        "scaler_path": str(exp_dir / "scaler.pkl"),
    }

def run_integrated_experiment(
    *,
    X_po_df: pd.DataFrame,
    y_po_lists,
    X_pa_train_df: pd.DataFrame,
    y_pa_train_lists,
    X_pa_test_df: pd.DataFrame,
    y_pa_test_lists,
    covariates,
    num_classes: int,
    output_dir: Path,
    seed: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    hidden_layers: int,
    project_name: str,
    w_po: float = 1.0,
    w_pa: float = 2.5,
    criterion_1 = DeepMaxEntLoss(),
    criterion_2 = BernoulliFromLogRateLoss(),
    name = "po_pa_integrated",
    add_po_cov = False

):
    print("\n=== Running PO+PA integrated experiment ===")

    if add_po_cov:
        X_pa_train_df['PO'] = 0
        X_pa_test_df['PO'] = 0
        X_po_df['PO'] = 1
        covariates = covariates + ['PO']

    print(f"Covariates used for {name}: {covariates}")

    X_pa_tr_df, y_pa_tr, X_pa_val_df, y_pa_val = split_train_val(
        X_pa_train_df,
        y_pa_train_lists,
        val_fraction=0.2,
        seed=seed,
    )

    # Fit scaler on both PO and PA train.
    scaler = StandardScaler().fit(
        pd.concat([X_po_df[covariates], X_pa_tr_df[covariates]], axis=0)
    )

    X_po = scaler.transform(X_po_df[covariates]).astype(np.float32)
    X_pa_tr = scaler.transform(X_pa_tr_df[covariates]).astype(np.float32)
    X_pa_val = scaler.transform(X_pa_val_df[covariates]).astype(np.float32)
    X_pa_test = scaler.transform(X_pa_test_df[covariates]).astype(np.float32)

    po_loader = make_loader(
        X_po,
        y_po_lists,
        num_classes=num_classes,
        batch_size=batch_size,
        shuffle=True,
    )

    pa_loader = make_loader(
        X_pa_tr,
        y_pa_tr,
        num_classes=num_classes,
        batch_size=batch_size,
        shuffle=True,
    )

    pa_val_loader = make_loader(
        X_pa_val,
        y_pa_val,
        num_classes=num_classes,
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(
        input_size=len(covariates),
        output_size=num_classes,
        hidden_size=hidden_dim,
        hidden_layers=hidden_layers,
    )
    

    criterion = IntegratedLoss(
        source1_loss=criterion_1,
        source2_loss=criterion_2,
        source1_weight=w_po,
        source2_weight=w_pa,
        bias_weight=0.0,
        clamp_source1=(-10, 10),
        clamp_source2=(-10, 10),
    )

    run = wandb.init(
        project=project_name,
        name="popa_integrated_deepmaxent_bernoulli_rate",
        reinit=True,
        config={
            "experiment_name": "popa_integrated",
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
    )

    # Optional validation loss on PA validation only.
    # This is not part of the integrated training loss; it is just a monitoring signal.
    val_logits = predict_logits(model, X_pa_val, device=device)
    val_aucs = per_species_auc_sparse(
        logits=val_logits,
        y_lists=y_pa_val,
        num_classes=num_classes,
    )
    val_avg_auc = float(np.nanmean(list(val_aucs.values())))

    logits_test = predict_logits(model, X_pa_test, device=device)
    species_auc = per_species_auc_sparse(
        logits=logits_test,
        y_lists=y_pa_test_lists,
        num_classes=num_classes,
    )
    avg_species_auc = float(np.nanmean(list(species_auc.values())))

    sites_auc = per_site_auc_sparse(
        logits=logits_test,
        y_lists=y_pa_test_lists,
        num_classes=num_classes,
    )
    avg_sites_auc = float(np.nanmean(list(sites_auc.values())))

    wandb.log(
        {
            "val/avg_auc": val_avg_auc,
            "test/avg_species_auc": avg_species_auc,
            "test/avg_sites_auc": avg_sites_auc,
        }
    )

    print(f"{name} val avg AUC : {val_avg_auc:.4f}")
    print(f"{name} test avg AUC: {avg_species_auc:.4f}")

    exp_dir = output_dir / "popa_integrated"
    exp_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), exp_dir / "model.pt")

    with open(exp_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "val_avg_auc": val_avg_auc,
                "avg_species_auc": avg_species_auc
            },
            f,
            indent=2,
        )

    run.finish()

    return {
        "source": name,
        "avg_species_auc": avg_species_auc,
        "avg_sites_auc": avg_sites_auc,
        "val_avg_auc": val_avg_auc,
        "model_path": str(exp_dir / "model.pt"),
        "scaler_path": str(exp_dir / "scaler.pkl"),
    }

def main(dataset_name: str, region: str):
    seed = 42
    set_all_seeds(seed)
    print(f"Running experiment for region: {region}")

    project_name = "sdm-integration"
    data_path = f"data/processed/{dataset_name}/{region}"
    output_dir = Path(f"outputs/baselines/{dataset_name}/{region}")
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 500
    epochs = 25
    lr = 1e-3
    weight_decay = 3e-4
    hidden_dim = 250
    hidden_layers = 2

    data = load_geoplant_processed(data_path)
    X_po, y_po = data.X_po, data.y_po
    X_pa_train, y_pa_train = data.X_pa_train, data.y_pa_train
    X_pa_test, y_pa_test = data.X_pa_test, data.y_pa_test
    covariates = data.covariates
    num_classes = len(data.species)

    shared = dict(
        covariates=covariates, num_classes=num_classes, output_dir=output_dir,
        seed=seed, batch_size=batch_size, epochs=epochs, lr=lr,
        weight_decay=weight_decay, hidden_dim=hidden_dim, hidden_layers=hidden_layers,
        project_name=project_name,
    )

    # --- single-source experiments ---
    single_source_configs = [
        dict(source_name="pa_balanced_bce", X_train_df=X_pa_train, y_train_lists=y_pa_train,
             X_test_df=X_pa_test, y_test_lists=y_pa_test,
             criterion=BalancedBCELoss(), criterion_name="balanced_bce"),

        dict(source_name="pa_bce", X_train_df=X_pa_train, y_train_lists=y_pa_train,
             X_test_df=X_pa_test, y_test_lists=y_pa_test,
             criterion=BCEWithLogitsLoss(), criterion_name="bce"),

        dict(source_name="pa_deepmaxent", X_train_df=X_pa_train, y_train_lists=y_pa_train,
             X_test_df=X_pa_test, y_test_lists=y_pa_test,
             criterion=DeepMaxEntLoss(), criterion_name="deepmaxent"),

        dict(source_name="po_deepmaxent", X_train_df=X_po, y_train_lists=y_po,
             X_test_df=X_pa_test, y_test_lists=y_pa_test,
             criterion=DeepMaxEntLoss(), criterion_name="deepmaxent"),

        dict(source_name="po_bce", X_train_df=X_po, y_train_lists=y_po,
             X_test_df=X_pa_test, y_test_lists=y_pa_test,
             criterion=BCEWithLogitsLoss(), criterion_name="bce"),

        dict(source_name="po_balanced_bce", X_train_df=X_po, y_train_lists=y_po,
             X_test_df=X_pa_test, y_test_lists=y_pa_test,
             criterion=BalancedBCELoss(), criterion_name="balanced_bce"),
    ]

    # --- integrated experiments ---
    integrated_configs = [
        # dict(name="po_dme_pa_bipp",       criterion_1=DeepMaxEntLoss(),   criterion_2=BernoulliFromLogRateLoss()),
        # # dict(name="po_dme_pa_balanced_bipp",       criterion_1=DeepMaxEntLoss(),   criterion_2=BernoulliFromLogRateLoss(balance_pos=True)),
        # # dict(name="po_balanced_bce_pa_balanced_bce",     criterion_1=BalancedBCELoss(),  criterion_2=BalancedBCELoss()),
        # # dict(name="po_bce_pa_bce",              criterion_1=BCEWithLogitsLoss(),criterion_2=BCEWithLogitsLoss()),
        # # dict(name="po_deepmaxent_pa_deepmaxent",       criterion_1=DeepMaxEntLoss(),   criterion_2=DeepMaxEntLoss()),
        # dict(name="po_deepmaxent_pa_balanced_bce", criterion_1=DeepMaxEntLoss(),   criterion_2=BalancedBCELoss()),
        # dict(name="po_deepmaxent_pa_bce",          criterion_1=DeepMaxEntLoss(),   criterion_2=BCEWithLogitsLoss()),
    ]

    # integrated_configs = [
    #     dict(name="po_dme_pa_dme",       criterion_1=DeepMaxEntLoss(),   criterion_2=DeepMaxEntLoss()),
    #     # TODO: Check that this doesnt alter the covariate list for the other experiments
    #     dict(name="po_dme_pa_dme_wpocov",       criterion_1=DeepMaxEntLoss(),   criterion_2=DeepMaxEntLoss(), add_po_cov=True),
        

    #     # dict(name="po_dme_pa_bipp",       criterion_1=DeepMaxEntLoss(),   criterion_2=BernoulliFromLogRateLoss()),
    #     # dict(name="po_dme_pa_balanced_bipp",       criterion_1=DeepMaxEntLoss(),   criterion_2=BernoulliFromLogRateLoss(balance_pos=True)),
    #     # dict(name="po_balanced_bce_pa_balanced_bce",     criterion_1=BalancedBCELoss(),  criterion_2=BalancedBCELoss()),
    #     # dict(name="po_bce_pa_bce",              criterion_1=BCEWithLogitsLoss(),criterion_2=BCEWithLogitsLoss()),
    #     # dict(name="po_deepmaxent_pa_deepmaxent",       criterion_1=DeepMaxEntLoss(),   criterion_2=DeepMaxEntLoss()),
    #     # dict(name="po_deepmaxent_pa_balanced_bce", criterion_1=DeepMaxEntLoss(),   criterion_2=BalancedBCELoss()),
    #     # dict(name="po_deepmaxent_pa_bce",          criterion_1=DeepMaxEntLoss(),   criterion_2=BCEWithLogitsLoss()),
    # ]

    results = []

    for cfg in single_source_configs:
        results.append(run_single_source_experiment(**shared, **cfg))

    for cfg in integrated_configs:
        results.append(run_integrated_experiment(
            X_po_df=X_po, y_po_lists=y_po,
            X_pa_train_df=X_pa_train, y_pa_train_lists=y_pa_train,
            X_pa_test_df=X_pa_test, y_pa_test_lists=y_pa_test,
            w_po=1.0, w_pa=1.0,
            **shared, **cfg,
        ))

    summary = pd.DataFrame(results)
    summary.to_csv(output_dir / "summary_single_source.csv", index=False)
    print("\n=== Summary ===")
    print(summary)


if __name__ == "__main__":

    dataset_name = "GeoPlant"
    region = "full"

    main(dataset_name, region)