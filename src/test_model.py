# from __future__ import annotations 

import os
import random
from typing import List, Tuple, Dict
from time import time

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# --- your models & loss
from src.models import deepmaxent_model, deepmaxent_loss, deepmaxent_model_w_bias


# =========================
# Config
# =========================
REGION = "AWT"                  # "AWT", "CAN", "NSW", "SWI", "NZ"
ADD_PO_VAR = True               # whether to add the presence-only indicator feature
ADD_PA_DATA = True              # whether to add half of the PA data to training¿
GROUP = "_plant"       
BIAS_MODEL = False            # whether to use the model with per-plot bias

# COVARIATES: List[str] = [
#     "age","deficit","dem","hillshade","mas","mat","r2pet","rain","slope","sseas","toxicats","tseas","vpd"
# ]

# Model / training
HIDDEN_SIZE = 250
HIDDEN_LAYERS = 2
LR = 1e-4
EPOCHS = 500              
BATCH_SIZE = 250
MAX_BATCH_PERCENTAGE = 1  # max batch size as percentage of training data               
PRINT_EVERY = 1000

# Reproducibility
SEED = 42

# SWI bias: 0.8469
# SWI no bias:

# =========================
# Utils
# =========================
def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_paths(region: str, group_filter: str) -> Tuple[str, str, str]:
    po_path = os.path.join("data", "processed", "Records", "train_po", f"{region}train_po{group_filter}.csv")
    pa_path = os.path.join("data", "raw", "Records", "test_pa", f"{region}test_pa{group_filter}.csv")
    env_path = os.path.join("data", "raw", "Records", "test_env", f"{region}test_env{group_filter}.csv")
    return po_path, pa_path, env_path


class XYDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, plot_ids: np.ndarray | None = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.plot_ids = None if plot_ids is None else torch.tensor(plot_ids, dtype=torch.long)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        if self.plot_ids is None:
            return self.X[idx], self.Y[idx]
        else:
            return self.X[idx], self.Y[idx], self.plot_ids[idx]



def safe_reindex_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Reindex df to the given columns, creating any missing columns (filled with 0)."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        df = df.copy()
        for c in missing:
            df[c] = 0
    return df[cols]


# =========================
# Data loading & processing
# =========================
def load_data(region: str, group_filter: str, add_po_var: bool):
              #, covariates: List[str]):
    po_path, pa_path, env_path = build_paths(region, group_filter)

    # Presence-only (PO) records
    df_po = pd.read_csv(po_path)
    # df_po = df_po[df_po["group"] == group_filter].copy()

    # Presence-absence (PA) labels + environmental covariates for test
    df_pa = pd.read_csv(pa_path)
    df_env = pd.read_csv(env_path)
    # covariates are from 5th column onward
    covariates = df_env.columns[4:].tolist()
    # print(f'Data loaded for {region}: PO Data {df_po.shape}, PA Data {df_pa.shape}, ENV Data {df_env.shape}')

    # Species list from PO data
    unique_species = df_po["spid"].unique().tolist()
    unique_species.sort()  # ensure stable column ordering

    # --- Build Y from PO: pivot to multi-label per (x, y)
    Y_po = (
        df_po
        .pivot_table(index=["x", "y"], columns="spid", aggfunc="size", fill_value=0)
        .reset_index()
    )
    # Make sure columns are in the same order as unique_species
    Y_po = safe_reindex_columns(Y_po, ["x", "y"] + unique_species)

    # --- X from PO: mean covariates at (x, y)
    X_po = (
        df_po
        .groupby(["x", "y"])[covariates]
        .mean()
        .reset_index()
    )
    if add_po_var:
        X_po["PO"] = 1

    # --- Build test (PA labels + ENV covariates)
    # Ensure test Y has the full species set (create missing columns if any)
    Y_test = safe_reindex_columns(df_pa.copy(), unique_species)
    X_test = df_env[covariates].copy()
    if add_po_var:
        X_test["PO"] = 0

    # Align X_po and Y_po by (x, y)
    
    XY_po = pd.merge(X_po, Y_po, on=["x", "y"], how="inner")


    # Split columns
    X = XY_po[["x", "y"] + covariates + (["PO"] if add_po_var and "PO" not in covariates else [])].copy()
    # Ensure covariate list includes PO if requested
    covs = covariates.copy()
    if add_po_var and "PO" not in covs:
        covs.append("PO")

    Y = XY_po[unique_species].copy()

    # As in your code: split test in two and add half to training

    if ADD_PA_DATA:
        # at least 2 rows extra for adding
        perc = 1-min(0.5, 1 - 2 / len(X_test))
        print('percentage for PA split:', perc)
        X_test_half, X_extra, Y_test_half, Y_extra = train_test_split(
            X_test, Y_test, test_size=perc, random_state=SEED
        )

        X_full = pd.concat([X[covs], X_extra], axis=0, ignore_index=True)
        Y_full = pd.concat([Y, Y_extra], axis=0, ignore_index=True)

    else:
        X_full = X[covs].copy()
        Y_full = Y.copy()
        X_test_half = X_test.copy()
        Y_test_half = Y_test.copy()

    # print final shapes (after aggregating), refering train (PO), test (PA)
    print(f"Final shapes for {region} {group_filter}:\n Train X {X_full.shape}, Train Y {Y_full.shape}, Test X {X_test_half.shape}, Test Y {Y_test_half.shape}")

    return X_full, Y_full, X_test_half, Y_test_half, unique_species, covs


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    covariates: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler().fit(X_train[covariates])
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[covariates] = scaler.transform(X_train[covariates])
    X_test_scaled[covariates] = scaler.transform(X_test[covariates])
    return X_train_scaled, X_test_scaled, scaler


# =========================
# Training / Evaluation
# =========================
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    epochs: int = 1000,
    lr: float = 1e-4,
    print_every: int = 100,
    dev: torch.device | None = None,
    verbose: bool = False
):
    dev = dev or device()
    model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-4)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for xb, yb, idx in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            idx = idx.to(dev)

            optimizer.zero_grad()
            if BIAS_MODEL:
                outputs = model(xb, idx)
            else:
                outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            avg_loss = running_loss / len(train_loader.dataset)
            if verbose: print(f"Epoch {epoch:5d}/{epochs} | Train Loss: {avg_loss:.4f}")


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    criterion: nn.Module,
    dev: torch.device | None = None,
) -> float:
    dev = dev or device()
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=dev)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=dev)
    if BIAS_MODEL:
        outputs = model(X_t, None)  # no bias if no plot indices
    else:
        outputs = model(X_t)
    loss = criterion(outputs, Y_t).item()
    return loss


@torch.no_grad()
def predict(
    model: nn.Module,
    X: np.ndarray,
    dev: torch.device | None = None,
) -> np.ndarray:
    dev = dev or device()
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=dev)
    if BIAS_MODEL:
        outputs = model(X_t, None)  # no bias if no plot indices
    else:
        outputs = model(X_t)
    return outputs.detach().cpu().numpy()


def per_species_auc(
    y_true: pd.DataFrame,
    y_score: np.ndarray,
    species: List[str]
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for i, sp in enumerate(species):
        # auc = roc_auc_score(y_true[sp].values, y_score[:, i])
        try:
            # Handle edge cases where only one class is present in y_true
            auc = roc_auc_score(y_true[sp].values, y_score[:, i])
        except ValueError:
            # print(np.unique(y_true[sp]))
            auc = np.nan  # not defined if only one class
        # print(f"Species: {sp}, AUC: {auc:.4f}" if not np.isnan(auc) else f"Species: {sp}, AUC: N/A")
        scores[sp] = auc
    return scores


# =========================
# Main
# =========================
def main():

    start_time = time()

    set_all_seeds(SEED)
    dev = device()
    print(f"Using device: {dev}")

    regions = ["AWT", "CAN", "NSW", "SA", "SWI", "NZ"]

    #regions = ['SWI']

    group_regions = {
        "AWT": ["_plant", "_bird"],
        "CAN": [""],
        "NSW": ["_ba", "_db", "_nb", "_ot", "_rt", "_ru", "_sr"],
        "SA" : [""],
        "SWI": [""],
        "NZ": [""]
    }  

    total_aucs = []

    for region in regions:

        avg_auc_region = []

        for group in group_regions[region]:

            print(f"\n=== REGION: {region}, GROUP: {group} ===")

            # Load & prepare data
            X_train, Y_train, X_test, Y_test, species, covs = load_data(
                region=region,
                group_filter=group,
                add_po_var=ADD_PO_VAR
            )

            # print head of each
            # print("\nTraining data sample (X):")
            # print(X_train.head())
            # print("\nTraining data sample (Y):")
            # print(Y_train.head())
            # print("\nTest data sample (X):")
            # print(X_test.head())
            # print("\nTest data sample (Y):")
            # print(Y_test.head())
            # exit()

            # Scale
            X_train, X_test, scaler = scale_features(X_train, X_test, covs)

            # arrays
            X_train_np = X_train[covs].values.astype(np.float32)
            Y_train_np = Y_train[species].values.astype(np.float32)
            X_test_np  = X_test[covs].values.astype(np.float32)
            Y_test_df  = Y_test[species].copy()

            # DataLoader
            batch_size_consolidated = min(BATCH_SIZE, int(len(X_train_np) * MAX_BATCH_PERCENTAGE))
            print(f"Using batch size: {batch_size_consolidated} (of {len(X_train_np)} training samples)")
            plots_idx = np.arange(len(X_train_np))  # dummy plot indices for bias model
            train_ds = XYDataset(X_train_np, Y_train_np, plots_idx)
            train_loader = DataLoader(train_ds, batch_size=batch_size_consolidated, shuffle=True, drop_last=False)

            # Model / loss
            if BIAS_MODEL:
                num_plots = len(X_train)  # assuming each row is a unique plot
                model = deepmaxent_model_w_bias(
                    input_size=len(covs),
                    hidden_size=HIDDEN_SIZE,
                    output_size=len(species),
                    hidden_nbr=HIDDEN_LAYERS,
                    num_plots=num_plots
                )
            else:
                model = deepmaxent_model(
                    input_size=len(covs),
                    hidden_size=HIDDEN_SIZE,
                    output_size=len(species),
                    hidden_nbr=HIDDEN_LAYERS
                )
            criterion = deepmaxent_loss()

            # Train
            train_model(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                epochs=EPOCHS,
                lr=LR,
                print_every=PRINT_EVERY,
                dev=dev,
                verbose=False
            )

            # Evaluate (loss)
            test_loss = evaluate_loss(
                model=model, X=X_test_np, Y=Y_test_df.values.astype(np.float32),
                criterion=criterion, dev=dev
            )
            # print(f"\nTest Loss: {test_loss:.4f}")

            # Evaluate (AUC per species)
            test_scores = predict(model, X_test_np, dev=dev)
            aucs = per_species_auc(Y_test_df, test_scores, species)

            avg_auc = np.nanmean(list(aucs.values()))  # ignore NaNs from single-class folds
            print(f"Average AUC (ignoring NaNs): {avg_auc:.4f}")
            avg_auc_region.append(avg_auc)

            # print number of parameters of the model
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # print(f"Total trainable parameters in model: {total_params}")

        
        region_avg_auc = np.nanmean(avg_auc_region)
        total_aucs.append(region_avg_auc)
        # print(f"\n=== **##AVERAGE AUC FOR REGION {region} ACROSS GROUPS: {region_avg_auc:.4f}##** ===")



            # Optional: save artifacts
            # torch.save(model.state_dict(), f"deepmaxent_{REGION}.pt")
            # import joblib; joblib.dump(scaler, f"scaler_{REGION}.joblib")

    overall_avg_auc = np.nanmean(total_aucs)
    print(f"\n=== OVERALL AVERAGE AUC ACROSS REGIONS & GROUPS: {overall_avg_auc:.4f} ===")

    end_time = time()
    elapsed_time = end_time - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
