# from __future__ import annotations

import os
import random
from typing import List, Tuple, Dict
from time import time

import numpy as np
import pandas as pd
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

# --- your models & loss
from src.models import deepmaxent_model, deepmaxent_loss, deepmaxent_model_w_bias, deepmaxent_domain, grad_reverse, DomainDiscriminator, DeepMaxentTwoHead

# =========================
# Config
# =========================
REGIONS = ["AWT"]             # regions to run
GROUPS_BY_REGION = {
    "AWT": ["_bird", "_plant"],              # groups to run per region
}
ADD_PO_VAR = True            # if True, add PO indicator covariate (it's like a source indicator)
BIAS_MODEL = False            # set True if you want per-plot bias
TEST_PA_FRACTION = 0.5        # PA split: test fraction
SEED = 42

# Model / training
HIDDEN_SIZE = 250
HIDDEN_LAYERS = 2
LR = 1e-4
EPOCHS = 500
BATCH_SIZE = 250
MAX_BATCH_PERCENTAGE = 1
PRINT_EVERY = 1000

RUN_PO = False
RUN_PA = False
RUN_POPA = True
RUN_DA = False   # domain adversarial training
RUN_TRANSFER = False
RUN_TH = True

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
            # keep triple to match training loop
            return self.X[idx], self.Y[idx], torch.tensor(idx, dtype=torch.long)
        else:
            return self.X[idx], self.Y[idx], self.plot_ids[idx]

def safe_reindex_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        df = df.copy()
        for c in missing:
            df[c] = 0
    return df[cols]

# =========================
# Data loading & processing
# =========================
def load_po_pa(region: str, group_filter: str, add_po_var: bool):
    """
    Load PO (aggregated by (x,y)) and PA (full table with covariates & labels),
    return aligned X/Y for PO and PA, with unified species columns and covariates.
    """
    po_path, pa_path, env_path = build_paths(region, group_filter)

    # Presence-only (PO) records
    df_po = pd.read_csv(po_path)

    # PA labels + environmental covariates
    df_pa = pd.read_csv(pa_path)
    df_env = pd.read_csv(env_path)

    # covariates from 5th col onward in env
    covariates = df_env.columns[4:].tolist()

    # Species 
    species = sorted(df_po["spid"].unique().tolist()) 

    # --- Build Y_po: pivot to multi-label per (x, y)
    Y_po = (
        df_po.pivot_table(index=["x", "y"], columns="spid", aggfunc="size", fill_value=0)
            .reset_index()
    )
    Y_po = safe_reindex_columns(Y_po, ["x", "y"] + species)

    # --- X_po: mean covariates at (x, y)
    X_po = (
        df_po.groupby(["x", "y"])[covariates]
            .mean()
            .reset_index()
    )
    if add_po_var:
        X_po["PO"] = 1

    # --- PA: labels and covariates
    Y_pa = safe_reindex_columns(df_pa.copy(), species)
    X_pa = df_env[covariates].copy()
    if add_po_var:
        X_pa["PO"] = 0

    # Align PO by (x,y) join
    XY_po = pd.merge(X_po, Y_po, on=["x", "y"], how="inner")

    # final lists
    covs = covariates.copy()
    if add_po_var and "PO" not in covs:
        covs.append("PO")

    # Split X and Y for PO
    X_po_final = XY_po[["x", "y"] + covs].copy()
    Y_po_final = XY_po[species].copy()

    # For PA we already have X_pa, Y_pa aligned to species
    return X_po_final, Y_po_final, X_pa, Y_pa, species, covs

def split_pa_train_test(X_pa: pd.DataFrame, Y_pa: pd.DataFrame, test_frac: float, seed: int):
    """
    Split PA into train/test on rows (locations).
    We avoid stratification (multi-label strat is non-trivial and dataset-dependent).
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_pa, Y_pa, test_size=test_frac, random_state=seed
    )
    return X_train, X_test, Y_train, Y_test

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    covariates: List[str],
    output_path: str | None = None,
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler().fit(X_train[covariates])
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[covariates] = scaler.transform(X_train[covariates])
    X_test_scaled[covariates] = scaler.transform(X_test[covariates])
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(scaler, f)
    if verbose:
        print("\nFeature scaling summary:")
        for col in covariates:
            print(f"{col}: Train mean={X_train_scaled[col].mean():.4f}, Train std={X_train_scaled[col].std():.4f}, "
                  f"Test mean={X_test_scaled[col].mean():.4f}, Test std={X_test_scaled[col].std():.4f}")
    return X_train_scaled, X_test_scaled, scaler

# =========================
# Train / Eval helpers
# =========================
def build_model(input_size: int, output_size: int, bias: bool, num_plots: int | None = None) -> nn.Module:
    if bias:
        if num_plots is None:
            raise ValueError("num_plots required for bias model.")
        return deepmaxent_model_w_bias(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            output_size=output_size,
            hidden_nbr=HIDDEN_LAYERS,
            num_plots=num_plots
        )
    else:
        return deepmaxent_model(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            output_size=output_size,
            hidden_nbr=HIDDEN_LAYERS
        )

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    epochs: int = 1000,
    lr: float = 1e-4,
    print_every: int = 1000,
    dev: torch.device | None = None,
    verbose: bool = False,
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
            outputs = model(xb, idx) if BIAS_MODEL else model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == epochs):
            avg_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch:5d}/{epochs} | Train Loss: {avg_loss:.4f}")






@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, dev: torch.device | None = None) -> np.ndarray:
    dev = dev or device()
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=dev)
    outputs = model(X_t, None) if BIAS_MODEL else model(X_t)
    return outputs.detach().cpu().numpy()

def per_species_auc(y_true: pd.DataFrame, y_score: np.ndarray, species: List[str]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for i, sp in enumerate(species):
        try:
            auc = roc_auc_score(y_true[sp].values, y_score[:, i])
        except ValueError:
            auc = np.nan
        scores[sp] = auc
    return scores

# --- NEW: make balanced loaders that only feed covs
def make_loader(X_df: pd.DataFrame, Y_df: pd.DataFrame, covs: List[str], species: List[str],
                batch_size: int = 256, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
    X = X_df[covs].values.astype(np.float32)
    Y = Y_df[species].values.astype(np.float32)
    idx = np.arange(len(X))
    ds = XYDataset(X, Y, idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

# --- NEW: fit scaler on PA_train (target domain), transform PO & PA & PA_test
def fit_target_scaler_and_transform(
    X_po: pd.DataFrame, X_pa_tr: pd.DataFrame, X_pa_te: pd.DataFrame, covs: List[str],
    scaler_path: str | None = None, verbose: bool = False,
    scale_w_po: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    if scale_w_po:
        X_po_s, X_pa_tr_s, scaler = scale_features(X_po, X_pa_tr, covs, output_path=scaler_path, verbose=verbose)
        X_pa_te_s = X_pa_te.copy()
        X_pa_te_s[covs] = scaler.transform(X_pa_te_s[covs])
    if not scale_w_po:
        X_pa_tr_s, X_pa_te_s, scaler = scale_features(X_pa_tr, X_pa_te, covs, output_path=scaler_path, verbose=verbose)
        X_po_s = X_po.copy()
        X_po_s[covs] = scaler.transform(X_po_s[covs])
    return X_po_s, X_pa_tr_s, X_pa_te_s, scaler

# --- NEW: standard DANN lambda schedule (0 -> 1)
def dann_lambda(epoch: int, total_epochs: int) -> float:
    # linear
    return epoch / total_epochs
    # p = epoch / max(1, total_epochs)
    # return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

# --- NEW: Domain Adversarial experiment runner
def run_experiment_da(
    name: str,
    X_po_df: pd.DataFrame, Y_po_df: pd.DataFrame,
    X_pa_tr_df: pd.DataFrame, Y_pa_tr_df: pd.DataFrame,
    X_pa_val_df: pd.DataFrame, Y_pa_val_df: pd.DataFrame,   # NEW
    X_pa_te_df: pd.DataFrame, Y_pa_te_df: pd.DataFrame,
    covs: List[str], species: List[str],
    output_dir: str, region: str, group: str,
    epochs: int = 300, lr: float = 1e-4, lambda_da_cap: float = 0.2,
    batch_size: int = 256, pretrain_epochs: int = 50, w_pa: float = 0.7, w_po: float = 0.3, w_dom: float = 0.2
):
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, f"scaler_{name}_{region}{group}.pkl")

    # Scale using PA_train; transform PO, PA_val, PA_test
    X_po_cov = X_po_df.drop(columns=["x","y"], errors="ignore")
    X_po_s, X_pa_tr_s, X_pa_te_s, scaler = fit_target_scaler_and_transform(
        X_po_cov, X_pa_tr_df, X_pa_te_df, covs, scaler_path=scaler_path, verbose=False, scale_w_po=True
    )
    X_pa_val_s = X_pa_val_df.copy()
    X_pa_val_s[covs] = scaler.transform(X_pa_val_s[covs])

    batch_size = min(batch_size, int(len(X_pa_tr_s) * MAX_BATCH_PERCENTAGE))

    po_loader = make_loader(X_po_s,    Y_po_df,    covs, species, batch_size=batch_size, shuffle=True, drop_last=True)
    pa_loader = make_loader(X_pa_tr_s, Y_pa_tr_df, covs, species, batch_size=batch_size, shuffle=True, drop_last=True)

    model = deepmaxent_domain(input_size=len(covs), hidden_size=HIDDEN_SIZE, output_size=len(species), hidden_nbr=HIDDEN_LAYERS)
    criterion_species = deepmaxent_loss()

    # train (with PA validation)
    X_val_np = X_pa_val_s[covs].values.astype(np.float32)
    train_domain_adversarial(
        model=model,
        po_loader=po_loader,
        pa_loader=pa_loader,
        criterion_species=criterion_species,
        feature_dim=HIDDEN_SIZE,
        lambda_da_cap=lambda_da_cap,
        epochs=epochs,
        lr=lr,
        pretrain_epochs=pretrain_epochs,
        w_pa=w_pa, w_po=w_po, w_dom=w_dom,
        pa_val=(X_val_np, Y_pa_val_df[species]),
        patience=20,
    )

    # evaluate on PA_test
    X_te_np = X_pa_te_s[covs].values.astype(np.float32)
    scores = predict(model, X_te_np, dev=device())
    aucs = per_species_auc(Y_pa_te_df[species], scores, species)
    avg_auc = np.nanmean(list(aucs.values()))

    model_path = os.path.join(output_dir, f"deepmaxent_DA_{region}{group}.pt")
    torch.save(model, model_path)
    return avg_auc, aucs, model_path, scaler_path

#### this is just a sketch to test the idea of domain adversarial training
def train_domain_adversarial(
    model: nn.Module,
    po_loader: DataLoader,
    pa_loader: DataLoader,
    criterion_species,
    feature_dim: int,
    lambda_da_cap: float = 0.3,    # cap for λ after warm-up
    epochs: int = 300,
    lr: float = 1e-4,
    pretrain_epochs: int = 50,     # ONLY species loss for first N epochs
    w_pa: float = 0.6,             # PA weight in species loss
    w_po: float = 0.4,             # PO weight in species loss
    w_dom: float = 0.1,            # weight for domain loss
    pa_val: tuple | None = None,   # (X_val_np, Y_val_df)
    patience: int = 20,            # early stopping on PA_val AUC
):
    dev = device()
    model = model.to(dev)
    domain_disc = DomainDiscriminator(feature_dim).to(dev)

    opt_main = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-4)
    opt_dom  = torch.optim.Adam(domain_disc.parameters(), lr=lr)

    # cyclical iterators to balance batches
    def cycle(loader):
        while True:
            for b in loader:
                yield b
    po_it, pa_it = cycle(po_loader), cycle(pa_loader)
    steps_per_epoch = min(len(po_loader), len(pa_loader))
    print(f"Training DANN for {epochs} epochs, {steps_per_epoch} steps/epoch.")
    print('len(po_loader):', len(po_loader)
          , 'len(pa_loader):', len(pa_loader)
            )

    best_auc = -1.0
    bad_epochs = 0
    X_val_np, Y_val_df = (pa_val if pa_val is not None else (None, None))

    for epoch in range(1, epochs + 1):
        model.train(); domain_disc.train()
        # λ warm-up (0 → cap). Keep λ=0 in pretrain phase.
        lam = 0.0 if epoch <= pretrain_epochs else min(lambda_da_cap, dann_lambda(epoch - pretrain_epochs, epochs - pretrain_epochs))
        ep_loss, dom_acc_accum = 0.0, 0.0

        for _ in range(steps_per_epoch):
            (x_po, y_po, _) = next(po_it)
            (x_pa, y_pa, _) = next(pa_it)
            x_po, y_po = x_po.to(dev), y_po.to(dev)
            x_pa, y_pa = x_pa.to(dev), y_pa.to(dev)

            # shared features
            z_po = model.get_features(x_po)
            z_pa = model.get_features(x_pa)

            # species preds
            y_pred_po = torch.sigmoid(model.output_layer(z_po))
            y_pred_pa = torch.sigmoid(model.output_layer(z_pa))

            # PA-prioritized species loss
            loss_sp = w_po * criterion_species(y_pred_po, y_po) + w_pa * criterion_species(y_pred_pa, y_pa)

            # try with a different loss for PA (convert outputs to probs)
            # y_pred_pa = y_pred_pa.clamp(1e-6, 1-1e-6) 
            # loss_sp = w_po * criterion_species(y_pred_po, y_po) + w_pa * F.binary_cross_entropy(y_pred_pa, y_pa)

            # domain loss (only after pretrain)
            if lam > 0.0:
                z = torch.cat([z_po, z_pa], 0)
                d = torch.cat([torch.ones(len(z_po), 1, device=dev), torch.zeros(len(z_pa), 1, device=dev)], 0)
                z_rev = grad_reverse(z, lam)
                d_pred = domain_disc(z_rev)
                loss_dom = F.binary_cross_entropy(d_pred, d)
                dom_acc = ((d_pred > 0.5).float() == d).float().mean().item()
            else:
                loss_dom = torch.tensor(0.0, device=dev)
                dom_acc = float('nan')

            opt_main.zero_grad(); opt_dom.zero_grad()
            loss_total = loss_sp + w_dom*loss_dom  # GRL flips sign internally
            loss_total.backward()
            opt_main.step(); opt_dom.step()

            ep_loss += float(loss_total.item())
            if lam > 0.0 and not np.isnan(dom_acc):
                dom_acc_accum += dom_acc

        # --- monitor every 20 epochs
        if epoch % 20 == 0 or epoch == epochs:
            if lam > 0.0 and dom_acc_accum > 0:
                dom_msg = f"dom_acc={dom_acc_accum/steps_per_epoch:.2f}"
            else:
                dom_msg = "dom_acc=NA"
            print(f"Epoch {epoch}/{epochs} | Loss: {ep_loss:.2f} | λ={lam:.3f} | {dom_msg}")
            # domain and species loss
            print(f"  Species loss: {loss_sp.item():.4f} | Domain loss: {loss_dom.item():.4f}")

        # # --- early stopping on PA validation AUC (when provided)
        # if X_val_np is not None:
        #     model.eval()
        #     with torch.no_grad():
        #         y_val_pred = predict(model, X_val_np, dev=dev)
        #     aucs = per_species_auc(Y_val_df, y_val_pred, list(Y_val_df.columns))
        #     avg_auc = float(np.nanmean(list(aucs.values())))
        #     if avg_auc > best_auc + 1e-1:
        #         best_auc = avg_auc
        #         bad_epochs = 0
        #     else:
        #         bad_epochs += 1
        #         if bad_epochs >= patience:
        #             print(f"Early stopping at epoch {epoch} | best PA_val AUC={best_auc:.4f}")
        #             break



# =========================
# Experiment runner
# =========================
def run_experiment(
    name: str,
    X_train_df: pd.DataFrame, Y_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,  Y_test_df: pd.DataFrame,
    covs: List[str], species: List[str],
    output_dir: str, region: str, group: str,
    verbose: bool = False
):
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, f"scaler_{name}_{region}{group}.pkl")

    # scale (fit on training split of this experiment)
    X_train_scaled, X_test_scaled, _ = scale_features(
        X_train_df, X_test_df, covs, output_path=scaler_path, verbose=False
    )

    # numpy arrays
    X_tr = X_train_scaled[covs].values.astype(np.float32)
    Y_tr = Y_train_df[species].values.astype(np.float32)
    X_te = X_test_scaled[covs].values.astype(np.float32)
    Y_te = Y_test_df[species].copy()

    # dataloader
    batch_size_consolidated = max(1, min(BATCH_SIZE, int(len(X_tr) * MAX_BATCH_PERCENTAGE)))
    plots_idx = np.arange(len(X_tr))  # dummy plot indices for bias model
    train_ds = XYDataset(X_tr, Y_tr, plots_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size_consolidated, shuffle=True, drop_last=False)

    # model
    model = build_model(input_size=len(covs), output_size=len(species), bias=BIAS_MODEL, num_plots=len(X_tr))
    criterion = deepmaxent_loss()

    # train
    train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        epochs=EPOCHS,
        lr=LR,
        print_every=PRINT_EVERY,
        dev=device(),
        verbose=verbose
    )

    # evaluate on shared PA_test
    scores = predict(model, X_te, dev=device())
    aucs = per_species_auc(Y_te, scores, species)
    avg_auc = np.nanmean(list(aucs.values()))

    # save model
    model_path = os.path.join(output_dir, f"deepmaxent_{name}_{region}{group}.pt")
    torch.save(model, model_path)

    return avg_auc, aucs, model_path, scaler_path



def domain_probe(X_po: pd.DataFrame, X_pa: pd.DataFrame, covs: list[str]):
    """
    Simple diagnostic to test covariate shift between PO and PA.
    Returns domain-AUC (1 = perfectly separable, 0.5 = identical distributions).
    """
    # 1 = PA, 0 = PO
    X = pd.concat([X_po[covs], X_pa[covs]], axis=0)
    y = np.concatenate([
        np.zeros(len(X_po)),  # PO
        np.ones(len(X_pa))    # PA
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Simple logistic probe
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print(f"\n[Domain probe] PO vs PA separability AUC = {auc:.3f}")
    print("≈0.5 → distributions similar | >0.7 → strong covariate shift\n")
    return auc


def fit_target_scaler_and_transform_3(
    X_po: pd.DataFrame, X_pa_tr: pd.DataFrame, X_pa_val: pd.DataFrame, X_pa_te: pd.DataFrame,
    covs: List[str], scaler_path: str | None = None, verbose: bool = False
):
    """
    Fit scaler on PA_train (target domain), transform PO, PA_val, PA_test.
    Returns (X_po_s, X_pa_tr_s, X_pa_val_s, X_pa_te_s, scaler)
    """
    X_pa_tr_s, X_pa_te_s, scaler = scale_features(X_pa_tr, X_pa_te, covs, output_path=scaler_path, verbose=verbose)
    X_pa_val_s = X_pa_val.copy()
    X_pa_val_s[covs] = scaler.transform(X_pa_val_s[covs])

    X_po_s = X_po.copy()
    X_po_s[covs] = scaler.transform(X_po_s[covs])
    return X_po_s, X_pa_tr_s, X_pa_val_s, X_pa_te_s, scaler


def reinit_linear(m: nn.Linear):
    """Reinitialize a linear layer (Kaiming uniform + bias uniform)."""
    nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
    if m.bias is not None:
        fan_in = m.weight.size(1)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(m.bias, -bound, bound)


def make_loader_simple(X_df: pd.DataFrame, Y_df: pd.DataFrame, covs: List[str], species: List[str],
                       batch_size: int = 256, shuffle: bool = True, drop_last: bool = False) -> DataLoader:
    X = X_df[covs].values.astype(np.float32)
    Y = Y_df[species].values.astype(np.float32)
    idx = np.arange(len(X))
    return DataLoader(XYDataset(X, Y, idx), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

def run_experiment_transfer(
    name: str,
    X_po_df: pd.DataFrame, Y_po_df: pd.DataFrame,
    X_pa_tr_df: pd.DataFrame, Y_pa_tr_df: pd.DataFrame,
    X_pa_val_df: pd.DataFrame, Y_pa_val_df: pd.DataFrame,
    X_pa_te_df: pd.DataFrame, Y_pa_te_df: pd.DataFrame,
    covs: List[str], species: List[str],
    output_dir: str, region: str, group: str,
    pretrain_epochs: int = 150, finetune_epochs: int = 200,
    lr_pretrain: float = 1e-4, lr_finetune: float = 5e-5,
    batch_size: int = 256, patience: int = 20,
    reinit_head: bool = True, unfreeze_top_after: int | None = None, lr_all: float = 1e-5
):
    """
    Pretrain on PO with DeepMaxEnt loss; reset head; freeze backbone; fine-tune on PA with BCEWithLogits.
    Optionally unfreeze entire net for a short all-layer fine-tune at the end (very small LR).
    """
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, f"scaler_{name}_{region}{group}.pkl")

    # --- scale using PA_train scaler; map PO into target space
    X_po_cov = X_po_df.drop(columns=["x","y"], errors="ignore")
    X_po_s, X_pa_tr_s, X_pa_val_s, X_pa_te_s, scaler = fit_target_scaler_and_transform_3(
        X_po_cov, X_pa_tr_df, X_pa_val_df, X_pa_te_df, covs, scaler_path=scaler_path, verbose=False
    )

    # --- loaders
    po_loader    = make_loader_simple(X_po_s,    Y_po_df,    covs, species, batch_size=batch_size, shuffle=True)
    pa_tr_loader = make_loader_simple(X_pa_tr_s, Y_pa_tr_df, covs, species, batch_size=batch_size, shuffle=True)
    # numpy for eval
    X_val_np = X_pa_val_s[covs].values.astype(np.float32)
    X_te_np  = X_pa_te_s[covs].values.astype(np.float32)

    # --- model
    # Use non-bias backbone for transfer simplicity (feature_extractor + output_layer expected).
    model = deepmaxent_model(
        input_size=len(covs), hidden_size=HIDDEN_SIZE,
        output_size=len(species), hidden_nbr=HIDDEN_LAYERS
    ).to(device())
    dev = device()
    # losses
    loss_po = deepmaxent_loss()           # your PO (DeepMaxEnt) loss possibly expects probs
    loss_pa = nn.BCEWithLogitsLoss()      # standard for PA fine-tune (logits)

    # --- PRETRAIN on PO
    opt_pre = torch.optim.Adam(model.parameters(), lr=lr_pretrain, weight_decay=3e-4)
    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb, _ in po_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt_pre.zero_grad()
            # If your deepmaxent_loss expects probabilities, pass sigmoid outputs.
            preds = model(xb)
            # NOTE: if deepmaxent_loss expects logits, remove sigmoid inside that function.
            # Here we assume deepmaxent_loss is okay with raw preds (your earlier code used sigmoid sometimes).
            # Safer route: apply sigmoid before the loss:
            preds_prob = torch.sigmoid(preds)
            l = loss_po(preds_prob, yb)
            l.backward()
            opt_pre.step()
            ep_loss += float(l.item())
        if epoch % 25 == 0:
            print(f"[Transfer: Pretrain PO] Epoch {epoch}/{pretrain_epochs} | loss: {ep_loss:.2f}")

    # --- Reset ONLY the head, freeze backbone
    assert hasattr(model, "feature_extractor") and hasattr(model, "output_layer"), \
        "deepmaxent_model must expose .feature_extractor and .output_layer for transfer."
    reinit_linear(model.output_layer)
    for p in model.feature_extractor.parameters():
        p.requires_grad = False

    # --- Fine-tune on PA (head only) with BCEWithLogits
    head_params = [p for p in model.parameters() if p.requires_grad]
    opt_ft = torch.optim.Adam(head_params, lr=lr_finetune, weight_decay=0.0)

    best_auc = -1.0
    bad = 0
    for epoch in range(1, finetune_epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb, _ in pa_tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt_ft.zero_grad()
            logits = model(xb)             # raw logits for BCEWithLogits
            l = loss_pa(logits, yb)
            l.backward()
            opt_ft.step()
            ep_loss += float(l.item())

        # validate on PA_val every 10 epochs
        if epoch % 10 == 0 or epoch == finetune_epochs:
            model.eval()
            with torch.no_grad():
                val_logits = predict(model, X_val_np, dev=dev)  # predict returns raw model(x)
            # roc_auc_score accepts scores (logits are fine; monotone with probs)
            aucs = per_species_auc(Y_pa_val_df[species], val_logits, species)
            val_auc = float(np.nanmean(list(aucs.values())))
            print(f"[Transfer: Finetune PA] Epoch {epoch}/{finetune_epochs} | loss: {ep_loss:.2f} | PA_val AUC: {val_auc:.4f}")

            if val_auc > best_auc + 1e-4:
                best_auc = val_auc
                bad = 0
                torch.save(model, os.path.join(output_dir, f"deepmaxent_TRANSFER_best_{region}{group}.pt"))
            else:
                bad += 1
                if bad >= patience:
                    print(f"[Transfer: Finetune PA] Early stop at epoch {epoch} | best PA_val AUC: {best_auc:.4f}")
                    break

    # --- Optional: brief all-layer fine-tune (very small LR) to squeeze a bit more
    if unfreeze_top_after is not None:
        for p in model.parameters():
            p.requires_grad = True
        opt_all = torch.optim.Adam(model.parameters(), lr=lr_all, weight_decay=0.0)
        for t in range(unfreeze_top_after):
            model.train()
            for xb, yb, _ in pa_tr_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                opt_all.zero_grad()
                logits = model(xb)
                l = loss_pa(logits, yb)
                l.backward()
                opt_all.step()
        print(f"[Transfer] All-layer fine-tune steps: {unfreeze_top_after}")

    # --- Evaluate on PA_test
    model.eval()
    with torch.no_grad():
        test_logits = predict(model, X_te_np, dev=dev)
    aucs_test = per_species_auc(Y_pa_te_df[species], test_logits, species)
    avg_auc = float(np.nanmean(list(aucs_test.values())))

    final_path = os.path.join(output_dir, f"deepmaxent_TRANSFER_{region}{group}.pt")
    torch.save(model, final_path)
    return avg_auc, aucs_test, final_path, scaler_path


# --- NEW: Trainer (no adversary) ---------------------------------------------
def train_twohead_integration(
    model: nn.Module,
    po_loader: DataLoader,
    pa_loader: DataLoader,
    criterion_species,
    epochs: int = 300,
    lr: float = 1e-4,
    w_pa: float = 0.7,        # PA weight (target domain)
    w_po: float = 0.3,        # PO weight (auxiliary)
    w_align: float = 0.05,    # optional alignment loss between heads on PA batch (logits-level)
    pa_val: tuple | None = None,  # (X_val_np, Y_val_df)
    patience: int = 20,
):
    dev = device()
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-4)

    # cyclical iterators to balance batches
    def cycle(loader):
        while True:
            for b in loader:
                yield b
    po_it, pa_it = cycle(po_loader), cycle(pa_loader)
    steps_per_epoch = min(len(po_loader), len(pa_loader))
    print(f"Training two-head integration for {epochs} epochs, {steps_per_epoch} steps/epoch.")
    print('len(po_loader):', len(po_loader), 'len(pa_loader):', len(pa_loader))

    best_auc = -1.0
    bad_epochs = 0
    X_val_np, Y_val_df = (pa_val if pa_val is not None else (None, None))

    mse = nn.MSELoss(reduction="mean")

    criterion_pa = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0

        for _ in range(steps_per_epoch):
            (x_po, y_po, _) = next(po_it)
            (x_pa, y_pa, _) = next(pa_it)
            x_po, y_po = x_po.to(dev), y_po.to(dev)
            x_pa, y_pa = x_pa.to(dev), y_pa.to(dev)

            # forward
            z_po = model.get_features(x_po)
            z_pa = model.get_features(x_pa)

            # logits
            logit_po = model.po_head(z_po)
            logit_pa = model.pa_head(z_pa)

            # probabilities
            y_pred_po = torch.sigmoid(logit_po)
            y_pred_pa = torch.sigmoid(logit_pa)

            # species losses (note: your deepmaxent_loss should work with probabilities)
            loss_pa = criterion_species(y_pred_pa, y_pa)
            # loss_pa = criterion_pa(logit_pa, y_pa)
            # for PA we can do a cross entropy ()
            loss_po = criterion_species(y_pred_po, y_po)

            # in a specific case do a print to see what's going on
            # if np.random.rand() < 0.1:
            #     print(f"Initial losses | PA: {loss_pa.item():.4f} | PO: {loss_po.item():.4f}")
            #     print('Y PO true:', y_po[0:5, 0])
            #     print('Y PA true:', y_pa[0:5, 0])
            #     print('Y pred PO:', y_pred_po[0:5, 0])
            #     print('Y pred PA:', y_pred_pa[0:5, 0])
            #     exit()
        

            # optional head-alignment on PA batch to keep suitability similar across heads
            # (penalize difference between heads' logits on SAME x_pa)
            logit_po_on_pa = model.po_head(z_pa).detach() if w_align < 0 else model.po_head(z_pa)
            # default: use symmetric MSE without detach (lets both heads meet in the middle)
            align_loss = mse(logit_pa, model.po_head(z_pa))

            loss = w_pa * loss_pa + w_po * loss_po + w_align * align_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_loss += float(loss.item())

        if epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs} | Loss: {ep_loss:.2f} | "
                  f"PA: {loss_pa.item():.4f} | PO: {loss_po.item():.4f} | Align: {align_loss.item():.4f}")

        # Early stopping on PA validation AUC
        if X_val_np is not None:
            model.eval()
            with torch.no_grad():
                # IMPORTANT: predict() uses model.output_layer → PA head
                y_val_pred = predict(model, X_val_np, dev=dev)
            aucs = per_species_auc(Y_val_df, y_val_pred, list(Y_val_df.columns))
            avg_auc = float(np.nanmean(list(aucs.values())))
            if avg_auc > best_auc + 1e-4:
                best_auc = avg_auc
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch} | best PA_val AUC={best_auc:.4f}")
                    break

# --- NEW: Experiment runner (PO+PA integration; no adversary) ----------------
def run_experiment_twohead(
    name: str,
    X_po_df: pd.DataFrame, Y_po_df: pd.DataFrame,
    X_pa_tr_df: pd.DataFrame, Y_pa_tr_df: pd.DataFrame,
    X_pa_val_df: pd.DataFrame, Y_pa_val_df: pd.DataFrame,
    X_pa_te_df: pd.DataFrame,  Y_pa_te_df: pd.DataFrame,
    covs: List[str], species: List[str],
    output_dir: str, region: str, group: str,
    epochs: int = 300, lr: float = 1e-4,
    batch_size: int = 256, pretrain_epochs: int = 0,  # kept for signature symmetry; unused here
    w_pa: float = 0.7, w_po: float = 0.3, w_align: float = 0.05
):
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, f"scaler_{name}_{region}{group}.pkl")

    # Scale using PA_train; transform PO, PA_val, PA_test
    X_po_cov = X_po_df.drop(columns=["x","y"], errors="ignore")
    X_po_s, X_pa_tr_s, X_pa_te_s, scaler = fit_target_scaler_and_transform(
        X_po_cov, X_pa_tr_df, X_pa_te_df, covs, scaler_path=scaler_path, verbose=False, scale_w_po=True
    )
    X_pa_val_s = X_pa_val_df.copy()
    X_pa_val_s[covs] = scaler.transform(X_pa_val_s[covs])

    batch_size = min(batch_size, int(len(X_pa_tr_s) * MAX_BATCH_PERCENTAGE))

    po_loader = make_loader(X_po_s,    Y_po_df,    covs, species, batch_size=batch_size, shuffle=True, drop_last=True)
    pa_loader = make_loader(X_pa_tr_s, Y_pa_tr_df, covs, species, batch_size=batch_size, shuffle=True, drop_last=True)

    model = DeepMaxentTwoHead(
        input_size=len(covs), hidden_size=HIDDEN_SIZE, output_size=len(species), hidden_nbr=HIDDEN_LAYERS
    )
    criterion_species = deepmaxent_loss()

    # train (validate on PA)
    X_val_np = X_pa_val_s[covs].values.astype(np.float32)
    train_twohead_integration(
        model=model,
        po_loader=po_loader,
        pa_loader=pa_loader,
        criterion_species=criterion_species,
        epochs=epochs,
        lr=lr,
        w_pa=w_pa, w_po=w_po, w_align=w_align,
        pa_val=(X_val_np, Y_pa_val_df[species]),
        patience=20,
    )

    # evaluate on PA_test using PA head
    X_te_np = X_pa_te_s[covs].values.astype(np.float32)
    scores = predict(model, X_te_np, dev=device())  # uses model.output_layer → PA head
    aucs = per_species_auc(Y_pa_te_df[species], scores, species)
    avg_auc = np.nanmean(list(aucs.values()))

    model_path = os.path.join(output_dir, f"deepmaxent_twohead_{region}{group}.pt")
    torch.save(model, model_path)
    return avg_auc, aucs, model_path, scaler_path

# =========================
# Main
# =========================
def main():
    start_time = time()
    set_all_seeds(SEED)
    dev = device()
    print(f"Using device: {dev}")

    output_root = os.path.join("output", "integration_schemes")
    os.makedirs(output_root, exist_ok=True)
    summary_rows = []

    for region in REGIONS:
        for group in GROUPS_BY_REGION[region]:
            print(f"\n=== REGION: {region}, GROUP: {group or '(all)'} ===")

            # 1) Load PO & PA, aligned species & covs
            X_po, Y_po, X_pa, Y_pa, species, covs = load_po_pa(
                region=region,
                group_filter=group,
                add_po_var=ADD_PO_VAR
            )

            # 2) Split PA → train/test (fixed for all experiments)
            X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te = split_pa_train_test(
                X_pa, Y_pa, test_frac=TEST_PA_FRACTION, seed=SEED
            )
            print(f"PA split → train: {len(X_pa_tr)}, test: {len(X_pa_te)}")

            # 3) Three experiments, same PA_test
            exp_dir = os.path.join(output_root, f"{region}{group}")

            if RUN_PO:
                # A) PO-only → PA_test
                auc_po, aucs_po, model_po, scaler_po = run_experiment(
                    name="PO_only",
                    X_train_df=X_po.drop(columns=["x","y"], errors="ignore"),
                    Y_train_df=Y_po,
                    X_test_df=X_pa_te,  # evaluate on PA_test covs
                    Y_test_df=Y_pa_te,  # evaluate on PA_test labels
                    covs=covs, species=species,
                    output_dir=exp_dir, region=region, group=group
                )
                print(f"[PO-only]   Average AUC on PA_test: {auc_po:.4f}")

            if RUN_PA:
                # B) PA_train-only → PA_test
                auc_pa, aucs_pa, model_pa, scaler_pa = run_experiment(
                    name="PA_only",
                    X_train_df=X_pa_tr,
                    Y_train_df=Y_pa_tr,
                    X_test_df=X_pa_te,
                    Y_test_df=Y_pa_te,
                    covs=covs, species=species,
                    output_dir=exp_dir, region=region, group=group
                )
                print(f"[PA-only]   Average AUC on PA_test: {auc_pa:.4f}")

            if RUN_POPA:
                # C) PO + PA_train → PA_test
                X_mix = pd.concat([X_po.drop(columns=["x","y"], errors="ignore"), X_pa_tr], axis=0, ignore_index=True)
                Y_mix = pd.concat([Y_po, Y_pa_tr], axis=0, ignore_index=True)
                auc_mix, aucs_mix, model_mix, scaler_mix = run_experiment(
                    name="PO_plus_PA",
                    X_train_df=X_mix,
                    Y_train_df=Y_mix,
                    X_test_df=X_pa_te,
                    Y_test_df=Y_pa_te,
                    covs=covs, species=species,
                    output_dir=exp_dir, region=region, group=group
                )

                
                print(f"[PO+PA]     Average AUC on PA_test: {auc_mix:.4f}")

            # ### test if separable
            # X_po = X_po.drop(columns=["x","y"], errors="ignore")
            # scaler = StandardScaler().fit(X_pa_tr[covs])
            # X_po_s = X_po.copy()
            # X_po_s[covs] = scaler.transform(X_po_s[covs])
            # X_pa_tr_s = X_pa_tr.copy()
            # X_pa_tr_s[covs] = scaler.transform(X_pa_tr_s[covs])
            # domain_auc = domain_probe(X_po_s, X_pa_tr_s, covs)
            # exit()



            if RUN_TRANSFER:

                # After: X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te = split_pa_train_test(...)
                # Add a small PA validation split
                X_pa_tr, X_pa_val, Y_pa_tr, Y_pa_val = train_test_split(
                    X_pa_tr, Y_pa_tr, test_size=0.1, random_state=SEED
                )

                exp_dir = os.path.join(output_root, f"{region}{group}")
                # auc_tr, aucs_tr, model_tr, scaler_tr = run_experiment_transfer(
                #     name="TRANSFER",
                #     X_po_df=X_po, Y_po_df=Y_po,
                #     X_pa_tr_df=X_pa_tr, Y_pa_tr_df=Y_pa_tr,
                #     X_pa_val_df=X_pa_val, Y_pa_val_df=Y_pa_val,
                #     X_pa_te_df=X_pa_te, Y_pa_te_df=Y_pa_te,
                #     covs=covs, species=species,
                #     output_dir=exp_dir, region=region, group=group,
                #     pretrain_epochs=150, finetune_epochs=200,
                #     lr_pretrain=1e-4, lr_finetune=5e-5,
                #     batch_size=256, patience=20,
                #     reinit_head=True,                # reset only the classifier
                #     unfreeze_top_after=None,         # set e.g. 20 to try a brief all-layer fine-tune
                #     lr_all=1e-5
                # )
                auc_tr, aucs_tr, model_tr, scaler_tr = run_experiment_transfer(
                    name="TRANSFER",
                    X_po_df=X_po, Y_po_df=Y_po,
                    X_pa_tr_df=X_pa_tr, Y_pa_tr_df=Y_pa_tr,
                    X_pa_val_df=X_pa_val, Y_pa_val_df=Y_pa_val,
                    X_pa_te_df=X_pa_te, Y_pa_te_df=Y_pa_te,
                    covs=covs, species=species,
                    output_dir=exp_dir, region=region, group=group,
                    pretrain_epochs=0, finetune_epochs=200,
                    lr_pretrain=1e-4, lr_finetune=5e-5,
                    batch_size=256, patience=20,
                    reinit_head=True,                # reset only the classifier
                    unfreeze_top_after=None,         # set e.g. 20 to try a brief all-layer fine-tune
                    lr_all=1e-4
                )
                print(f"[TRANSFER] Average AUC on PA_test: {auc_tr:.4f}")


            if RUN_DA:
                # --- NEW: PA train/valid (for early stopping & λ warm-up control)
                # remove "PA" cov if present
                X_pa_tr = X_pa_tr.drop(columns=["PA"], errors="ignore")
                X_pa_te = X_pa_te.drop(columns=["PA"], errors="ignore")
                X_po = X_po.drop(columns=["PA"], errors="ignore")

                X_pa_tr, X_pa_val, Y_pa_tr, Y_pa_val = train_test_split(
                    X_pa_tr, Y_pa_tr, test_size=0.1, random_state=SEED
                )


                # auc_da, aucs_da, model_da, scaler_da = run_experiment_da(
                #     name="DA",
                #     X_po_df=X_po, Y_po_df=Y_po,
                #     X_pa_tr_df=X_pa_tr, Y_pa_tr_df=Y_pa_tr,
                #     X_pa_val_df=X_pa_val, Y_pa_val_df=Y_pa_val,   # NEW
                #     X_pa_te_df=X_pa_te, Y_pa_te_df=Y_pa_te,
                #     covs=covs, species=species,
                #     output_dir=exp_dir, region=region, group=group,
                #     epochs=300, lr=1e-3, lambda_da_cap=0.2,
                #     batch_size=256, pretrain_epochs=50, w_pa=0.7, w_po=0.3,
                #     w_dom = 0.3
                # )
                auc_da, aucs_da, model_da, scaler_da = run_experiment_da(
                    name="DA",
                    X_po_df=X_po, Y_po_df=Y_po,
                    X_pa_tr_df=X_pa_tr, Y_pa_tr_df=Y_pa_tr,
                    X_pa_val_df=X_pa_val, Y_pa_val_df=Y_pa_val,   # NEW
                    X_pa_te_df=X_pa_te, Y_pa_te_df=Y_pa_te,
                    covs=covs, species=species,
                    output_dir=exp_dir, region=region, group=group,
                    epochs=300, lr=1e-3, lambda_da_cap=0.5,
                    batch_size=256, pretrain_epochs=1, w_pa=0.65, w_po=0.35,
                    w_dom = 0.2
                )
                print(f"[DA]        Average AUC on PA_test: {auc_da:.4f}")



            ## TWO HEAD
            if RUN_TH:
                # separate
                X_pa_tr, X_pa_val, Y_pa_tr, Y_pa_val = train_test_split(X_pa_tr, Y_pa_tr, test_size=0.1, random_state=SEED)

                auc_twohead, aucs_twohead, model_twohead, scaler_twohead = run_experiment_twohead(
                    name="TWOHEAD",
                    X_po_df=X_po, Y_po_df=Y_po,
                    X_pa_tr_df=X_pa_tr, Y_pa_tr_df=Y_pa_tr,
                    X_pa_val_df=X_pa_val, Y_pa_val_df=Y_pa_val,
                    X_pa_te_df=X_pa_te, Y_pa_te_df=Y_pa_te,
                    covs=covs, species=species,
                    output_dir=exp_dir, region=region, group=group,
                    epochs=EPOCHS, lr=LR,
                    batch_size=256, pretrain_epochs=0,  # kept for signature symmetry; unused here
                    w_pa=0.75, w_po=0.25, w_align=0.05
                )
                print(f"[TWO-HEAD]  Average AUC on PA_test: {auc_twohead:.4f}")




            summary_rows.append({
                "region": region,
                "group": group or "(all)",
                "TEST_PA_FRACTION": TEST_PA_FRACTION,
                "AUC_PO_only": float(np.round(auc_po, 6)) if RUN_PO else np.nan,
                "AUC_PA_only": float(np.round(auc_pa, 6)) if RUN_PA else np.nan,
                "AUC_TRANSFER": float(np.round(auc_tr, 6)) if RUN_TRANSFER else np.nan,
                "AUC_DA": float(np.round(auc_da, 6)) if RUN_DA else np.nan,
                "AUC_PO_plus_PA": float(np.round(auc_mix, 6)) if RUN_POPA else np.nan,
                "AUC_TWOHEAD": float(np.round(auc_twohead, 6)) if RUN_TH else np.nan,
            })

    summary = pd.DataFrame(summary_rows)
    print("\n=== Summary (Average AUC on shared PA_test) ===")
    print(summary.to_string(index=False))

    elapsed = time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
