# Just keeps the relevant 3 methods which so far make more sense, discarding the rest (still in test_data_integration.py)

import os
import random
from typing import List, Tuple, Dict
from time import time

import numpy as np
import pandas as pd
import pickle

from load_data import build_paths_nceas
from utils import safe_reindex_columns, load_po_pa_nceas, split_pa_train_test_spatially, scale_features

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

# --- your models & loss
from models import deepmaxent_model, deepmaxent_loss, deepmaxent_model_w_bias, deepmaxent_domain, grad_reverse, DomainDiscriminator, DeepMaxentTwoHead

# =========================
# Config
# =========================
# REGIONS = ['SWI']
REGIONS = ["AWT", "CAN", "NSW", "SA", "SWI", "NZ"]             # regions to run
GROUPS_BY_REGION = {
        "AWT": ["_plant", "_bird"],
        "CAN": [""],
        "NSW": ["_ba", "_db", "_nb", "_ot", "_rt", "_ru", "_sr"],
        "SA" : [""],
        "SWI": [""],
        "NZ": [""]
    }  
# General Experiment settings
ADD_PO_VAR = True            # if True, add PO indicator covariate (it's like a Tsource indicator)
BIAS_MODEL = False            # set True if you want per-plot bias
KEEP_XY = True  # if True, keep x,y in covariates
TEST_PA_FRACTION = 0.3        # PA split: test fraction
# FILTER_PO = True

# Model / training
HIDDEN_SIZE = 250
HIDDEN_LAYERS = 2
LR = 1e-4
EPOCHS = 500
BATCH_SIZE = 250
MAX_BATCH_PERCENTAGE = 1
PRINT_EVERY = 1000
SEED = 42

RUN_PO = False
RUN_PA = False
RUN_POPA, ADD_INTERACTIONS = False, False
RUN_IMPUTED_POPA = False
RUN_POPA_ENSAMBLE = True
RUN_POPA_WEIGHTED = False


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
    criterion: str,
    epochs: int = 1000,
    lr: float = 1e-4,
    print_every: int = 1000,
    dev: torch.device | None = None,
    verbose: bool = False,
):
    dev = dev or device()
    model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-4)

    if criterion == 'bce':
        loss_f = torch.nn.BCEWithLogitsLoss()
    elif criterion == 'deepmaxent':
        loss_f = deepmaxent_loss()

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for xb, yb, idx in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            idx = idx.to(dev)

            optimizer.zero_grad()
            outputs = model(xb, idx) if BIAS_MODEL else model(xb)
            # if criterion == 'bce':
            #     outputs = torch.sigmoid(outputs)
            loss = loss_f(outputs, yb)
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



# =========================
# Experiment runner
# =========================
def run_experiment(
    name: str,
    X_train_df: pd.DataFrame, Y_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,  Y_test_df: pd.DataFrame,
    covs: List[str], species: List[str],
    output_dir: str, region: str, group: str,
    criterion = 'deepmaxent',
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

def run_experiment_popa(
    name: str,
    X_po_df: pd.DataFrame, Y_po_df: pd.DataFrame,
    X_pa_tr_df: pd.DataFrame, Y_pa_tr_df: pd.DataFrame,
    X_pa_te_df: pd.DataFrame, Y_pa_te_df: pd.DataFrame,
    covs: List[str], species: List[str],
    output_dir: str, region: str, group: str,
    epochs: int = 300, lr: float = 1e-4,
    batch_size: int = 256, w_po: float = .5, w_pa: float = .5,
):
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, f"scaler_{name}_{region}{group}.pkl")

    # scale (fit on combo PO + PA train)
    scaler = StandardScaler().fit(
        pd.concat([X_po_df[covs], X_pa_tr_df[covs]], axis=0)
    )
    X_po_s = X_po_df.copy()
    X_pa_tr_s = X_pa_tr_df.copy()
    X_pa_te_s = X_pa_te_df.copy()
    X_po_s[covs] = scaler.transform(X_po_s[covs])
    X_pa_tr_s[covs] = scaler.transform(X_pa_tr_s[covs])
    X_pa_te_s[covs] = scaler.transform(X_pa_te_s[covs])

    batch_size = min(batch_size, int(len(X_pa_tr_s) * MAX_BATCH_PERCENTAGE))

   

    model = deepmaxent_model(input_size=len(covs), hidden_size=HIDDEN_SIZE, output_size=len(species), hidden_nbr=HIDDEN_LAYERS)
    criterion_species = deepmaxent_loss()

    # loaders separately (proportional sizes)
    proportion_po = len(X_po_s) / (len(X_po_s) + len(X_pa_tr_s))
    proportion_pa = len(X_pa_tr_s) / (len(X_po_s) + len(X_pa_tr_s))
    batch_size_po = max(1, int(batch_size * w_po))
    batch_size_pa = max(1, int(batch_size * w_pa))
    print('Batch for PO: ', batch_size_po)
    print('Batch for PA: ', batch_size_pa)
    po_ds = XYDataset(X_po_s[covs].values.astype(np.float32), Y_po_df[species].values.astype(np.float32))
    pa_ds = XYDataset(X_pa_tr_s[covs].values.astype(np.float32), Y_pa_tr_df[species].values.astype(np.float32))
    po_loader = DataLoader(po_ds, batch_size=batch_size_po, shuffle=True, drop_last=True)
    pa_loader = DataLoader(pa_ds, batch_size=batch_size_pa, shuffle=True, drop_last=True)

    def train_popa_model(
        model: nn.Module,
        po_loader: DataLoader,
        pa_loader: DataLoader,
        criterion_species: nn.Module,
        epochs: int = 300,
        lr: float = 1e-4,
        dev: torch.device | None = None,
        w_po: float = .5,
        w_pa: float = .5,
    ):
        dev = dev or device()
        model.to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-4)

        criterion_po = deepmaxent_loss()
        criterion_pa = deepmaxent_loss()#torch.nn.BCEWithLogitsLoss()

        model.train()
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            for (xb_po, yb_po, _), (xb_pa, yb_pa, _) in zip(po_loader, pa_loader):
                xb_po = xb_po.to(dev)
                yb_po = yb_po.to(dev)
                xb_pa = xb_pa.to(dev)
                yb_pa = yb_pa.to(dev)

                optimizer.zero_grad()
                outputs_po= model(xb_po)
                outputs_pa= model(xb_pa)
                # concat outputs and labels
                output_mix = torch.cat([outputs_po, outputs_pa], dim=0)
                yb_mix = torch.cat([yb_po, yb_pa], dim=0)
                loss = criterion_species(output_mix, yb_mix)

                # loss_po = criterion_po(outputs_po, yb_po)
                # loss_pa = criterion_pa(outputs_pa, yb_pa)
                # loss = w_po * loss_po + w_pa * loss_pa

                loss.backward()
                optimizer.step()
                running_loss += loss.item() * (xb_po.size(0) + xb_pa.size(0))

        avg_loss = running_loss / (len(po_loader.dataset) + len(pa_loader.dataset))
    
    train_popa_model(
        model=model,
        po_loader=po_loader,
        pa_loader=pa_loader,
        criterion_species=criterion_species,
        epochs=epochs,
        lr=lr,
        dev=device(),
        w_po=w_po,
        w_pa=w_pa
    )

    # evaluate on PA_test
    X_te_np = X_pa_te_s[covs].values.astype(np.float32)
    scores = predict(model, X_te_np, dev=device())
    aucs = per_species_auc(Y_pa_te_df[species], scores, species)
    avg_auc = np.nanmean(list(aucs.values()))

    model_path = os.path.join(output_dir, f"deepmaxent_DA_{region}{group}.pt")
    torch.save(model, model_path)
    return avg_auc, aucs, model_path, scaler_path



def run_experiment_popa_ensamble(
    name: str,
    X_po_df: pd.DataFrame, Y_po_df: pd.DataFrame,
    X_pa_tr_df: pd.DataFrame, Y_pa_tr_df: pd.DataFrame,
    X_pa_te_df: pd.DataFrame, Y_pa_te_df: pd.DataFrame,
    covs: List[str], species: List[str],
    output_dir: str, region: str, group: str,
    epochs: int = 300, lr: float = 1e-4,
    batch_size: int = 256, w_po: float = .5, w_pa: float = .5,
):
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, f"scaler_{name}_{region}{group}.pkl")

    # ---- scale (fit on combo PO + PA train) ----
    scaler = StandardScaler().fit(
        pd.concat([X_po_df[covs], X_pa_tr_df[covs]], axis=0)
    )
    X_po_s = X_po_df.copy()
    X_pa_tr_s = X_pa_tr_df.copy()
    X_pa_te_s = X_pa_te_df.copy()
    X_po_s[covs] = scaler.transform(X_po_s[covs])
    X_pa_tr_s[covs] = scaler.transform(X_pa_tr_s[covs])
    X_pa_te_s[covs] = scaler.transform(X_pa_te_s[covs])

    # cap batch size by % of PA train size (as in your codebase)
    batch_size_po = min(batch_size, int(len(X_po_s) * MAX_BATCH_PERCENTAGE))
    batch_size_pa = min(batch_size, int(len(X_pa_tr_s) * MAX_BATCH_PERCENTAGE))

    # ---- data loaders (PO and PA task-specific) ----
    batch_size_po = max(1, int(batch_size * w_po))
    batch_size_pa = max(1, int(batch_size_pa * w_pa))
    print('Batch for PO: ', batch_size_po)
    print('Batch for PA: ', batch_size_pa)

    po_ds = XYDataset(
        X_po_s[covs].values.astype(np.float32),
        Y_po_df[species].values.astype(np.float32)
    )
    pa_ds = XYDataset(
        X_pa_tr_s[covs].values.astype(np.float32),
        Y_pa_tr_df[species].values.astype(np.float32)
    )
    po_loader = DataLoader(po_ds, batch_size=batch_size_po, shuffle=True, drop_last=True)
    pa_loader = DataLoader(pa_ds, batch_size=batch_size_pa, shuffle=True, drop_last=True)

    # ---- model for PA (multi-label BCE) ----
    model_pa = deepmaxent_model(
        input_size=len(covs), hidden_size=HIDDEN_SIZE,
        output_size=len(species), hidden_nbr=HIDDEN_LAYERS
    )
    train_model(
        model=model_pa,
        train_loader=pa_loader,
        criterion='bce',
        epochs=epochs,
        lr=lr,
        print_every=PRINT_EVERY,
        dev=device(),
        verbose=False
    )

    # ---- model for PO (DeepMaxEnt loss) ----
    model_po = deepmaxent_model(
        input_size=len(covs), hidden_size=HIDDEN_SIZE,
        output_size=len(species), hidden_nbr=HIDDEN_LAYERS
    )
    train_model(
        model=model_po,
        train_loader=po_loader,
        criterion='deepmaxent',
        epochs=epochs,
        lr=lr,
        print_every=PRINT_EVERY,
        dev=device(),
        verbose=False
    )

    # ---- third model: discriminator PO vs PA (binary) ----
    # build combined train set: PO labeled 0, PA labeled 1
    X_disc_tr = np.vstack([
        X_po_s[covs].values.astype(np.float32),
        X_pa_tr_s[covs].values.astype(np.float32)
    ])
    y_disc_tr = np.concatenate([
        np.zeros(len(X_po_s), dtype=np.float32),
        np.ones(len(X_pa_tr_s), dtype=np.float32)
    ])[:, None]  # shape (N,1) for BCE

    disc_ds = XYDataset(X_disc_tr, y_disc_tr)
    # keep batch roughly aligned with base batch_size
    disc_loader = DataLoader(disc_ds, batch_size=max(32, batch_size), shuffle=True, drop_last=True)

    model_disc = deepmaxent_model(
        input_size=len(covs), hidden_size=HIDDEN_SIZE,
        output_size=1, hidden_nbr=HIDDEN_LAYERS
    )
    train_model(
        model=model_disc,
        train_loader=disc_loader,
        criterion='bce',            # binary cross-entropy
        epochs=epochs,
        lr=lr,
        print_every=PRINT_EVERY,
        dev=device(),
        verbose=False
    )

    # ---- evaluate on PA_test with weighted ensemble ----
    X_te_np = X_pa_te_s[covs].values.astype(np.float32)

 
    # raw logits
    logits_pa = predict(model_pa, X_te_np, dev=device())
    logits_po = predict(model_po, X_te_np, dev=device())
    logits_disc = predict(model_disc, X_te_np, dev=device())

    # ---- normalization ----
    # PA + PO: row-wise softmax or normalize by sum
    exp_pa = np.exp(logits_pa - np.max(logits_pa, axis=1, keepdims=True))
    scores_pa = exp_pa / np.clip(exp_pa.sum(axis=1, keepdims=True), 1e-8, None)

    exp_po = np.exp(logits_po - np.max(logits_po, axis=1, keepdims=True))
    scores_po = exp_po / np.clip(exp_po.sum(axis=1, keepdims=True), 1e-8, None)

    # discriminator: sigmoid for probability
    p_is_pa = 1 / (1 + np.exp(-logits_disc))
    if p_is_pa.ndim == 2 and p_is_pa.shape[1] == 1:
        p_is_pa = p_is_pa.reshape(-1, 1)

    # ---- weighted ensemble ----
    scores_ens = p_is_pa * scores_pa + (1.0 - p_is_pa) * scores_po

    # AUCs on PA test
    aucs = per_species_auc(Y_pa_te_df[species], scores_ens, species)
    avg_auc = np.nanmean(list(aucs.values()))

    # ---- save artifacts ----
    scaler_path = os.path.join(output_dir, f"scaler_{name}_{region}{group}.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    model_dir = os.path.join(output_dir, f"deepmaxent_DA_{region}{group}")
    os.makedirs(model_dir, exist_ok=True)
    model_pa_path = os.path.join(model_dir, "model_pa.pt")
    model_po_path = os.path.join(model_dir, "model_po.pt")
    model_disc_path = os.path.join(model_dir, "model_disc.pt")
    torch.save(model_pa, model_pa_path)
    torch.save(model_po, model_po_path)
    torch.save(model_disc, model_disc_path)

    # For backward compatibility keep a single path; point it to the directory
    model_path = model_dir

    return avg_auc, aucs, model_path, scaler_path


def domain_probe(X_po: pd.DataFrame, X_pa: pd.DataFrame, covs: list[str], verbose: bool = True) -> float:
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

    if verbose:
        print(f"\n[Domain probe] PO vs PA separability AUC = {auc:.3f}")
        print("≈0.5 → distributions similar | >0.7 → strong covariate shift\n")
    return auc

def run_minimal_filter(
    X_po: pd.DataFrame, Y_po: pd.DataFrame,
    X_pa: pd.DataFrame, Y_pa: pd.DataFrame,
    covs: List[str], species: List[str],
    *, epochs: int = 50, batch_size: int = 256, lr: float = 1e-3,
    threshold: float = 0.5, require_all_labels: bool = True,
    ):
    """Trains on PA, filters PO by correctness, returns (X_po_keep, Y_po_keep, model, scaler, keep_mask)."""

    # scale (fit on combo PO + PA train)
    scaler = StandardScaler().fit(
        pd.concat([X_po[covs], X_pa[covs]], axis=0)
    )
    X_po_s = X_po.copy()
    X_pa_s = X_pa.copy()
    X_po_s[covs] = scaler.transform(X_po[covs])
    X_pa_s[covs] = scaler.transform(X_pa[covs])

    train_ds = XYDataset(X_pa_s[covs].values.astype(np.float32), Y_pa[species].values.astype(np.float32))
    train_loader = DataLoader(train_ds, batch_size=100, shuffle=True, drop_last=False)

    # model
    model = build_model(input_size=len(covs), output_size=len(species), 
                        bias=BIAS_MODEL, num_plots=len(X_pa))
    criterion = 'bce'

    # train
    print('\n Fitting PA Model for Filtering')
    train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        epochs=EPOCHS,
        lr=LR,
        print_every=PRINT_EVERY,
        dev=device(),
        verbose=False
    )


    # logits on PO → relative intensities (N x S)
    I = predict(model, X_po_s[covs].values.astype(np.float32), dev=device()) 
    I = 1/(1+np.exp(-I))


    # keep site if ANY present species is in species-wise top quantile of intensity
    # pres = (Y_po[species].to_numpy() > 0)
    # print(pres)
    # thr = np.quantile(I, q=0.5, axis=0)               # per-species threshold
    # hits = pres & (I >= thr)                           # N x S
    # print(hits)
    # keep = hits.any(axis=1) & (pres.any(axis=1))       # not too restrictive

    keep = [False]*len(I)

    if False:
        print('Filtering with correlations')
        corr_points = []
        for j in range(len(I)):
            corr_score = np.corrcoef(Y_po.loc[j,species], I[j,:])[0,1]
            if corr_score > 0.1:
                keep[j] = True
                corr_points.append(j)
        print(f'{len(corr_points)} added from Corr')

    if True:
        pred_po = []
        I_round = I.copy()
        I_round = 1*(I_round>0.5)
        # print(np.mean(I, axis = 0))
        # print(I_round)
        Y_po_np = np.array(Y_po)
        for j in range(len(I)):
            args_po = Y_po_np[j] == 1
            # print(np.sum(args_po))
            if np.sum(np.abs(Y_po_np[j,args_po] - I_round[j,args_po])) == 0:
                pred_po.append(j)
                keep[j] = True
                I[j,args_po] = 1
            else:
                keep[j] = True
                I[j,args_po] = 1
                # I[j,~args_po] = np.mean(Y_po_np[:,~args_po], axis = 0)
                I[j,~args_po] = np.mean(Y_pa.values[:,~args_po], axis = 0)
            # if np.sum(args_po) >=2:
            #     keep[j] = True
            #     I[j] = Y_po_np[j]
            

    if False:
        for j in X_pa_s.index:
            x_pa_j = X_pa_s.loc[j,'x']
            y_pa_j = X_pa_s.loc[j,'y']
            dists = (x_pa_j - X_po_s['x'])**2 + (y_pa_j - X_po_s['y'])**2
            # choosen_index = np.argmin(dists)
            # # print(f'For PA sample {j} we take PO sample {choosen_index}')
            # keep[choosen_index] = True}

                
            # Convert squared distances to probabilities (closer = higher weight)
            sigma = np.percentile(dists, 10) + 1e-8  # scale: 10th percentile distance
            probs = np.exp(-dists / (2 * sigma))
            probs = probs / probs.sum()

            # Randomly sample one PO index using these probabilities
            chosen_index = np.random.choice(X_po_s.index, p=probs)
            keep[chosen_index] = True
    

    print(f'---{np.sum(keep)} PO samples where kept out of {len(X_po)}---')

    # I = (I + Y_po)/2 

    return keep, I, I_round



def po_inputer(
    X_po: pd.DataFrame, Y_po: pd.DataFrame,
    X_pa: pd.DataFrame, Y_pa: pd.DataFrame,
    covs: List[str], species: List[str],
    *, epochs: int = 50, batch_size: int = 256, lr: float = 1e-3,
    threshold: float = 0.5, require_all_labels: bool = True,
    ):
    """Trains on PA, filters PO by correctness, returns (X_po_keep, Y_po_keep, model, scaler, keep_mask)."""

    # scale (fit on combo PO + PA train)
    scaler = StandardScaler().fit(
        pd.concat([X_po[covs], X_pa[covs]], axis=0)
    )
    X_po_s = X_po.copy()
    X_pa_s = X_pa.copy()
    X_po_s[covs] = scaler.transform(X_po[covs])
    X_pa_s[covs] = scaler.transform(X_pa[covs])

    train_ds = XYDataset(X_pa_s[covs].values.astype(np.float32), Y_pa[species].values.astype(np.float32))
    train_loader = DataLoader(train_ds, batch_size=100, shuffle=True, drop_last=False)

    # model
    model = build_model(input_size=len(covs), output_size=len(species), 
                        bias=BIAS_MODEL, num_plots=len(X_pa))
    criterion = 'bce'

    # train
    print('\n Fitting PA Model for Filtering')
    train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        epochs=EPOCHS,
        lr=LR,
        print_every=PRINT_EVERY,
        dev=device(),
        verbose=False
    )


    # logits on PO → relative intensities (N x S)
    I = predict(model, X_po_s[covs].values.astype(np.float32), dev=device()) 
    I = 1/(1+np.exp(-I))
    
    keep = [False]*len(I)
    if True:
        pred_po = []
        I_round = I.copy()
        I_round = 1*(I_round>0.5)
        # print(np.mean(I, axis = 0))
        # print(I_round)
        Y_po_np = np.array(Y_po)
        for j in range(len(I)):
            args_po = Y_po_np[j] == 1
            # print(np.sum(args_po))
            if np.sum(np.abs(Y_po_np[j,args_po] - I_round[j,args_po])) == 0:
                pred_po.append(j)
                keep[j] = True
                I[j,args_po] = 1
            else:
                keep[j] = True
                I[j,args_po] = 1
                # I[j,~args_po] = np.mean(Y_po_np[:,~args_po], axis = 0)
                I[j,~args_po] = np.mean(Y_pa.values[:,~args_po], axis = 0)
            # if np.sum(args_po) >=2:
            #     keep[j] = True
            #     I[j] = Y_po_np[j]
            

    print(f'---{np.sum(keep)} PO samples where kept out of {len(X_po)}---')

    # I = (I + Y_po)/2 

    return keep, I, I_round



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
            X_po, Y_po, X_pa, Y_pa, species, covs = load_po_pa_nceas(
                region=region,
                group_filter=group,
                add_po_var=ADD_PO_VAR
            )

            # 2) Split PA → train/test (fixed for all experiments)
            X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te = split_pa_train_test_spatially(
                X_pa, Y_pa, test_frac=TEST_PA_FRACTION, seed=SEED
            )
            print(f"PA split → train: {len(X_pa_tr)}, test: {len(X_pa_te)}")

            exp_dir = os.path.join(output_root, f"{region}{group}")

            ### test if separable
            # X_po = X_po.drop(columns=["x","y"], errors="ignore")
            scaler = StandardScaler().fit(X_pa_tr[covs])
            X_po_s = X_po.copy()
            X_po_s[covs] = scaler.transform(X_po_s[covs])
            X_pa_tr_s = X_pa_tr.copy()
            X_pa_tr_s[covs] = scaler.transform(X_pa_tr_s[covs])
            # drop PO column if present
            X_po_s = X_po_s.drop(columns=["PO"], errors="ignore")
            X_pa_tr_s = X_pa_tr_s.drop(columns=["PO"], errors="ignore")
            covs_copy = [c for c in covs if c != "PO"]
            # domain_auc = domain_probe(X_po_s, X_pa_tr_s, covs_copy, verbose=True)

           
    


            if RUN_PO:
                print("\n--- Running PO-only experiment ---")
                # A) PO-only → PA_test
   
     
                auc_po, aucs_po, model_po, scaler_po = run_experiment(
                    name="PO_only",
                    X_train_df=X_po,
                    Y_train_df=Y_po,
                    X_test_df=X_pa_te,  # evaluate on PA_test covs
                    Y_test_df=Y_pa_te,  # evaluate on PA_test labels
                    covs=covs, species=species,
                    output_dir=exp_dir, region=region, group=group
                )
                print(f"[PO-only]   Average AUC on PA_test: {auc_po:.4f}")

            if RUN_PA:
                print("\n--- Running PA-only experiment ---")
                
                loss_criterion = 'bce'
                # B) PA_train-only → PA_test
                auc_pa, aucs_pa, model_pa, scaler_pa = run_experiment(
                    name="PA_only",
                    X_train_df=X_pa_tr,
                    Y_train_df=Y_pa_tr,
                    X_test_df=X_pa_te,
                    Y_test_df=Y_pa_te,
                    covs=covs, species=species,
                    output_dir=exp_dir, region=region, group=group,
                    criterion = loss_criterion
                )
                print(f"[PA-only]   Average AUC on PA_test: {auc_pa:.4f}")

            ### Filtering based on PA-PO difference (Done for methods that join PO and PA, so from 3rd on)
            if RUN_IMPUTED_POPA:
                keep_po, I, I_round = po_inputer(X_po, Y_po, X_pa_tr, Y_pa_tr, covs_copy, species)
                X_po_imputed = X_po.copy().loc[keep_po]
                Y_po_imputed = Y_po.copy().loc[keep_po]

                #Y_po[species] = I_round[keep_po]
                Y_po_imputed[species] = I[keep_po]
                # X_po['']
                # covs for POPA
                print("\n--- Running PO+PA (IMPUTED) integration experiment ---")

                # C) PO + PA_train → PA_test
                X_mix = pd.concat([X_po_imputed, X_pa_tr], axis=0, ignore_index=True)
                Y_mix = pd.concat([Y_po_imputed, Y_pa_tr], axis=0, ignore_index=True)

                auc_mix_imp, aucs_mix_imp, model_mix_imp, scaler_mix_imp = run_experiment(
                        name="PO_plus_PA_imputed",
                        X_train_df=X_mix,
                        Y_train_df=Y_mix,
                        X_test_df=X_pa_te,
                        Y_test_df=Y_pa_te,
                        covs=covs, species=species,
                        output_dir=exp_dir, region=region, group=group,
                        criterion = 'bce'
                    )
                print(f"[PO+PA (IMPUTED)] Average AUC on PA_test: {auc_mix_imp:.4f}")


            if RUN_POPA_ENSAMBLE:
                
                print("\n--- Running PO+PA (ENSAMBLE) integration experiment ---")
                auc_mix_e, aucs_mix_e, model_mix_e, scaler_mix_e = run_experiment_popa_ensamble(
                    name="PO_plus_PA_Ensamble",
                    X_po_df=X_po,
                    Y_po_df=Y_po,
                    X_pa_tr_df=X_pa_tr,
                    Y_pa_tr_df=Y_pa_tr,
                    X_pa_te_df=X_pa_te,
                    Y_pa_te_df=Y_pa_te,
                    covs=covs, species=species,
                    output_dir=exp_dir, region=region, group=group,
                    epochs=EPOCHS,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    w_po=.5,
                    w_pa=.5
                )
                print(f"[Ensamble PO+PA]     Average AUC on PA_test: {auc_mix_e:.4f}")

                

            if RUN_POPA:
                # covs for POPA
                print("\n--- Running PO+PA integration experiment ---")
                print('The covariates used are:', covs)

                # C) PO + PA_train → PA_test
                X_mix = pd.concat([X_po, X_pa_tr], axis=0, ignore_index=True)
                Y_mix = pd.concat([Y_po, Y_pa_tr], axis=0, ignore_index=True)

                # print(X_po)
                # print('LEN X_PA TRAIN', len(X_pa_tr))
                # print('LEN X_MIN TRAIN', len(X_mix))
                # exit()

                if ADD_INTERACTIONS:
                    # X_expanded (PO interaction with the rest)
                    non_po_covs = [c for c in covs if c != "PO"]
                    X_pa_te_copy = X_pa_te.copy()
                    for c in non_po_covs:
                        X_mix[f"PO_{c}"] = X_mix["PO"] * X_mix[c]
                        X_pa_te_copy[f"PO_{c}"] = X_pa_te_copy["PO"] * X_pa_te_copy[c]
                    total_covs = X_mix.columns.tolist()
                    print('After adding interactions, the covariates used are:', total_covs)
                    auc_mix, aucs_mix, model_mix, scaler_mix = run_experiment(
                        name="PO_plus_PA_interactions",
                        X_train_df=X_mix,
                        Y_train_df=Y_mix,
                        X_test_df=X_pa_te_copy,
                        Y_test_df=Y_pa_te,
                        covs=total_covs, species=species,
                        output_dir=exp_dir, region=region, group=group
                    )
                else:

                    auc_mix, aucs_mix, model_mix, scaler_mix = run_experiment(
                        name="PO_plus_PA",
                        X_train_df=X_mix,
                        Y_train_df=Y_mix,
                        X_test_df=X_pa_te,
                        Y_test_df=Y_pa_te,
                        covs=covs, species=species,
                        output_dir=exp_dir, region=region, group=group,
                        criterion = 'deepmaxent'
                    )

                    
                print(f"[PO+PA]     Average AUC on PA_test: {auc_mix:.4f}")

            # weighted PO+PA
            if RUN_POPA_WEIGHTED:

                print("\n--- Running Weighted PO+PA integration experiment ---")

                # use domain_auc as weights
                w_po = .5#1-domain_auc # if they are easily separable domain_auc ~ 1, so we want to downweight PO
                w_pa = .5#domain_auc

                auc_mix_w, aucs_mix_w, model_mix_w, scaler_mix_w = run_experiment_popa(
                    name="PO_plus_PA_weighted",
                    X_po_df=X_po,
                    Y_po_df=Y_po,
                    X_pa_tr_df=X_pa_tr,
                    Y_pa_tr_df=Y_pa_tr,
                    X_pa_te_df=X_pa_te,
                    Y_pa_te_df=Y_pa_te,
                    covs=covs, species=species,
                    output_dir=exp_dir, region=region, group=group,
                    epochs=EPOCHS,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    w_po=w_po,
                    w_pa=w_pa
                )
                print(f"[Weighted PO+PA]     Average AUC on PA_test: {auc_mix_w:.4f}")




            summary_rows.append({
                "region": region,
                "group": group or "(all)",
                "TEST_PA_FRACTION": TEST_PA_FRACTION,
                "AUC_PO_only": float(np.round(auc_po, 4)) if RUN_PO else np.nan,
                "AUC_PA_only": float(np.round(auc_pa, 4)) if RUN_PA else np.nan,
                "AUC_PO_plus_PA": float(np.round(auc_mix, 4)) if RUN_POPA else np.nan,
                "AUC_PO_plus_PA_imputed": float(np.round(auc_mix_imp, 4)) if RUN_IMPUTED_POPA else np.nan,
                "AUC_PO_plus_PA_ensamble": float(np.round(auc_mix_e, 4)) if RUN_POPA_ENSAMBLE else np.nan,
                "AUC_Weighted_PO_plus_PA": float(np.round(auc_mix_w, 4)) if RUN_POPA_WEIGHTED else np.nan,
            })

    summary = pd.DataFrame(summary_rows)
    print("\n=== Summary (Average AUC on shared PA_test) ===")
    print(summary.to_string(index=False))
    summary.to_csv('output/data_integration_results.csv')

    elapsed = time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
