# Just keeps the relevant 3 methods which so far make more sense, discarding the rest (still in test_data_integration.py)

## Weights and Biases

import os
import random
from typing import List, Tuple, Dict
from time import time

import numpy as np
import pandas as pd
import pickle

from src.load_data import build_paths_nceas, load_po_pa_nceas, load_po_pa_geoplant
from src.utils import safe_reindex_columns, split_pa_train_test_spatially, scale_features

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# --- your models & loss
from src.models import DeepMaxEntModel, deepmaxent_loss, deepmaxent_model_w_bias
import src.pa_split as pa_split
from src.model_training import smooth_targets_v3

# =========================
# Config
# =========================
# REGIONS = ["AWT", "NZ", "SWI"]
# REGIONS = ['AWT']

# REGIONS = ["AWT", "CAN", "NSW", "SA", "SWI", "NZ"]             # regions to run
REGIONS = ['SWI']
GROUPS_BY_REGION = {
        # "AWT": ["_bird"],
        "AWT": ["_plant", "_bird"],
        "CAN": [""],
        # "NSW": ['_plant', ],
        "NSW": ['_bat', '_bird', '_plant', '_reptile'],
        "SA" : [""],
        "SWI": [""],
        "NZ": [""]
    }  




# General Experiment settings
ADD_PO_VAR = False            # if True, add PO indicator covariate (it's like a Tsource indicator)
BIAS_MODEL = False            # set True for per-plot bias
KEEP_XY = True  # if True, keep x,y in covariates
TEST_PA_FRACTION = 0.3        # PA split: test fraction
# FILTER_PO = True

# Model / training
HIDDEN_SIZE = 100
# HIDDEN_BIAS_SIZE = 3000
HIDDEN_LAYERS = 2
LR = 1e-4
EPOCHS = 100
BATCH_SIZE = 250
MAX_BATCH_PERCENTAGE = 1
PRINT_EVERY = 1000
SEED = 42

RUN_PO = True
RUN_PA = True
RUN_POPA, ADD_INTERACTIONS = True, False
RUN_POPA_SMOOTHED = True
RUN_POPA_ENSEMBLE = True


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
    def __init__(self, X: np.ndarray, Y: np.ndarray, plot_ids = None):
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


class XZYDataset(Dataset):
    def __init__(
        self,
        X_species: np.ndarray,     # (N, Cx)
        Z_bias: np.ndarray,        # (N, Cz)
        Y: np.ndarray,             # (N, K)
        # mask: np.ndarray | None = None  # (N, K) boolean or 0/1 for missing labels
    ):
        assert len(X_species) == len(Z_bias) == len(Y)
        self.Xs = torch.tensor(X_species, dtype=torch.float32)
        self.Zb = torch.tensor(Z_bias, dtype=torch.float32)
        self.Y  = torch.tensor(Y, dtype=torch.float32)
        # self.mask = None if mask is None else torch.tensor(mask.astype(bool))

    def __len__(self) -> int:
        return self.Xs.shape[0]

    def __getitem__(self, idx: int):
        # if self.mask is None:
        #     return self.Xs[idx], self.Zb[idx], self.Y[idx]#, None
        # else:
        return self.Xs[idx], self.Zb[idx], self.Y[idx]#, self.mask[idx]




# =========================
# Train / Eval helpers
# =========================
def build_model(input_size: int, output_size: int, bias: bool, num_plots = None) -> nn.Module:
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
        return DeepMaxEntModel(
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
    dev: torch.device = None,
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

    if BIAS_MODEL:
        print(model.plot_bias(idx))










@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, dev: torch.device = None) -> np.ndarray:
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
    #     print(f"  Species {sp}: AUC = {auc:.4f}")
    # exit()
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

   

    model = DeepMaxEntModel(input_size=len(covs), hidden_size=HIDDEN_SIZE, output_size=len(species), hidden_nbr=HIDDEN_LAYERS)
    # criterion_species = deepmaxent_loss()

    # loaders separately (proportional sizes)
    proportion_po = len(X_po_s) / (len(X_po_s) + len(X_pa_tr_s))
    proportion_pa = len(X_pa_tr_s) / (len(X_po_s) + len(X_pa_tr_s))
    batch_size_po = max(1, int(batch_size*proportion_po))
    batch_size_pa = max(1, int(batch_size*proportion_pa))
    print('Batch for PO: ', batch_size_po)
    print('Batch for PA: ', batch_size_pa)
    po_ds = XYDataset(X_po_s[covs].values.astype(np.float32), Y_po_df[species].values.astype(np.float32))
    pa_ds = XYDataset(X_pa_tr_s[covs].values.astype(np.float32), Y_pa_tr_df[species].values.astype(np.float32))
    po_loader = DataLoader(po_ds, batch_size=batch_size_po, shuffle=True, drop_last=False)
    pa_loader = DataLoader(pa_ds, batch_size=batch_size_pa, shuffle=True, drop_last=False)

    def train_popa_model(
        model: nn.Module,
        po_loader: DataLoader,
        pa_loader: DataLoader,
        po_ds: Dataset,
        pa_ds: Dataset,
        criterion_species: nn.Module,
        epochs: int = 300,
        lr: float = 1e-4,
        dev: torch.device = None,
        w_po: float = .5,
        w_pa: float = .5,
    ):
        dev = dev or device()
        model.to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-4)
        # criterion_po = deepmaxent_loss()

        # criterion_pa = deepmaxent_loss()#torch.nn.BCEWithLogitsLoss()

        beta = 0.5  # initial smoothing parameter

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

                # smooth PO targets
                yb_po_smooth = smooth_targets_v3(yb_po,outputs_po,beta)
                

                # concat outputs and labels
                output_mix = torch.cat([outputs_po, outputs_pa], dim=0)
                yb_mix = torch.cat([yb_po_smooth, yb_pa], dim=0)
                loss = criterion_species(output_mix, yb_mix)

                # loss_po = criterion_po(outputs_po, yb_po)
                # loss_pa = criterion_pa(outputs_pa, yb_pa)
                # loss = w_po * loss_po + w_pa * loss_pa

                loss.backward()
                optimizer.step()
                running_loss += loss.item() * (xb_po.size(0) + xb_pa.size(0))

            with torch.no_grad():
                    # update beta with M-step

                ## access the full Y
                y = po_ds.Y.to(dev)
                logits_full = model(po_ds.X.to(dev))
                # reshape beta to have same number as columns as y
                # beta = np.array([beta] * y.shape[1]) # could be better
                y_soft_full = smooth_targets_v3(y, logits_full, beta)

                y_pa = pa_ds.Y.to(dev)

                y_complete = torch.cat([y, y_pa], dim=0)
                y_complete_soft = torch.cat([y_soft_full, y_pa], dim=0)

                # apply the threshold for zeros as well
                # threshold_zero = epoch / train_cfg['epochs'] * 0.01
                # y_soft_full = torch.where((y == 0) & (y_soft_full < threshold_zero), torch.zeros_like(y_soft_full), y_soft_full)

                
                numer = (y_complete_soft * (1 - y_complete)).sum(dim=0)
                denom = y_complete_soft.sum(dim=0)
                # numer = (y_soft * (1 - y)).sum(dim=0)
                # denom = y_soft.sum(dim=0)

                theta = torch.zeros_like(numer)

                # only update where denom > 0
                mask = denom > 0
                theta[mask] = numer[mask] / denom[mask]

                theta = theta.clamp(0.0, 1.0)

                beta = 1-theta.cpu().numpy()
         

        avg_loss = running_loss / (len(po_loader.dataset) + len(pa_loader.dataset))
    
    # for loss use BCE
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_popa_model(
        model=model,
        po_loader=po_loader,
        pa_loader=pa_loader,
        po_ds=po_ds,
        pa_ds=pa_ds,
        criterion_species=loss_fn,
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

    # model_path = os.path.join(output_dir, f"deepmaxent_DA_{region}{group}.pt")
    # torch.save(model, model_path)
    model_path = None
    return avg_auc, aucs, model_path, scaler_path



def run_experiment_popa_ensemble(
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
    model_pa = DeepMaxEntModel(
        input_size=len(covs), hidden_size=HIDDEN_SIZE,
        output_size=len(species), hidden_nbr=HIDDEN_LAYERS
    )
    train_model(
        model=model_pa,
        train_loader=pa_loader,
        criterion='deepmaxent',
        epochs=epochs,
        lr=lr,
        print_every=PRINT_EVERY,
        dev=device(),
        verbose=False
    )

    # ---- model for PO (DeepMaxEnt loss) ----
    model_po = DeepMaxEntModel(
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
    disc_loader = DataLoader(disc_ds, batch_size=max(32, batch_size), shuffle=True, drop_last=False)

    model_disc = DeepMaxEntModel(
        input_size=len(covs), hidden_size=int(HIDDEN_SIZE/3),
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
    # exp_pa = np.exp(logits_pa - np.max(logits_pa, axis=1, keepdims=True))
    # scores_pa = exp_pa / np.clip(exp_pa.sum(axis=1, keepdims=True), 1e-8, None)
    # sum_sites_pa = np.sum(scores_pa, axis = 1)

    # exp_po = np.exp(logits_po - np.max(logits_po, axis=1, keepdims=True))
    # scores_po = exp_po / np.clip(exp_po.sum(axis=1, keepdims=True), 1e-8, None)

    # # discriminator: sigmoid for probability
    p_is_pa = 1 / (1 + np.exp(-logits_disc))
    if p_is_pa.ndim == 2 and p_is_pa.shape[1] == 1:
        p_is_pa = p_is_pa.reshape(-1, 1)

    # ---- weighted ensemble ----
    scores_ens = p_is_pa * logits_pa + (1.0 - p_is_pa) * logits_po

    # ---- debug print ----    
    # for i in range(scores_ens.shape[0]):
    #     print(f"\nSpecies {i}:")
    #     print(f"  p_is_pa = {p_is_pa[i].item():.3f}")
    #     print(f"  scores_pa = {np.round(scores_pa[i], 3)}")
    #     print(f"  scores_po = {np.round(scores_po[i], 3)}")
    #     print(f"  -> scores_ens = {np.round(scores_ens[i], 3)}")


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












# =========================
# Main
# =========================
def main():
    start_time = time()
    set_all_seeds(SEED)
    dev = device()
    print(f"Using device: {dev}")

    output_root = os.path.join("output", "integration_geoplant")
    os.makedirs(output_root, exist_ok=True)
    summary_rows = []

    detailed_summary = []

    for region in ['france']:
        for group in ['plants']:
            print(f"\n=== REGION: {region}, GROUP: {group or '(all)'} ===")



            # use these paths for now
            # data/processed/geoplant/geoplant_po_mediterranean_withcovs.csv
            po_path = os.path.join(
                "data", "processed", "geoplant", f"geoplant_po_{region.lower()}_withcovs.csv"
            )
            pa_path = os.path.join(
                "data", "processed", "geoplant", f"geoplant_pa_{region.lower()}_withcovs.csv"
            )

            # 1) Load PO & PA, aligned species & covs
            X_po, Y_po, X_pa, Y_pa, species, covs = load_po_pa_geoplant(
                po_path, pa_path
            )
            print(f"Loaded PO: {len(X_po)} samples, PA: {len(X_pa)} samples")
            print(f"Species: {len(species)}, Covariates: {len(covs)}")
            # using the following covariates and species 
            print(f"Covariates: {covs}")
            print(f"Species: {species}")


            covs = [c for c in covs if c not in ['x','y']] 
            covs_xy = ['x','y'] if KEEP_XY else []
            covs_total = covs_xy + covs
            

            if ADD_PO_VAR:  
                X_po['PO'] = 1
                X_pa_tr['PO'] = 0
                X_pa_te['PO'] = 0




            # # 2) Split PA → train/test (fixed for all experiments)
            # X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te = split_pa_train_test_spatially(
            #     X_pa, Y_pa, test_frac=TEST_PA_FRACTION, seed=SEED
            # )
            # print(f"PA split → train: {len(X_pa_tr)}, test: {len(X_pa_te)}")
            # 2) Generate multiple PA train/test splits via partition sweep

            exp_dir = os.path.join(output_root, f"{region}{group}")

            ### test if separable
            # X_po = X_po.drop(columns=["x","y"], errors="ignore")
            scaler = StandardScaler().fit(X_pa[covs_total])
            X_po_s = X_po.copy()
            X_po_s[covs_total] = scaler.transform(X_po_s[covs_total])
            X_pa_s = X_pa.copy()
            X_pa_s[covs_total] = scaler.transform(X_pa_s[covs_total])
            # # drop PO column if present
            X_po_s = X_po_s.drop(columns=["PO"], errors="ignore")
            X_pa_s = X_pa_s.drop(columns=["PO"], errors="ignore")
            covs_no_po = [c for c in covs if c != "PO"]
            # domain_auc = domain_probe(X_po_s, X_pa_tr_s, covs_copy, verbose=True)


            # IMPORTANT FOR V2: This gives circulas partitions, which may be less desirable (still not finished)
            # pa_splits, split_type_list = pa_split.partition_sweep_ranges_v2(
            #     X_pa_s, Y_pa, covs_xy, covs_xy, K_clusters=20, select_subset=1, train_proportion=.4, distance_metric='euclidean')


 
            # # spatial case, only use xy for partitioning and distance
            pa_splits, split_type_list = pa_split.partition_sweep_ranges(
                X_pa_s, Y_pa, covs_xy, covs_xy, K_clusters=100, select_subset=10, train_proportion=.4, distance_metric='euclidean')


            for split_id, (X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te, d_metric) in enumerate(pa_splits):
                print(f'---- RUNNING SPLIT NUMBER {split_id} ----')


                if RUN_PO:
                    print("\n--- Running PO-only experiment ---")
                    # A) PO-only → PA_test
    
        
                    auc_po, aucs_po, model_po, scaler_po = run_experiment(
                        name="PO_only",
                        X_train_df=X_po_s,
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

    
                    

                if RUN_POPA:
                    # covs for POPA
                    print("\n--- Running PO+PA integration experiment ---")
                    print('The covariates used are:', covs)
                    



                    # C) PO + PA_train → PA_test
                    X_mix = pd.concat([X_po_s, X_pa_tr], axis=0, ignore_index=True)
                    Y_mix = pd.concat([Y_po, Y_pa_tr], axis=0, ignore_index=True)

                    # print(X_po)
                    # print('LEN X_PA TRAIN', len(X_pa_tr))
                    # print('LEN X_MIN TRAIN', len(X_mix))
                    # exit()

                    covs_to_use = covs + (['PO'] if ADD_PO_VAR else [])

            
                    auc_mix, aucs_mix, model_mix, scaler_mix = run_experiment(
                        name="PO_plus_PA",
                        X_train_df=X_mix,
                        Y_train_df=Y_mix,
                        X_test_df=X_pa_te,
                        Y_test_df=Y_pa_te,
                        covs=covs_to_use, species=species,
                        output_dir=exp_dir, region=region, group=group,
                        criterion = 'deepmaxent'
                    )

                    
                    print(f"[PO+PA]     Average AUC on PA_test: {auc_mix:.4f}")


                if RUN_POPA_SMOOTHED:

                    covs_to_use = covs + (['PO'] if ADD_PO_VAR else [])

                    auc_mix_smooth, aucs_mix_smooth, model_mix_smooth, scaler_mix_smooth = run_experiment_popa(
                        name="PO_plus_PA_smooth",
                        X_po_df=X_po_s,
                        Y_po_df=Y_po,
                        X_pa_tr_df=X_pa_tr,
                        Y_pa_tr_df=Y_pa_tr,
                        X_pa_te_df=X_pa_te,
                        Y_pa_te_df=Y_pa_te,
                        covs=covs_to_use, species=species,
                        output_dir=exp_dir, region=region, group=group,
                        epochs=EPOCHS,
                        lr=LR,
                        batch_size=BATCH_SIZE,
                        w_po=0.5,
                        w_pa=0.5
                    )
                    print(f"[PO+PA Smooth] Average AUC on PA_test: {auc_mix_smooth:.4f}")   

                if RUN_POPA_ENSEMBLE:
                    auc_mix_ens, aucs_mix_ens, model_mix_ens, scaler_mix_ens = run_experiment_popa_ensemble(
                        name="PO_plus_PA_ens",
                        X_po_df=X_po_s,
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
                        w_po=0.5,
                        w_pa=0.5
                    )
                    print(f"[PO+PA Ens] Average AUC on PA_test: {auc_mix_ens:.4f}")
     
                row_info = {
                    "region": region,
                    "group": group or "(all)",
                    "split_id": split_id,
                    # "K": k_list[split_id] if k_list is not None else np.nan,
                    "train_test_dist": d_metric,
                    "split_type": split_type_list[split_id][0], #if split_type_list else "unknown",
                    "test_id": split_type_list[split_id][1], #if split_type_list else "unknown",
                    'train_indexes': X_pa_tr.index.tolist(),
                    'test_indexes': X_pa_te.index.tolist(),
                    "TEST_PA_FRACTION": TEST_PA_FRACTION,
                    "AUC_PO_only": float(np.round(auc_po, 4)) if RUN_PO else np.nan,
                    "AUC_PA_only": float(np.round(auc_pa, 4)) if RUN_PA else np.nan,
                    "AUC_PO_plus_PA": float(np.round(auc_mix, 4)) if RUN_POPA else np.nan,
                    "AUC_PO_plus_PA_smooth": float(np.round(auc_mix_smooth, 4)) if RUN_POPA_SMOOTHED else np.nan,
                    "AUC_PO_plus_PA_ens": float(np.round(auc_mix_ens, 4)) if RUN_POPA_ENSEMBLE else np.nan,
                }

                summary_rows.append(row_info)

                # detailed per-species
                for sp in species:
                    detailed_row = row_info.copy()
                    detailed_row.update({
                        "species": sp,
                        "AUC_PO_only_sp": float(np.round(aucs_po[sp], 4)) if RUN_PO else np.nan,
                        "AUC_PA_only_sp": float(np.round(aucs_pa[sp], 4)) if RUN_PA else np.nan,
                        "AUC_PO_plus_PA_sp": float(np.round(aucs_mix[sp], 4)) if RUN_POPA else np.nan,
                        "AUC_PO_plus_PA_smooth_sp": float(np.round(aucs_mix_smooth[sp], 4)) if RUN_POPA_SMOOTHED else np.nan,
                        "AUC_PO_plus_PA_ens_sp": float(np.round(aucs_mix_ens[sp], 4)) if RUN_POPA_ENSEMBLE else np.nan,
                    })
                    detailed_summary.append(detailed_row)

                # make summary per species as well

    summary = pd.DataFrame(summary_rows)
    print("\n=== Summary (Average AUC on shared PA_test) ===")
    print(summary.to_string(index=False))
    # make dir if not exists
    # os.makedirs('output/integration_analysis', exist_ok=True)
    summary_path = os.path.join(output_root, 'partition_results.csv')
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    detailed_summary_df = pd.DataFrame(detailed_summary)
    detailed_summary_path = os.path.join(output_root, 'partition_results_detailed.csv')
    detailed_summary_df.to_csv(detailed_summary_path, index=False)
    print(f"Detailed summary saved to: {detailed_summary_path}")

    elapsed = time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
