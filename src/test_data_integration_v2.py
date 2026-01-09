# Just keeps the relevant 3 methods which so far make more sense, discarding the rest (still in test_data_integration.py)

## Weights and Biases

import os
import random
from typing import List, Tuple, Dict, Optional
from time import time

import numpy as np
import pandas as pd
import pickle

from src.load_data import build_paths_nceas, load_po_pa_nceas
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
# from k_means_constrained import KMeansConstrained
from src.model_training import smooth_targets_v3

# --- your models & loss
from src.models import DeepMaxEntModel, deepmaxent_loss, deepmaxent_model_w_bias, deepmaxent_domain, grad_reverse, DomainDiscriminator, DeepMaxentTwoHead, SDMWithBias, deepmaxent_loss_w_bias, PoissonCountAndPresenceLoss


# NEW: TabPFN
try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False

# =========================
# Config
# =========================
# REGIONS = ['SWI', 'AWT', 'CAN']
# REGIONS = ["AWT", "CAN", "NSW", "SA", "SWI", "NZ"]             # regions to run
# # REGIONS = ["SWI"]             # regions to run
# GROUPS_BY_REGION = {
#         "AWT": ["_plant", "_bird"],
#         "CAN": [""],
#         "NSW": ['_bat', '_bird', '_plant', '_reptile'],
#         # "NSW": ["_ba", "_db", "_nb", "_ot", "_rt", "_ru", "_sr"],
#         "SA" : [""],
#         "SWI": [""],
#         "NZ": [""]
#     } 

REGIONS = ["CAN", "NSW", "NZ", "SWI"]             # regions to run
GROUPS_BY_REGION = {
        "CAN": [""],
        "NSW": ['_bird'],
        "NZ": [""],
        "SWI": [""]
        }

# General Experiment settings
ADD_PO_VAR = False            # if True, add PO indicator covariate (it's like a Tsource indicator)
BIAS_MODEL = False            # set True for per-plot bias
KEEP_XY = True  # if True, keep x,y in covariates
TEST_PA_FRACTION = 0.5        # PA split: test fraction
# FILTER_PO = True

# Model / training
HIDDEN_SIZE = 250
# HIDDEN_BIAS_SIZE = 3000
HIDDEN_LAYERS = 2
LR = 1e-4
EPOCHS = 200
BATCH_SIZE = 250
MAX_BATCH_PERCENTAGE = 1
PRINT_EVERY = 1000
SEED = 42

RUN_PO = True
RUN_PO_LOGREG = False
RUN_PO_W_BIAS = True
RUN_PA = True
RUN_PA_LOGREG = False
RUN_POPA, ADD_INTERACTIONS = True, False
RUN_IMPUTED_POPA = False
RUN_POPA_ENSEMBLE = False
RUN_POPA_SMOOTHED = False
RUN_POPA_WEIGHTED = False
RUN_POPA_SMOOTHED_W_PRIOR = False
RUN_POPA_BIAS = True

RUN_PA_TABPFN = False

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
    def __init__(self, X: np.ndarray, Y: np.ndarray, plot_ids: Optional[np.ndarray] = None):
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
def build_model(input_size: int, output_size: int, bias: bool, num_plots: Optional[int] = None) -> nn.Module:
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
    dev: Optional[torch.device] = None,
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


def run_experiment_bias(
    name: str,
    X_train_df: pd.DataFrame, Y_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,  Y_test_df: pd.DataFrame,
    covs_species: List[str], covs_bias: List[str], species: List[str],
    output_dir: str, region: str, group: str,
    criterion: str = "bce_from_lambda",   # or "poisson"
    link: str = "logadd",
    verbose: bool = False
):
    os.makedirs(output_dir, exist_ok=True)
    # Independent scalers for the two inputs (you can share if desired)
    scaler_sp_path = os.path.join(output_dir, f"scaler_species_{name}_{region}{group}.pkl")
    scaler_bi_path = os.path.join(output_dir, f"scaler_bias_{name}_{region}{group}.pkl")

    # scale species covariates
    Xtr_sp, Xte_sp, _ = scale_features(
        X_train_df, X_test_df, covs_species, output_path=scaler_sp_path, verbose=False
    )
    # scale bias covariates
    Xtr_bi, Xte_bi, _ = scale_features(
        X_train_df, X_test_df, covs_bias, output_path=scaler_bi_path, verbose=False
    )

    Xs_tr = Xtr_sp[covs_species].values.astype(np.float32)
    Zb_tr = Xtr_bi[covs_bias].values.astype(np.float32)
    Y_tr  = Y_train_df[species].values.astype(np.float32)

    Xs_te = Xte_sp[covs_species].values.astype(np.float32)
    Zb_te = Xte_bi[covs_bias].values.astype(np.float32)
    Y_te  = Y_test_df[species].copy()

    train_ds = XZYDataset(Xs_tr, Zb_tr, Y_tr)
    train_loader = DataLoader(train_ds, batch_size=max(1, min(BATCH_SIZE, int(len(Xs_tr) * MAX_BATCH_PERCENTAGE))),
                              shuffle=True, drop_last=False)

    n_neurons_bias = int(len(Zb_tr)*0.3)
    model = SDMWithBias(len(covs_species), len(species), len(covs_bias), 
                        hidden_species = (500,500), hidden_bias = (n_neurons_bias,int(n_neurons_bias/2)))

    train_bias_model(
        model=model,
        loader=train_loader,
        criterion=criterion,
        epochs=EPOCHS,
        lr=LR,
        print_every=PRINT_EVERY,
        dev=device(),
        verbose=verbose,
    )
    # print('Finished model training')

    # # Evaluate
    # if criterion == "bce_from_lambda":
    #     scores = predict_twohead_presence_prob(model, Xs_te, Zb_te, dev=device())
    # else:  # "poisson"
    #     scores = predict_twohead_lambda(model, Xs_te, Zb_te, dev=device())
    Xs_te = torch.tensor(Xs_te, dtype=torch.float32, device=device())
    Zb_te = torch.tensor(Zb_te, dtype=torch.float32, device=device())
    scores, _ = model(Xs_te, Zb_te)
    scores = scores.detach().cpu().numpy()

    aucs = per_species_auc(Y_te, scores, species)
    avg_auc = np.nanmean(list(aucs.values()))

    auc_site = per_site_auc(Y_te[species], scores)
    avg_auc_site = np.nanmean(list(auc_site.values()))

    # Save
    model_path = os.path.join(output_dir, f"twohead_{name}_{region}{group}.pt")
    torch.save(model, model_path)

    return avg_auc, aucs, avg_auc_site, model_path, (scaler_sp_path, scaler_bi_path)

def run_experiment_popa_smoothed_w_prior(
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

   

    model_pa = DeepMaxEntModel(input_size=len(covs), hidden_size=HIDDEN_SIZE, output_size=len(species), hidden_nbr=HIDDEN_LAYERS)
    criterion_pa = 'bce'

    train_model(
        model=model_pa,
        train_loader=DataLoader(
            XYDataset(X_pa_tr_s[covs].values.astype(np.float32), Y_pa_tr_df[species].values.astype(np.float32)),
            batch_size=max(1, min(batch_size, int(len(X_pa_tr_s) * MAX_BATCH_PERCENTAGE))),
            shuffle=True,
            drop_last=False
        ),
        criterion=criterion_pa,
        epochs=epochs,
        lr=lr,
        dev=device(),
        verbose=False
    )
    print('Finished PA pre-training')
    beta = np.zeros(len(species))
    # for each species in PO, compute prediction for zero values
    with torch.no_grad():
        X_po_zero = X_po_s[covs].values.astype(np.float32).copy()
        for i, sp in enumerate(species):
            mask_zero = Y_po_df[sp].values == 0
            mask_one = Y_po_df[sp].values == 1
            if np.sum(mask_zero) > 0:
                X_po_zero_sp = X_po_zero[mask_zero]
                predicted_scores = predict(model_pa, X_po_s[covs].values.astype(np.float32), dev=device())[:, i]
                # applied sigmoid
                predicted_scores = 1 / (1 + np.exp(-predicted_scores))
                # 0.5 threshold
                # predicted_scores = (predicted_scores >= 0.5).astype(np.float32)

                # print a sample of the scores
                print(f"Sample scores for species {sp}: ", predicted_scores[:5])
            # compute beta as (predicted/(predicted+real))
            
            # # threshold predicted scores at 0.5 to get predicted presences
            # predicted_presences = (predicted_scores >= 0.5).astype(np.float32)

            # # # for predicted as one, check how many are zeros to get theta
            # arg_predicted_ones = np.where(predicted_presences == 1)[0]

            # pred_one_observed_zeros = np.sum(Y_po_df[sp].values[arg_predicted_ones] == 0)

            # theta = pred_one_observed_zeros / (len(arg_predicted_ones) + 1e-6)

            # beta[i] = theta # now use same notation

            # predicted presences in mask zero
            predicted_presences = predicted_scores[mask_zero]

            # see how many are above 0.5 threshold
            # predicted_one_in_absence = (predicted_presences >= 0.5).astype(np.float32)

            ratio = np.sum(predicted_presences) / (len(predicted_presences) + 1e-6)
            beta[i] = ratio




            # real = np.sum(Y_po_df[sp].values)
            # captured_ratio = real / (real + predicted_zeros)
            # # cap to 1
            # if captured_ratio > 1:
            #     captured_ratio = 1.0
            # beta[i] = 1 - captured_ratio
            print('For species ', sp, ' beta: ', beta[i])
            print('Predicted presences sum: ', np.sum(predicted_presences), ' Real presences sum: ', np.sum(Y_po_df[sp].values))
            # print('Predicted PO zeros sum: ', predicted_zeros, ' Real PO sum: ', real)
            
    # exit()

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

    # train model with PA only first
    model = DeepMaxEntModel(input_size=len(covs), hidden_size=HIDDEN_SIZE, output_size=len(species), hidden_nbr=HIDDEN_LAYERS)
    criterion_species = 'bce'
    

    def train_popa_model(
        model: nn.Module,
        po_loader: DataLoader,
        pa_loader: DataLoader,
        po_ds: Dataset,
        pa_ds: Dataset,
        criterion_species: nn.Module,
        epochs: int = 300,
        lr: float = 1e-4,
        dev: Optional[torch.device] = None,
        w_po: float = .5,
        w_pa: float = .5,
    ):
        dev = dev or device()
        model.to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-4)
        # criterion_po = deepmaxent_loss()

        # criterion_pa = deepmaxent_loss()#torch.nn.BCEWithLogitsLoss()

        # beta = 0.8  # initial smoothing parameter

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

            # with torch.no_grad():
            #         # update beta with M-step

            #     ## access the full Y
            #     y = po_ds.Y.to(dev)
            #     logits_full = model(po_ds.X.to(dev))
            #     # reshape beta to have same number as columns as y
            #     # beta = np.array([beta] * y.shape[1]) # could be better
            #     y_soft_full = smooth_targets_v3(y, logits_full, beta)

            #     y_pa = pa_ds.Y.to(dev)

            #     y_complete = torch.cat([y, y_pa], dim=0)
            #     y_complete_soft = torch.cat([y_soft_full, y_pa], dim=0)

            #     # apply the threshold for zeros as well
            #     # threshold_zero = epoch / train_cfg['epochs'] * 0.01
            #     # y_soft_full = torch.where((y == 0) & (y_soft_full < threshold_zero), torch.zeros_like(y_soft_full), y_soft_full)

                
            #     numer = (y_complete_soft * (1 - y_complete)).sum(dim=0)
            #     denom = y_complete_soft.sum(dim=0)
            #     # numer = (y_soft * (1 - y)).sum(dim=0)
            #     # denom = y_soft.sum(dim=0)

            #     theta = torch.zeros_like(numer)

            #     # only update where denom > 0
            #     mask = denom > 0
            #     theta[mask] = numer[mask] / denom[mask]

            #     theta = theta.clamp(0.0, 1.0)

            #     beta = 1-theta.cpu().numpy()
         

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

    aucs_site = per_site_auc(Y_pa_te_df[species], scores)
    avg_auc_site = np.nanmean(list(aucs_site.values()))

    # model_path = os.path.join(output_dir, f"deepmaxent_DA_{region}{group}.pt")
    # torch.save(model, model_path)
    model_path = None
    return avg_auc, aucs, avg_auc_site, model_path, scaler_path

def run_experiment_popa_smoothed(
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
        dev: Optional[torch.device] = None,
        w_po: float = .5,
        w_pa: float = .5,
    ):
        dev = dev or device()
        model.to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-4)
        # criterion_po = deepmaxent_loss()

        # criterion_pa = deepmaxent_loss()#torch.nn.BCEWithLogitsLoss()

        beta = 0.8  # initial smoothing parameter

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

    aucs_site = per_site_auc(Y_pa_te_df[species], scores)
    avg_auc_site = np.nanmean(list(aucs_site.values()))




    # model_path = os.path.join(output_dir, f"deepmaxent_DA_{region}{group}.pt")
    # torch.save(model, model_path)
    model_path = None
    return avg_auc, aucs, avg_auc_site, model_path, scaler_path


def run_experiment_popa_bias(
    name: str,
    X_po_df: pd.DataFrame, Y_po_df: pd.DataFrame,
    X_pa_tr_df: pd.DataFrame, Y_pa_tr_df: pd.DataFrame,
    X_pa_te_df: pd.DataFrame, Y_pa_te_df: pd.DataFrame,
    covs: List[str], covs_bias: List[str], species: List[str],
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
    scaler_bias = StandardScaler().fit(
        pd.concat([X_po_df[covs_bias], X_pa_tr_df[covs_bias]], axis=0)
    )

    X_po_s = X_po_df.copy()
    X_pa_tr_s = X_pa_tr_df.copy()
    X_pa_te_s = X_pa_te_df.copy()
    X_po_s[covs] = scaler.transform(X_po_s[covs]) 
    X_pa_tr_s[covs] = scaler.transform(X_pa_tr_s[covs])
    X_pa_te_s[covs] = scaler.transform(X_pa_te_s[covs])

    X_po_s_bias = X_po_df.copy()
    X_po_s_bias[covs_bias] = scaler_bias.transform(X_po_s_bias[covs_bias])
    X_pa_tr_s_bias = X_pa_tr_df.copy()
    X_pa_tr_s_bias[covs_bias] = scaler_bias.transform(X_pa_tr_s_bias[covs_bias])
    X_pa_te_s_bias = X_pa_te_df.copy()
    X_pa_te_s_bias[covs_bias] = scaler_bias.transform(X_pa_te_s_bias[covs_bias])

    batch_size = min(batch_size, int(len(X_pa_tr_s) * MAX_BATCH_PERCENTAGE))

   

    # model = deepmaxent_model(input_size=len(covs), hidden_size=HIDDEN_SIZE, output_size=len(species), hidden_nbr=HIDDEN_LAYERS)

    model = SDMWithBias(
        n_species_covariates=len(covs),
        n_species=len(species),
        n_bias_covariates=len(covs_bias),
        hidden_species=(128, 64),
        hidden_bias=(500, 250),
        link="logadd"
    )

    # criterion_species = deepmaxent_loss()

    # loaders separately (proportional sizes)
    proportion_po = len(X_po_s) / (len(X_po_s) + len(X_pa_tr_s))
    proportion_pa = len(X_pa_tr_s) / (len(X_po_s) + len(X_pa_tr_s))
    batch_size_po = max(1, int(batch_size*proportion_po))
    batch_size_pa = max(1, int(batch_size*proportion_pa))
    print('Batch for PO: ', batch_size_po)
    print('Batch for PA: ', batch_size_pa)
    # po_ds = XYDataset(X_po_s[covs].values.astype(np.float32), Y_po_df[species].values.astype(np.float32))
    po_ds = XZYDataset(X_po_s[covs].values.astype(np.float32), X_po_s_bias[covs_bias].values.astype(np.float32), Y_po_df[species].values.astype(np.float32))
    pa_ds = XYDataset(X_pa_tr_s[covs].values.astype(np.float32), Y_pa_tr_df[species].values.astype(np.float32))
    po_loader = DataLoader(po_ds, batch_size=batch_size_po, shuffle=True, drop_last=False)
    pa_loader = DataLoader(pa_ds, batch_size=batch_size_pa, shuffle=True, drop_last=False)



    def train_popa_model_bias(
        model: nn.Module,
        po_loader: DataLoader,
        pa_loader: DataLoader,
        po_ds: Dataset,
        pa_ds: Dataset,
        criterion_species: nn.Module,
        epochs: int = 300,
        lr: float = 1e-4,
        dev: Optional[torch.device] = None,
        w_po: float = .5,
        w_pa: float = .5,
    ):
        dev = dev or device()
        model.to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-4)
        # criterion_po = deepmaxent_loss()

        # criterion_pa = deepmaxent_loss()#torch.nn.BCEWithLogitsLoss()

        model.train()
        loss_fn = PoissonCountAndPresenceLoss()
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            for (xb_po, zb_po, yb_po), (xb_pa, yb_pa, _) in zip(po_loader, pa_loader):
                xb_po = xb_po.to(dev)
                zb_po = zb_po.to(dev)
                yb_po = yb_po.to(dev)
                xb_pa = xb_pa.to(dev)
                yb_pa = yb_pa.to(dev)

                optimizer.zero_grad()

                score_po, bias_pred_po = model(xb_po, zb_po)
                score_pa, _ = model(xb_pa)


                loss = loss_fn(score_po, score_pa, bias_pred_po, yb_po, yb_pa)

                loss.backward()
                optimizer.step()
                running_loss += loss.item() * (xb_po.size(0) + xb_pa.size(0))


    # for loss use BCE
    loss_fn = None

    train_popa_model_bias(
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
    scores = predict(model, X_te_np, dev=device(), bias_model=True)
    aucs = per_species_auc(Y_pa_te_df[species], scores, species)
    avg_auc = np.nanmean(list(aucs.values()))

    aucs_site = per_site_auc(Y_pa_te_df[species], scores)
    avg_auc_site = np.nanmean(list(aucs_site.values()))




    # model_path = os.path.join(output_dir, f"deepmaxent_DA_{region}{group}.pt")
    # torch.save(model, model_path)
    model_path = None
    return avg_auc, aucs, avg_auc_site, model_path, scaler_path


def train_bias_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: str = "poisson",       # "poisson" or "bce_from_lambda"
    epochs: int = 200,
    lr: float = 1e-3,
    print_every: int = 1000,
    dev: Optional[torch.device] = None,
    verbose: bool = False
):
    dev = dev or device()
    model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-4)

    loss_f = deepmaxent_loss_w_bias()

    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        nobs = 0
        for Xs, Zb, Yb in loader:
            Xs = Xs.to(dev); Zb = Zb.to(dev); Yb = Yb.to(dev)
            # M = None if M is None else M.to(dev)

            optimizer.zero_grad()
            S, bias = model(Xs, Zb)           # (B, K), (B, K)
            loss = loss_f(S, bias, Yb)

            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # what is this
            optimizer.step()

            bs = Xs.size(0)
            running += loss.item() * bs
            nobs += bs

        if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == epochs):
            print(f"Epoch {epoch:5d}/{epochs} | Train Loss: {running / max(nobs,1):.4f}")


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, dev: Optional[torch.device] = None,
            bias_model: bool = False) -> np.ndarray:
    dev = dev or device()
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=dev)
    if bias_model:
        outputs, _ = model(X_t, None)
    else:
        outputs = model(X_t)
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


def per_site_auc(
    y_true: pd.DataFrame,
    y_score: np.ndarray
    # sites: List[str]
) -> Dict[str, float]:
    
    scores: Dict[str, float] = {}


    sites = y_true.index.tolist()

    for i, site in enumerate(sites):
        try:
            auc = roc_auc_score(
                y_true.iloc[i].values,
                y_score[i, :]
            )
        except ValueError:
            auc = np.nan

        scores[site] = auc

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

    avg_auc_site = per_site_auc(Y_te[species], scores)
    avg_auc_site = np.nanmean(list(avg_auc_site.values()))

    # save model
    model_path = os.path.join(output_dir, f"deepmaxent_{name}_{region}{group}.pt")
    torch.save(model, model_path)

    return avg_auc, aucs, avg_auc_site, model_path, scaler_path

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
        dev: Optional[torch.device] = None,
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

    avg_auc_site = per_site_auc(Y_pa_te_df[species], scores)
    avg_auc_site = np.nanmean(list(avg_auc_site.values()))

    model_path = os.path.join(output_dir, f"deepmaxent_DA_{region}{group}.pt")
    torch.save(model, model_path)
    return avg_auc, aucs, avg_auc_site, model_path, scaler_path


def run_experiment_logreg(
    name: str,
    X_train_df: pd.DataFrame, Y_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,  Y_test_df: pd.DataFrame,
    covs: List[str], species: List[str],
    output_dir: str, region: str, group: str,
    criterion: str = 'logreg',       # kept for API symmetry; unused
    verbose: bool = False,
    penalty: str = "l2",
    C: float = 1.0,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    n_jobs: int = -1,
    random_state: int = 42,
) -> Tuple[float, Dict[str, float], str, str]:
    """
    Run an experiment using one binary Logistic Regression per species.

    Returns
    -------
    avg_auc : float
        Mean AUC across species.
    aucs : dict
        Per-species AUCs, keyed by species name.
    model_path : str
        Path to the saved models (a dict of species -> LogisticRegression).
    scaler_path : str
        Path to the saved scaler used for covariate scaling.
    """

    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, f"scaler_{name}_{region}{group}.pkl")

    # ---- scale (fit on training split of this experiment) ----
    X_train_scaled, X_test_scaled, _ = scale_features(
        X_train_df, X_test_df, covs, output_path=scaler_path, verbose=False
    )

    # ---- numpy arrays ----
    X_tr = X_train_scaled[covs].values.astype(np.float32)
    X_te = X_test_scaled[covs].values.astype(np.float32)

    # we keep Y as DataFrame for convenience; values are used per species
    Y_tr_df = Y_train_df[species]
    Y_te_df = Y_test_df[species].copy()

    # ---- train one LogisticRegression per species ----
    models: Dict[str, LogisticRegression] = {}
    scores_mat = np.zeros((len(X_te), len(species)), dtype=np.float32)

    for j, sp in enumerate(species):
        if verbose:
            print(f"Training Logistic Regression for species: {sp}")

        # sklearn expects 1D array of labels
        y_tr = Y_tr_df[sp].values
        # make sure labels are 0/1 ints (or bool)
        # if they are floats in {0.0, 1.0}, this is safe
        y_tr = y_tr.astype(int)

        clf = LogisticRegression(
            penalty=penalty,
            C=C,
            max_iter=max_iter,
            solver=solver,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        clf.fit(X_tr, y_tr)
        models[sp] = clf

        # predict probability of presence (class 1)
        # handle edge case where only one class is present in training
        if len(clf.classes_) == 2:
            prob_pos = clf.predict_proba(X_te)[:, list(clf.classes_).index(1)]
        else:
            # model degenerated to a single class: all probs are 0 or 1
            # choose 1.0 if the single class is 1, else 0.0
            single_class = clf.classes_[0]
            prob_pos = np.full(len(X_te), float(single_class), dtype=np.float32)

        scores_mat[:, j] = prob_pos

    # ---- evaluate ----
    scores_df = pd.DataFrame(scores_mat, columns=species, index=Y_te_df.index)

    # per_species_auc is assumed to work with DataFrames or arrays + species list
    aucs = per_species_auc(Y_te_df, scores_mat, species)
    avg_auc = np.nanmean(list(aucs.values()))

    # # ---- save models ----
    # model_path = os.path.join(output_dir, f"logreg_{name}_{region}{group}.pkl")
    # joblib.dump(models, model_path)

    return avg_auc, aucs, None, None# model_path, scaler_path


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


def run_experiment_tabpfn(
    name: str,
    X_train_df: pd.DataFrame, Y_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,  Y_test_df: pd.DataFrame,
    covs: List[str], species: List[str],
    output_dir: str, region: str, group: str,
    criterion: str = 'tabpfn',    # kept for API symmetry; unused
    verbose: bool = False,
    N_ensemble_configurations: int = 16,
) -> Tuple[float, Dict[str, float], Optional[str], Optional[str]]:
    """
    Run an experiment using one TabPFNClassifier per species (binary task).

    Returns
    -------
    avg_auc : float
        Mean AUC across species.
    aucs : dict
        Per-species AUCs, keyed by species name.
    model_path : Optional[str]
        (Not saved currently; return None for symmetry.)
    scaler_path : Optional[str]
        Path to the saved scaler used for covariate scaling.
    """
    if not HAS_TABPFN:
        raise RuntimeError(
            "TabPFN is not installed. Install it with `pip install tabpfn` "
            "before running TabPFN experiments."
        )

    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, f"scaler_{name}_{region}{group}.pkl")

    # ---- scale (fit on training split of this experiment) ----
    X_train_scaled, X_test_scaled, _ = scale_features(
        X_train_df, X_test_df, covs, output_path=scaler_path, verbose=False
    )

    # ---- numpy arrays ----
    X_tr = X_train_scaled[covs].values.astype(np.float32)
    X_te = X_test_scaled[covs].values.astype(np.float32)

    Y_tr_df = Y_train_df[species]
    Y_te_df = Y_test_df[species].copy()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    scores_mat = np.zeros((len(X_te), len(species)), dtype=np.float32)

    for j, sp in enumerate(species):
        if verbose:
            print(f"Training TabPFN for species: {sp}")

        y_tr = Y_tr_df[sp].values.astype(int)
        unique_labels = np.unique(y_tr)

        # If there is only one class in training, TabPFN cannot train a classifier.
        # Fallback: constant prediction equal to that class.
        if unique_labels.size == 1:
            const_val = float(unique_labels[0])
            scores_mat[:, j] = const_val
            if verbose:
                print(f"  Species {sp}: only one class in train ({const_val}), "
                    "using constant prediction.")
            continue

        # Current TabPFN API – no N_ensemble_configurations, no device kwarg
        clf = TabPFNClassifier()        # uses default v2.5 model, auto-selects GPU/CPU
        clf.fit(X_tr, y_tr)

        # Probability of presence (class 1)
        prob_pos = clf.predict_proba(X_te)[:, 1]
        scores_mat[:, j] = prob_pos.astype(np.float32)



    # ---- evaluate ----
    aucs = per_species_auc(Y_te_df, scores_mat, species)
    avg_auc = np.nanmean(list(aucs.values()))

    # Not saving TabPFN models here (they are heavy & not trivially picklable).
    model_path = None

    return avg_auc, aucs, model_path, scaler_path




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
                add_po_var=False,
                keep_xy=True, 
                index_col = ['siteid']
            )
            
 

            # 2) Split PA → train/test (fixed for all experiments)
            X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te = split_pa_train_test_spatially(
                X_pa, Y_pa, test_frac=TEST_PA_FRACTION, seed=SEED, K = 100
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
            
            # domain_auc = domain_probe(X_po_s, X_pa_tr_s, covs_copy, verbose=True)

            if ADD_PO_VAR:
                # add PO presence covariate
                X_po = X_po.copy()
                X_pa_tr = X_pa_tr.copy()
                X_pa_te = X_pa_te.copy()
                X_po["PO"] = 1.0
                X_pa_tr["PO"] = 0.0
                X_pa_te["PO"] = 0.0
                if "PO" not in covs:
                    covs.append("PO")

            if not KEEP_XY:
                X_po = X_po.drop(columns=["x","y"], errors="ignore")
                X_pa_tr = X_pa_tr.drop(columns=["x","y"], errors="ignore")
                X_pa_te = X_pa_te.drop(columns=["x","y"], errors="ignore")
                covs = [c for c in covs if c not in ["x","y"]]
            
            env_covs = [c for c in covs if c not in ["x", "y"]]

            covs_no_po = [c for c in env_covs if c != "PO"]


            if RUN_PO:
                print("\n--- Running PO-only experiment ---")
                # A) PO-only → PA_test
   
     
                auc_po, aucs_po, auc_po_site, model_po, scaler_po = run_experiment(
                    name="PO_only",
                    X_train_df=X_po,
                    Y_train_df=Y_po,
                    X_test_df=X_pa_te,  # evaluate on PA_test covs
                    Y_test_df=Y_pa_te,  # evaluate on PA_test labels
                    covs=covs_no_po, species=species,
                    output_dir=exp_dir, region=region, group=group
                )
                print(f"[PO-only]   Average AUC on PA_test: {auc_po:.4f}")

            if RUN_PO_LOGREG:
                print("\n--- Running PO-only (LogReg) experiment ---")
                auc_po_logreg, aucs_po_logreg, model_po_logreg, scaler_po_logreg = run_experiment_logreg(
                    name="PO_only_logreg",
                    X_train_df=X_po,
                    Y_train_df=Y_po,
                    X_test_df=X_pa_te,  # evaluate on PA_test covs
                    Y_test_df=Y_pa_te,  # evaluate on PA_test labels
                    covs=covs_no_po, species=species,
                    output_dir=exp_dir, region=region, group=group
                )
                print(f"[PO-only (LogReg)]   Average AUC on PA_test: {auc_po_logreg:.4f}")

            if RUN_PO_W_BIAS:
                print("\n--- Running PO-only (with BIAS) experiment ---")
                # A) PO-only → PA_test
     
                auc_po_bias, aucs_po_bias, auc_po_bias_site, _, _ = run_experiment_bias(
                    name="PO_only",
                    X_train_df=X_po,
                    Y_train_df=Y_po,
                    X_test_df=X_pa_te,  # evaluate on PA_test covs
                    Y_test_df=Y_pa_te,  # evaluate on PA_test labels
                    covs_species=env_covs, 
                    covs_bias=['x', 'y'],
                    species=species,
                    output_dir=exp_dir, region=region, group=group,
                    criterion = ''
                )
                print(f"[PO-only]   Average AUC on PA_test: {auc_po_bias:.4f}")

            if RUN_PA:
                print("\n--- Running PA-only experiment ---")
                
                loss_criterion = 'bce'
                # B) PA_train-only → PA_test
                auc_pa, aucs_pa, auc_pa_site, model_pa, scaler_pa = run_experiment(
                    name="PA_only",
                    X_train_df=X_pa_tr,
                    Y_train_df=Y_pa_tr,
                    X_test_df=X_pa_te,
                    Y_test_df=Y_pa_te,
                    covs=covs_no_po, species=species,
                    output_dir=exp_dir, region=region, group=group,
                    criterion = loss_criterion
                )
                print(f"[PA-only]   Average AUC on PA_test: {auc_pa:.4f}")

            if RUN_PA_TABPFN:
                print("\n--- Running PA-only (TabPFN) experiment ---")
                auc_pa_tabpfn, aucs_pa_tabpfn, model_pa_tabpfn, scaler_pa_tabpfn = run_experiment_tabpfn(
                    name="PA_only_tabpfn",
                    X_train_df=X_pa_tr,
                    Y_train_df=Y_pa_tr,
                    X_test_df=X_pa_te,
                    Y_test_df=Y_pa_te,
                    covs=covs_no_po, species=species,
                    output_dir=exp_dir, region=region, group=group
                )
                print(f"[PA-only (TabPFN)]   Average AUC on PA_test: {auc_pa_tabpfn:.4f}")


            if RUN_PA_LOGREG:
                print("\n--- Running PA-only (LogReg) experiment ---")
                
                loss_criterion = 'logreg'
                # B) PA_train-only → PA_test
                auc_pa_logreg, aucs_pa_logreg, model_pa_logreg, scaler_pa_logreg = run_experiment_logreg(
                    name="PA_only_logreg",
                    X_train_df=X_pa_tr,
                    Y_train_df=Y_pa_tr,
                    X_test_df=X_pa_te,
                    Y_test_df=Y_pa_te,
                    covs=covs_no_po, species=species,
                    output_dir=exp_dir, region=region, group=group,
                    criterion = loss_criterion
                )
                print(f"[PA-only (LogReg)]   Average AUC on PA_test: {auc_pa_logreg:.4f}")



            if RUN_POPA_ENSEMBLE:
                
                print("\n--- Running PO+PA (ENSEMBLE) integration experiment ---")
                auc_mix_e, aucs_mix_e, model_mix_e, scaler_mix_e = run_experiment_popa_ensemble(
                    name="PO_plus_PA_ensemble",
                    X_po_df=X_po,
                    Y_po_df=Y_po,
                    X_pa_tr_df=X_pa_tr,
                    Y_pa_tr_df=Y_pa_tr,
                    X_pa_te_df=X_pa_te,
                    Y_pa_te_df=Y_pa_te,
                    covs=covs_no_po, species=species,
                    output_dir=exp_dir, region=region, group=group,
                    epochs=EPOCHS,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    w_po=.5,
                    w_pa=.5
                )
                print(f"[ENSEMBLE PO+PA]     Average AUC on PA_test: {auc_mix_e:.4f}")

                

            if RUN_POPA:
                # covs for POPA
                print("\n--- Running PO+PA integration experiment ---")
                print('The covariates used are:', covs)



                # C) PO + PA_train → PA_test
                X_mix = pd.concat([X_po, X_pa_tr], axis=0, ignore_index=True)
                Y_mix = pd.concat([Y_po, Y_pa_tr], axis=0, ignore_index=True)


                auc_mix, aucs_mix, auc_mix_site, model_mix, scaler_mix = run_experiment(
                    name="PO_plus_PA",
                    X_train_df=X_mix,
                    Y_train_df=Y_mix,
                    X_test_df=X_pa_te,
                    Y_test_df=Y_pa_te,
                    covs=env_covs, species=species,
                    output_dir=exp_dir, region=region, group=group,
                    criterion = 'bce'
                )

                
                print(f"[PO+PA]     Average AUC on PA_test: {auc_mix:.4f}")


            ### Filtering based on PA-PO difference (Done for methods that join PO and PA, so from 3rd on)
            if RUN_IMPUTED_POPA:
                keep_po, I, I_round = po_inputer(X_po, Y_po, X_pa_tr, Y_pa_tr, covs_no_po, species)
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
                        covs=env_covs, species=species,
                        output_dir=exp_dir, region=region, group=group,
                        criterion = 'bce'
                    )
                print(f"[PO+PA (IMPUTED)] Average AUC on PA_test: {auc_mix_imp:.4f}")


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
                    covs=env_covs, species=species,
                    output_dir=exp_dir, region=region, group=group,
                    epochs=EPOCHS,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    w_po=w_po,
                    w_pa=w_pa
                )
                print(f"[Weighted PO+PA]     Average AUC on PA_test: {auc_mix_w:.4f}")
            
            if RUN_POPA_SMOOTHED:
                print("\n--- Running Smoothed PO+PA integration experiment ---")

                auc_mix_s, aucs_mix_s, auc_mix_s_site, model_mix_s, scaler_mix_s = run_experiment_popa_smoothed(
                    name="PO_plus_PA_smoothed",
                    X_po_df=X_po,
                    Y_po_df=Y_po,
                    X_pa_tr_df=X_pa_tr,
                    Y_pa_tr_df=Y_pa_tr,
                    X_pa_te_df=X_pa_te,
                    Y_pa_te_df=Y_pa_te,
                    covs=env_covs, species=species,
                    output_dir=exp_dir, region=region, group=group,
                    epochs=EPOCHS,
                    lr=LR,
                    batch_size=BATCH_SIZE,
             
                )
                print(f"[Smoothed PO+PA]     Average AUC on PA_test: {auc_mix_s:.4f}")

            if RUN_POPA_SMOOTHED_W_PRIOR:
                print("\n--- Running Smoothed PO+PA with Prior integration experiment ---")

                auc_mix_sp, aucs_mix_sp, auc_mix_sp_site, model_mix_sp, scaler_mix_sp = run_experiment_popa_smoothed_w_prior(
                    name="PO_plus_PA_smoothed_with_prior",
                    X_po_df=X_po,
                    Y_po_df=Y_po,
                    X_pa_tr_df=X_pa_tr,
                    Y_pa_tr_df=Y_pa_tr,
                    X_pa_te_df=X_pa_te,
                    Y_pa_te_df=Y_pa_te,
                    covs=env_covs, species=species,
                    output_dir=exp_dir, region=region, group=group,
                    epochs=EPOCHS,
                    lr=LR,
                    batch_size=BATCH_SIZE,
             
                )
                print(f"[Smoothed with Prior PO+PA]     Average AUC on PA_test: {auc_mix_sp:.4f}")


            if RUN_POPA_BIAS:
                auc_mix_b, aucs_mix_b, auc_mix_b_site, model_mix_b, scaler_mix_b = run_experiment_popa_bias(
                    name="PO_plus_PA_bias",
                    X_po_df=X_po,
                    Y_po_df=Y_po,
                    X_pa_tr_df=X_pa_tr,
                    Y_pa_tr_df=Y_pa_tr,
                    X_pa_te_df=X_pa_te,
                    Y_pa_te_df=Y_pa_te,
                    covs=env_covs,  covs_bias=["x", "y"], species=species,
                    output_dir=exp_dir, region=region, group=group,
                    epochs=EPOCHS,
                    lr=LR,
                    batch_size=BATCH_SIZE,
             
                )
                print(f"[Bias PO+PA]     Average AUC on PA_test: {auc_mix_b:.4f}")

            row = {
                "region": region,
                "group": group or "(all)",
                "TEST_PA_FRACTION": TEST_PA_FRACTION,
            }
            if RUN_PO:
                row["AUC_PO_only"] = float(np.round(auc_po, 4))
                row["AUC_PO_only_site"] = float(np.round(auc_po_site, 4))
            if RUN_PO_LOGREG:
                row["AUC_PO_only_logreg"] = float(np.round(auc_po_logreg, 4))
            if RUN_PO_W_BIAS:
                row["AUC_PO_w_bias"] = float(np.round(auc_po_bias, 4))
                row["AUC_PO_w_bias_site"] = float(np.round(auc_po_bias_site, 4))
            if RUN_PA:
                row["AUC_PA_only"] = float(np.round(auc_pa, 4))
                row["AUC_PA_only_site"] = float(np.round(auc_pa_site, 4))
            if RUN_PA_LOGREG:
                row["AUC_PA_only_logreg"] = float(np.round(auc_pa_logreg, 4))
            if RUN_PA_TABPFN:
                row["AUC_PA_only_tabpfn"] = float(np.round(auc_pa_tabpfn, 4))
            if RUN_POPA:
                row["AUC_PO_plus_PA"] = float(np.round(auc_mix, 4))
                row["AUC_PO_plus_PA_site"] = float(np.round(auc_mix_site, 4))
            if RUN_IMPUTED_POPA:
                row["AUC_PO_plus_PA_imputed"] = float(np.round(auc_mix_imp, 4))
            if RUN_POPA_ENSEMBLE:
                row["AUC_PO_plus_PA_ensemble"] = float(np.round(auc_mix_e, 4))
            if RUN_POPA_WEIGHTED:
                row["AUC_Weighted_PO_plus_PA"] = float(np.round(auc_mix_w, 4))
            if RUN_POPA_SMOOTHED:
                row["AUC_Smoothed_PO_plus_PA"] = float(np.round(auc_mix_s, 4))
                row["AUC_Smoothed_PO_plus_PA_site"] = float(np.round(auc_mix_s_site, 4))
            if RUN_POPA_SMOOTHED_W_PRIOR:
                row["AUC_Smoothed_with_Prior_PO_plus_PA"] = float(np.round(auc_mix_sp, 4))
                row["AUC_Smoothed_with_Prior_PO_plus_PA_site"] = float(np.round(auc_mix_sp_site, 4))
            if RUN_POPA_BIAS:
                row["AUC_PO_plus_PA_bias"] = float(np.round(auc_mix_b, 4))
                row["AUC_PO_plus_PA_bias_site"] = float(np.round(auc_mix_b_site, 4))

            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    print("\n=== Summary (Average AUC on shared PA_test) ===")
    print(summary.to_string(index=False))
    # print(summary[['AUC_PO_only','AUC_PO_w_bias']].mean())
    summary.to_csv('output/data_integration_results.csv')

    elapsed = time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
