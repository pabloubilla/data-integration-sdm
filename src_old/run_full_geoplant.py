# Full version has a different representation for Y

# TODO: Work on this, adapt bias to work well, it still missing coordinates
import os
import random
from typing import List, Tuple, Dict, Optional
from time import time
import tqdm

import wandb

import numpy as np
import pandas as pd
import pickle

from src_old.load_data import build_paths_nceas, load_po_pa_nceas, load_po_pa_geoplant, load_po_pa_geoplant_full
from src_old.utils import safe_reindex_columns, split_pa_train_test_spatially, scale_features

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
# from k_means_constrained import KMeansConstrained
from src_old.model_training import smooth_targets_v3, device, XZYDataset, XYDataset, set_all_seeds

# --- your models & loss
from src_old.models import DeepMaxEntModel, DeepMaxEntLoss, SDMWithBias, deepmaxent_loss_w_bias, IntegratedLoss, BalancedBCELoss


# General Experiment settings
ADD_PO_VAR = False            # if True, add PO indicator covariate (it's like a Tsource indicator)
BIAS_MODEL = False            # set True for per-plot bias
KEEP_XY = True  # if True, keep x,y in covariates
TEST_PA_FRACTION = 0.3        # PA split: test fraction
SUBSAMPLE_PO = 1
# FILTER_PO = True

# Model / training
HIDDEN_SIZE = 250
# HIDDEN_BIAS_SIZE = 3000
HIDDEN_LAYERS = 2
LR = 1e-3 # 1e-3
LR_PO = 0.00005 #1e-3
EPOCHS = 30
BATCH_SIZE = 500
WEIGHT_DECAY = 3e-4
MAX_BATCH_PERCENTAGE = 1
PRINT_EVERY = 1
SEED = 40


RUN_PO = False
RUN_PA = True
RUN_POPA_WEIGHTED = True
RUN_PO_LOGREG = False
RUN_PO_W_BIAS = False
RUN_PA_LOGREG = False
RUN_POPA, ADD_INTERACTIONS = False, False
RUN_IMPUTED_POPA = False
RUN_POPA_ENSEMBLE = False
RUN_POPA_SMOOTHED = False
RUN_POPA_SMOOTHED_W_PRIOR = False
RUN_POPA_BIAS = False
RUN_POPA_OMISSION_BIAS = False
 
RUN_PA_TABPFN = False

WANDB_KEY = "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"





# # =========================
# # Train / Eval helpers
# # =========================
# def build_model(input_size: int, output_size: int, bias: bool, num_plots: Optional[int] = None) -> nn.Module:
#     if bias:
#         if num_plots is None:
#             raise ValueError("num_plots required for bias model.")
#         return DeepMaxEntPlotBias(
#             input_size=input_size,
#             hidden_size=HIDDEN_SIZE,
#             output_size=output_size,
#             hidden_nbr=HIDDEN_LAYERS,
#             num_plots=num_plots
#         )
#     else:
#         return DeepMaxEntModel(
#             input_size=input_size,
#             hidden_size=HIDDEN_SIZE,
#             output_size=output_size,
#             hidden_nbr=HIDDEN_LAYERS
#         )

class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, X, y_lists):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = [torch.tensor(l, dtype=torch.int64) for l in y_lists]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]
def collate_multilabel(batch, num_classes: int):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)  # [B, D]

    B = len(ys)
    y = torch.zeros((B, num_classes), dtype=torch.float32)

    for i, lbls in enumerate(ys):
        if lbls.numel() > 0:
            y[i, lbls] = 1.0

    return x, y


class MultiLabelXZDataset(torch.utils.data.Dataset):
    """
    Sparse multilabel dataset with 2 inputs:
      - X: float features
      - Z: float bias features (Fourier features, coords, etc.)
      - Y: list-of-lists of class indices
    """
    def __init__(self, X, Z, y_lists):
        assert len(X) == len(Z) == len(y_lists)
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Z = torch.as_tensor(Z, dtype=torch.float32)
        self.Y = [torch.tensor(l, dtype=torch.int64) for l in y_lists]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Z[i], self.Y[i]


def collate_multilabel_xz(batch, num_classes: int):
    xs, zs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)  # [B, Dx]
    z = torch.stack(zs, dim=0)  # [B, Dz]

    B = len(ys)
    y = torch.zeros((B, num_classes), dtype=torch.float32)
    for i, lbls in enumerate(ys):
        if lbls.numel() > 0:
            y[i, lbls] = 1.0

    return x, z, y


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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if criterion == 'bce':
        loss_f = torch.nn.BCEWithLogitsLoss()
    elif criterion == 'deepmaxent':
        loss_f = DeepMaxEntLoss()

    model.train()
    for epoch in tqdm.tqdm(range(1, epochs + 1)):
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)

            optimizer.zero_grad()
            outputs = model(xb)
            # if criterion == 'bce':
            #     outputs = torch.sigmoid(outputs)
            loss = loss_f(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == epochs):
            avg_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch:5d}/{epochs} | Train Loss: {avg_loss:.4f}")




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
                # print(f"Sample scores for species {sp}: ", predicted_scores[:5])
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
    # proportion_po = len(X_po_s) / (len(X_po_s) + len(X_pa_tr_s))
    # proportion_pa = len(X_pa_tr_s) / (len(X_po_s) + len(X_pa_tr_s))
    # batch_size_po = max(1, int(batch_size*proportion_po))
    # batch_size_pa = max(1, int(batch_size*proportion_pa))
    # print('Batch for PO: ', batch_size_po)
    # print('Batch for PA: ', batch_size_pa)
    batch_size_pa = min(batch_size, int(len(X_pa_tr_s) * MAX_BATCH_PERCENTAGE))
    proportion_batch = batch_size / len(X_pa_tr_s)
    batch_size_po = max(1, int(len(X_po_s) * proportion_batch))
    print('Batch for PA: ', batch_size_pa)
    print('Batch for PO: ', batch_size_po)

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


def run_experiment_popa_omission_bias(
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

    # -------------------------
    # 2) Scale bias covs, then expand with Fourier features
    # -------------------------
    scaler_bias = MinMaxScaler().fit(pd.concat([X_po_df[covs_bias], X_pa_tr_df[covs_bias]], axis=0))

    # scaled raw bias covs
    X_po_bias_scaled = scaler_bias.transform(X_po_df[covs_bias])
    X_pa_tr_bias_scaled = scaler_bias.transform(X_pa_tr_df[covs_bias])
    X_pa_te_bias_scaled = scaler_bias.transform(X_pa_te_df[covs_bias])

    # Fourier-expanded bias covs (this is what the bias net will see)
    X_po_bias_ff = _fourier_features(
        X_po_bias_scaled, n_freqs=6, include_input=True
    )


    batch_size = min(batch_size, int(len(X_pa_tr_s) * MAX_BATCH_PERCENTAGE))

   

    # model = deepmaxent_model(input_size=len(covs), hidden_size=HIDDEN_SIZE, output_size=len(species), hidden_nbr=HIDDEN_LAYERS)

    hidden_species_arquitecture = (HIDDEN_SIZE,) * HIDDEN_LAYERS

    model = SDMWithBias(
        n_species_covariates=len(covs),
        n_species=len(species),
        n_bias_covariates=X_po_bias_ff.shape[1],  # UPDATED,
        hidden_species=hidden_species_arquitecture,
        hidden_bias=(100, 100),
        output_bias=1,
    )

    # criterion_species = deepmaxent_loss()

    # loaders separately (proportional sizes)
    # proportion_po = len(X_po_s) / (len(X_po_s) + len(X_pa_tr_s))
    # proportion_pa = len(X_pa_tr_s) / (len(X_po_s) + len(X_pa_tr_s))
    # batch_size_po = max(1, int(batch_size*proportion_po))
    # batch_size_pa = max(1, int(batch_size*proportion_pa))
    # print('Batch for PO: ', batch_size_po)
    # print('Batch for PA: ', batch_size_pa)

    batch_size_pa = min(batch_size, int(len(X_pa_tr_s) * MAX_BATCH_PERCENTAGE))
    proportion_batch = batch_size / len(X_pa_tr_s)
    batch_size_po = max(1, int(len(X_po_s) * proportion_batch))
    print('Batch for PA: ', batch_size_pa)
    print('Batch for PO: ', batch_size_po)
    # po_ds = XYDataset(X_po_s[covs].values.astype(np.float32), Y_po_df[species].values.astype(np.float32))
    po_ds = XZYDataset(X_po_s[covs].values.astype(np.float32), X_po_bias_ff, Y_po_df[species].values.astype(np.float32))
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
        loss_fn = IntegratedLoss(w_pa = 1, po_loss=False)
        # loss_fn = torch.nn.BCEWithLogitsLoss()
        for epoch in tqdm.tqdm(range(1, epochs + 1)):
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

                # transform bias to probabilities with sigmoid (is dim of species)
                bias_prob_po = torch.sigmoid(bias_pred_po)
                # bias_prob_po = 0.05
                # bias_prob_po = 0.1

                ### TODO: check the methodology here
                # smooth y_po with bias_prob_po using the formula bias_prob_po * yb_po  / (bias_prob_po * yb_po + (1 - bias_prob_po))
                yb_pred_po = score_po.sigmoid()
                
                yb_po_smooth = bias_prob_po * yb_pred_po  / (bias_prob_po * yb_pred_po + (1 - bias_prob_po))
                # for values where yb_po is 1, keep as 1
                yb_po_smooth = torch.where(yb_po == 1, torch.ones_like(yb_po_smooth), yb_po_smooth)


                # loss = loss_fn(score_po, yb_po_smooth) + loss_fn(score_pa, yb_pa)
                loss = loss_fn(score_po, score_pa, 0, yb_po_smooth, yb_pa)

                loss.backward()
                optimizer.step()
                running_loss += loss.item() * (xb_po.size(0) + xb_pa.size(0))

            if epoch % PRINT_EVERY == 0:
                avg_loss = running_loss / (len(po_loader.dataset) + len(pa_loader.dataset))
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")


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




def _fourier_features(
    X: np.ndarray,
    n_freqs: int = 5,
    include_input: bool = True,
) -> np.ndarray:
    """
    Simple Fourier features: [x, sin(2^k x), cos(2^k x)] for k=0..n_freqs-1, per dimension.
    X is expected to be scaled already (roughly zero mean, unit var).
    """
    feats = []
    if include_input:
        feats.append(X)

    # Frequencies: 1,2,4,... (works well as a simple default)
    for k in range(n_freqs):
        w = 2.0 ** k
        feats.append(np.sin(w * X))
        feats.append(np.cos(w * X))

    return np.concatenate(feats, axis=1).astype(np.float32)


def run_experiment_popa_bias(
    name: str,
    X_po_df: pd.DataFrame, Y_po_df: pd.DataFrame,
    X_pa_tr_df: pd.DataFrame, Y_pa_tr_df: pd.DataFrame,
    X_pa_te_df: pd.DataFrame, Y_pa_te_df: pd.DataFrame,
    covs: List[str], covs_bias: List[str], area_cov: str, species: List[str],
    output_dir: str, region: str, group: str,
    epochs: int = 300, lr: float = 1e-4,
    batch_size: int = 256, w_po: float = .5, w_pa: float = .5,
    # NEW: Fourier feature controls (safe defaults)
    bias_fourier_freqs: int = 10,
    bias_fourier_include_raw: bool = True,
    print_every: int = 10,
):
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, f"scaler_{name}_{region}{group}.pkl")


    # -------------------------
    # 1) Scale species covs (fit on PO + PA train)
    # -------------------------
    scaler = StandardScaler().fit(pd.concat([X_po_df[covs], X_pa_tr_df[covs]], axis=0))

    X_po_s = X_po_df.copy()
    X_pa_tr_s = X_pa_tr_df.copy()
    X_pa_te_s = X_pa_te_df.copy()
    X_po_s[covs] = scaler.transform(X_po_s[covs])
    X_pa_tr_s[covs] = scaler.transform(X_pa_tr_s[covs])
    X_pa_te_s[covs] = scaler.transform(X_pa_te_s[covs])

    # -------------------------
    # 2) Scale bias covs, then expand with Fourier features
    # -------------------------
    # Should Standard or MinMax make a difference here?
    scaler_bias = MinMaxScaler().fit(pd.concat([X_po_df[covs_bias], X_pa_tr_df[covs_bias]], axis=0))

    # scaled raw bias covs
    X_po_bias_scaled = scaler_bias.transform(X_po_df[covs_bias])
    X_pa_tr_bias_scaled = scaler_bias.transform(X_pa_tr_df[covs_bias])
    X_pa_te_bias_scaled = scaler_bias.transform(X_pa_te_df[covs_bias])

    # Fourier-expanded bias covs (this is what the bias net will see)
    X_po_bias_ff = _fourier_features(
        X_po_bias_scaled, n_freqs=bias_fourier_freqs, include_input=bias_fourier_include_raw
    )
    X_pa_tr_bias_ff = _fourier_features(
        X_pa_tr_bias_scaled, n_freqs=bias_fourier_freqs, include_input=bias_fourier_include_raw
    )
    X_pa_te_bias_ff = _fourier_features(
        X_pa_te_bias_scaled, n_freqs=bias_fourier_freqs, include_input=bias_fourier_include_raw
    )

    # keep original dataframes around for plotting coords etc.
    # (we only need numpy arrays for datasets)

    # -------------------------
    # 3) Model (only change: n_bias_covariates)
    # -------------------------


    hidden_species_arquitecture = (HIDDEN_SIZE,) * HIDDEN_LAYERS

    model = SDMWithBias(
        n_species_covariates=len(covs),
        n_species=len(species),
        n_bias_covariates=X_po_bias_ff.shape[1],  # UPDATED
        hidden_species=hidden_species_arquitecture,
        hidden_bias=(500, 500), 
        output_bias=1,
    )

    # -------------------------
    # 4) Loaders (same idea as before)
    # -------------------------
    # proportion_po = len(X_po_s) / (len(X_po_s) + len(X_pa_tr_s))
    # proportion_pa = len(X_pa_tr_s) / (len(X_po_s) + len(X_pa_tr_s))
    # batch_size_po = max(1, int(batch_size * proportion_po))
    # batch_size_pa = max(1, int(batch_size * proportion_pa))
    
    # -------------------------
    # 4) Loaders (SPARSE labels)
    # -------------------------
    batch_size_pa = min(batch_size, int(len(X_pa_tr_s) * MAX_BATCH_PERCENTAGE))
    proportion_batch = batch_size / max(len(X_pa_tr_s), 1)
    batch_size_po = max(1, int(len(X_po_s) * proportion_batch))

    print('Batch for PO: ', batch_size_po)
    print('Batch for PA: ', batch_size_pa)

    # PO: (X, Z_bias_ff, sparse Y list-of-lists)
    po_ds = MultiLabelXZDataset(
        X_po_s[covs].values.astype(np.float32),
        X_po_bias_ff.astype(np.float32),
        Y_po_df  # <-- this must be list-of-lists (same as your run_experiment_popa uses)
    )

    # PA: (X, sparse Y list-of-lists)  [area NOT USED]
    pa_ds = MultiLabelDataset(
        X_pa_tr_s[covs].values.astype(np.float32),
        Y_pa_tr_df  # <-- list-of-lists
    )

    po_loader = DataLoader(
        po_ds,
        batch_size=batch_size_po,
        shuffle=True,
        drop_last=False,
        collate_fn=lambda b: collate_multilabel_xz(b, num_classes=len(species))
    )

    pa_loader = DataLoader(
        pa_ds,
        batch_size=batch_size_pa,
        shuffle=True,
        drop_last=False,
        collate_fn=lambda b: collate_multilabel(b, num_classes=len(species))
    )
    # -------------------------
    # 5) Train (unchanged)
    # -------------------------
    def train_popa_model_bias(
        model: torch.nn.Module,
        po_loader: DataLoader,
        pa_loader: DataLoader,
        po_ds: Dataset,
        pa_ds: Dataset,
        criterion_species: torch.nn.Module,
        epochs: int = 300,
        lr: float = 1e-4,
        dev: Optional[torch.device] = None,
        w_po: float = .5,
        w_pa: float = .5,
    ):
        dev = dev or device()
        model.to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-4)

        model.train()


        loss_fn = IntegratedLoss(w_pa=2, add_area=False, po_loss='deepmaxent')

        for epoch in tqdm.tqdm(range(1, epochs + 1), desc="Training epochs"):
            running_loss = 0.0

            for (xb_po, zb_po, yb_po), (xb_pa, yb_pa) in zip(po_loader, pa_loader):
                xb_po = xb_po.to(dev)
                zb_po = zb_po.to(dev)
                yb_po = yb_po.to(dev)

                xb_pa = xb_pa.to(dev)
                yb_pa = yb_pa.to(dev)

                optimizer.zero_grad()

                score_po, bias_pred_po = model(xb_po, zb_po)
                score_pa, _ = model(xb_pa)

                # area not used
                area_b_pa = None

                loss = loss_fn(score_po, score_pa, bias_pred_po, yb_po, yb_pa, area_b_pa)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * (xb_po.size(0) + xb_pa.size(0))

            if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
                avg_loss = running_loss / (len(po_loader.dataset) + len(pa_loader.dataset))
                print(f"Epoch {epoch:5d}/{epochs} | Train Loss: {avg_loss:.4f}")


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


    # -------------------------
    # 6) Evaluate (SPARSE)
    # -------------------------
    X_te_np = X_pa_te_s[covs].values.astype(np.float32)
    scores = predict(model, X_te_np, dev=device(), bias_model=True)

    aucs = aucs_per_species_sparse(scores, Y_pa_te_df, species_len=len(species))  # Y_pa_te_df is list-of-lists
    avg_auc = np.nanmean(list(aucs.values()))
    avg_auc_site = np.nan

    # -------------------------
    # 7) Predict bias on PO points (UPDATED: use Fourier features)
    # -------------------------
    X_po_np = X_po_s[covs].values.astype(np.float32)
    X_po_tensor = torch.tensor(X_po_np, dtype=torch.float32, device=device())

    X_po_bias_tensor = torch.tensor(X_po_bias_ff, dtype=torch.float32, device=device())
    scores_t, bias_scores_t = model(X_po_tensor, X_po_bias_tensor)
    bias_scores = bias_scores_t.detach().cpu().numpy()

    # max, min, mean and median
    print(f"Bias scores - max: {bias_scores.max():.4f}, min: {bias_scores.min():.4f}, mean: {bias_scores.mean():.4f}, median: {np.median(bias_scores):.4f}")

    # # plot the bias (same as before)
    import matplotlib.pyplot as plt
    # sigmoid_bias_scores = 1 / (1 + np.exp(-bias_scores))  # convert logits to probabilities for better visualizations
    sigmoid_bias_scores = bias_scores  # if bias_scores are already probabilities (e.g., if output_bias=1 and no activation), use directly
    plt.figure(figsize=(8, 6))
    plt.scatter(X_po_s['x'], X_po_s['y'], c=sigmoid_bias_scores, cmap='viridis', s=.1, alpha=0.7)
    plt.colorbar(label='Bias Score (Probability)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Predicted Bias Scores at PO Locations')
    # add parameters to filename (bias arquitecture)
    # bias_shape = '_'.join([str(h) for h in model.hidden_bias])
    # file_name = f"bias_map_{name}_{region}{group}_biasnet{bias_shape}.png"
    file_name = f"bias_map_{name}_{region}{group}.png"
    plt.savefig(file_name)

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
    nan_count = 0
    for i, sp in enumerate(species):
        try:
            # if only one class keep as nan
            if len(np.unique(y_true[sp].values)) < 2:
                auc = np.nan
                nan_count += 1
            else:
                auc = roc_auc_score(y_true[sp].values, y_score[:, i])
        except ValueError:
            auc = np.nan
        scores[sp] = auc
    print(f"Warning: {nan_count} species had only one class in true labels, AUC set to NaN for these.")
    return scores


def per_site_auc(
    y_true: pd.DataFrame,
    y_score: np.ndarray
    # sites: List[str]
) -> Dict[str, float]:
    
    scores: Dict[str, float] = {}


    sites = y_true.index.tolist()

    nan_count = 0
    for i, site in enumerate(sites):
        try:
            if len(np.unique(y_true.iloc[i].values)) < 2:
                auc = np.nan
                nan_count += 1
            else:
                auc = roc_auc_score(
                    y_true.iloc[i].values,
                    y_score[i, :]
                )
        except ValueError:
            auc = np.nan

        scores[site] = auc

    print(f"Warning: {nan_count} sites had only one class in true labels, AUC set to NaN for these.")
    return scores

def aucs_per_species_sparse(
    scores, 
    Y_te, 
    species_len, 
):
        # --- convert list-of-lists -> multi-hot matrix for eval only ---
    Y_te_mat = np.zeros((len(Y_te), species_len), dtype=np.uint8)
    for i, lbls in enumerate(Y_te):
        if lbls:  # non-empty list
            Y_te_mat[i, np.asarray(lbls, dtype=np.int64)] = 1

    print(Y_te_mat[:10,:10])  # print a subset of the multi-hot labels for verification

    # --- compute per-class AUC (column-wise) ---
    aucs = {}
    how_many_in_test = 0
    for j in range(species_len):
        yj = Y_te_mat[:, j]
        # AUC is undefined if only one class present (all 0s or all 1s)
        if yj.min() == yj.max():
            aucs[j] = np.nan
        else:
            aucs[j] = roc_auc_score(yj, scores[:, j])
            how_many_in_test += 1

    print(f"Computed AUC for {how_many_in_test} species (out of {species_len}).")
    return aucs

# =========================
# Experiment runner
# =========================
def run_experiment(
    name: str,
    X_train_df: pd.DataFrame, Y_tr: pd.DataFrame,
    X_test_df: pd.DataFrame,  Y_te: pd.DataFrame,
    covs: List[str], species: List[str],
    output_dir: str, region: str, group: str,
    criterion = 'deepmaxent',
    verbose: bool = False, 
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
    weight_decay: float = WEIGHT_DECAY,
):
    
    run = wandb.init(project="sdm-integration",
        entity = "pabloubilla-inria",         
        name=name, reinit=True, config={
        "region": region,
        "group": group,
        "criterion": criterion,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
    })

    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, f"scaler_{name}_{region}{group}.pkl")

    # scale (fit on training split of this experiment)
    X_train_scaled, X_test_scaled, _ = scale_features(
        X_train_df, X_test_df, covs, output_path=scaler_path, verbose=False
    )

    # sample for each
    print(f"Training samples: {len(X_train_scaled)}, Testing samples: {len(X_test_scaled)}")
    # heads
    print("Training covariates sample:")
    print(X_train_scaled[covs].head())
    print("Testing covariates sample:")
    print(X_test_scaled[covs].head())



    # numpy arrays
    X_tr = X_train_scaled[covs].values.astype(np.float32)
    # Y_tr = Y_train_df[species].values.astype(np.float32)
    X_te = X_test_scaled[covs].values.astype(np.float32)
    # Y_te = Y_test_df[species].copy()


    # dataloader
    batch_size_consolidated = max(1, min(batch_size, int(len(X_tr) * MAX_BATCH_PERCENTAGE)))
    # plots_idx = np.arange(len(X_tr))  # dummy plot indices for bias model
    # train_ds = XYDataset(X_tr, Y_tr, plots_idx)
    # train_loader = DataLoader(train_ds, batch_size=batch_size_consolidated, shuffle=True, drop_last=False)
    train_ds = MultiLabelDataset(X_tr, Y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size_consolidated, shuffle=True, drop_last=False, 
                              collate_fn=lambda b: collate_multilabel(b, num_classes=len(species)))

    # model
    model = DeepMaxEntModel(input_size=len(covs), hidden_size=HIDDEN_SIZE, output_size=len(species), hidden_nbr=HIDDEN_LAYERS)


    # split in train and validation
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_tr, Y_tr, test_size=0.2, random_state=42)

    val_ds = MultiLabelDataset(X_val, Y_val)
    val_loader = DataLoader(val_ds, batch_size=batch_size_consolidated, shuffle=True, drop_last=True,
                            collate_fn=lambda b: collate_multilabel(b, num_classes=len(species)))
                            
                            
    #build_model(input_size=len(covs), output_size=len(species), bias=BIAS_MODEL, num_plots=len(X_tr))
    def train_model_(
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
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


        C = len(species)          # number of classes
        N = len(Y_tr)             # Y_tr is list-of-lists of class indices

        # Flatten all labels into one 1D array (memory is proportional to #positive labels, not N*C)
        flat = np.fromiter(
            (lbl for row in Y_tr for lbl in row),
            dtype=np.int64
        )

        pos_counts = np.bincount(flat, minlength=C).astype(np.float32)   # [C]
        neg_counts = (N - pos_counts).astype(np.float32)

        # avoid div by zero for classes with 0 positives
        pos_weight = neg_counts / (pos_counts + 1e-6)

        # optional: clamp to avoid insane weights for ultra-rare classes
        pos_weight = np.clip(pos_weight, 1.0, 200.0)

        pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32, device=dev)

        print("Pos weights for loss:", pos_weight)

        if criterion == 'deepmaxent':
            loss_f = DeepMaxEntLoss()
        if criterion == 'bce':
            # loss_f = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
            loss_f = BalancedBCELoss()
            
        model.train()
        step = 0
        for epoch in tqdm.tqdm(range(1, epochs + 1)):
            running_loss_train = 0.0
            for xb, yb in train_loader:
                step += len(xb)
                xb = xb.to(dev)
                yb = yb.to(dev)

    
                optimizer.zero_grad()
                outputs = model(xb)
                # if criterion == 'bce':
                #     outputs = torch.sigmoid(outputs)
                loss = loss_f(outputs, yb)
                loss.backward()
                optimizer.step()
                running_loss_train += loss.item() * xb.size(0)

                run.log({"train_loss": loss.item(),
                         "epoch": epoch    
                         }, step=step)
            run.log({"epoch_loss": running_loss_train / len(train_loader.dataset)}, step=step)

            # ADD VALIDATION LOSS
            running_loss_val = 0.0
            for xb_val, yb_val in val_loader:
                xb_val = xb_val.to(dev)
                yb_val = yb_val.to(dev)
                with torch.no_grad():
                    outputs_val = model(xb_val)
                    loss_val = loss_f(outputs_val, yb_val)
                    running_loss_val += loss_val.item() * xb_val.size(0)
            run.log({"val_loss": running_loss_val / len(val_loader.dataset)}, step=step)


            if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == epochs):
                avg_loss = running_loss_train / len(train_loader.dataset)
                print(f"Epoch {epoch:5d}/{epochs} | Train Loss: {avg_loss:.4f}")


    # train
    train_model_(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        epochs=epochs,
        lr=lr,
        print_every=PRINT_EVERY,
        dev=device(),
        verbose=verbose
    )



    # # evaluate on shared PA_test
    # scores = predict(model, X_te, dev=device())

    # # print a subset of scores

    # # aucs = per_species_auc(Y_te, scores, species)
    # # avg_auc = np.nanmean(list(aucs.values()))


    # N_test = len(Y_te)
    # species_len = len(species)

    # # --- convert list-of-lists -> multi-hot matrix for eval only ---
    # Y_te_mat = np.zeros((N_test, species_len), dtype=np.uint8)
    # for i, lbls in enumerate(Y_te):
    #     if lbls:  # non-empty list
    #         Y_te_mat[i, np.asarray(lbls, dtype=np.int64)] = 1

    # print(Y_te_mat[:10,:10])  # print a subset of the multi-hot labels for verification

    # # --- compute per-class AUC (column-wise) ---
    # aucs = {}
    # how_many_in_test = 0
    # for j, sp in enumerate(species):
    #     yj = Y_te_mat[:, j]
    #     # AUC is undefined if only one class present (all 0s or all 1s)
    #     if yj.min() == yj.max():
    #         aucs[sp] = np.nan
    #     else:
    #         aucs[sp] = roc_auc_score(yj, scores[:, j])
    #         # print(f"AUC for {sp}: {aucs[sp]:.4f}, with {yj.sum()} positives and {len(yj) - yj.sum()} negatives.")
    #         how_many_in_test += 1

    # print(f"Computed AUC for {how_many_in_test} species (out of {len(species)}).")
    scores = predict(model, X_te, dev=device())
    aucs = aucs_per_species_sparse(scores, Y_te, species_len=len(species))

    avg_auc = np.nanmean(list(aucs.values()))
    print("avg_auc:", avg_auc)

    # avg_auc_site = per_site_auc(Y_te[species], scores)
    # avg_auc_site = np.nanmean(list(avg_auc_site.values()))
    avg_auc_site = np.nan



    # save model
    model_path = os.path.join(output_dir, f"deepmaxent_{name}_{region}{group}.pt")
    torch.save(model, model_path)


    # add lengths to config
    run.config.update({
        "train_samples": len(X_tr),
        "test_samples": len(X_te),
        "validation_samples": len(X_val),
        "avg_auc": avg_auc,
        "avg_auc_site": avg_auc_site,
    })

    return avg_auc, aucs, avg_auc_site, model_path, scaler_path

def run_experiment_popa(
    name: str,
    X_po_df: pd.DataFrame, Y_po: pd.DataFrame,
    X_pa_tr_df: pd.DataFrame, Y_pa_tr: pd.DataFrame,
    X_pa_te_df: pd.DataFrame, Y_pa_te: pd.DataFrame,
    covs: List[str], species: List[str],
    output_dir: str, region: str, group: str,
    epochs: int = 300, lr: float = 1e-4,
    batch_size: int = 256, w_po: float = .5, w_pa: float = .5,
    weight_decay: float = WEIGHT_DECAY,
    print_every = PRINT_EVERY
):
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, f"scaler_{name}_{region}{group}.pkl")

    run = wandb.init(project="sdm-integration",
        entity = "pabloubilla-inria",
        name=f"{name}", reinit=True, config={
        "region": region,
        "group": group,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "w_po": w_po,
        "w_pa": w_pa,
        "weight_decay": weight_decay
    })


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
    criterion_species = DeepMaxEntLoss()

    # loaders separately (proportional sizes)
    batch_size_pa = batch_size
    batch_size_po = batch_size
    # proportion_batch = batch_size / len(X_pa_tr_s)
    # batch_size_po = int(len(X_po_s)*proportion_batch)
    print('Batch for PO: ', batch_size_po)
    print('Batch for PA: ', batch_size_pa)
    # po_ds = XYDataset(X_po_s[covs].values.astype(np.float32), Y_po_df[species].values.astype(np.float32))
    # pa_ds = XYDataset(X_pa_tr_s[covs].values.astype(np.float32), Y_pa_tr_df[species].values.astype(np.float32))
    # po_loader = DataLoader(po_ds, batch_size=batch_size_po, shuffle=True, drop_last=True)
    # pa_loader = DataLoader(pa_ds, batch_size=batch_size_pa, shuffle=True, drop_last=True)

    # use sparse datasets for both
    po_ds = MultiLabelDataset(X_po_s[covs].values.astype(np.float32), Y_po)
    pa_ds = MultiLabelDataset(X_pa_tr_s[covs].values.astype(np.float32), Y_pa_tr)

    # sampler_po = RandomSampler(po_ds, replacement=True, num_samples=batch_size_po * 50)
    # sampler_pa = RandomSampler(pa_ds, replacement=True, num_samples=batch_size_pa * 50)

    ## CHECK HOW TO DO RANDOM SAMPLING
    po_loader = DataLoader(po_ds, batch_size=batch_size_po, shuffle=True, drop_last=False,
                            collate_fn=lambda b: collate_multilabel(b, num_classes=len(species)))
    pa_loader = DataLoader(pa_ds, batch_size=batch_size_pa, shuffle=True, drop_last=False,
                            collate_fn=lambda b: collate_multilabel(b, num_classes=len(species)))

    
    #### WEIGHTS
    C = len(species)          # number of classes
    N = len(Y_pa_tr)             # Y_pa_tr is list-of-lists of class indices

    # # Flatten all labels into one 1D array (memory is proportional to #positive labels, not N*C)
    # flat = np.fromiter(
    #     (lbl for row in Y_pa_tr for lbl in row),
    #     dtype=np.int64
    # )

    # pos_counts = np.bincount(flat, minlength=C).astype(np.float32)   # [C]
    # neg_counts = (N - pos_counts).astype(np.float32)

    # # avoid div by zero for classes with 0 positives
    # pos_weight = neg_counts / (pos_counts + 1e-6)

    # # optional: clamp to avoid insane weights for ultra-rare classes
    # pos_weight = np.clip(pos_weight, 1.0, 200.0)

    # pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32, device=device())

    # print("Pos weights for loss:", pos_weight)


    def train_popa_model(
        model: nn.Module,
        po_loader: DataLoader,
        pa_loader: DataLoader,
        criterion_species: nn.Module,
        epochs: int = 300,
        lr: float = 1e-4,
        dev: Optional[torch.device] = None,
        w_po: float = 1,
        w_pa: float = 1,
    ):
        dev = dev or device()
        model.to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        criterion_po = DeepMaxEntLoss()
        # poisson loss
        # criterion_po = torch.nn.PoissonNLLLoss() 
        # criterion_pa = torch.nn.BCELoss()
        criterion_pa = BalancedBCELoss()

        model.train()
        for epoch in tqdm.tqdm(range(1, epochs + 1), desc="Training epochs"):
            running_loss = 0.0
            for (xb_po, yb_po), (xb_pa, yb_pa) in zip(po_loader, pa_loader):
                xb_po = xb_po.to(dev)
                yb_po = yb_po.to(dev)
                xb_pa = xb_pa.to(dev)
                yb_pa = yb_pa.to(dev)

                # multiply yb_pa by pos_weight_t to handle class imbalance
                # yb_pa_weighted = yb_pa 

                optimizer.zero_grad()
                outputs_po= model(xb_po)
                outputs_pa= model(xb_pa)
                
                lambda_pa = torch.exp(outputs_pa)  # convert logits to lambda 
                probs_pa = -torch.expm1(-lambda_pa) + 1e-12  # convert logits to probabilities with numerical stability
                loss_pa = criterion_pa(outputs_pa, yb_pa)
                loss_po = criterion_po(outputs_po, yb_po)

                loss = w_po * loss_po + w_pa * loss_pa

                # loss = w_po * loss_po + w_pa * loss_pa

                loss.backward()
                optimizer.step()
                running_loss += loss.item() * (xb_po.size(0) + xb_pa.size(0))

                run.log({"train_loss": loss.item(),
                         "epoch": epoch
                            }, step=epoch)
                # save loss po and pa
                run.log({"loss_po": loss_po.item(), "loss_pa": loss_pa.item()}, step=epoch)

                # size of each batch po and pa
                run.log({"batch_size_po": xb_po.size(0), "batch_size_pa": xb_pa.size(0)}, step=epoch)

            run.log({"epoch_loss": running_loss / (len(po_loader.dataset) + len(pa_loader.dataset))}, step=epoch)

            if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
                print(f"Epoch {epoch}/{epochs}, Loss: {running_loss / (len(po_loader.dataset) + len(pa_loader.dataset))}")

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
    aucs = aucs_per_species_sparse(scores, Y_pa_te, species_len=len(species))
    avg_auc = np.nanmean(list(aucs.values()))

    # avg_auc_site = per_site_auc(Y_pa_te[species], scores)
    # avg_auc_site = np.nanmean(list(avg_auc_site.values()))
    avg_auc_site = np.nan

    model_path = os.path.join(output_dir, f"deepmaxent_DA_{region}{group}.pt")
    torch.save(model, model_path)
    return avg_auc, aucs, avg_auc_site, model_path, scaler_path


# EXPERIMENTAL VERSION To do Adam per SOURCE
# def run_experiment_popa(
#     name: str,
#     X_po_df: pd.DataFrame, Y_po: pd.DataFrame,
#     X_pa_tr_df: pd.DataFrame, Y_pa_tr: pd.DataFrame,
#     X_pa_te_df: pd.DataFrame, Y_pa_te: pd.DataFrame,
#     covs: List[str], species: List[str],
#     output_dir: str, region: str, group: str,
#     epochs: int = 300, lr: float = 1e-4,
#     batch_size: int = 256, w_po: float = .5, w_pa: float = .5,
#     weight_decay: float = WEIGHT_DECAY,
#     print_every=PRINT_EVERY
# ):

#     os.makedirs(output_dir, exist_ok=True)
#     scaler_path = os.path.join(output_dir, f"scaler_{name}_{region}{group}.pkl")

#     run = wandb.init(
#         project="sdm-integration",
#         entity="pabloubilla-inria",
#         name=f"{name}",
#         reinit=True,
#         config={
#             "region": region,
#             "group": group,
#             "epochs": epochs,
#             "lr": lr,
#             "batch_size": batch_size,
#             "w_po": w_po,
#             "w_pa": w_pa,
#             "weight_decay": weight_decay,
#             "optimizer": "separate_task_adam"
#         }
#     )

#     # ------------------------------------------------------------------
#     # scale (fit on combo PO + PA train)
#     # ------------------------------------------------------------------
#     scaler = StandardScaler().fit(
#         pd.concat([X_po_df[covs], X_pa_tr_df[covs]], axis=0)
#     )

#     X_po_s = X_po_df.copy()
#     X_pa_tr_s = X_pa_tr_df.copy()
#     X_pa_te_s = X_pa_te_df.copy()

#     X_po_s[covs] = scaler.transform(X_po_s[covs])
#     X_pa_tr_s[covs] = scaler.transform(X_pa_tr_s[covs])
#     X_pa_te_s[covs] = scaler.transform(X_pa_te_s[covs])

#     batch_size = min(batch_size, max(1, int(len(X_pa_tr_s) * MAX_BATCH_PERCENTAGE)))

#     model = DeepMaxEntModel(
#         input_size=len(covs),
#         hidden_size=HIDDEN_SIZE,
#         output_size=len(species),
#         hidden_nbr=HIDDEN_LAYERS
#     )
#     criterion_species = DeepMaxEntLoss()

#     # ------------------------------------------------------------------
#     # datasets / loaders
#     # ------------------------------------------------------------------
#     batch_size_pa = batch_size
#     batch_size_po = batch_size

#     print("Batch for PO: ", batch_size_po)
#     print("Batch for PA: ", batch_size_pa)

#     po_ds = MultiLabelDataset(X_po_s[covs].values.astype(np.float32), Y_po)
#     pa_ds = MultiLabelDataset(X_pa_tr_s[covs].values.astype(np.float32), Y_pa_tr)

#     po_loader = DataLoader(
#         po_ds,
#         batch_size=batch_size_po,
#         shuffle=True,
#         drop_last=False,
#         collate_fn=lambda b: collate_multilabel(b, num_classes=len(species))
#     )
#     pa_loader = DataLoader(
#         pa_ds,
#         batch_size=batch_size_pa,
#         shuffle=True,
#         drop_last=False,
#         collate_fn=lambda b: collate_multilabel(b, num_classes=len(species))
#     )

#     # ------------------------------------------------------------------
#     # Separate-momentum Adam:
#     # one Adam state for PO and one Adam state for PA
#     # ------------------------------------------------------------------
#     class SeparateTaskAdam:
#         def __init__(
#             self,
#             params,
#             lr=1e-4,
#             betas=(0.9, 0.999),
#             eps=1e-8,
#             weight_decay=0.0
#         ):
#             self.params = [p for p in params if p.requires_grad]
#             self.lr = lr
#             self.beta1, self.beta2 = betas
#             self.eps = eps
#             self.weight_decay = weight_decay

#             self.state = {}
#             for p in self.params:
#                 self.state[p] = {
#                     "po": {
#                         "step": 0,
#                         "m": torch.zeros_like(p.data),
#                         "v": torch.zeros_like(p.data),
#                     },
#                     "pa": {
#                         "step": 0,
#                         "m": torch.zeros_like(p.data),
#                         "v": torch.zeros_like(p.data),
#                     }
#                 }

#         @torch.no_grad()
#         def zero_grad(self):
#             for p in self.params:
#                 p.grad = None

#         @torch.no_grad()
#         def _task_update(self, p, grad, task_name):
#             """
#             Returns the normalized Adam update for one task, using that task's own moments.
#             """
#             if grad is None:
#                 return torch.zeros_like(p.data)

#             st = self.state[p][task_name]
#             st["step"] += 1

#             m = st["m"]
#             v = st["v"]

#             m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
#             v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)

#             bias_correction1 = 1.0 - self.beta1 ** st["step"]
#             bias_correction2 = 1.0 - self.beta2 ** st["step"]

#             m_hat = m / bias_correction1
#             v_hat = v / bias_correction2

#             update = m_hat / (torch.sqrt(v_hat) + self.eps)
#             return update

#         @torch.no_grad()
#         def step(self, grads_po, grads_pa, w_po=1.0, w_pa=1.0):
#             for p, g_po, g_pa in zip(self.params, grads_po, grads_pa):
#                 if g_po is None and g_pa is None:
#                     continue

#                 # decoupled weight decay
#                 if self.weight_decay > 0:
#                     p.data.mul_(1.0 - self.lr * self.weight_decay)

#                 upd_po = self._task_update(p, g_po, "po")
#                 upd_pa = self._task_update(p, g_pa, "pa")

#                 total_update = w_po * upd_po + w_pa * upd_pa
#                 p.data.add_(total_update, alpha=-self.lr)

#     def cycle_loader(loader):
#         while True:
#             for batch in loader:
#                 yield batch

#     def train_popa_model(
#         model: nn.Module,
#         po_loader: DataLoader,
#         pa_loader: DataLoader,
#         criterion_species: nn.Module,
#         epochs: int = 300,
#         lr: float = 1e-4,
#         dev: Optional[torch.device] = None,
#         w_po: float = 1.0,
#         w_pa: float = 1.0,
#     ):
#         dev = dev or device()
#         model.to(dev)

#         optimizer = SeparateTaskAdam(
#             model.parameters(),
#             lr=lr,
#             betas=(0.9, 0.999),
#             eps=1e-8,
#             weight_decay=weight_decay
#         )

#         criterion_po = DeepMaxEntLoss()
#         criterion_pa = BalancedBCELoss()

#         model.train()

#         # Use the longest loader so both data sources contribute every epoch
#         steps_per_epoch = max(len(po_loader), len(pa_loader))
#         po_iter = cycle_loader(po_loader)
#         pa_iter = cycle_loader(pa_loader)

#         params = [p for p in model.parameters() if p.requires_grad]

#         for epoch in tqdm.tqdm(range(1, epochs + 1), desc="Training epochs"):
#             running_loss = 0.0
#             running_loss_po = 0.0
#             running_loss_pa = 0.0
#             seen_po = 0
#             seen_pa = 0

#             for step in range(steps_per_epoch):
#                 xb_po, yb_po = next(po_iter)
#                 xb_pa, yb_pa = next(pa_iter)

#                 xb_po = xb_po.to(dev)
#                 yb_po = yb_po.to(dev)
#                 xb_pa = xb_pa.to(dev)
#                 yb_pa = yb_pa.to(dev)

#                 optimizer.zero_grad()

#                 # Forward PO
#                 outputs_po = model(xb_po)
#                 loss_po = criterion_po(outputs_po, yb_po)

#                 # Forward PA
#                 outputs_pa = model(xb_pa)
#                 loss_pa = criterion_pa(outputs_pa, yb_pa)

#                 # Separate gradients for each task
#                 grads_po_raw = torch.autograd.grad(
#                     loss_po,
#                     params,
#                     retain_graph=False,
#                     create_graph=False,
#                     allow_unused=True
#                 )

#                 grads_pa_raw = torch.autograd.grad(
#                     loss_pa,
#                     params,
#                     retain_graph=False,
#                     create_graph=False,
#                     allow_unused=True
#                 )

#                 # Replace None with zeros so zip/step stays simple and safe
#                 grads_po = [
#                     g if g is not None else torch.zeros_like(p)
#                     for p, g in zip(params, grads_po_raw)
#                 ]
#                 grads_pa = [
#                     g if g is not None else torch.zeros_like(p)
#                     for p, g in zip(params, grads_pa_raw)
#                 ]

#                 optimizer.step(grads_po=grads_po, grads_pa=grads_pa, w_po=w_po, w_pa=w_pa)

#                 total_loss = w_po * loss_po.item() + w_pa * loss_pa.item()

#                 running_loss += total_loss * (xb_po.size(0) + xb_pa.size(0))
#                 running_loss_po += loss_po.item() * xb_po.size(0)
#                 running_loss_pa += loss_pa.item() * xb_pa.size(0)
#                 seen_po += xb_po.size(0)
#                 seen_pa += xb_pa.size(0)

#                 run.log({
#                     "train_loss": total_loss,
#                     "loss_po": loss_po.item(),
#                     "loss_pa": loss_pa.item(),
#                     "batch_size_po": xb_po.size(0),
#                     "batch_size_pa": xb_pa.size(0),
#                     "epoch": epoch
#                 })

#             denom = max(1, seen_po + seen_pa)
#             epoch_loss = running_loss / denom
#             epoch_loss_po = running_loss_po / max(1, seen_po)
#             epoch_loss_pa = running_loss_pa / max(1, seen_pa)

#             run.log({
#                 "epoch": epoch,
#                 "epoch_loss": epoch_loss,
#                 "epoch_loss_po": epoch_loss_po,
#                 "epoch_loss_pa": epoch_loss_pa
#             })

#             if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
#                 print(
#                     f"Epoch {epoch}/{epochs} | "
#                     f"Loss: {epoch_loss:.6f} | "
#                     f"PO: {epoch_loss_po:.6f} | "
#                     f"PA: {epoch_loss_pa:.6f}"
#                 )

#         avg_loss = epoch_loss
#         return avg_loss

#     train_popa_model(
#         model=model,
#         po_loader=po_loader,
#         pa_loader=pa_loader,
#         criterion_species=criterion_species,
#         epochs=epochs,
#         lr=lr,
#         dev=device(),
#         w_po=w_po,
#         w_pa=w_pa
#     )

#     # ------------------------------------------------------------------
#     # evaluate on PA_test
#     # ------------------------------------------------------------------
#     X_te_np = X_pa_te_s[covs].values.astype(np.float32)
#     scores = predict(model, X_te_np, dev=device())
#     aucs = aucs_per_species_sparse(scores, Y_pa_te, species_len=len(species))
#     avg_auc = np.nanmean(list(aucs.values()))
#     avg_auc_site = np.nan

#     model_path = os.path.join(output_dir, f"deepmaxent_DA_{region}{group}.pt")
#     torch.save(model, model_path)

#     return avg_auc, aucs, avg_auc_site, model_path, scaler_path

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






# =========================
# Main
# =========================
def main():
    wandb.login(key=WANDB_KEY)
    start_time = time()
    set_all_seeds(SEED)
    dev = device()
    print(f"Using device: {dev}")

    regions = ['']

    
    data_folder = "full_data"

    output_root = os.path.join("output", f"integration_geoplant_{data_folder}")
    os.makedirs(output_root, exist_ok=True)
    summary_rows = []

    

    for region in regions:
        for group in ['plants']:
            print(f"\n=== REGION: {region}, GROUP: {group or '(all)'} ===")


            # use these paths for now
            species_path = 'data/processed/GeoPlant/full_data'
            covariates_path = 'data/processed/GeoPlant/climatic'

            
            X_po, Y_po, X_pa_tr, Y_pa_tr, X_pa_te, Y_pa_te, species, covs = load_po_pa_geoplant_full(
                species_path=species_path,
                covariates_path=covariates_path
            )

            print(f"PA split → train: {len(X_pa_tr)}, test: {len(X_pa_te)}")

            exp_dir = os.path.join(output_root, f"{region}{group}")

            if SUBSAMPLE_PO < 1:
                n_po = len(X_po)
                n_subsample = int(n_po * SUBSAMPLE_PO)
                print(f"Subsampling PO from {n_po} to {n_subsample} samples...")

                subsample_idx = np.random.choice(n_po, n_subsample, replace=False)

                # Subsample X (pandas)
                X_po = X_po.iloc[subsample_idx].reset_index(drop=True)

                # Subsample Y (list-of-lists)
                Y_po = [Y_po[i] for i in subsample_idx]



            


            ### test if separable
            # # X_po = X_po.drop(columns=["x","y"], errors="ignore")
            # scaler = StandardScaler().fit(X_pa_tr[covs])
            # X_po_s = X_po.copy()
            # X_po_s[covs] = scaler.transform(X_po_s[covs])
            # X_pa_tr_s = X_pa_tr.copy()
            # X_pa_tr_s[covs] = scaler.transform(X_pa_tr_s[covs])
            # # drop PO column if present
            # X_po_s = X_po_s.drop(columns=["PO"], errors="ignore")
            # X_pa_tr_s = X_pa_tr_s.drop(columns=["PO"], errors="ignore")
            
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

                auc_po, aucs_po, auc_po_site, model_po, scaler_po = run_experiment(
                    name="PO_only",
                    X_train_df=X_po,
                    Y_tr=Y_po,
                    X_test_df=X_pa_te,  # evaluate on PA_test covs
                    Y_te=Y_pa_te,  # evaluate on PA_test labels
                    covs=covs_no_po, species=species,
                    output_dir=exp_dir, region=region, group=group,
                    verbose=True, criterion='deepmaxent',
                    lr=LR_PO
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
                    Y_tr=Y_pa_tr,
                    X_test_df=X_pa_te,
                    Y_te=Y_pa_te,
                    covs=covs_no_po, species=species,
                    output_dir=exp_dir, region=region, group=group,
                    criterion = loss_criterion, verbose = True
                )
                print(f"[PA-only]   Average AUC on PA_test: {auc_pa:.4f}")



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
                # print('The covariates used are:', covs)



                # C) PO + PA_train → PA_test
                X_mix = pd.concat([X_po, X_pa_tr], axis=0, ignore_index=True)
                Y_mix = pd.concat([Y_po, Y_pa_tr], axis=0, ignore_index=True)


                auc_mix, aucs_mix, auc_mix_site, model_mix, scaler_mix = run_experiment(
                    name="PO_plus_PA",
                    X_train_df=X_mix,
                    Y_tr=Y_mix,
                    X_test_df=X_pa_te,
                    Y_te=Y_pa_te,
                    covs=env_covs, species=species,
                    output_dir=exp_dir, region=region, group=group,
                    criterion = 'bce'
                )

                
                print(f"[PO+PA]     Average AUC on PA_test: {auc_mix:.4f}")

            # weighted PO+PA
            if RUN_POPA_WEIGHTED:

                print("\n--- Running Weighted PO+PA integration experiment ---")

                # use domain_auc as weights
                w_po = 1
                w_pa = 2

                auc_mix_w, aucs_mix_w, _, model_mix_w, scaler_mix_w = run_experiment_popa(
                    name="PO_plus_PA_weighted",
                    X_po_df=X_po,
                    Y_po=Y_po,
                    X_pa_tr_df=X_pa_tr,
                    Y_pa_tr=Y_pa_tr,
                    X_pa_te_df=X_pa_te,
                    Y_pa_te=Y_pa_te,
                    covs=env_covs, species=species,
                    output_dir=exp_dir, region=region, group=group,
                    epochs=EPOCHS,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    w_po=w_po,
                    w_pa=w_pa,
                    weight_decay=WEIGHT_DECAY
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
                print("\n--- Running PO+PA (with BIAS) integration experiment ---")
                auc_mix_b, aucs_mix_b, auc_mix_b_site, model_mix_b, scaler_mix_b = run_experiment_popa_bias(
                    name="PO_plus_PA_bias",
                    X_po_df=X_po,
                    Y_po_df=Y_po,
                    X_pa_tr_df=X_pa_tr,
                    Y_pa_tr_df=Y_pa_tr,
                    X_pa_te_df=X_pa_te,
                    Y_pa_te_df=Y_pa_te,
                    covs=env_covs,  covs_bias=["x", "y"], area_cov=None, species=species,
                    output_dir=exp_dir, region=region, group=group,
                    epochs=EPOCHS,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    print_every=PRINT_EVERY
             
                )
                print(f"[Bias PO+PA]     Average AUC on PA_test: {auc_mix_b:.4f}")

            if RUN_POPA_OMISSION_BIAS:
                auc_mix_ob, aucs_mix_ob, auc_mix_ob_site, model_mix_ob, scaler_mix_ob = run_experiment_popa_omission_bias(
                    name="PO_plus_PA_omission_bias",
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
                print(f"[Omission Bias PO+PA]     Average AUC on PA_test: {auc_mix_ob:.4f}")

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
            if RUN_POPA:
                row["AUC_PO_plus_PA"] = float(np.round(auc_mix, 4))
                row["AUC_PO_plus_PA_site"] = float(np.round(auc_mix_site, 4))
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
            if RUN_POPA_OMISSION_BIAS:
                row["AUC_PO_plus_PA_omission_bias"] = float(np.round(auc_mix_ob, 4))
                row["AUC_PO_plus_PA_omission_bias_site"] = float(np.round(auc_mix_ob_site, 4))

            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    # print(summary[['AUC_PO_only','AUC_PO_w_bias']].mean())
    # summary.to_csv('output/data_integration_results.csv')

    summary_path = os.path.join(output_root, "summary_integration_geoplant.csv")
    summary.to_csv(summary_path, index=False)

    # transpose and show summary, AUC is rows now
    summary_t = summary.set_index(['region', 'group', 'TEST_PA_FRACTION']).T
    print("\n=== Summary of Results ===")
    print(summary_t)


    elapsed = time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")



if __name__ == "__main__":
    main()
