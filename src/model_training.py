import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Sequence

from src.load_data import load_po_pa_nceas
from src.utils import scale_features
from src.models import (DeepMaxEntModel, deepmaxent_loss, 
                        deepmaxent_model_w_bias, deepmaxent_loss_w_bias, 
                        SDMWithBias, sdm_model_abn, sdm_loss_abn)

from torch.nn import BCEWithLogitsLoss

from sklearn.metrics import roc_auc_score


class SDMDataset(Dataset):
    """
    Flexible dataset for combinations of inputs/labels:
      required:  X  (N, Cx)  float32   (Covariates)
                 Y  (N, K)   float32   (Species observations)
      optional:  Z  (N, Cz)  float32   (bias covariates)
                 I (N,)     int64     (plot ids)
                 mask (N, K)  bool/uint8 (missing labels, if you use it)

    The set of keys returned is fixed at construction (no runtime branching).
    """
    _FLOAT_KEYS = {"X", "Y", "Z"}
    _INT_KEYS   = {"I"}
    _BOOL_KEYS  = {"mask"}

    def __init__(self, **arrays):
        if "X" not in arrays or "Y" not in arrays:
            raise ValueError("MultiInputDataset requires at least xs=..., yb=...")

        # --- validate lengths ---
        N = len(next(iter(arrays.values())))
        for k, v in arrays.items():
            if len(v) != N:
                raise ValueError(f"All arrays must have the same length; '{k}' has {len(v)} vs {N}")

        # --- stash tensors with correct dtypes ---
        self._keys: List[str] = sorted(arrays.keys())  # fixed output order
        self.N = N
        self.data: Dict[str, torch.Tensor] = {}
        for k, v in arrays.items():
            t = torch.as_tensor(v)  # zero-copy when possible
            if k in self._FLOAT_KEYS:
                t = t.to(torch.float32)
            elif k in self._INT_KEYS:
                t = t.to(torch.long)
            elif k in self._BOOL_KEYS:
                t = t.to(torch.bool)
            self.data[k] = t

        # cheap sanity checks
        if self.data["Y"].dim() == 1:
            self.data["Y"] = self.data["Y"].unsqueeze(1)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        # Return only the keys present at construction, in stable order
        return {k: self.data[k][i] for k in self._keys}

def per_species_auc(y_true: pd.DataFrame, y_score: np.ndarray, species: List[str]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for i, sp in enumerate(species):
        try:
            auc = roc_auc_score(y_true[sp].values, y_score[:, i])
        except ValueError:
            auc = np.nan
        scores[sp] = auc
    return scores

@torch.no_grad()
def predict_logits(model, X: np.ndarray, *, model_type: str,
                   Z: Optional[np.ndarray] = None,
                   I: Optional[np.ndarray] = None,
                   device=None) -> np.ndarray:
    """
    Returns logits as np.ndarray. If the model outputs (logits, bias),
    we keep only the logits part.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(dev)

    X_t = torch.tensor(X, dtype=torch.float32, device=dev)

    if model_type == "PO_Only_Cov_Bias":
        if Z is None:
            raise ValueError("Z must be provided for PO_Only_Cov_Bias prediction")
        Z_t = torch.tensor(Z, dtype=torch.float32, device=dev)
        out = model(X_t, Z_t)          # (logits, bias)
        logits = out[0] if isinstance(out, (tuple, list)) else out

    elif model_type in ("PO_Only_Plot_Bias", "PO_Only_Plot_Bias_Normalized"):
        if I is None:
            raise ValueError("I must be provided for plot-bias prediction")
        I_t = torch.tensor(I, dtype=torch.long, device=dev)
        out = model(X_t, None)          # logits or (logits, b)
        logits = out[0] if isinstance(out, (tuple, list)) else out

    elif model_type in ("PO_Only", "PO_Only_Smooth", "PO_Only_BCE", "PO_Only_Weak_Neg"):
        logits = model(X_t)

    elif model_type == "PO_Only_ABN":
        out = model(X_t)          # (p, theta)
        logits = out[0] if isinstance(out, (tuple, list)) else out

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return logits.detach().cpu().numpy()

def smooth_targets(y_true: torch.Tensor, logits: torch.Tensor, beta) -> torch.Tensor:
    """
    y_true: (B,) or (B,S)  binary labels
    logits: same shape as y_true
    beta:   scalar OR (S,) per-species
    """
    if y_true.ndim == 1:
        y_true, logits = y_true[:, None], logits[:, None]

    with torch.no_grad():
        p = torch.sigmoid(logits)
        if not torch.is_tensor(beta):
            beta = torch.as_tensor(beta, dtype=y_true.dtype, device=y_true.device)
        beta = beta.to(y_true.device, y_true.dtype)
        beta = beta if beta.ndim else beta.expand_as(y_true)              # scalar -> (B,S)
        if beta.ndim == 1:                                                # (S,) -> (B,S)
            beta = beta[None, :].expand_as(y_true)

    return torch.where(y_true > 0, y_true, beta * p)

def smooth_targets_v2(y_true: torch.Tensor, logits: torch.Tensor, beta) -> torch.Tensor:
    """
    y_true: (B,) or (B,S)  binary labels
    logits: same shape as y_true
    beta:   scalar OR (S,) per-species (not used in this version)
    """
    if y_true.ndim == 1:
        y_true, logits = y_true[:, None], logits[:, None]

    with torch.no_grad():
        p = torch.sigmoid(logits)
        if not torch.is_tensor(beta):
            beta = torch.as_tensor(beta, dtype=y_true.dtype, device=y_true.device)
        beta = beta.to(y_true.device, y_true.dtype)
        beta = beta if beta.ndim else beta.expand_as(y_true)              # scalar -> (B,S)
        if beta.ndim == 1:                                                # (S,) -> (B,S)
            beta = beta[None, :].expand_as(y_true)

    return torch.where(y_true > 0, y_true, p**2)

"""
Version with the adjustment using Bayes for the smooth targets (makes more sense probably)
"""
def smooth_targets_v3(y_true: torch.Tensor, logits: torch.Tensor, theta) -> torch.Tensor:
    """
    y_true: (B,) or (B,S)  binary labels
    logits: same shape as y_true
    beta:   scalar OR (S,) per-species 
    """
    if y_true.ndim == 1:
        y_true, logits = y_true[:, None], logits[:, None]

    with torch.no_grad():
        p = torch.sigmoid(logits)
        if not torch.is_tensor(theta):
            theta = torch.as_tensor(theta, dtype=y_true.dtype, device=y_true.device)
        theta = theta.to(y_true.device, y_true.dtype)
        theta = theta if theta.ndim else theta.expand_as(y_true)              # scalar -> (B,S)
        if theta.ndim == 1:                                                # (S,) -> (B,S)
            theta = theta[None, :].expand_as(y_true)

    smooth_targets = torch.where(y_true > 0, y_true, p*(theta/(1 - p + p*(theta))))

    return smooth_targets

def weaken_negatives(y_true: torch.Tensor,
                    lamb: float) -> torch.Tensor:
    """
    Build soft targets:
      if y==1 -> keep 1
      if y==0 -> use (1-alpha) * p_model (DETACHED)
    """
    with torch.no_grad():
        p = torch.sigmoid(logits)
    # y_smooth = y*1 + (1-y) * (1-alpha) * p
    y_smooth = torch.where(y_true > 0, y_true, (1 - lamb) * p)
    return y_smooth

# def smooth_targets(y_true: torch.Tensor,
#                     logits: torch.Tensor,
#                     beta: float | np.array) -> torch.Tensor:
#     """
#     Build soft targets:
#       if y==1 -> keep 1
#       if y==0 -> use beta * p_model (DETACHED)
#     """
#     with torch.no_grad():
#         p = torch.sigmoid(logits)
#     # y_smooth = y*1 + (1-y) * beta * p
#     y_smooth = torch.where(y_true > 0, y_true, beta * p)
#     return y_smooth


'''
More generic version of the training:
receives a model, loader, loss_fn, 
then train_cfg can be different parameters depending on the setting
'''
def train_loop(model, loader, loss_fn, dataset = None, *, train_cfg: dict, device=None):
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on {dev}')
    model.to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.get('lr',1e-4), weight_decay=train_cfg.get('weight_decay',3e-4))
    # opt = make_optimizer(model, train_cfg) ## this line can be used if we would like to try different optimizer (but needs work)
    grad_clip = train_cfg.get("grad_clip")
    n = len(loader.dataset)

    # # if model is ABN, remove theta from weight_decay
    # if train_cfg['model_type'] == 'PO_Only_ABN':
    #     # remove theta from weight decay using its name
    #     opt = torch.optim.Adam([
    #         {'params': [param for name, param in model.named_parameters() if name != 'logit_theta'], 'weight_decay': train_cfg.get('weight_decay',3e-4)},
    #         {'params': [model.logit_theta], 'weight_decay': 0.0}
    #     ], lr=train_cfg.get('lr',1e-4))

    beta = np.array(train_cfg.get("beta")) # only used for Smooth Model
    if beta: print(f"Initial beta: {beta}")
    for epoch in range(1, train_cfg.get("epochs", 200) + 1):
        running = 0.0
        for batch in loader:
            # batch can be dict-like: {"X","Y","Z","I"}
            xb, yb = batch["X"].to(dev), batch["Y"].to(dev)
            opt.zero_grad()

            if train_cfg['model_type'] == "PO_Only_Cov_Bias":  
                logits, bias = model(xb, batch["Z"].to(dev))
                loss = loss_fn(logits, bias, yb)

            elif train_cfg['model_type'] == "PO_Only_Plot_Bias_Normalized": 
                logits, b = model(xb, batch["I"].to(dev))
                loss = loss_fn(logits, b, yb)

            elif train_cfg['model_type'] == "PO_Only_Plot_Bias":
                logits = model(xb, batch["I"].to(dev))
                loss = loss_fn(logits, yb)

            elif train_cfg['model_type'] in ("PO_Only", "PO_Only_BCE"):
                logits = model(xb)
                loss = loss_fn(logits, yb)

            elif train_cfg['model_type'] in ("PO_Only_Smooth", "PO_Only_Weak_Neg"):
                logits = model(xb)

                if train_cfg['model_type'] == 'PO_Only_Weak_Neg':
                    # se lambda to 1/Batch size
                    # y_soft = weaken_negatives(yb, logits, lamb=1.0/xb.size(0))
                    L = xb.size(0)
                    # if L <= 1:
                    #     L = 2
                    weight_y_zero = 1.0/(L-1)
                    weights_y_one = 1.0
                    weights = torch.where(yb == 1, torch.ones_like(yb) * weights_y_one, torch.ones_like(yb) * weight_y_zero)
                    loss_function = torch.nn.BCEWithLogitsLoss(weight=weights)
                    loss = loss_function(logits, yb)

                    # loss = loss_fn(logits, yb, weight=weights)
                else:
                    y_soft = smooth_targets_v3(yb, logits, beta)
                    loss = loss_fn(logits, y_soft)

                # trheshold for soft as a function of the epoch
                # threshold_zero = epoch / train_cfg['epochs'] * 0.01
                # y_soft = torch.where((yb == 0) & (y_soft < threshold_zero), torch.zeros_like(y_soft), y_soft)

                # IMPORTANT: pass the soft targets into the same loss
                # Works with BCEWithLogitsLoss (soft labels) and your deepmaxent losses (soft targets)
                

                            # if train_cfg['model_type'] == 'PO_Only_Smooth':


            elif train_cfg['model_type'] == 'PO_Only_ABN':
                logit_p, logit_theta = model(xb)                     # p: [B, C], theta: [C]
                eps = 1e-5

                p = torch.sigmoid(logit_p)                           # [B, C]
                theta = torch.sigmoid(logit_theta)                   # [C]
                # take the mean (1 value)
                # theta = theta.mean(dim=0)                             # [1]
       

                # (optional but helps stability a lot)
                p = p.clamp(eps, 1 - eps)               # avoid exact 0/1 probs

                with torch.no_grad():
                    # broadcast theta to batch
                    theta_b = theta.unsqueeze(0).expand_as(p)  # [B, C]

                    # omission-only model:
                    # if yobs == 1 → q = 1
                    # if yobs == 0 → q = ((1 - theta) * p) / ((1 - theta) * p + (1 - p))
                    numer = (1 - theta_b) * p
                    denom = numer + (1 - p) + eps        # +eps to avoid 0

                    q_neg = numer / denom
                    q = torch.where(yb == 1, torch.ones_like(p), q_neg)

                    # keep q in (0,1) to avoid log(0)
                    q = q.clamp(eps, 1 - eps)
                
                # # for q being nan print everything
                # if torch.isnan(q).any():
                #     with torch.no_grad():
                #         # print logits
                #         print(f"  logit_p sample: {logit_p[0:5].cpu().numpy()}")
                #         print(f"  logit_theta sample: {logit_theta.cpu().numpy()}")
                #         print(f"  p sample: {p[0:5].cpu().numpy()}")
                #         print(f"  theta sample: {theta.cpu().numpy()}")
                #         print(f"  q sample: {q[0:5].cpu().numpy()}")
                #         print('epoch:', epoch)
                #         exit(1)

                loss = loss_fn(p, theta, yb, q)




            


            # elif train_cfg['model_type'] == 'PO_Only_ABN':
            #     p, theta = model(xb)
                
                
            #     with torch.no_grad():
            #         numer = (1 - theta) * p              # (1-theta)*p
            #         denom = numer + (1 - p) + 1e-8         # + eps for stability
            #         q_neg = numer / denom                  # when yb == 0

            #         q = torch.where(yb == 1, torch.ones_like(p), q_neg)


            #     loss = loss_fn(p, theta, yb, q)
            #     if epoch % 50 == 0:
            #         with torch.no_grad():
            #             theta_for_print = theta.cpu().numpy()
            #             print(f"  Theta Update {epoch}: {theta_for_print}")
            #             # if nan print everything
            #             if np.any(np.isnan(theta_for_print)):
            #                 print(f"  p sample: {p[0:5].cpu().numpy()}")
            #                 print(f"  q sample: {q[0:5].cpu().numpy()}")


            else:
                raise ValueError(f"Unknown model_type: {train_cfg['model_type']}")
            
            ## add base line BCE



            loss.backward()
            if grad_clip: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
    
            opt.step()
            running += loss.item() * xb.size(0)

            # if train_cfg['model_type'] == 'PO_Only_ABN':
            #     with torch.no_grad():
            #         theta_prob = torch.sigmoid(model.logit_theta)      # [C]
            #         theta_prob = theta_prob.clamp(0.05, 0.95)         # keep away from 0/1
            #         model.logit_theta.data = torch.log(theta_prob / (1 - theta_prob))



                    # print beta for each column
        #         # print(f"Updated beta: {beta.cpu().numpy()}")
        # something that updates more often the closer we are to the last epoch 
        # make a paramater that controls a weight from the previous beta and the new one
        # alpha = min(1.0, epoch / train_cfg.get("epochs", 200))
        # alpha_initial = 1
        # alpha_last = .1
        # alpha = alpha_initial - (alpha_initial - alpha_last) * (epoch / train_cfg.get("epochs", 200))
        if False:
            if train_cfg['model_type'] == 'PO_Only_Smooth':
                # print('Updating beta at epoch', epoch)
                with torch.no_grad():
                    # update beta with M-step

                    ## access the full Y
                    y = loader.dataset.data['Y'].to(dev)
                    logits_full = model(loader.dataset.data['X'].to(dev))
                    # reshape beta to have same number as columns as y
                    # beta = np.array([beta] * y.shape[1]) # could be better
                    y_soft_full = smooth_targets_v3(y, logits_full, beta)

                    # apply the threshold for zeros as well
                    # threshold_zero = epoch / train_cfg['epochs'] * 0.01
                    # y_soft_full = torch.where((y == 0) & (y_soft_full < threshold_zero), torch.zeros_like(y_soft_full), y_soft_full)

                    
                    numer = (y_soft_full * (1 - y)).sum(dim=0)
                    denom = y_soft_full.sum(dim=0)
                    # numer = (y_soft * (1 - y)).sum(dim=0)
                    # denom = y_soft.sum(dim=0)

                    theta = torch.zeros_like(numer)

                    # only update where denom > 0
                    mask = denom > 0
                    theta[mask] = numer[mask] / denom[mask]

                    theta = theta.clamp(0.0, 1.0)

                    beta = 1-theta.cpu().numpy()
         
                    # beta_new = 1 - theta.cpu().numpy()

                    # beta = (1-alpha) * beta + (alpha) * beta_new

                    # beta = beta.clip(0.0, .95)


        if (epoch == 1) or (epoch % train_cfg.get("print_every", 1000) == 0) or (epoch == train_cfg["epochs"]):
            print(f"Epoch {epoch:5d}/{train_cfg['epochs']} | Loss: {running/max(1,n):.4f}")
            # if get smooth
            if beta is not None:
                print(f"  Beta Update {epoch}: {beta}")
            if train_cfg['model_type'] == 'PO_Only_ABN':
                with torch.no_grad():
                    theta_for_print = torch.sigmoid(model.logit_theta).cpu().numpy()
                    print(f"  Theta Update {epoch}: {theta_for_print}")




#### test init ####

if __name__ == '__main__':

    train_config = {'hidden_size' : 250,
    # HIDDEN_BIAS_SIZE : 3000
    'hidden_layers' : 2,
    'lr' : 1e-4,
    'epochs' : 250,
    'bs' : 250,
    'max_batch_percentage' : 1,
    'print_every' : 1000,
    'seed' : 42}

    X_po, Y_po, X_pa, Y_pa, species, covs = load_po_pa_nceas(
        region='NSW',
        group_filter='_ba',
        add_po_var=False
    )

    print(X_po)
    print(Y_po)
    print(Y_pa)
    print(species)
    print(covs)

    X_po_scaled, X_pa_scaled, _ = scale_features(X_po, X_pa, covs, None, False)

    # ===== PO_Only =====
    train_config['model_type'] = 'PO_Only'
    ds = SDMDataset(X = X_po_scaled[covs].values, Y = Y_po[species].values)
    loader = DataLoader(ds, train_config['bs'], shuffle=True)
    model = DeepMaxEntModel(len(covs), train_config['hidden_size'], len(species), train_config['hidden_layers'])
    train_loop(model, loader=loader, loss_fn=deepmaxent_loss(), train_cfg=train_config)

    logits = predict_logits(model, X_pa_scaled[covs].values,
                            model_type='PO_Only', device=None)
    
    aucs = per_species_auc(Y_pa[species], logits, species)
    print("[PO_Only] Avg AUC on PA:", np.nanmean(list(aucs.values())))



    # ===== PO_Only_Smooth =====
    train_config['model_type'] = 'PO_Only_Smooth'
    ds = SDMDataset(X = X_po_scaled[covs].values, Y = Y_po[species].values)
    loader = DataLoader(ds, train_config['bs'], shuffle=True)
    model = DeepMaxEntModel(len(covs), train_config['hidden_size'], len(species), train_config['hidden_layers'])
    train_loop(model, loader=loader, loss_fn=deepmaxent_loss(), train_cfg=train_config)

    logits = predict_logits(model, X_pa_scaled[covs].values,
                            model_type='PO_Only_Smooth', device=None)
    aucs = per_species_auc(Y_pa[species], logits, species)
    print("[PO_Only_Smooth] Avg AUC on PA:", np.nanmean(list(aucs.values())))


    # ===== PO_Only_Plot_Bias =====
    train_config['model_type'] = 'PO_Only_Plot_Bias'
    ds = SDMDataset(X = X_po_scaled[covs].values, Y = Y_po[species].values,
                    I = np.arange(len(X_po_scaled)))
    loader = DataLoader(ds, train_config['bs'], shuffle=True)
    model = deepmaxent_model_w_bias(len(covs), train_config['hidden_size'], len(species),
                                    train_config['hidden_layers'], num_plots=len(X_po_scaled), separate=False)
    # you used BCE here; keep it if that's intentional
    train_loop(model, loader=loader, loss_fn=BCEWithLogitsLoss(), train_cfg=train_config)

    I_pa = np.arange(len(X_pa_scaled))
    logits = predict_logits(model, X_pa_scaled[covs].values,
                            model_type='PO_Only_Plot_Bias', I=I_pa, device=None)
    aucs = per_species_auc(Y_pa[species], logits, species)
    print("[PO_Only_Plot_Bias] Avg AUC on PA:", np.nanmean(list(aucs.values())))


    # ===== PO_Only_Plot_Bias_Normalized (two-output; use logits part) =====
    train_config['model_type'] = 'PO_Only_Plot_Bias_Normalized'
    ds = SDMDataset(X = X_po_scaled[covs].values, Y = Y_po[species].values,
                    I = np.arange(len(X_po_scaled)))
    loader = DataLoader(ds, train_config['bs'], shuffle=True)
    model = deepmaxent_model_w_bias(len(covs), train_config['hidden_size'], len(species),
                                    train_config['hidden_layers'], num_plots=len(X_po_scaled), separate=True)
    train_loop(model, loader=loader, loss_fn=deepmaxent_loss_w_bias(), train_cfg=train_config)

    I_pa = np.arange(len(X_pa_scaled))
    logits = predict_logits(model, X_pa_scaled[covs].values,
                            model_type='PO_Only_Plot_Bias_Normalized', I=I_pa, device=None)
    aucs = per_species_auc(Y_pa[species], logits, species)
    print("[PO_Only_Plot_Bias_Normalized] Avg AUC on PA:", np.nanmean(list(aucs.values())))


    # ===== PO_Only_Cov_Bias (two-output; pass Z and use logits part) =====
    train_config['model_type'] = 'PO_Only_Cov_Bias'
    train_config['bias_architecture'] = (1000, 500)
    train_config['species_architecture'] = (500, 500)

    covs_bias = ['x', 'y']
    # FIXED: was `if covs not in ['x','y']`
    covs_species = [c for c in covs if c not in covs_bias]

    ds = SDMDataset(
        X = X_po_scaled[covs_species].values,
        Y = Y_po[species].values,
        Z = X_po_scaled[covs_bias].values
    )
    loader = DataLoader(ds, train_config['bs'], shuffle=True)
    model = SDMWithBias(len(covs_species), len(species), len(covs_bias),
                        train_config['species_architecture'], train_config['bias_architecture'])
    train_loop(model, loader=loader, loss_fn=deepmaxent_loss_w_bias(), train_cfg=train_config)

    logits = predict_logits(model,
                            X_pa_scaled[covs_species].values,
                            model_type='PO_Only_Cov_Bias',
                            Z=X_pa_scaled[covs_bias].values,
                            device=None)
    aucs = per_species_auc(Y_pa[species], logits, species)
    print("[PO_Only_Cov_Bias] Avg AUC on PA:", np.nanmean(list(aucs.values())))
