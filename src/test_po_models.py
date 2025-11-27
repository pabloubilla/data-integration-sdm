#!/usr/bin/env python3
import os
import json
import yaml
import pickle
import argparse
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

# ---- your core building blocks (change module name as needed) ----
from src.model_training import (
    SDMDataset,
    train_loop,
    predict_logits,
    per_species_auc,
    deepmaxent_model,
    deepmaxent_model_w_bias,
    deepmaxent_loss,
    deepmaxent_loss_w_bias,
    SDMWithBias,
    sdm_model_abn,
    sdm_loss_abn,
)

from src.load_data import load_po_pa_nceas
from src.utils import scale_features


# -------------------------------
# utilities
# -------------------------------
def set_all_seeds(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


############### TEST FUNCTION FOR BETA ###########
def make_beta_abs_from_prevalence(Y):
    # n = len(Y)
    # # logarithmic scaling with diminishing returns
    # size_factor = np.log10(max(n, 10)) / np.log10(scale)
    # beta_abs = a + b * size_factor
    # beta_abs = max(beta_min, min(beta_abs, beta_max))
    # return float(beta_abs)
    return .8

# -------------------------------
# experiment runner (one method)
# -------------------------------
def run_one_method(
    *,
    model_type: str,
    train_cfg: Dict,
    X_po_scaled, Y_po,              # training (PO)
    X_pa_scaled, Y_pa,              # evaluation (PA)
    covs: List[str],
    species: List[str],
    out_dir: str,
) -> Tuple[float, Dict[str, float], str]:
    """
    Trains the given model_type on PO (scaled), evaluates on PA (scaled),
    returns (avg_auc, per_species_auc, model_path).
    Saves model + metadata in out_dir/model_type.
    """
    ensure_dir(out_dir)
    method_dir = os.path.join(out_dir, model_type)
    ensure_dir(method_dir)

    # ----- build dataset/loader, model, loss -----
    bs = int(train_cfg.get("bs", 256))
    hidden_size = int(train_cfg.get("hidden_size", 250))
    hidden_layers = int(train_cfg.get("hidden_layers", 2))
    print_every = int(train_cfg.get("print_every", 1000))

    # default cov splits for bias model with covariates
    covs_bias = train_cfg.get("covs_bias", ["x", "y"])
    covs_species = [c for c in covs if c not in covs_bias]

    if model_type == "PO_Only":
        ds = SDMDataset(X=X_po_scaled[covs].values, Y=Y_po[species].values)
        loader = DataLoader(ds, bs, shuffle=True)
        model = deepmaxent_model(len(covs), hidden_size, len(species), hidden_layers)
        loss_fn = deepmaxent_loss()

        train_loop(model, loader=loader, loss_fn=loss_fn, dataset=ds, train_cfg={**train_cfg, "model_type": model_type})
        logits = predict_logits(model, X_pa_scaled[covs].values, model_type=model_type)
        model_path = os.path.join(method_dir, "model.pt")

    elif model_type == "PO_Only_BCE":
        
        ds = SDMDataset(X=X_po_scaled[covs].values, Y=Y_po[species].values)
        loader = DataLoader(ds, bs, shuffle=True)
        model = deepmaxent_model(len(covs), hidden_size, len(species), hidden_layers)
        # deepmaxent_loss supports soft targets as you built
        loss_fn = BCEWithLogitsLoss()

        train_loop(model, loader=loader, loss_fn=loss_fn, train_cfg={**train_cfg, "model_type": model_type})
        logits = predict_logits(model, X_pa_scaled[covs].values, model_type=model_type)
        model_path = os.path.join(method_dir, "model.pt")

    elif model_type == "PO_Only_Smooth":
        
        # train_cfg['beta'] = make_beta_abs_from_prevalence(Y_po[species].values)
        # train_cfg['beta'] = .8    
        beta = train_cfg.get('beta')

        print(f'Beta has been set to: {beta}')

        ds = SDMDataset(X=X_po_scaled[covs].values, Y=Y_po[species].values)
        loader = DataLoader(ds, bs, shuffle=True)
        model = deepmaxent_model(len(covs), hidden_size, len(species), hidden_layers)
        # deepmaxent_loss supports soft targets as you built
        # loss_fn = BCEWithLogitsLoss()
        loss_fn = deepmaxent_loss()

        train_loop(model, loader=loader, loss_fn=loss_fn, train_cfg={**train_cfg, "model_type": model_type})
        logits = predict_logits(model, X_pa_scaled[covs].values, model_type=model_type)
        model_path = os.path.join(method_dir, "model.pt")

    elif model_type == "PO_Only_ABN":
        ds = SDMDataset(X=X_po_scaled[covs].values, Y=Y_po[species].values)
        loader = DataLoader(ds, bs, shuffle=True)
        model = sdm_model_abn(len(covs), hidden_size, len(species), hidden_layers)
        loss_fn = sdm_loss_abn()

        train_loop(model, loader=loader, loss_fn=loss_fn, train_cfg={**train_cfg, "model_type": model_type})
        logits = predict_logits(model, X_pa_scaled[covs].values, model_type=model_type)
        model_path = os.path.join(method_dir, "model.pt")

    elif model_type == "PO_Only_Plot_Bias":
        I_train = np.arange(len(X_po_scaled))
        ds = SDMDataset(X=X_po_scaled[covs].values, Y=Y_po[species].values, I=I_train)
        loader = DataLoader(ds, bs, shuffle=True)
        model = deepmaxent_model_w_bias(len(covs), hidden_size, len(species), hidden_layers,
                                        num_plots=len(X_po_scaled), separate=False)
        # your example uses BCE for this variant
        loss_fn = deepmaxent_loss()

        train_loop(model, loader=loader, loss_fn=loss_fn, train_cfg={**train_cfg, "model_type": model_type})
        I_eval = np.arange(len(X_pa_scaled))
        logits = predict_logits(model, X_pa_scaled[covs].values, model_type=model_type, I=I_eval)
        model_path = os.path.join(method_dir, "model.pt")

    elif model_type == "PO_Only_Plot_Bias_Normalized":
        I_train = np.arange(len(X_po_scaled))
        ds = SDMDataset(X=X_po_scaled[covs].values, Y=Y_po[species].values, I=I_train)
        loader = DataLoader(ds, bs, shuffle=True)
        model = deepmaxent_model_w_bias(len(covs), hidden_size, len(species), hidden_layers,
                                        num_plots=len(X_po_scaled), separate=True)
        loss_fn = deepmaxent_loss_w_bias()

        train_loop(model, loader=loader, loss_fn=loss_fn, train_cfg={**train_cfg, "model_type": model_type})
        I_eval = np.arange(len(X_pa_scaled))
        logits = predict_logits(model, X_pa_scaled[covs].values, model_type=model_type, I=I_eval)
        model_path = os.path.join(method_dir, "model.pt")

    elif model_type == "PO_Only_Cov_Bias":
        # species inputs vs bias inputs
        ds = SDMDataset(
            X=X_po_scaled[covs_species].values,
            Y=Y_po[species].values,
            Z=X_po_scaled[covs_bias].values
        )
        loader = DataLoader(ds, bs, shuffle=True)
        sp_arch = tuple(train_cfg.get("species_architecture", (500, 500)))
        bi_arch = tuple(train_cfg.get("bias_architecture", (1000, 500)))
        model = SDMWithBias(len(covs_species), len(species), len(covs_bias),
                            sp_arch, bi_arch)
        loss_fn = deepmaxent_loss_w_bias()

        train_loop(model, loader=loader, loss_fn=loss_fn, train_cfg={**train_cfg, "model_type": model_type})
        logits = predict_logits(
            model,
            X_pa_scaled[covs_species].values,
            model_type=model_type,
            Z=X_pa_scaled[covs_bias].values
        )
        model_path = os.path.join(method_dir, "model.pt")

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # ----- evaluate -----
    aucs = per_species_auc(Y_pa[species], logits, species)
    avg_auc = float(np.nanmean(list(aucs.values())))

    # ----- save artifacts -----
    torch.save(model, model_path)
    with open(os.path.join(method_dir, "train_cfg.json"), "w") as f:
        json.dump(train_cfg, f, indent=2)
    with open(os.path.join(method_dir, "aucs.json"), "w") as f:
        json.dump({"avg_auc": avg_auc, "per_species_auc": aucs}, f, indent=2)

    return avg_auc, aucs, model_path


# -------------------------------
# main loop over regions/groups
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cpu or cuda:0")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    print(cfg)

    seed = int(cfg.get("seed", 42))
    set_all_seeds(seed)

    output_root = cfg.get("output_root", "output/region_runs")
    ensure_dir(output_root)

    regions_cfg: Dict[str, List[str]] = cfg["regions"]  # e.g., {"NSW": ["_ba","_db"], "SWI": [""]}
    experiments: List[Dict] = cfg["experiments"]

    summary_rows = []

    summary_per_species = []

    for region, groups in regions_cfg.items():
        for group in groups:
            print(f"\n=== REGION: {region} | GROUP: {group or '(all)'} ===")

            # ----- load data -----
            X_po, Y_po, X_pa, Y_pa, species, covs = load_po_pa_nceas(
                region=region,
                group_filter=group,
                add_po_var=cfg.get("add_po_var", False),
                categorical_vars=['ontveg', 'vegsys'] # this could be done in a different way (preprocessing)
            )
            # TODO: Try to generalize how different methods might use different covariates 
            # print(covs)
            # exit()
            env_covs = covs[2:]
            spatial_covs = covs[:2]
            # print(covs)


            # ----- scaling: fit on PA, transform both (your earlier style) -----
            X_po_scaled, X_pa_scaled, scaler = scale_features(X_po, X_pa, covs, output_path=None, verbose=False)

            # save scaler for this region/group
            rg_dir = os.path.join(output_root, f"{region}{group or ''}")
            ensure_dir(rg_dir)
            with open(os.path.join(rg_dir, "scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)

            # ----- run all experiments -----
            for exp in experiments:
                model_type = exp["model_type"]
                # merge global defaults with exp overrides
                exp_cfg = {**cfg.get("train_defaults", {}), **exp}
                exp_dir = os.path.join(rg_dir, model_type)

                if model_type == 'PO_Only_Cov_Bias':
                    covs_to_use = covs.copy()
                else:
                    covs_to_use = env_covs.copy()
                
                print(f'Running {region}{group} | {model_type}')
                print(f'With covariates: {covs_to_use}')

                avg_auc, aucs, model_path = run_one_method(
                    model_type=model_type,
                    train_cfg=exp_cfg,
                    X_po_scaled=X_po_scaled,
                    Y_po=Y_po,
                    X_pa_scaled=X_pa_scaled,
                    Y_pa=Y_pa,
                    covs=covs_to_use,
                    species=species,
                    out_dir=exp_dir,
                )

                print(f"{region}{group} | {model_type} -> {model_type}: AVG AUC = {avg_auc:.4f}")

            

                summary_rows.append({
                    "region": region,
                    "group": group,
                    "model_type": model_type,
                    "avg_auc": avg_auc,
                    # "model_path": model_path,
                })

                for ix, sp in enumerate(species):
                    summary_per_species.append({
                    "region": region,
                    "group": group,
                    "species": sp,
                    "model_type": model_type,
                    "auc": aucs[sp],
                    # "model_path": model_path,
                })

    # ----- save summary -----
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_root, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    summary_species_df = pd.DataFrame(summary_per_species)
    summary_species_csv = os.path.join(output_root, "summary_species.csv")
    summary_species_df.to_csv(summary_species_csv)
    print(summary_species_df.groupby(['model_type','region'])['auc'].mean())

    
    # print("\n=== Summary saved to:", summary_csv)
    # print(summary_df.pivot_table(index=["region", "group"], columns="model_type", values="avg_auc"))
    # print('\n')
    # print(summary_df.groupby('model_type')['avg_auc'].mean())
    
    # print("\nDone.")


if __name__ == "__main__":
    main()
