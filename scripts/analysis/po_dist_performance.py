"""
For the best POPA config, loads already-trained models/scalers from the
final split sweep (no retraining), computes per-site AUC, and relates it
to each site's distance to the nearest PO point.

Usage:
    python scripts/analyze_po_distance_effect.py \
        --dataset_name GeoPlant --split_type geographical --region france \
        --loss_po_name deep_maxent --loss_pa_name balanced_bce
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from isdm.load_data import load_geoplant_processed
from isdm.splits import load_split_specs, load_split, filter_and_remap, subset_list
from isdm.models import MLP
from isdm.evaluation import predict_logits, per_site_auc_sparse
from isdm.utils import get_device


def load_best_popa_row(tune_dir: Path, loss_po_name: str, loss_pa_name: str) -> dict:
    df = pd.read_csv(tune_dir / "best_popa_avg_over_distance.csv")
    row = df[(df["param_loss_po_name"] == loss_po_name) & (df["param_loss_pa_name"] == loss_pa_name)]
    if row.empty:
        raise ValueError(f"No tuned entry for {loss_po_name}/{loss_pa_name}")
    return row.iloc[0].to_dict()


def prepare_test_set(split_row, split_dir, data, use_overlapping_species=True):
    """Mirrors the PA-test-prep block inside run_one_split_popa, but also
    keeps lon/lat (NOT part of `covariates`, so extracted separately here,
    before any covariate-only subsetting happens downstream)."""
    split = load_split(split_dir, split_row["split_file"])
    test_idx = split["test_idx"]
    overlapping_species_list = split["species_list"]
    species = overlapping_species_list if use_overlapping_species else data.species
    num_classes = len(species)
    covariates = data.covariates  
    covariates = [c for c in covariates if c not in ["lon", "lat"]]

    X_pa_all = data.X_pa_train.reset_index(drop=True)
    y_pa_all = data.y_pa_train

    X_pa_test_df = X_pa_all.iloc[test_idx].copy()   # full row: covariates + lon/lat + anything else
    y_pa_test = subset_list(y_pa_all, test_idx)
    if use_overlapping_species:
        y_pa_test = filter_and_remap(overlapping_species_list, y_pa_test)

    site_lonlat = X_pa_test_df[["lon", "lat"]].to_numpy()  # pulled out BEFORE covariate subsetting

    return X_pa_test_df, y_pa_test, num_classes, covariates, site_lonlat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="GeoPlant")
    parser.add_argument("--split_type", type=str, default="geographical")
    parser.add_argument("--region", type=str, default="france")
    parser.add_argument("--use_overlapping_species", action="store_true", default=True)
    parser.add_argument("--loss_po_name", type=str, default="deep_maxent")
    parser.add_argument("--loss_pa_name", type=str, default="bce_ippp")
    parser.add_argument("--hidden_dim", type=int, default=None, help="override; else read from tune results")
    parser.add_argument("--hidden_layers", type=int, default=None)
    args = parser.parse_args()

    overlap_name = "intersect" if args.use_overlapping_species else "union"
    data_path = f"data/processed/{args.dataset_name}/{args.region}"
    split_dir = Path(f"outputs/splits/{args.dataset_name}/{args.region}_bands/{args.split_type}")
    sweep_dir = Path(f"outputs/split_sweep/{args.dataset_name}/{args.region}_bands/{args.split_type}/{overlap_name}")
    tune_dir = Path(f"outputs/tune/{args.dataset_name}/{args.region}/{args.split_type}/test_0")

    best = load_best_popa_row(tune_dir, args.loss_po_name, args.loss_pa_name)
    hidden_dim = args.hidden_dim or int(best["param_hidden_dim"])
    hidden_layers = args.hidden_layers or int(best["param_hidden_layers"])

    # print
    print(f"Using best config from {tune_dir}:")
    print(f"  loss_po_name={args.loss_po_name}, loss_pa_name={args.loss_pa_name}, hidden_dim={hidden_dim}, hidden_layers={hidden_layers}")
    exit()

    # add_coordinates=True is required — lon/lat are NOT loaded by default,
    # since they're not model covariates, only needed here for distance calc
    data = load_geoplant_processed(data_path, add_coordinates=True)
    device = get_device()

    # PO tree built once — PO training data doesn't vary by split
    po_lonlat = data.X_po[["lon", "lat"]].dropna().to_numpy()
    po_tree = cKDTree(po_lonlat)

    split_table = load_split_specs(split_dir)
    split_table = split_table[~split_table["option"].str.contains("val", case=False, na=False)]

    config_name = f"po_{args.loss_po_name}_pa_{args.loss_pa_name}"
    all_rows = []

    for _, split_row in split_table.iterrows():
        exp_dir = sweep_dir / "popa_split_sweep" / split_row["split_id"]
        model_path, scaler_path = exp_dir / "model.pt", exp_dir / "scaler.pkl"
        if not model_path.exists():
            print(f"skip {split_row['split_id']}: no saved model at {model_path}")
            continue

        X_pa_test_df, y_pa_test, num_classes, covariates, site_lonlat = prepare_test_set(
            split_row, split_dir, data, args.use_overlapping_species
        )

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X_pa_test = scaler.transform(X_pa_test_df[covariates]).astype(np.float32)  # covariates only, no lon/lat

        model = MLP(input_size=len(covariates), output_size=num_classes,
                    hidden_size=hidden_dim, hidden_layers=hidden_layers)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()

        logits = predict_logits(model, X_pa_test, device=device)
        aucs_site = per_site_auc_sparse(logits=logits, y_lists=y_pa_test, num_classes=num_classes)

        dist_to_po, _ = po_tree.query(site_lonlat, k=1)

        for local_idx, (site_key, auc) in enumerate(aucs_site.items()):
            all_rows.append({
                "split_id": split_row["split_id"],
                "option": split_row["option"],
                "test_number": split_row["test_number"],
                "site_local_idx": local_idx,
                "lon": site_lonlat[local_idx, 0],
                "lat": site_lonlat[local_idx, 1],
                "distance_to_po": dist_to_po[local_idx],
                "site_auc": auc,
            })

        print(f"{split_row['split_id']}: {len(aucs_site)} sites processed")

    df = pd.DataFrame(all_rows)
    out_dir = sweep_dir / "po_distance_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{config_name}_site_auc_vs_po_distance.csv", index=False)

    df_valid = df.dropna(subset=["site_auc"])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df_valid["distance_to_po"], df_valid["site_auc"], s=4, alpha=0.15, c="#2166AC")

    bins = np.quantile(df_valid["distance_to_po"], np.linspace(0, 1, 11))
    df_valid = df_valid.copy()
    df_valid["dist_bin"] = pd.cut(df_valid["distance_to_po"], bins, include_lowest=True)
    binned = df_valid.groupby("dist_bin", observed=True)["site_auc"].mean()
    bin_centers = [interval.mid for interval in binned.index]
    ax.plot(bin_centers, binned.values, color="#B2182B", linewidth=2, marker="o", label="binned mean")

    ax.set_xlabel("distance to nearest PO point (° lon/lat, unprojected)")
    ax.set_ylabel("site-level AUC")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{config_name}_site_auc_vs_po_distance.png", dpi=200)
    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()