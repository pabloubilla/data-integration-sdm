"""
For the best POPA config, loads trained models/scalers from the final split
sweep, computes per-site AUC on the PA test set, and relates it to each
site's great-circle distance (km) to the nearest PO record and to the
nearest PA training site.

Usage:
    python scripts/analyze_po_distance_effect.py \
        --dataset_name GeoPlant --split_type geographical --region france \
        --loss_po_name deep_maxent --loss_pa_name balanced_bce
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from isdm.evaluation import predict_logits, per_site_auc_sparse
from isdm.utils import haversine_tree, nearest_km
from isdm.load_data import load_geoplant_processed
from isdm.models import MLP
from isdm.splits import filter_and_remap, load_split, load_split_specs, subset_list
from isdm.utils import get_device

COORD_COLS = ["lon", "lat"]  # never fed to the model
DISTANCES = {"distance_to_po_km": "nearest PO record", "distance_to_pa_train_km": "nearest PA training site"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="GeoPlant")
    p.add_argument("--split_type", default="geographical")
    p.add_argument("--region", default="sparse_pa")
    p.add_argument("--tune_region", default="france")
    p.add_argument("--use_overlapping_species", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--loss_po_name", default="deep_maxent")
    p.add_argument("--loss_pa_name", default="bce_ippp")
    p.add_argument("--hidden_dim", type=int, default=None, help="override; else read from tune results")
    p.add_argument("--hidden_layers", type=int, default=None)
    args = p.parse_args()

    overlap = "intersect" if args.use_overlapping_species else "union"
    args.data_path = f"data/processed/{args.dataset_name}/{args.region}"
    args.split_dir = Path(f"outputs/splits/{args.dataset_name}/{args.region}_bands/{args.split_type}")
    args.sweep_dir = Path(f"outputs/split_sweep/{args.dataset_name}/{args.region}_bands/{args.split_type}/{overlap}")
    args.tune_dir = Path(f"outputs/tune/{args.dataset_name}/{args.tune_region}/{args.split_type}/test_0")
    args.config_name = f"po_{args.loss_po_name}_pa_{args.loss_pa_name}"
    return args


def load_best_params(args):
    df = pd.read_csv(args.tune_dir / "best_popa_avg_over_distance.csv")
    match = df[(df.param_loss_po_name == args.loss_po_name) & (df.param_loss_pa_name == args.loss_pa_name)]
    if match.empty:
        raise ValueError(f"No tuned params found for {args.config_name}")
    return match.iloc[0]


def prepare_split(split_row, split_dir, data, use_overlapping_species=True):
    split = load_split(split_dir, split_row["split_file"])
    train_idx, test_idx = split["train_idx"], split["test_idx"]
    overlap = split["species_list"]
    X_pa = data.X_pa_train.reset_index(drop=True)

    y_test = subset_list(data.y_pa_train, test_idx)
    if use_overlapping_species:
        y_test = filter_and_remap(overlap, y_test)

    return dict(
        X_test=X_pa.iloc[test_idx],
        y_test=y_test,
        train_lonlat=X_pa.iloc[train_idx][COORD_COLS].to_numpy(),
        num_classes=len(overlap if use_overlapping_species else data.species),
    )


def load_model(exp_dir, n_features, num_classes, hidden_dim, hidden_layers, device):
    with open(exp_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    model = MLP(input_size=n_features, output_size=num_classes,
                hidden_size=hidden_dim, hidden_layers=hidden_layers)
    model.load_state_dict(torch.load(exp_dir / "model.pt", map_location=device))
    return model.to(device).eval(), scaler


def plot_auc_vs_distances(df, path, n_bins=10):
    fig, axes = plt.subplots(1, len(DISTANCES), figsize=(6 * len(DISTANCES), 5), sharey=True)
    for ax, (col, label) in zip(axes, DISTANCES.items()):
        ax.scatter(df[col], df["site_auc"], s=4, alpha=0.15, c="#2166AC")

        bins = pd.qcut(df[col], n_bins, duplicates="drop")
        binned = df.groupby(bins, observed=True)["site_auc"].mean()
        ax.plot([b.mid for b in binned.index], binned.values,
                color="#B2182B", linewidth=2, marker="o", label="binned mean")

        ax.set_xlabel(f"distance to {label} (km)")
        ax.legend()
    axes[0].set_ylabel("site-level AUC")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    best = load_best_params(args)
    hidden_dim = args.hidden_dim or int(best["param_hidden_dim"])
    hidden_layers = args.hidden_layers or int(best["param_hidden_layers"])
    print(f"{args.config_name}: hidden_dim={hidden_dim}, hidden_layers={hidden_layers}")

    data = load_geoplant_processed(args.data_path, add_coordinates=True)
    covariates = [c for c in data.covariates if c not in COORD_COLS]
    device = get_device()

    po_tree = haversine_tree(data.X_po[COORD_COLS].dropna().to_numpy())

    splits = load_split_specs(args.split_dir)
    splits = splits[~splits["option"].str.contains("val", case=False, na=False)]

    results = []
    for _, split_row in splits.iterrows():
        exp_dir = args.sweep_dir / "popa_split_sweep" / split_row["split_id"]
        if not (exp_dir / "model.pt").exists():
            print(f"skip {split_row['split_id']}: no saved model in {exp_dir}")
            continue

        s = prepare_split(split_row, args.split_dir, data, args.use_overlapping_species)
        model, scaler = load_model(exp_dir, len(covariates), s["num_classes"], hidden_dim, hidden_layers, device)

        X_test = scaler.transform(s["X_test"][covariates]).astype(np.float32)
        logits = predict_logits(model, X_test, device=device)
        aucs = per_site_auc_sparse(logits=logits, y_lists=s["y_test"], num_classes=s["num_classes"])
        aucs = pd.Series(dict(aucs.items()), dtype=float)  # keys = row positions in X_test

        site_idx = aucs.index.to_numpy(dtype=int)
        lonlat = s["X_test"][COORD_COLS].to_numpy()[site_idx]
        pa_train_tree = haversine_tree(s["train_lonlat"])

        results.append(pd.DataFrame({
            "split_id": split_row["split_id"],
            "option": split_row["option"],
            "test_number": split_row["test_number"],
            "site_local_idx": site_idx,
            "lon": lonlat[:, 0],
            "lat": lonlat[:, 1],
            "distance_to_po_km": nearest_km(po_tree, lonlat),
            "distance_to_pa_train_km": nearest_km(pa_train_tree, lonlat),
            "site_auc": aucs.to_numpy(),
        }))
        print(f"{split_row['split_id']}: {len(aucs)} sites")

    df = pd.concat(results, ignore_index=True)
    out_dir = args.sweep_dir / "po_distance_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.config_name}_site_auc_vs_distance"
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    plot_auc_vs_distances(df.dropna(subset=["site_auc"]), out_dir / f"{stem}.png")
    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()