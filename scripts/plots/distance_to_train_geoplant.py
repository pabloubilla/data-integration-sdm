"""
Distance-to-PO vs site-AUC for a given POPA loss combo. Model can be
reloaded from the final split sweep's saved checkpoints, or retrained.

Usage:
    python scripts/analyze_po_distance_effect.py --loss_po_name deep_maxent --loss_pa_name balanced_bce --mode reload
    python scripts/analyze_po_distance_effect.py --loss_po_name deep_maxent --loss_pa_name balanced_bce --mode retrain
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

from isdm.load_data import load_geoplant_processed
from isdm.splits import load_split_specs, load_split, filter_and_remap, subset_list, make_loader, train_double_source
from isdm.models import MLP
from isdm.losses import LOSS_REGISTRY, IntegratedLoss
from isdm.evaluation import predict_logits, per_site_auc_sparse
from isdm.utils import get_device

COORD_COLS = ["lon", "lat"]  # loaded via add_coordinates, never fed to the model


def infer_mlp_architecture(state_dict):
    idx = [int(k.split(".")[1]) for k in state_dict if k.startswith("hidden_layers.") and k.endswith(".weight")]
    return state_dict["hidden_layers.0.weight"].shape[0], max(idx) + 1


def prepare_split(split_row, split_dir, data, covariates, use_overlapping_species):
    split = load_split(split_dir, split_row["split_file"])
    train_idx, test_idx = split["train_idx"], split["test_idx"]
    overlap = split["species_list"]
    num_classes = len(overlap if use_overlapping_species else data.species)

    X_pa_all = data.X_pa_train.reset_index(drop=True)
    y_pa_all = data.y_pa_train
    X_pa_train_df = X_pa_all.iloc[train_idx].copy()
    X_pa_test_df = X_pa_all.iloc[test_idx].copy()
    y_pa_train = subset_list(y_pa_all, train_idx)
    y_pa_test = subset_list(y_pa_all, test_idx)
    if use_overlapping_species:
        y_pa_train = filter_and_remap(overlap, y_pa_train)
        y_pa_test = filter_and_remap(overlap, y_pa_test)

    site_lonlat = X_pa_test_df[COORD_COLS].to_numpy()  # kept separately, never part of `covariates`
    return X_pa_train_df, y_pa_train, X_pa_test_df, y_pa_test, site_lonlat, num_classes, overlap


def get_model_and_scaler(mode, split_row, split_dir, data, covariates, args, best, device):
    if mode == "reload":
        exp_dir = args.sweep_dir / "popa_split_sweep" / args.config_name / split_row["split_id"]
        state_dict = torch.load(exp_dir / "model.pt", map_location=device)
        with open(exp_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        hidden_dim, hidden_layers = infer_mlp_architecture(state_dict)

        split = load_split(split_dir, split_row["split_file"])
        num_classes = len(split["species_list"] if args.use_overlapping_species else data.species)
        model = MLP(input_size=len(covariates), output_size=num_classes,
                    hidden_size=hidden_dim, hidden_layers=hidden_layers)
        model.load_state_dict(state_dict)
        return model.to(device).eval(), scaler

    X_pa_train_df, y_pa_train, _, _, _, num_classes, overlap = prepare_split(
        split_row, split_dir, data, covariates, args.use_overlapping_species)

    X_po_df = data.X_po.reset_index(drop=True)
    y_po = data.y_po
    if args.use_overlapping_species:
        y_po = filter_and_remap(overlap, y_po)

    scaler = StandardScaler().fit(pd.concat([X_po_df[covariates], X_pa_train_df[covariates]]))
    X_po = scaler.transform(X_po_df[covariates]).astype(np.float32)
    X_pa_train = scaler.transform(X_pa_train_df[covariates]).astype(np.float32)

    batch_size = int(best["param_batch_size"])
    po_loader = make_loader(X_po, y_po, num_classes=num_classes, batch_size=batch_size, shuffle=True)
    pa_loader = make_loader(X_pa_train, y_pa_train, num_classes=num_classes, batch_size=batch_size, shuffle=True)

    model = MLP(input_size=len(covariates), output_size=num_classes,
                hidden_size=int(best["param_hidden_dim"]), hidden_layers=int(best["param_hidden_layers"]))
    criterion = IntegratedLoss(
        source1_loss=LOSS_REGISTRY[args.loss_po_name](), source2_loss=LOSS_REGISTRY[args.loss_pa_name](),
        source1_weight=float(best["param_w_po"]), source2_weight=float(best["param_w_pa"]),
        clamp_source1=(-10, 10), clamp_source2=(-10, 10),
    )
    train_double_source(
        model=model, criterion=criterion, source1_loader=po_loader, source2_loader=pa_loader,
        device=device, epochs=int(round(best["best_epoch"])), lr=float(best["param_lr"]),
        weight_decay=float(best["param_weight_decay"]), source1_name="po", source2_name="pa",
        max_steps_per_epoch=max(len(po_loader), len(pa_loader)),
    )
    return model.to(device).eval(), scaler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="GeoPlant")
    p.add_argument("--split_type", default="geographical")
    p.add_argument("--region", default="france")
    p.add_argument("--use_overlapping_species", action="store_true", default=True)
    p.add_argument("--loss_po_name", default="deep_maxent")
    p.add_argument("--loss_pa_name", default="bce_ippp")
    p.add_argument("--mode", choices=["reload", "retrain"], default="retrain")
    args = p.parse_args()

    overlap_name = "intersect" if args.use_overlapping_species else "union"
    data_path = f"data/processed/{args.dataset_name}/{args.region}"
    split_dir = Path(f"outputs/splits/{args.dataset_name}/{args.region}_bands/{args.split_type}")
    args.sweep_dir = Path(f"outputs/split_sweep/{args.dataset_name}/{args.region}_bands/{args.split_type}/{overlap_name}")
    tune_dir = Path(f"outputs/tune/{args.dataset_name}/{args.region}/{args.split_type}/test_0")
    args.config_name = f"po_{args.loss_po_name}_pa_{args.loss_pa_name}"

    best_df = pd.read_csv(tune_dir / "best_popa_avg_over_distance.csv")
    best = best_df[(best_df.param_loss_po_name == args.loss_po_name)
                   & (best_df.param_loss_pa_name == args.loss_pa_name)].iloc[0]

    data = load_geoplant_processed(data_path, add_coordinates=True)
    covariates = [c for c in data.covariates if c not in COORD_COLS]  # lon/lat excluded from model input, always
    device = get_device()
    po_tree = cKDTree(data.X_po[COORD_COLS].dropna().to_numpy())

    split_table = load_split_specs(split_dir)
    split_table = split_table[~split_table["option"].str.contains("val", case=False, na=False)]

    rows = []
    for _, split_row in split_table.iterrows():
        model, scaler = get_model_and_scaler(args.mode, split_row, split_dir, data, covariates, args, best, device)

        _, _, X_pa_test_df, y_pa_test, site_lonlat, num_classes, _ = prepare_split(
            split_row, split_dir, data, covariates, args.use_overlapping_species)
        X_pa_test = scaler.transform(X_pa_test_df[covariates]).astype(np.float32)

        logits = predict_logits(model, X_pa_test, device=device)
        aucs_site = per_site_auc_sparse(logits=logits, y_lists=y_pa_test, num_classes=num_classes)
        dist_to_po, _ = po_tree.query(site_lonlat, k=1)

        for i, (_, auc) in enumerate(aucs_site.items()):
            rows.append({"split_id": split_row["split_id"], "option": split_row["option"],
                         "lon": site_lonlat[i, 0], "lat": site_lonlat[i, 1],
                         "distance_to_po": dist_to_po[i], "site_auc": auc})
        print(f"{split_row['split_id']} ({args.mode}): {len(aucs_site)} sites")

    df = pd.DataFrame(rows).dropna(subset=["site_auc"])
    out_dir = args.sweep_dir / "po_distance_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{args.config_name}_site_auc_vs_po_distance.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["distance_to_po"], df["site_auc"], s=4, alpha=0.15, c="#2166AC")
    bins = np.quantile(df["distance_to_po"], np.linspace(0, 1, 11))
    df["dist_bin"] = pd.cut(df["distance_to_po"], bins, include_lowest=True)
    binned = df.groupby("dist_bin", observed=True)["site_auc"].mean()
    ax.plot([b.mid for b in binned.index], binned.values, color="#B2182B", linewidth=2, marker="o", label="binned mean")
    ax.set_xlabel("distance to nearest PO point (° lon/lat, unprojected)")
    ax.set_ylabel("site-level AUC")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{args.config_name}_site_auc_vs_po_distance.png", dpi=200)
    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()