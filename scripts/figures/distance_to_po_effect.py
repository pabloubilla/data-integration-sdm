"""
A posteriori analysis of the effect of PO distance to performance
"""
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess

from isdm.evaluation import predict_logits, per_site_auc_sparse
from isdm.load_data import load_geoplant_processed
from isdm.losses import LOSS_REGISTRY, IntegratedLoss
from isdm.models import MLP
from isdm.splits import (filter_and_remap, load_split, load_split_specs, make_loader, subset_list,
                         train_double_source, train_single_source)
from isdm.utils import get_device, haversine_tree, nearest_km

COORD_COLS = ["lon", "lat"]                         # never fed to the model
FIG_DIR = Path("outputs/figures/po_analysis")
BINS = (0, 10, 100, np.inf)                         # distance to nearest PA training site (km)
METHODS = {"POPA": ("site_auc", "#2166AC", "-"),    # name: (column, colour, linestyle)
           "PA": ("pa_site_auc", "#B2182B", "--")}
FRAC = 0.1                                          # LOWESS span (share of sites per local fit)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="GeoPlant")
    p.add_argument("--split_type", default="geographical")
    p.add_argument("--region", default="sparse_pa")
    p.add_argument("--tune_region", default="france")
    p.add_argument("--use_overlapping_species", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--loss_po_name", default="deep_maxent")
    p.add_argument("--loss_pa_name", default="bce")
    p.add_argument("--mode", choices=["reload", "retrain"], default="reload")
    p.add_argument("--recompute", action="store_true", help="recompute per-site AUCs even if the CSV exists")
    args = p.parse_args()

    overlap = "intersect" if args.use_overlapping_species else "union"
    args.data_path = f"data/processed/{args.dataset_name}/{args.region}"
    args.split_dir = Path(f"outputs/splits/{args.dataset_name}/{args.region}_bands/{args.split_type}")
    args.sweep_dir = Path(f"outputs/split_sweep/{args.dataset_name}/{args.region}_bands/{args.split_type}/{overlap}")
    args.tune_dir = Path(f"outputs/tune/{args.dataset_name}/{args.tune_region}/{args.split_type}/test_0")
    args.config_name = f"po_{args.loss_po_name}_pa_{args.loss_pa_name}"
    args.pa_config_name = f"pa_{args.loss_pa_name}"
    return args


# ---------------------------------------------------------------- models

def load_best_params(args):
    popa = pd.read_csv(args.tune_dir / "best_popa_avg_over_distance.csv")
    popa = popa[(popa.param_loss_po_name == args.loss_po_name) & (popa.param_loss_pa_name == args.loss_pa_name)]
    pa = pd.read_csv(args.tune_dir / "best_pa_avg_over_distance.csv")
    pa = pa[pa.param_loss_name == args.loss_pa_name]
    if popa.empty or pa.empty:
        raise ValueError(f"No tuned params found for {args.config_name} / {args.pa_config_name}")
    return popa.iloc[0], pa.iloc[0]


def prepare_split(split_row, args, data):
    split = load_split(args.split_dir, split_row["split_file"])
    overlap = split["species_list"]
    X_pa = data.X_pa_train.reset_index(drop=True)

    def take(idx):
        y = subset_list(data.y_pa_train, idx)
        if args.use_overlapping_species:
            y = filter_and_remap(overlap, y)
        return X_pa.iloc[idx], y

    X_train, y_train = take(split["train_idx"])
    X_test, y_test = take(split["test_idx"])
    num_classes = len(overlap if args.use_overlapping_species else data.species)
    return dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                overlap=overlap, num_classes=num_classes)


def reload_model(exp_dir, n_features, num_classes, device):
    state_dict = torch.load(exp_dir / "model.pt", map_location=device)
    with open(exp_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    idx = [int(k.split(".")[1]) for k in state_dict if k.startswith("hidden_layers.") and k.endswith(".weight")]
    model = MLP(input_size=n_features, output_size=num_classes,
                hidden_size=state_dict["hidden_layers.0.weight"].shape[0], hidden_layers=max(idx) + 1)
    model.load_state_dict(state_dict)
    return model, scaler


def retrain_popa_model(s, args, data, covariates, best, device):
    X_po_df = data.X_po.reset_index(drop=True)
    y_po = filter_and_remap(s["overlap"], data.y_po) if args.use_overlapping_species else data.y_po

    scaler = StandardScaler().fit(pd.concat([X_po_df[covariates], s["X_train"][covariates]]))
    X_po = scaler.transform(X_po_df[covariates]).astype(np.float32)
    X_pa = scaler.transform(s["X_train"][covariates]).astype(np.float32)

    batch_size = int(best["param_batch_size"])
    po_loader = make_loader(X_po, y_po, num_classes=s["num_classes"], batch_size=batch_size, shuffle=True)
    pa_loader = make_loader(X_pa, s["y_train"], num_classes=s["num_classes"], batch_size=batch_size, shuffle=True)

    model = MLP(input_size=len(covariates), output_size=s["num_classes"],
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
    return model, scaler


def retrain_pa_model(s, args, covariates, pa_best, device):
    scaler = StandardScaler().fit(s["X_train"][covariates])
    X_pa = scaler.transform(s["X_train"][covariates]).astype(np.float32)
    pa_loader = make_loader(X_pa, s["y_train"], num_classes=s["num_classes"],
                            batch_size=int(pa_best["param_batch_size"]), shuffle=True)

    model = MLP(input_size=len(covariates), output_size=s["num_classes"],
                hidden_size=int(pa_best["param_hidden_dim"]), hidden_layers=int(pa_best["param_hidden_layers"]))
    train_single_source(
        model=model, criterion=LOSS_REGISTRY[args.loss_pa_name](), train_loader=pa_loader, val_loader=None,
        device=device, epochs=int(round(pa_best["best_epoch"])), lr=float(pa_best["param_lr"]),
        weight_decay=float(pa_best["param_weight_decay"]),
    )
    return model, scaler


def site_aucs(model, scaler, s, covariates, device):
    model = model.to(device).eval()
    X_test = scaler.transform(s["X_test"][covariates]).astype(np.float32)
    logits = predict_logits(model, X_test, device=device)
    aucs = per_site_auc_sparse(logits=logits, y_lists=s["y_test"], num_classes=s["num_classes"])
    return pd.Series(dict(aucs.items()), dtype=float)  # keys = row positions in X_test


def compute_site_aucs(args):
    device = get_device()
    data = load_geoplant_processed(args.data_path, add_coordinates=True)
    covariates = [c for c in data.covariates if c not in COORD_COLS]
    po_tree = haversine_tree(data.X_po[COORD_COLS].dropna().to_numpy())
    if args.mode == "retrain":
        best, pa_best = load_best_params(args)

    splits = load_split_specs(args.split_dir)
    splits = splits[~splits["option"].str.contains("val", case=False, na=False)]

    results = []
    for _, split_row in splits.iterrows():
        s = prepare_split(split_row, args, data)
        if args.mode == "reload":
            popa = reload_model(args.sweep_dir / args.config_name / "popa_split_sweep" / split_row["split_id"],
                                len(covariates), s["num_classes"], device)
            pa = reload_model(args.sweep_dir / args.pa_config_name / "pa_split_sweep" / split_row["split_id"],
                              len(covariates), s["num_classes"], device)
        else:
            popa = retrain_popa_model(s, args, data, covariates, best, device)
            pa = retrain_pa_model(s, args, covariates, pa_best, device)

        aucs = site_aucs(*popa, s, covariates, device)
        pa_aucs = site_aucs(*pa, s, covariates, device).reindex(aucs.index)

        lonlat = s["X_test"][COORD_COLS].to_numpy()[aucs.index.to_numpy(dtype=int)]
        pa_train_tree = haversine_tree(s["X_train"][COORD_COLS].to_numpy())
        results.append(pd.DataFrame({
            "split_id": split_row["split_id"],
            "option": split_row["option"],
            "lon": lonlat[:, 0],
            "lat": lonlat[:, 1],
            "distance_to_po_km": nearest_km(po_tree, lonlat),
            "distance_to_pa_train_km": nearest_km(pa_train_tree, lonlat),
            "site_auc": aucs.to_numpy(),
            "pa_site_auc": pa_aucs.to_numpy(),
        }))
        print(f"{split_row['split_id']} ({args.mode}): {len(aucs)} sites")

    df = pd.concat(results, ignore_index=True)
    df["delta_auc"] = df["site_auc"] - df["pa_site_auc"]
    return df


# ---------------------------------------------------------------- figure

def lowess_curve(x, y):
    """LOWESS fitted in log10(distance), returned in km for plotting on a log axis."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = x > 0
    lx = np.log10(x[m])
    fit = lowess(y[m], lx, frac=FRAC, delta=0.01 * np.ptp(lx))
    return 10 ** fit[:, 0], fit[:, 1]


def plot_popa_vs_pa(df, name):
    df = df.dropna(subset=["distance_to_po_km", "distance_to_pa_train_km", "site_auc", "pa_site_auc"])
    labels = [f"{lo:g}–{hi:g} km" if np.isfinite(hi) else f">{lo:g} km" for lo, hi in zip(BINS[:-1], BINS[1:])]
    df = df.assign(pa_bin=pd.cut(df["distance_to_pa_train_km"], bins=BINS, labels=labels, include_lowest=True))

    fig, axes = plt.subplots(1, len(labels), figsize=(3.4 * len(labels), 3.6), sharex=True, sharey=True)
    for ax, lab in zip(axes, labels):
        g = df[df["pa_bin"] == lab]
        ax.scatter(g["distance_to_po_km"], g["site_auc"], s=1, alpha=0.08, color="grey", rasterized=True)
        for method, (col, color, ls) in METHODS.items():
            ax.plot(*lowess_curve(g["distance_to_po_km"], g[col]), color=color, ls=ls, lw=2, label=method)
        ax.set_xscale("log")
        ax.set_title(f"Closest PA {lab} (n={len(g)})", fontsize=9)
        ax.set_xlabel("Closest PO (km)")
    axes[0].set_ylabel("AUC-site")
    axes[0].legend(frameon=False, fontsize=8)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=400)
    plt.close(fig)


def main():
    args = parse_args()
    csv = args.sweep_dir / "po_distance_analysis" / f"po_distance_analysis_{args.config_name}_{args.mode}.csv"

    if args.recompute or not csv.exists():
        df = compute_site_aucs(args)
        csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv, index=False)
        print(f"Saved site AUCs -> {csv}")
    else:
        df = pd.read_csv(csv)
        print(f"Loaded cached site AUCs <- {csv} (use --recompute to regenerate)")

    plot_popa_vs_pa(df, f"po_distance_popa_vs_pa_{args.config_name}")
    print(f"Saved figure -> {FIG_DIR}")


if __name__ == "__main__":
    main()