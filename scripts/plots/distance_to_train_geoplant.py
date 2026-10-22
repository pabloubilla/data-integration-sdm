import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from isdm.evaluation import predict_logits, per_site_auc_sparse
from isdm.utils import haversine_tree, nearest_km
from isdm.load_data import load_geoplant_processed
from isdm.losses import LOSS_REGISTRY, IntegratedLoss
from isdm.models import MLP
from isdm.splits import filter_and_remap, load_split, load_split_specs, make_loader, subset_list, train_double_source, train_single_source
from isdm.utils import get_device
from statsmodels.nonparametric.smoothers_lowess import lowess



COORD_COLS = ["lon", "lat"]  # never fed to the model


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
    args = p.parse_args()

    overlap = "intersect" if args.use_overlapping_species else "union"
    args.data_path = f"data/processed/{args.dataset_name}/{args.region}"
    args.split_dir = Path(f"outputs/splits/{args.dataset_name}/{args.region}_bands/{args.split_type}")
    args.sweep_dir = Path(f"outputs/split_sweep/{args.dataset_name}/{args.region}_bands/{args.split_type}/{overlap}")
    args.tune_dir = Path(f"outputs/tune/{args.dataset_name}/{args.tune_region}/{args.split_type}/test_0")
    args.config_name = f"po_{args.loss_po_name}_pa_{args.loss_pa_name}"
    args.pa_config_name = f"pa_{args.loss_pa_name}"
    return args



def load_best_params(args):
    df = pd.read_csv(args.tune_dir / "best_popa_avg_over_distance.csv")
    match = df[(df.param_loss_po_name == args.loss_po_name) & (df.param_loss_pa_name == args.loss_pa_name)]
    if match.empty:
        raise ValueError(f"No tuned params found for {args.config_name}")
    return match.iloc[0]


def load_best_pa_params(args):
    df = pd.read_csv(args.tune_dir / "best_pa_avg_over_distance.csv")
    match = df[df.param_loss_name == args.loss_pa_name]
    if match.empty:
        raise ValueError(f"No tuned PA params found for {args.pa_config_name}")
    return match.iloc[0]


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


def infer_mlp_architecture(state_dict):
    idx = [int(k.split(".")[1]) for k in state_dict if k.startswith("hidden_layers.") and k.endswith(".weight")]
    return state_dict["hidden_layers.0.weight"].shape[0], max(idx) + 1


def reload_model(split_row, s, args, n_features, device):
    exp_dir = args.sweep_dir / args.config_name / 'popa_split_sweep' / split_row["split_id"]
    state_dict = torch.load(exp_dir / "model.pt", map_location=device)
    with open(exp_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    hidden_dim, hidden_layers = infer_mlp_architecture(state_dict)
    model = MLP(input_size=n_features, output_size=s["num_classes"],
                hidden_size=hidden_dim, hidden_layers=hidden_layers)
    model.load_state_dict(state_dict)
    return model, scaler


def retrain_model(s, args, data, covariates, best, device):
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


def reload_pa_model(split_row, s, args, n_features, device):
    exp_dir = args.sweep_dir / args.pa_config_name / 'pa_split_sweep' / split_row["split_id"]
    state_dict = torch.load(exp_dir / "model.pt", map_location=device)
    with open(exp_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    hidden_dim, hidden_layers = infer_mlp_architecture(state_dict)
    model = MLP(input_size=n_features, output_size=s["num_classes"],
                hidden_size=hidden_dim, hidden_layers=hidden_layers)
    model.load_state_dict(state_dict)
    return model, scaler


def retrain_pa_model(s, args, covariates, pa_best, device):
    scaler = StandardScaler().fit(s["X_train"][covariates])
    X_pa = scaler.transform(s["X_train"][covariates]).astype(np.float32)

    batch_size = int(pa_best["param_batch_size"])
    pa_loader = make_loader(X_pa, s["y_train"], num_classes=s["num_classes"], batch_size=batch_size, shuffle=True)

    model = MLP(input_size=len(covariates), output_size=s["num_classes"],
                hidden_size=int(pa_best["param_hidden_dim"]), hidden_layers=int(pa_best["param_hidden_layers"]))
    train_single_source(
        model=model, criterion=LOSS_REGISTRY[args.loss_pa_name](), train_loader=pa_loader, val_loader=None,
        device=device, epochs=int(round(pa_best["best_epoch"])), lr=float(pa_best["param_lr"]),
        weight_decay=float(pa_best["param_weight_decay"]),
    )
    return model, scaler


def add_smoothed_mean(ax, x, y, n_bins=100, window=3, color="black", label="smoothed mean"):
    d = pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y)}).dropna()
    d["bin"] = pd.qcut(d["x"], n_bins, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(x=("x", "median"), y=("y", "mean"))
    y_s = g["y"].rolling(window, center=True, min_periods=1).mean()
    ax.plot(g["x"], y_s, color=color, linewidth=2, label=label)
    ax.legend(loc="best", frameon=False)


def add_lowess(ax, x, y, log_x=True, frac=0.05, color="#E08214", label="LOWESS"):
    d = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}).dropna()
    if log_x:
        d = d[d["x"] > 0]  # log10 undefined at 0; these sites are not visible on a log axis anyway
        xs = np.log10(d["x"].to_numpy())
    else:
        xs = d["x"].to_numpy()
    fit = lowess(d["y"].to_numpy(), xs, frac=frac, delta=0.01 * np.ptp(xs))
    x_fit = 10 ** fit[:, 0] if log_x else fit[:, 0]
    ax.plot(x_fit, fit[:, 1], color=color, linewidth=2, label=label)
    ax.legend(loc="best", frameon=False)


def plot_auc_vs_distance(df, path, log_x=True, smooth_mean=True, smooth_lowess=True, lowess_frac=0.05):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["distance_to_po_km"], df["site_auc"], s=1, alpha=0.15, c="#2166AC")
    if smooth_mean:
        add_smoothed_mean(ax, df["distance_to_po_km"], df["site_auc"])
    if smooth_lowess:
        add_lowess(ax, df["distance_to_po_km"], df["site_auc"], log_x=log_x, frac=lowess_frac)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("distance to nearest PO record (km)")
    ax.set_ylabel("site-level AUC")
    fig.tight_layout()
    fig.savefig(path / "po_distance_analysis.png", dpi=200)
    plt.close(fig)


def plot_delta_vs_distance(df, path, log_x=True):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["distance_to_po_km"], df["delta_auc"], s=1, alpha=0.15, c="#B2182B")
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("distance to nearest PO record (km)")
    ax.set_ylabel("site AUC delta (POPA - PA)")
    fig.tight_layout()
    fig.savefig(path / "po_distance_delta_analysis.png", dpi=200)
    plt.close(fig)


def plot_auc_with_both_distances(df, path, log_scale=True):
    fig, ax = plt.subplots(figsize=(6, 5))
    if log_scale:
        df['distance_to_pa_train_km_log'] = np.log10(df['distance_to_pa_train_km'] + 1e-6)
        sc = ax.scatter(df["distance_to_po_km"], df["site_auc"], s=.2, alpha=1,
                        c=df["distance_to_pa_train_km_log"], cmap="magma", vmin=df["distance_to_pa_train_km_log"].min(), vmax=df["distance_to_pa_train_km_log"].max())
    else:   
        sc = ax.scatter(df["distance_to_po_km"], df["site_auc"], s=1, alpha=0.2,
                    c=df["distance_to_pa_train_km"], cmap="magma", vmin=0, vmax=df["distance_to_pa_train_km"].max())
    if log_scale:
        ax.set_xscale("log")
    ax.set_xlabel("distance to nearest PO record (km)")
    ax.set_ylabel("site-level AUC")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("distance to nearest PA train record (km)")
    fig.tight_layout()
    fig.savefig(path / "po_distance_analysis_with_pa_distance.png", dpi=200)
    plt.close(fig)


def plot_auc_by_pa_distance_bins(df, path, bins=(0, 10, 100, np.inf), log_x=True, lowess_frac=0.1, min_sites=50):
    d = df.dropna(subset=["distance_to_po_km", "distance_to_pa_train_km", "site_auc"])
    labels = [f"{lo:g}–{hi:g} km" if np.isfinite(hi) else f">{lo:g} km" for lo, hi in zip(bins[:-1], bins[1:])]
    d = d.assign(pa_bin=pd.cut(d["distance_to_pa_train_km"], bins=bins, labels=labels, include_lowest=True))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(labels)))

    n = len(labels) + 1
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharex=True, sharey=True)
    for ax, lab, color in zip(axes, labels, colors):
        g = d[d["pa_bin"] == lab]
        ax.scatter(g["distance_to_po_km"], g["site_auc"], s=1, alpha=0.2, c=[color])
        ax.set_title(f"PA dist {lab} (n={len(g)})", fontsize=9)
        if len(g) >= min_sites:
            add_lowess(ax, g["distance_to_po_km"], g["site_auc"], log_x=log_x, frac=lowess_frac, color="black")
            add_lowess(axes[-1], g["distance_to_po_km"], g["site_auc"], log_x=log_x, frac=lowess_frac, color=color, label=lab)
    axes[-1].set_title("LOWESS per PA-distance bin", fontsize=9)

    for ax in axes:
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel("distance to nearest PO (km)")
    axes[0].set_ylabel("site-level AUC")
    fig.tight_layout()
    fig.savefig(path / "po_distance_by_pa_bins.png", dpi=200)
    plt.close(fig)


def plot_auc_delta_by_pa_distance_bins(df, path, bins=(0, 10, 100, np.inf), log_x=True, lowess_frac=0.1, min_sites=50):
    d = df.dropna(subset=["distance_to_po_km", "distance_to_pa_train_km", "delta_auc"])
    labels = [f"{lo:g}–{hi:g} km" if np.isfinite(hi) else f">{lo:g} km" for lo, hi in zip(bins[:-1], bins[1:])]
    d = d.assign(pa_bin=pd.cut(d["distance_to_pa_train_km"], bins=bins, labels=labels, include_lowest=True))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(labels)))

    n = len(labels) + 1
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharex=True, sharey=True)
    for ax, lab, color in zip(axes, labels, colors):
        g = d[d["pa_bin"] == lab]
        ax.scatter(g["distance_to_po_km"], g["delta_auc"], s=1, alpha=0.2, c=[color])
        ax.set_title(f"PA dist {lab} (n={len(g)})", fontsize=9)
        if len(g) >= min_sites:
            add_lowess(ax, g["distance_to_po_km"], g["delta_auc"], log_x=log_x, frac=lowess_frac, color="black")
            add_lowess(axes[-1], g["distance_to_po_km"], g["delta_auc"], log_x=log_x, frac=lowess_frac, color=color, label=lab)
    axes[-1].set_title("LOWESS per PA-distance bin", fontsize=9)

    for ax in axes:
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel("distance to nearest PO (km)")
    axes[0].set_ylabel("site-level AUC delta")
    fig.tight_layout()
    fig.savefig(path / "po_distance_delta_by_pa_bins.png", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    best = load_best_params(args)
    pa_best = load_best_pa_params(args)
    device = get_device()

    data = load_geoplant_processed(args.data_path, add_coordinates=True)
    covariates = [c for c in data.covariates if c not in COORD_COLS]
    po_lonlat = data.X_po[COORD_COLS].dropna().to_numpy()
    po_tree = haversine_tree(po_lonlat)

    splits = load_split_specs(args.split_dir)
    splits = splits[~splits["option"].str.contains("val", case=False, na=False)]

    results = []
    for _, split_row in splits.iterrows():
        s = prepare_split(split_row, args, data)

        if args.mode == "reload":
            model, scaler = reload_model(split_row, s, args, len(covariates), device)
            pa_model, pa_scaler = reload_pa_model(split_row, s, args, len(covariates), device)
        else:
            model, scaler = retrain_model(s, args, data, covariates, best, device)
            pa_model, pa_scaler = retrain_pa_model(s, args, covariates, pa_best, device)
        model = model.to(device).eval()
        pa_model = pa_model.to(device).eval()

        X_test = scaler.transform(s["X_test"][covariates]).astype(np.float32)
        X_test_pa = pa_scaler.transform(s["X_test"][covariates]).astype(np.float32)

        logits = predict_logits(model, X_test, device=device)
        pa_logits = predict_logits(pa_model, X_test_pa, device=device)
        aucs = per_site_auc_sparse(logits=logits, y_lists=s["y_test"], num_classes=s["num_classes"])
        pa_aucs = per_site_auc_sparse(logits=pa_logits, y_lists=s["y_test"], num_classes=s["num_classes"])
        aucs = pd.Series(dict(aucs.items()), dtype=float)  # keys = row positions in X_test
        pa_aucs = pd.Series(dict(pa_aucs.items()), dtype=float)

        lonlat_pa_test = s["X_test"][COORD_COLS].to_numpy()[aucs.index.to_numpy(dtype=int)]
        lonlat_pa_train = s["X_train"][COORD_COLS].to_numpy()
        pa_train_tree = haversine_tree(lonlat_pa_train)


        results.append(pd.DataFrame({
            "split_id": split_row["split_id"],
            "option": split_row["option"],
            "lon": lonlat_pa_test[:, 0],
            "lat": lonlat_pa_test[:, 1],
            "distance_to_po_km": nearest_km(po_tree, lonlat_pa_test),
            "distance_to_pa_train_km": nearest_km(pa_train_tree, lonlat_pa_test),
            "site_auc": aucs.to_numpy(),
            "pa_site_auc": pa_aucs.reindex(aucs.index).to_numpy(),
        }))
        print(f"{split_row['split_id']} ({args.mode}): {len(aucs)} sites")

    df = pd.concat(results, ignore_index=True)
    df["delta_auc"] = df["site_auc"] - df["pa_site_auc"]

    out_dir = args.sweep_dir / "po_distance_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"po_distance_analysis.csv", index=False)
    plot_auc_vs_distance(df, out_dir)
    plot_delta_vs_distance(df, out_dir)
    plot_auc_with_both_distances(df, out_dir)
    plot_auc_by_pa_distance_bins(df, out_dir, bins=(0, 10, 100, np.inf), log_x=True, lowess_frac=0.1, min_sites=50)
    plot_auc_delta_by_pa_distance_bins(df, out_dir, bins=(0, 10, 100, np.inf), log_x=True, lowess_frac=0.1, min_sites=50)

    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()


# """
# Site-level AUC vs great-circle distance (km) to the nearest PO record,
# for one PO/PA loss combination.

# Usage:
#     python scripts/analyze_po_distance_effect.py --loss_po_name deep_maxent --loss_pa_name balanced_bce --mode reload
#     python scripts/analyze_po_distance_effect.py --loss_po_name deep_maxent --loss_pa_name balanced_bce --mode retrain
# """

# import argparse
# import pickle
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import torch
# from sklearn.preprocessing import StandardScaler

# from isdm.evaluation import predict_logits, per_site_auc_sparse
# from isdm.utils import haversine_tree, nearest_km
# from isdm.load_data import load_geoplant_processed
# from isdm.losses import LOSS_REGISTRY, IntegratedLoss
# from isdm.models import MLP
# from isdm.splits import filter_and_remap, load_split, load_split_specs, make_loader, subset_list, train_double_source
# from isdm.utils import get_device

# COORD_COLS = ["lon", "lat"]  # never fed to the model


# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--dataset_name", default="GeoPlant")
#     p.add_argument("--split_type", default="geographical")
#     p.add_argument("--region", default="bene")
#     p.add_argument("--tune_region", default="france")
#     p.add_argument("--use_overlapping_species", action=argparse.BooleanOptionalAction, default=True)
#     p.add_argument("--loss_po_name", default="deep_maxent")
#     p.add_argument("--loss_pa_name", default="bce_ippp")
#     p.add_argument("--mode", choices=["reload", "retrain"], default="retrain")
#     args = p.parse_args()

#     overlap = "intersect" if args.use_overlapping_species else "union"
#     args.data_path = f"data/processed/{args.dataset_name}/{args.region}"
#     args.split_dir = Path(f"outputs/splits/{args.dataset_name}/{args.region}_bands/{args.split_type}")
#     args.sweep_dir = Path(f"outputs/split_sweep/{args.dataset_name}/{args.region}_bands/{args.split_type}/{overlap}")
#     args.tune_dir = Path(f"outputs/tune/{args.dataset_name}/{args.tune_region}/{args.split_type}/test_0")
#     args.config_name = f"po_{args.loss_po_name}_pa_{args.loss_pa_name}"
#     return args


# def load_best_params(args):
#     df = pd.read_csv(args.tune_dir / "best_popa_avg_over_distance.csv")
#     match = df[(df.param_loss_po_name == args.loss_po_name) & (df.param_loss_pa_name == args.loss_pa_name)]
#     if match.empty:
#         raise ValueError(f"No tuned params found for {args.config_name}")
#     return match.iloc[0]


# def prepare_split(split_row, args, data):
#     split = load_split(args.split_dir, split_row["split_file"])
#     overlap = split["species_list"]
#     X_pa = data.X_pa_train.reset_index(drop=True)

#     def take(idx):
#         y = subset_list(data.y_pa_train, idx)
#         if args.use_overlapping_species:
#             y = filter_and_remap(overlap, y)
#         return X_pa.iloc[idx], y

#     X_train, y_train = take(split["train_idx"])
#     X_test, y_test = take(split["test_idx"])
#     num_classes = len(overlap if args.use_overlapping_species else data.species)
#     return dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
#                 overlap=overlap, num_classes=num_classes)


# def infer_mlp_architecture(state_dict):
#     idx = [int(k.split(".")[1]) for k in state_dict if k.startswith("hidden_layers.") and k.endswith(".weight")]
#     return state_dict["hidden_layers.0.weight"].shape[0], max(idx) + 1


# def reload_model(split_row, s, args, n_features, device):
    
#     exp_dir = args.sweep_dir / args.config_name / 'popa_split_sweep' / split_row["split_id"]
#     state_dict = torch.load(exp_dir / "model.pt", map_location=device)
#     with open(exp_dir / "scaler.pkl", "rb") as f:
#         scaler = pickle.load(f)

#     hidden_dim, hidden_layers = infer_mlp_architecture(state_dict)
#     model = MLP(input_size=n_features, output_size=s["num_classes"],
#                 hidden_size=hidden_dim, hidden_layers=hidden_layers)
#     model.load_state_dict(state_dict)
#     return model, scaler


# def retrain_model(s, args, data, covariates, best, device):
#     X_po_df = data.X_po.reset_index(drop=True)
#     y_po = filter_and_remap(s["overlap"], data.y_po) if args.use_overlapping_species else data.y_po

#     scaler = StandardScaler().fit(pd.concat([X_po_df[covariates], s["X_train"][covariates]]))
#     X_po = scaler.transform(X_po_df[covariates]).astype(np.float32)
#     X_pa = scaler.transform(s["X_train"][covariates]).astype(np.float32)

#     batch_size = int(best["param_batch_size"])
#     po_loader = make_loader(X_po, y_po, num_classes=s["num_classes"], batch_size=batch_size, shuffle=True)
#     pa_loader = make_loader(X_pa, s["y_train"], num_classes=s["num_classes"], batch_size=batch_size, shuffle=True)

#     model = MLP(input_size=len(covariates), output_size=s["num_classes"],
#                 hidden_size=int(best["param_hidden_dim"]), hidden_layers=int(best["param_hidden_layers"]))
#     criterion = IntegratedLoss(
#         source1_loss=LOSS_REGISTRY[args.loss_po_name](), source2_loss=LOSS_REGISTRY[args.loss_pa_name](),
#         source1_weight=float(best["param_w_po"]), source2_weight=float(best["param_w_pa"]),
#         clamp_source1=(-10, 10), clamp_source2=(-10, 10),
#     )
#     train_double_source(
#         model=model, criterion=criterion, source1_loader=po_loader, source2_loader=pa_loader,
#         device=device, epochs=int(round(best["best_epoch"])), lr=float(best["param_lr"]),
#         weight_decay=float(best["param_weight_decay"]), source1_name="po", source2_name="pa",
#         max_steps_per_epoch=max(len(po_loader), len(pa_loader)),
#     )
#     return model, scaler


# def plot_auc_vs_distance(df, path, n_cats=40):
#     fig, ax = plt.subplots(figsize=(6, 5))

#     # create 5 categories from distance_to_pa_train_km (call it cat_dist)
#     df["cat_dist"] = pd.qcut(df["distance_to_pa_train_km"], n_cats, duplicates="drop")


#     ax.scatter(df["distance_to_po_km"], df["site_auc"], s=4, alpha=0.15
#                # color from category
#                 , c=pd.Categorical(df["cat_dist"]).codes, cmap="tab10"
#                )

#     ax.set_xlabel("distance to nearest PO record (km)")
#     ax.set_ylabel("site-level AUC")
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig(path / "po_distance_analysis.png", dpi=200)

#     for i, (cat, group) in enumerate(df.groupby("cat_dist", observed=True)):
#         fig, ax = plt.subplots(figsize=(6, 5))
#         ax.scatter(group["distance_to_po_km"], group["site_auc"], s=4, alpha=0.15, c="#2166AC")
#         ax.set_xlabel("distance to nearest PO record (km)")
#         ax.set_ylabel("site-level AUC")
#         ax.set_title(f"Distance to PA train: {cat.left:.1f} - {cat.right:.1f} km")
#         ax.legend()
#         fig.tight_layout()
#         fig.savefig(path / f"po_distance_analysis_cat_{i}.png", dpi=200)

#     plt.close(fig)



# def main():
#     args = parse_args()
#     best = load_best_params(args)
#     device = get_device()

#     data = load_geoplant_processed(args.data_path, add_coordinates=True)
#     covariates = [c for c in data.covariates if c not in COORD_COLS]
#     po_tree = haversine_tree(data.X_po[COORD_COLS].dropna().to_numpy())

#     splits = load_split_specs(args.split_dir)
#     splits = splits[~splits["option"].str.contains("val", case=False, na=False)]

#     results = []
#     for _, split_row in splits.iterrows():
#         s = prepare_split(split_row, args, data)
#         if args.mode == "reload":
#             model, scaler = reload_model(split_row, s, args, len(covariates), device)
#         else:
#             model, scaler = retrain_model(s, args, data, covariates, best, device)
#         model = model.to(device).eval()

#         X_test = scaler.transform(s["X_test"][covariates]).astype(np.float32)

#         logits = predict_logits(model, X_test, device=device)
#         aucs = per_site_auc_sparse(logits=logits, y_lists=s["y_test"], num_classes=s["num_classes"])
#         aucs = pd.Series(dict(aucs.items()), dtype=float)  # keys = row positions in X_test

#         lonlat_pa_test = s["X_test"][COORD_COLS].to_numpy()[aucs.index.to_numpy(dtype=int)]
#         lonlat_pa_train = s["X_train"][COORD_COLS].to_numpy()
#         pa_train_tree = haversine_tree(lonlat_pa_train)
#         results.append(pd.DataFrame({
#             "split_id": split_row["split_id"],
#             "option": split_row["option"],
#             "lon": lonlat_pa_test[:, 0],
#             "lat": lonlat_pa_test[:, 1],
#             "distance_to_po_km": nearest_km(po_tree, lonlat_pa_test),
#             "distance_to_pa_train_km": nearest_km(pa_train_tree, lonlat_pa_test),
#             "site_auc": aucs.to_numpy(),
#         }))
#         print(f"{split_row['split_id']} ({args.mode}): {len(aucs)} sites")

#     df = pd.concat(results, ignore_index=True)

#     out_dir = args.sweep_dir / "po_distance_analysis"
#     out_dir.mkdir(parents=True, exist_ok=True)
#     df.to_csv(out_dir / f"po_distance_analysis.csv", index=False)
#     plot_auc_vs_distance(df, out_dir)


#     print(f"Saved -> {out_dir}")


# if __name__ == "__main__":
#     main()