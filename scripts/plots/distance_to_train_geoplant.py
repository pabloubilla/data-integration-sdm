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


def plot_auc_vs_distance(df, path, log_x=True):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["distance_to_po_km"], df["site_auc"], s=4, alpha=0.15, c="#2166AC")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("distance to nearest PO record (km)")
    ax.set_ylabel("site-level AUC")
    fig.tight_layout()
    fig.savefig(path / "po_distance_analysis.png", dpi=200)
    plt.close(fig)


def plot_delta_vs_distance(df, path, log_x=True):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["distance_to_po_km"], df["delta_auc"], s=4, alpha=0.15, c="#B2182B")
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("distance to nearest PO record (km)")
    ax.set_ylabel("site AUC delta (POPA - PA)")
    fig.tight_layout()
    fig.savefig(path / "po_distance_delta_analysis.png", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    best = load_best_params(args)
    pa_best = load_best_pa_params(args)
    device = get_device()

    data = load_geoplant_processed(args.data_path, add_coordinates=True)
    covariates = [c for c in data.covariates if c not in COORD_COLS]
    po_tree = haversine_tree(data.X_po[COORD_COLS].dropna().to_numpy())

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