"""Print-only diagnosis of the tuning summaries, for both split types.
Usage: python diagnose_tuning.py [region] [test_number]"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

REGION = sys.argv[1] if len(sys.argv) > 1 else "france"
TEST = sys.argv[2] if len(sys.argv) > 2 else "0"
TARGET = "harmonic_mean_auc"
LOSS_COLS = {"pa": ["param_loss_name"], "po": ["param_loss_name"],
             "popa": ["param_loss_po_name", "param_loss_pa_name"]}
BASE_HP = ["param_lr", "param_weight_decay", "param_hidden_dim", "param_hidden_layers", "param_batch_size"]
GRID = {  # current grid in 03_tune_split.py
    "param_lr": [5e-3, 1e-3, 5e-4, 1e-4],
    "param_weight_decay": [1e-3, 1e-4, 1e-5, 0],
    "param_hidden_dim": [100, 250, 500, 1000],
    "param_hidden_layers": [1, 2, 3],
    "param_batch_size": [50, 100, 200, 500, 1000],
}

pd.set_option("display.width", 200)


def n_shared(complete, key, n_losses):
    per_config = complete.groupby(key)["loss"].nunique()
    return int((per_config == n_losses).sum()), len(per_config)


for split_type in ("geographical", "environmental"):
    tune_dir = Path(f"outputs/tune/GeoPlant/{REGION}/{split_type}/test_{TEST}")
    print("\n" + "#" * 80 + f"\n# {split_type.upper()}  ({tune_dir})\n" + "#" * 80)

    for method, loss_cols in LOSS_COLS.items():
        path = tune_dir / f"summary_{method}.csv"
        if not path.exists():
            print(f"\n[{method}] missing {path}")
            continue
        df = pd.read_csv(path)
        hp = BASE_HP + (["param_w_po"] if "param_w_po" in df.columns else [])
        for c in hp + (["param_w_pa"] if "param_w_pa" in df.columns else []):
            df[c] = df[c].astype(float).round(10)
        df["loss"] = df[loss_cols].astype(str).agg(" + ".join, axis=1)
        df["config"] = df[hp].astype(str).agg("|".join, axis=1)
        df["base"] = df[BASE_HP].astype(str).agg("|".join, axis=1)
        df["in_grid"] = np.all([np.isclose(df[c].values[:, None], np.array(v, float)).any(1)
                                for c, v in GRID.items()], axis=0)
        df["edge"] = df["param_w_po"].isin([0.0, 1.0]) if "param_w_po" in df.columns else False

        print(f"\n=== {method} ===  rows {len(df)} | distances {sorted(df.option.unique())}")
        if "error" in df.columns:
            print(f"  error rows: {df['error'].notna().sum()}")
        print(f"  NaN target rows: {df[TARGET].isna().sum()}")
        print(f"  duplicate (config, loss, distance) rows (reruns): "
              f"{df.duplicated(['loss', 'config', 'option']).sum()}")
        if {"param_w_po", "param_w_pa"} <= set(df.columns):
            print(f"  rows with w_po + w_pa != 1: {((df.param_w_po + df.param_w_pa - 1).abs() > 1e-6).sum()}")

        # complete = valid score on every distance present for this method
        n_req = df.option.nunique()
        ok = df.dropna(subset=[TARGET])
        n_opt = ok.groupby(["loss", "config"]).option.nunique()
        complete_keys = n_opt[n_opt == n_req].index
        avg = ok.set_index(["loss", "config"]).loc[complete_keys].groupby(["loss", "config"]) \
            .agg(score=(TARGET, "mean"), in_grid=("in_grid", "first"), edge=("edge", "first"),
                 base=("base", "first")).reset_index()

        rows = []
        for loss, sub in df.groupby("loss"):
            a = avg[avg.loss == loss]
            g = a[a.in_grid]
            rows.append(dict(
                loss=loss, tried=sub.config.nunique(), complete=len(a), in_grid=len(g),
                no_edges=int((~a.edge).sum()),
                wd_values=" ".join(f"{v:g}" for v in sorted(sub.param_weight_decay.unique())),
                best_all=round(a.score.max(), 4), best_in_grid=round(g.score.max(), 4) if len(g) else np.nan,
            ))
        print(pd.DataFrame(rows).set_index("loss").to_string())

        n_losses = df.loss.nunique()
        s, tot = n_shared(avg, "config", n_losses)
        print(f"  complete for all {n_losses} losses: {s} of {tot} configs")
        if method == "popa":
            s, tot = n_shared(avg, "base", n_losses)
            print(f"  ... ignoring w_po: {s} of {tot}")
            no_bb = avg[avg.loss != "bce + bce"]
            s, tot = n_shared(no_bb, "config", no_bb.loss.nunique())
            print(f"  excluding bce + bce, complete for all {no_bb.loss.nunique()} losses: {s} of {tot}")
            s, tot = n_shared(no_bb[no_bb.in_grid], "config", no_bb.loss.nunique())
            print(f"  ... and only in current grid: {s} of {tot}")