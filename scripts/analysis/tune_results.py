"""
Hyperparameter sensitivity figures from the validation tuning runs.

  w_po_by_distance_<split>   w_PO sensitivity per distance (main text)
  hyperparams_<split>        min/mean/max per hyperparameter value over all other
                                 hyperparameters, selected loss of each method (appendix)
  loss_tuning_<split> (+ .csv)   spread over configurations for every loss (appendix)

Scores are averaged over the close/medium/far validation splits (the model selection
criterion), except in the per-distance panels of the w_PO figure.

Figures go to --out_dir (default outputs/figures/tuning) and to tune_dir/figures, as .png and .pdf.

Usage:
    python tune_results.py --split_type geographical
    python tune_results.py --split_type environmental
    python tune_results.py --tune_dir outputs/tune/GeoPlant/france/geographical/test_0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TARGET = "harmonic_mean_auc"
OPTIONS = ("closest_val", "middle_val", "farthest_val")
LOSS_COLS = {
    "pa": ["param_loss_name"],
    "po": ["param_loss_name"],
    "popa": ["param_loss_po_name", "param_loss_pa_name"],
}
BEST_LOSS = {  # selected loss per method (Table 2)
    "pa": ("bce",),
    "po": ("balanced_bce",),
    "popa": ("deep_maxent", "bce_ippp"),
}
DROP_WPO_EDGES = True  # w_po in {0, 1} are single-source models: left out of the appendix figures

METHOD_LABELS = {"pa": "PA only", "po": "PO only", "popa": "PO & PA"}
LOSS_LABELS = {
    "bce": "BCE", "balanced_bce": "Bal. BCE", "deep_maxent": "DeepMaxent",
    "bce_ippp": "BCE (IPP)", "balanced_bce_ippp": "Bal. BCE (IPP)",
}
PARAM_LABELS = {
    "param_lr": "Learning rate",
    "param_weight_decay": "Weight decay",
    "param_hidden_dim": "Hidden layer size",
    "param_hidden_layers": "Hidden layers",
    "param_batch_size": "Batch size",
    "param_w_po": "$w_{PO}$",
}
DISTANCE_LABELS = {"closest": "close", "middle": "medium", "farthest": "far", "average": "average"}
YLABEL = "Harmonic mean AUC"

CMAP = plt.get_cmap("Blues")
RANGE_STYLE = {
    "min":  dict(color=CMAP(0.40), linestyle=":",  linewidth=1.6, marker="o", markersize=4, zorder=2),
    "mean": dict(color=CMAP(0.65), linestyle="--", linewidth=1.8, marker="s", markersize=5, zorder=3),
    "max":  dict(color=CMAP(0.95), linestyle="-",  linewidth=2.0, marker="D", markersize=6, zorder=4),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def param_cols(df):
    return [c for c in df.columns if c.startswith("param_") and c != "param_w_pa"]  # w_pa = 1 - w_po


def hyperparams(df, method):
    return [c for c in param_cols(df) if c not in LOSS_COLS[method]]


def load(tune_dir: Path, method: str):
    path = tune_dir / f"summary_{method}.csv"
    if not path.exists():
        print(f"[{method}] {path} not found, skipping.")
        return None
    df = pd.read_csv(path)
    for c in param_cols(df):
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(10)  # float noise would split configurations
    return df[df["option"].isin(OPTIONS)]


def check(df, method):
    """Quick counts: rows, failures, duplicates, configurations per loss and pairing."""
    hp = hyperparams(df, method)
    loss = df[LOSS_COLS[method]].astype(str).agg(" + ".join, axis=1)
    config = df[hp].astype(str).agg("|".join, axis=1)
    n_dup = pd.DataFrame({"l": loss, "c": config, "o": df["option"]}).duplicated().sum()
    per_loss = config.groupby(loss).nunique()
    n_all = (loss.groupby(config).nunique() == loss.nunique()).sum()

    print(f"[{method}] {len(df)} rows | {df['option'].nunique()} distance(s) | "
          f"{df[TARGET].isna().sum()} failed | {n_dup} duplicate rows")
    print(f"[{method}]   configs per loss: " + ", ".join(f"{l}={n}" for l, n in per_loss.items()))
    print(f"[{method}]   configs run for all {loss.nunique()} losses: {n_all}")


def average_over_distance(df):
    """One row per configuration. Configurations missing a distance are dropped."""
    out = (df.dropna(subset=[TARGET])
             .groupby(param_cols(df), dropna=False)
             .agg(score=(TARGET, "mean"), n=("option", "nunique"))
             .reset_index())
    out = out[out["n"] == df["option"].nunique()]
    return out.drop(columns="n").rename(columns={"score": TARGET})


def select_loss(df, method, loss):
    mask = np.ones(len(df), dtype=bool)
    for col, name in zip(LOSS_COLS[method], loss):
        mask &= (df[col] == name).values
    if not mask.any():
        print(f"[{method}] loss {loss} not found. Available:\n"
              f"{df[LOSS_COLS[method]].drop_duplicates().to_string(index=False)}")
    return df[mask]


def drop_wpo_edges(df):
    if DROP_WPO_EDGES and "param_w_po" in df.columns:
        return df[(df["param_w_po"] > 0) & (df["param_w_po"] < 1)]
    return df


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def param_label(param):
    return PARAM_LABELS.get(param, param.replace("param_", "").replace("_", " ").capitalize())


def loss_label(loss):
    names = [LOSS_LABELS.get(name, name) for name in loss]
    if len(names) == 2 and names[0] == names[1]:
        return f"{names[0]} (both)"
    return " + ".join(names)


def fmt_tick(v):
    """0.0005 -> '5e-4', 1e-05 -> '1e-5', 500 -> '500'."""
    if v == 0:
        return "0"
    if abs(v) < 0.01:
        mantissa, exp = f"{v:.1e}".split("e")
        return f"{mantissa.rstrip('0').rstrip('.')}e{int(exp)}"
    return f"{v:g}"


def save(paths: list):
    """Save for multiple paths, and also it could be more formats if needed"""
    for path in paths:
        for ext in ["png"]:
            plt.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close()


def range_panel(ax, df, param, legend=False):
    agg = df.groupby(param)[TARGET].agg(["min", "mean", "max"]).sort_index()
    x = [fmt_tick(v) for v in agg.index]
    for stat in ("min", "mean", "max"):
        ax.plot(x, agg[stat], label=stat, **RANGE_STYLE[stat])
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if legend:
        ax.legend(fontsize=8, frameon=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_wpo_by_distance(df, out: list):
    """One panel per distance + 'average' (configurations averaged over distance first,
    so its spread comes from hyperparameters only)."""
    sub = select_loss(df, "popa", BEST_LOSS["popa"])
    if sub.empty:
        return
    panels = [(opt.replace("_val", ""), sub[sub["option"] == opt]) for opt in OPTIONS]
    panels.append(("average", average_over_distance(sub)))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.4), sharey=True)
    for j, (label, d) in enumerate(panels):
        range_panel(axes[j], d, "param_w_po", legend=(j == len(panels) - 1))
        axes[j].set_title(DISTANCE_LABELS[label], fontsize=12, fontweight="bold")
        axes[j].set_xlabel("$w_{PO}$", fontsize=12)
    axes[0].set_ylabel(YLABEL, fontsize=12)
    plt.tight_layout()
    save(out)


def fig_hyperparams(dfs_avg: dict, out: list):
    """Rows = methods (selected loss), columns = hyperparameters (w_PO is in its own figure)."""
    rows = [(m, drop_wpo_edges(select_loss(df, m, BEST_LOSS[m]))) for m, df in dfs_avg.items()]
    rows = [(m, sub) for m, sub in rows if not sub.empty]
    if not rows:
        return
    varying = lambda sub, m: [p for p in hyperparams(sub, m) if sub[p].nunique() > 1 and p != "param_w_po"]
    found = {p for m, sub in rows for p in varying(sub, m)}
    params = [p for p in PARAM_LABELS if p in found] + sorted(found - set(PARAM_LABELS))

    fig, axes = plt.subplots(len(rows), len(params), figsize=(3.2 * len(params), 2.8 * len(rows)),
                             sharey="row", squeeze=False)
    for i, (method, sub) in enumerate(rows):
        for j, param in enumerate(params):
            ax = axes[i][j]
            if param not in varying(sub, method):
                ax.axis("off")
                continue
            range_panel(ax, sub, param, legend=(i == 0 and j == len(params) - 1))
            ax.set_xlabel(param_label(param), fontsize=10)
        axes[i][0].set_ylabel(f"{METHOD_LABELS[method]}\n{loss_label(BEST_LOSS[method])}\n{YLABEL}",
                              fontsize=9)
    plt.tight_layout()
    save(out)


def fig_loss_tuning(dfs_avg: dict, out: list):
    """One panel per method, one box per loss, over all sampled configurations."""
    panels, summary = [], []
    for method, df in dfs_avg.items():
        df = drop_wpo_edges(df)
        groups = [(loss_label(key if isinstance(key, tuple) else (key,)), sub[TARGET].values)
                  for key, sub in df.groupby(LOSS_COLS[method])]
        groups.sort(key=lambda g: -np.median(g[1]))
        panels.append((method, groups))
        summary += [dict(method=METHOD_LABELS[method], loss=label, n_configs=len(s),
                         best=s.max(), median=np.median(s), std=s.std(),
                         gap_best_median=s.max() - np.median(s)) for label, s in groups]

    widths = [len(groups) for _, groups in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(1.2 * sum(widths) + 2, 4.5),
                             gridspec_kw=dict(width_ratios=widths), squeeze=False)
    for ax, (method, groups) in zip(axes[0], panels):
        ax.boxplot([s for _, s in groups], widths=0.6, medianprops=dict(color="black"),
                   flierprops=dict(marker="o", markersize=3, alpha=0.5))
        ax.set_xticks(range(1, len(groups) + 1), [label for label, _ in groups],
                      rotation=30, ha="right", fontsize=9)
        ax.set_title(METHOD_LABELS[method], fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel(YLABEL, fontsize=10)
    plt.tight_layout()
    save(out)

    table = pd.DataFrame(summary).round(4)
    for path in out:
        table.to_csv(path.with_suffix(".csv"), index=False)
    print("\n" + table.to_string(index=False))


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune_dir", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default="GeoPlant")
    parser.add_argument("--region", type=str, default="france")
    parser.add_argument("--split_type", type=str, default="geographical")
    parser.add_argument("--test_number", type=int, default=0)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures/tuning"))
    args = parser.parse_args()

    tune_dir = args.tune_dir or Path(
        f"outputs/tune/{args.dataset}/{args.region}/{args.split_type}/test_{args.test_number}")
    out_dirs = [args.out_dir, tune_dir / "figures"]  # paper folder + next to the data
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
    name = lambda fig: [d / f"{fig}_{args.split_type}" for d in out_dirs]

    dfs = {m: df for m in ("pa", "po", "popa") if (df := load(tune_dir, m)) is not None}
    for m, df in dfs.items():
        check(df, m)
    dfs_avg = {m: average_over_distance(df) for m, df in dfs.items()}

    if "popa" in dfs:
        fig_wpo_by_distance(dfs["popa"], name("w_po_by_distance"))
    fig_hyperparams(dfs_avg, name("hyperparams"))
    fig_loss_tuning(dfs_avg, name("loss_tuning"))
    print(f"\nFigures written to {', '.join(map(str, out_dirs))}")


if __name__ == "__main__":
    main()