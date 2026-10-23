"""
Hyperparameter tuning figures

  w_po_by_distance_<split>   w_PO sensitivity per distance (Fig in main paper)
  hyperparams_<split>        min/mean/max per hyperparameter value over all other
                             hyperparameters, best loss of each source (appendix)
  loss_tuning_<split>        spread over configurations for every loss (appendix)

Scores are averaged over the close/medium/far validation splits (the model selection
criterion), except in the per-distance panels of Fig. 7. Integrated runs with w_PO in {0, 1}
are single-source models and are left out of the appendix figures.

Each figure is written to outputs/figures/tuning and next to the tuning results.
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

DATASET, REGION, TEST_NUMBER = "GeoPlant", "france", 0
SPLIT_TYPES = ["geographical", "environmental"]
FIG_DIR = Path("outputs/figures/tuning")

TARGET = "harmonic_mean_auc"
OPTIONS = {"closest_val": "close", "middle_val": "medium", "farthest_val": "far"}
LOSS_COLS = {
    "po": ["param_loss_name"],
    "pa": ["param_loss_name"],
    "popa": ["param_loss_po_name", "param_loss_pa_name"],
}
# best loss per source and split type, as in Fig. 5 / Table 2
BEST_LOSS = {
    "geographical": {"po": ("balanced_bce",), "pa": ("bce",), "popa": ("deep_maxent", "bce")},
    "environmental": {"po": ("balanced_bce",), "pa": ("bce",), "popa": ("deep_maxent", "bce_ippp")},
}
# order and labels of the losses, as in the tables
LOSSES = {
    "po": {
        ("deep_maxent",): "DeepMaxent",
        ("bce",): "BCE",
        ("balanced_bce",): "Bal. BCE",
    },
    "pa": {
        ("bce",): "BCE",
        ("balanced_bce",): "Bal. BCE",
    },
    "popa": {
        ("deep_maxent", "deep_maxent"): "DeepMaxent (both)",
        ("bce", "bce"): "BCE (both)",
        ("balanced_bce", "balanced_bce"): "Bal. BCE (both)",
        ("deep_maxent", "bce"): "DeepMaxent + BCE (Sigmoid)",
        ("deep_maxent", "bce_ippp"): "DeepMaxent + BCE (IPP)",
        ("deep_maxent", "balanced_bce"): "DeepMaxent + Bal. BCE (Sigmoid)",
        ("deep_maxent", "balanced_bce_ippp"): "DeepMaxent + Bal. BCE (IPP)",
    },
}
SOURCE_LABELS = {"po": "PO only", "pa": "PA only", "popa": "PO & PA"}
PARAM_LABELS = {
    "param_lr": "Learning rate",
    "param_weight_decay": "Weight decay",
    "param_hidden_dim": "Hidden layer size",
    "param_hidden_layers": "Hidden layers",
    "param_batch_size": "Batch size",
}
YLABEL = "Harmonic mean AUC"

CMAP = plt.colormaps["Blues"]
RANGE_STYLE = {
    "min":  dict(color=CMAP(0.40), linestyle=":",  linewidth=1.6, marker="o", markersize=4, zorder=2),
    "mean": dict(color=CMAP(0.65), linestyle="--", linewidth=1.8, marker="s", markersize=5, zorder=3),
    "max":  dict(color=CMAP(0.95), linestyle="-",  linewidth=2.0, marker="D", markersize=6, zorder=4),
}


def hp_cols(df, source):
    """Hyperparameter columns without the loss (w_pa = 1 - w_po is redundant)."""
    return [c for c in df.columns
            if c.startswith("param_") and c != "param_w_pa" and c not in LOSS_COLS[source]]


def load(tune_dir, source):
    """Read a tuning summary and print quick checks (failures, duplicates, configs per loss)."""
    path = tune_dir / f"summary_{source}.csv"
    if not path.exists():
        print(f"[{source}] {path} not found, skipping.")
        return None
    df = pd.read_csv(path)
    df = df[df["option"].isin(OPTIONS)]
    for c in hp_cols(df, source):
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(10)  # float noise would split configurations

    loss = df[LOSS_COLS[source]].astype(str).agg(" + ".join, axis=1)
    config = df[hp_cols(df, source)].astype(str).agg("|".join, axis=1)
    n_dup = pd.DataFrame({"l": loss, "c": config, "o": df["option"]}).duplicated().sum()
    n_all = (loss.groupby(config).nunique() == loss.nunique()).sum()
    print(f"[{source}] {len(df)} rows | {df['option'].nunique()} distance(s) | "
          f"{df[TARGET].isna().sum()} failed | {n_dup} duplicates | "
          f"{n_all} configs run for all {loss.nunique()} losses")
    return df


def average_over_distance(df, source):
    """One row per configuration; configurations missing a distance are dropped."""
    out = (df.dropna(subset=[TARGET])
             .groupby(hp_cols(df, source) + LOSS_COLS[source], dropna=False)
             .agg(score=(TARGET, "mean"), n=("option", "nunique"))
             .reset_index())
    return out[out["n"] == df["option"].nunique()].rename(columns={"score": TARGET})


def only_loss(df, source, loss):
    return df[(df[LOSS_COLS[source]] == list(loss)).all(axis=1)]


def drop_wpo_edges(df):
    if "param_w_po" in df.columns:
        return df[(df["param_w_po"] > 0) & (df["param_w_po"] < 1)]
    return df


def fmt_tick(v):
    """0.0005 -> '5e-4', 1e-05 -> '1e-5', 500 -> '500'."""
    if v == 0:
        return "0"
    if abs(v) < 0.01:
        mantissa, exp = f"{v:.1e}".split("e")
        return f"{mantissa.rstrip('0').rstrip('.')}e{int(exp)}"
    return f"{v:g}"


def range_panel(ax, df, param, legend=False):
    agg = df.groupby(param)[TARGET].agg(["min", "mean", "max"]).sort_index()
    x = [fmt_tick(v) for v in agg.index]
    for stat in ("min", "mean", "max"):
        ax.plot(x, agg[stat], label=stat, **RANGE_STYLE[stat])
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if legend:
        ax.legend(fontsize=8, frameon=False)


def save(fig, paths):
    fig.tight_layout()
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_wpo_by_distance(df, best_loss, paths):
    """Fig. 7: one panel per distance + 'average' (each configuration averaged over distance first)."""
    sub = only_loss(df, "popa", best_loss)
    panels = [(label, sub[sub["option"] == opt]) for opt, label in OPTIONS.items()]
    panels.append(("average", average_over_distance(sub, "popa")))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.4), sharey=True)
    for ax, (label, d) in zip(axes, panels):
        range_panel(ax, d, "param_w_po", legend=(label == "average"))
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("$w_{PO}$", fontsize=12)
    axes[0].set_ylabel(YLABEL, fontsize=12)
    save(fig, paths)


def fig_hyperparams(dfs_avg, best_loss, paths):
    """Rows = sources (best loss), columns = hyperparameters (w_PO is in Fig. 7)."""
    rows = [(s, drop_wpo_edges(only_loss(df, s, best_loss[s]))) for s, df in dfs_avg.items()]
    n_cols = len(PARAM_LABELS)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(3.2 * n_cols, 2.8 * len(rows)),
                             sharey="row", squeeze=False)
    for i, (source, sub) in enumerate(rows):
        for j, (param, label) in enumerate(PARAM_LABELS.items()):
            if param not in sub.columns or sub[param].nunique() < 2:
                axes[i][j].axis("off")
                continue
            range_panel(axes[i][j], sub, param, legend=(i == 0 and j == n_cols - 1))
            axes[i][j].set_xlabel(label, fontsize=10)
        axes[i][0].set_ylabel(f"{SOURCE_LABELS[source]}\n{LOSSES[source][best_loss[source]]}\n{YLABEL}",
                              fontsize=9)
    save(fig, paths)


def fig_loss_tuning(dfs_avg, paths):
    """One panel per source, one box per loss over all sampled configurations."""
    panels = []
    for source, df in dfs_avg.items():
        scores = {(k if isinstance(k, tuple) else (k,)): sub[TARGET].values
                  for k, sub in drop_wpo_edges(df).groupby(LOSS_COLS[source])}
        panels.append((source, [(label, scores[k]) for k, label in LOSSES[source].items() if k in scores]))

    widths = [len(groups) for _, groups in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(1.2 * sum(widths) + 2, 4.5),
                             gridspec_kw=dict(width_ratios=widths), squeeze=False, sharey=True)
    for ax, (source, groups) in zip(axes[0], panels):
        ax.boxplot([s for _, s in groups], widths=0.6, medianprops=dict(color="black"),
                   flierprops=dict(marker="o", markersize=3, alpha=0.5))
        ax.set_xticks(range(1, len(groups) + 1), [label for label, _ in groups],
                      rotation=30, ha="right", fontsize=9)
        ax.set_title(SOURCE_LABELS[source], fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel():
        ax.tick_params(axis="y", labelleft=True)
    axes[0][0].set_ylabel(YLABEL, fontsize=10)
    save(fig, paths)


# ---------------------------------------------------------------------------
def run(split_type):
    print(f"\n=== {split_type} ===")
    tune_dir = Path(f"outputs/tune/{DATASET}/{REGION}/{split_type}/test_{TEST_NUMBER}")
    paths = lambda name: [FIG_DIR / f"{name}_{split_type}.png", tune_dir / "figures" / f"{name}_{split_type}.png"]
    best_loss = BEST_LOSS[split_type]

    dfs = {s: df for s in SOURCE_LABELS if (df := load(tune_dir, s)) is not None}
    dfs_avg = {s: average_over_distance(df, s) for s, df in dfs.items()}

    if "popa" in dfs:
        fig_wpo_by_distance(dfs["popa"], best_loss["popa"], paths("w_po_by_distance"))
    fig_hyperparams(dfs_avg, best_loss, paths("hyperparams"))
    fig_loss_tuning(dfs_avg, paths("loss_tuning"))
    print(f"Figures written to {FIG_DIR} and {tune_dir / 'figures'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_types", default=",".join(SPLIT_TYPES))
    for split_type in parser.parse_args().split_types.split(","):
        run(split_type)


if __name__ == "__main__":
    main()