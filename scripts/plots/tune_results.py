"""
Plots for grid search results produced by scripts/tune_split.py.

Reads a summary_*.csv (e.g. outputs/tune/test_1/summary_popa.csv) and
produces a handful of standard diagnostic plots into an output folder:

    01_auc_vs_distance.png       — how performance degrades with distance,
                                    one line per param combo
    02_metric_by_param_<p>.png   — one boxplot per tunable param, showing
                                    how each value of that param affects
                                    harmonic_mean_auc (one file per param)
    03_heatmap_<p1>_vs_<p2>.png  — 2D heatmap of mean harmonic_mean_auc
                                    over the two params you care about most
    04_loss_combo_comparison.png — if multiple loss combos are present,
                                    boxplot comparing them
    05_site_vs_species_scatter.png — avg_auc_site vs avg_auc_species,
                                    colored by option, to see which combos
                                    are balanced vs lopsided
    06_regression_coefficients.png — standardized linear regression of the
                                    metric on all params (+ distance), to see
                                    which params matter most and in which
                                    direction, all else held equal. Prints R^2
                                    to the console so you know how much of the
                                    variance this actually explains.

Usage: 
    python scripts/plots/tune_results.py \
        --csv outputs/tune/GeoPlant/france/geographical/test_0/summary_popa.csv \
        --out outputs/tune/GeoPlant/france/geographical/test_0/plots

    # to control which two params go into the heatmap:
    python scripts/plots/tune_results.py \
        --csv outputs/tune/GeoPlant/france/geographical/test_0/summary_popa.csv \
        --out outputs/tune/GeoPlant/france/geographical/test_0/plots \
        --heatmap_params param_lr param_w_pa
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "harmonic_mean_auc" not in df.columns:
        raise ValueError(
            f"'{csv_path}' has no 'harmonic_mean_auc' column — "
            "is this a summary_*.csv produced by tune_split.py?"
        )
    return df


def param_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("param_")]


def combo_id(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Short human-readable label identifying a param combo, for legends."""
    return df[cols].astype(str).agg(",".join, axis=1)


# ─────────────────────────────────────────────
#  1. AUC vs distance, one line per param combo
# ─────────────────────────────────────────────
def plot_auc_vs_distance(df: pd.DataFrame, out_dir: Path, metric: str = "harmonic_mean_auc"):
    cols = param_cols(df)
    df = df.copy()
    df["_combo"] = combo_id(df, cols)

    fig, ax = plt.subplots(figsize=(9, 6))
    for combo_label, group in df.groupby("_combo"):
        group = group.sort_values("distance")
        ax.plot(group["distance"], group[metric], marker="o", alpha=0.6, linewidth=1)

    ax.set_xlabel("distance")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs distance (one line per param combo)")
    # too many combos to fit a legend readably — omit it, this plot is for
    # eyeballing the overall shape/spread, not identifying individual lines
    fig.tight_layout()
    fig.savefig(out_dir / "01_auc_vs_distance.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────
#  2. Boxplot of metric by each param's values
# ─────────────────────────────────────────────
def plot_metric_by_param(df: pd.DataFrame, out_dir: Path, metric: str = "harmonic_mean_auc"):
    cols = param_cols(df)
    for col in cols:
        if df[col].nunique() < 2:
            continue  # nothing to compare if the param was fixed

        groups = [g[metric].values for _, g in df.groupby(col)]
        labels = [str(v) for v in sorted(df[col].unique(), key=str)]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot(groups, tick_labels=labels)
        ax.set_xlabel(col.replace("param_", ""))
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by {col.replace('param_', '')}")
        fig.tight_layout()
        safe_name = col.replace("param_", "")
        fig.savefig(out_dir / f"02_metric_by_param_{safe_name}.png", dpi=150)
        plt.close(fig)


# ─────────────────────────────────────────────
#  3. Heatmap of mean metric over two chosen params
# ─────────────────────────────────────────────
def plot_heatmap(df: pd.DataFrame, out_dir: Path, p1: str, p2: str, metric: str = "harmonic_mean_auc"):
    if p1 not in df.columns or p2 not in df.columns:
        print(f"  skipping heatmap: {p1} or {p2} not found in columns")
        return
    if df[p1].nunique() < 2 or df[p2].nunique() < 2:
        print(f"  skipping heatmap: {p1} or {p2} has only one unique value")
        return

    pivot = df.pivot_table(index=p1, columns=p2, values=metric, aggfunc="mean")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(p2.replace("param_", ""))
    ax.set_ylabel(p1.replace("param_", ""))
    ax.set_title(f"mean {metric}")

    # annotate cells with the value
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        color="white" if val < pivot.values[~pd.isna(pivot.values)].mean() else "black",
                        fontsize=8)

    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    safe1, safe2 = p1.replace("param_", ""), p2.replace("param_", "")
    fig.savefig(out_dir / f"03_heatmap_{safe1}_vs_{safe2}.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────
#  4. Compare loss combos, if more than one is present
# ─────────────────────────────────────────────
def plot_loss_combo_comparison(df: pd.DataFrame, out_dir: Path, metric: str = "harmonic_mean_auc"):
    loss_cols = [c for c in ["param_loss_po_name", "param_loss_pa_name"] if c in df.columns]
    if not loss_cols:
        return

    df = df.copy()
    df["_loss_combo"] = combo_id(df, loss_cols)

    if df["_loss_combo"].nunique() < 2:
        print("  skipping loss combo comparison: only one loss combo present")
        return

    groups = [g[metric].values for _, g in df.groupby("_loss_combo")]
    labels = sorted(df["_loss_combo"].unique())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(groups, tick_labels=labels)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by loss combo")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "04_loss_combo_comparison.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────
#  5. Site AUC vs species AUC scatter, colored by option
# ─────────────────────────────────────────────
def plot_site_vs_species(df: pd.DataFrame, out_dir: Path):
    if "avg_auc_site" not in df.columns or "avg_auc_species" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    for option, group in df.groupby("option"):
        ax.scatter(group["avg_auc_species"], group["avg_auc_site"], label=option, alpha=0.6, s=25)

    lims = [
        min(df["avg_auc_species"].min(), df["avg_auc_site"].min()) - 0.02,
        max(df["avg_auc_species"].max(), df["avg_auc_site"].max()) + 0.02,
    ]
    ax.plot(lims, lims, linestyle="--", color="gray", linewidth=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("avg_auc_species")
    ax.set_ylabel("avg_auc_site")
    ax.set_title("Site AUC vs Species AUC (points below the line: species-biased)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "05_site_vs_species_scatter.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────
#  6. Regression: which params matter, and in which direction?
# ─────────────────────────────────────────────
def plot_regression_coefficients(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str = "harmonic_mean_auc",
    include_distance: bool = True,
):
    """
    Fit metric ~ params (+ distance) as a linear regression on standardized
    numeric features and one-hot-encoded categorical features. The resulting
    coefficients say: "holding everything else fixed, how much does this
    param's effect move the metric, in units of one standard deviation of
    the metric" — which makes coefficients directly comparable across params
    on very different scales (e.g. lr vs hidden_dim vs a loss name).

    This does NOT capture interactions (e.g. "lr only matters when w_pa is
    high") — that's what the heatmap is for. Low R^2 here is a signal that
    interactions/nonlinearity matter more than individual param effects.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    cols = param_cols(df)
    if include_distance and "distance" in df.columns:
        cols = cols + ["distance"]

    # keep only columns that actually vary — a constant column contributes
    # nothing to a regression and one-hot encoding it would just add a
    # useless all-zero or all-one dummy
    varying_cols = [c for c in cols if df[c].nunique() > 1]
    if len(varying_cols) == 0:
        print("  skipping regression: no params vary in this grid")
        return

    X_raw = df[varying_cols].copy()
    numeric_cols = [c for c in varying_cols if pd.api.types.is_numeric_dtype(X_raw[c])]
    categorical_cols = [c for c in varying_cols if c not in numeric_cols]

    X_parts = []
    feature_names = []

    if numeric_cols:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_raw[numeric_cols])
        X_parts.append(X_num)
        feature_names += [c.replace("param_", "") for c in numeric_cols]

    if categorical_cols:
        X_cat = pd.get_dummies(X_raw[categorical_cols], drop_first=True)
        X_parts.append(X_cat.values.astype(float))
        feature_names += [c.replace("param_", "") for c in X_cat.columns]

    X = np.concatenate(X_parts, axis=1) if len(X_parts) > 1 else X_parts[0]

    y = df[metric].values
    y_std = (y - y.mean()) / (y.std() if y.std() > 0 else 1.0)

    model = LinearRegression()
    model.fit(X, y_std)
    r2 = model.score(X, y_std)

    coefs = pd.Series(model.coef_, index=feature_names).sort_values()

    print(f"  regression R^2 = {r2:.3f} (fraction of variance in {metric} explained linearly)")

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(coefs))))
    colors = ["tab:red" if c < 0 else "tab:blue" for c in coefs.values]
    ax.barh(coefs.index, coefs.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"standardized effect on {metric}")
    ax.set_title(f"Linear regression coefficients (R² = {r2:.3f})")
    fig.tight_layout()
    fig.savefig(out_dir / "06_regression_coefficients.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, help="Path to summary_*.csv",
                        default='outputs/tune/test_0/summary_popa.csv')
    parser.add_argument("--out", type=Path, default=None, help="Output dir for plots (default: <csv_dir>/plots)")
    parser.add_argument("--metric", type=str, default="harmonic_mean_auc",
                         help="Metric column to use for plots 1-4 (default: harmonic_mean_auc)")
    parser.add_argument("--heatmap_params", nargs=2, default=None,
                         help="Two param_* columns for the heatmap, e.g. --heatmap_params param_lr param_w_pa")
    args = parser.parse_args()

    out_dir = args.out or (args.csv.parent / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Param columns: {param_cols(df)}")

    print("Plotting 01: AUC vs distance ...")
    plot_auc_vs_distance(df, out_dir, metric=args.metric)

    print("Plotting 02: metric by each param ...")
    plot_metric_by_param(df, out_dir, metric=args.metric)

    if args.heatmap_params:
        p1, p2 = args.heatmap_params
        print(f"Plotting 03: heatmap {p1} vs {p2} ...")
        plot_heatmap(df, out_dir, p1, p2, metric=args.metric)
    else:
        # best-effort default: pick the two params with the most unique values
        cols = param_cols(df)
        nuniques = {c: df[c].nunique() for c in cols}
        varying = [c for c, n in nuniques.items() if n > 1]
        if len(varying) >= 2:
            varying_sorted = sorted(varying, key=lambda c: -nuniques[c])
            p1, p2 = varying_sorted[0], varying_sorted[1]
            print(f"Plotting 03: heatmap {p1} vs {p2} (auto-selected) ...")
            plot_heatmap(df, out_dir, p1, p2, metric=args.metric)
        else:
            print("  skipping heatmap: fewer than 2 params vary in this grid")

    print("Plotting 04: loss combo comparison ...")
    plot_loss_combo_comparison(df, out_dir, metric=args.metric)

    print("Plotting 05: site vs species scatter ...")
    plot_site_vs_species(df, out_dir)

    print("Plotting 06: regression coefficients ...")
    plot_regression_coefficients(df, out_dir, metric=args.metric)

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()