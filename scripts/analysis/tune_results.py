"""
Runs regression + RF importance + line plots for PA, PO, and POPA in one go.

    python analyze_tune.py --dataset GeoPlant --region france --split_type geographical --test_number 0
    python analyze_tune.py --tune_dir outputs/tune/GeoPlant/france/geographical/test_0

Writes everything under tune_dir/analysis/{method}/{regression,plots}/.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

TARGET = "harmonic_mean_auc"
GROUP_COLS = {
    "pa": ["param_loss_name"],
    "po": ["param_loss_name"],
    "popa": ["param_loss_po_name", "param_loss_pa_name"],
}
EXCLUDE_COLS = {"param_w_pa"}          # redundant with param_w_po (w_pa = 1 - w_po)
EXCLUDE_WPO_EDGES = False               # drop w_po in {0, 1} (single-source degenerate cases)
BEST_LOSS_COMBO = ("deep_maxent", "bce_ippp")   # pick from your best_popa_avg_over_distance.csv
OPTIONS_ORDER = ("closest_val", "middle_val", "farthest_val")


def _group_label(key) -> str:
    return "_".join(map(str, key)) if isinstance(key, tuple) else str(key)


def _design_matrix(df, covariates):
    X_raw = df[covariates]
    numeric = [c for c in covariates if pd.api.types.is_numeric_dtype(X_raw[c]) and df[c].nunique() > 1]
    categorical = [c for c in covariates if c not in numeric and df[c].nunique() > 1]

    parts, names = [], []
    if numeric:
        parts.append((numeric, X_raw[numeric].values))
        names += [c.replace("param_", "") for c in numeric]
    if categorical:
        dummies = pd.get_dummies(X_raw[categorical], drop_first=False)
        parts.append((categorical, dummies.values.astype(float)))
        names += list(dummies.columns)
    return parts, names, numeric


def _fit_regression(df, covariates, target):
    parts, names, numeric = _design_matrix(df, covariates)
    if not parts:
        return pd.Series(dtype=float), float("nan")

    mats = [StandardScaler().fit_transform(v) if cols == numeric else v for cols, v in parts]
    X = np.concatenate(mats, axis=1)
    y = df[target].values
    y_std = (y - y.mean()) / (y.std() or 1)

    model = LinearRegression().fit(X, y_std)
    r2 = model.score(X, y_std)
    coefs = pd.Series(model.coef_, index=names).sort_values()
    return coefs, r2


def _fit_rf(df, covariates, target, n_estimators=300, max_depth=4, seed=0):
    parts, names, _ = _design_matrix(df, covariates)
    if not parts:
        return pd.Series(dtype=float), float("nan")

    X = np.concatenate([v for _, v in parts], axis=1)
    y = df[target].values

    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                oob_score=True, random_state=seed)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)
    return importances, rf.oob_score_


def _run_per_group(df, covariates, target, group_cols, fit_fn):
    if not group_cols:
        return {"all": fit_fn(df, covariates, target)}
    return {_group_label(key): fit_fn(sub, covariates, target) for key, sub in df.groupby(group_cols)}


def plot_importance(results: dict, out: Path):
    fig, axes = plt.subplots(1, len(results), figsize=(4.5 * len(results), 4), squeeze=False)
    for ax, (label, (importances, oob_r2)) in zip(axes[0], results.items()):
        importances.sort_values().plot.barh(ax=ax, color="#4C72B0")
        ax.set_title(f"{label}\nOOB R²={oob_r2:.2f}", fontsize=9)
        ax.set_xlabel("importance")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_simple_line(df, param, target, group_cols, out: Path):
    groups = list(df.groupby(group_cols)) if group_cols else [("all", df)]
    fig, ax = plt.subplots(figsize=(5, 4))
    for key, sub in groups:
        agg = sub.groupby(param)[target].mean().sort_index()
        ax.plot(agg.index.astype(str), agg.values, marker="o")
    ax.set_title(param.replace("param_", ""))
    ax.set_ylabel(target)
    if group_cols:
        ax.legend([_group_label(k) for k, _ in groups], fontsize=7)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def _plot_range_panels(panels: list, target, out: Path, cmap_name="Blues"):
    """panels: list of (label, sub_df, param) — one panel per entry."""
    cmap = plt.get_cmap(cmap_name)
    style = {
        "min":  dict(color=cmap(0.40), linestyle=":", linewidth=1.6, marker="o", markersize=4, alpha=1.0, zorder=2),
        "mean": dict(color=cmap(0.65), linestyle="--", linewidth=1.8, marker="s", markersize=5, alpha=1.0, zorder=3),
        "max":  dict(color=cmap(0.95), linestyle="-", linewidth=2, marker="D", markersize=6, alpha=1.0, zorder=4),
    }

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.4), squeeze=False, sharey="row")
    for j, (label, sub, param) in enumerate(panels):
        ax = axes[0][j]
        agg = sub.groupby(param)[target].agg(["min", "mean", "max"]).sort_index()
        x = agg.index.astype(str)
        for stat in ("min", "mean", "max"):
            ax.plot(x, agg[stat], label=stat, **style[stat])
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if j == 0:
            ax.set_ylabel('Harmonic mean AUC', fontsize=9)
        if j == len(panels) - 1:
            ax.legend(fontsize=8, frameon=False)
    
    # x axis
    for ax in axes[0]:
        ax.set_xlabel("$w_{PO}$", fontsize=9)

    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_range(df, param, target, group_cols, out: Path, cmap_name="Blues"):
    """One panel per group (e.g. loss combo) — min/mean/max vs param."""
    groups = list(df.groupby(group_cols)) if group_cols else [("all", df)]
    panels = [(_group_label(key) if group_cols else target, sub, param) for key, sub in groups]
    _plot_range_panels(panels, target, out, cmap_name)


def plot_wpo_by_distance(df_raw, loss_po_name, loss_pa_name, target, out: Path,
                          options=OPTIONS_ORDER, cmap_name="Blues"):
    """One panel per distance (option) plus one 'average' panel, for a
    single fixed loss combo — the featured w_po plot for the main text.

    The 'average' panel first averages over option PER COMBO (same as
    df_avg elsewhere), then takes min/mean/max ACROSS combos of that
    averaged value — not a pool of raw rows across options, which would
    mix distance-driven spread into what should be hyperparameter-driven
    spread only.
    """
    sub_loss = df_raw[(df_raw["param_loss_po_name"] == loss_po_name)
                       & (df_raw["param_loss_pa_name"] == loss_pa_name)]

    param_cols = [c for c in sub_loss.columns if c.startswith("param_") and c not in EXCLUDE_COLS]
    sub_loss_avg = sub_loss.groupby(param_cols, as_index=False)[target].mean()  # collapse option first

    panels = [(opt.replace("_val", ""), sub_loss[sub_loss["option"] == opt], "param_w_po") for opt in options]
    panels.append(("average", sub_loss_avg, "param_w_po"))
    _plot_range_panels(panels, target, out, cmap_name)


def resolve_covariates(df, group_cols):
    return [c for c in df.columns if c.startswith("param_") and c not in group_cols
            and c not in EXCLUDE_COLS and df[c].nunique() > 1]


def run_method(tune_dir: Path, method: str):
    csv_path = tune_dir / f"summary_{method}.csv"
    if not csv_path.exists():
        print(f"[{method}] no summary CSV found, skipping.")
        return

    df_raw = pd.read_csv(csv_path)
    if EXCLUDE_WPO_EDGES and "param_w_po" in df_raw.columns:
        n_before = len(df_raw)
        df_raw = df_raw[(df_raw["param_w_po"] > 0) & (df_raw["param_w_po"] < 1)]
        print(f"[{method}] excluded w_po edges: {n_before} -> {len(df_raw)} rows")


    param_cols = [c for c in df_raw.columns if c.startswith("param_") and c not in EXCLUDE_COLS]
    df_avg = df_raw.groupby(param_cols, as_index=False)[TARGET].mean()

    ### little plot to see sensitivity of losses    
    # print(method)
    # print(df_avg)
    # #
    # std = df_avg.groupby('param_loss_name')[TARGET].std()
    # print(std)
    
    # #boxplot each loss_name
    # df_avg.boxplot(column=TARGET, by='param_loss_name', grid=False, figsize=(8,6))
    # plt.title(f"{method} - {TARGET} by loss_name")
    # plt.suptitle("")
    # plt.xlabel("Loss Name")
    # plt.ylabel(TARGET)
    # plt.savefig(tune_dir / f"{method}_boxplot_{TARGET}_by_loss_name.png", dpi=150)
    # # where saved
    # print(f"[{method}] boxplot saved to {tune_dir / f'{method}_boxplot_{TARGET}_by_loss_name.png'}")
    # plt.close()

    # exit()

    group_cols = GROUP_COLS[method]
    covariates = resolve_covariates(df_avg, group_cols)
    print(f"[{method}] covariates: {covariates}")

    out_dir = tune_dir / "analysis" / method
    reg_dir, plot_dir = out_dir / "regression", out_dir / "plots"
    reg_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    reg_results = _run_per_group(df_avg, covariates, TARGET, group_cols, _fit_regression)
    with open(reg_dir / "regression.txt", "w") as f:
        for label, (coefs, r2) in reg_results.items():
            f.write(f"[{label}] R^2={r2:.3f}\n{coefs.to_string()}\n\n")

    rf_results = _run_per_group(df_avg, covariates, TARGET, group_cols, _fit_rf)
    with open(reg_dir / "rf_importance.txt", "w") as f:
        for label, (imp, oob) in rf_results.items():
            f.write(f"[{label}] OOB R^2={oob:.3f}\n{imp.to_string()}\n\n")
    plot_importance(rf_results, reg_dir / "rf_importance.png")

    for c in covariates:
        plot_simple_line(df_avg, c, TARGET, group_cols, plot_dir / f"{c.replace('param_', '')}.png")

    if method == "popa" and "param_w_po" in df_raw.columns:
        loss_po, loss_pa = BEST_LOSS_COMBO
        plot_wpo_by_distance(df_raw, loss_po, loss_pa, TARGET, plot_dir / "w_po_by_distance.png")

    print(f"[{method}] done -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune_dir", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default="GeoPlant")
    parser.add_argument("--region", type=str, default="france")
    parser.add_argument("--split_type", type=str, default="geographical")
    parser.add_argument("--test_number", type=int, default=0)
    args = parser.parse_args()

    tune_dir = args.tune_dir or Path(
        f"outputs/tune/{args.dataset}/{args.region}/{args.split_type}/test_{args.test_number}"
    )

    for method in ("pa", "po", "popa"):
        run_method(tune_dir, method)


if __name__ == "__main__":
    main()