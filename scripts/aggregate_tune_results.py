"""
Aggregate tuning results from a directory of trials.

Usage:
    python scripts/aggregate_tune_results.py \
        --tune_dir outputs/tune/GeoPlant/france/geographical/test_0
"""

import argparse
from pathlib import Path
import pandas as pd
from isdm.tune import load_all_trial_results
import numpy as np

LOSS_COLS_POPA = ["param_loss_po_name", "param_loss_pa_name"]


def build_summary(results: list[dict], method: str) -> pd.DataFrame:
    """
    Build a structured summary DataFrame from raw result dicts.

    Adds:
        mean_auc           — arithmetic mean of avg_auc_site and avg_auc_species
        harmonic_mean_auc  — harmonic mean of the two (0 if either metric is 0)
    """
    df = pd.DataFrame(results)
    df["method"] = method

    df["mean_auc"] = df[["avg_auc_site", "avg_auc_species"]].mean(axis=1)

    site, species = df["avg_auc_site"], df["avg_auc_species"]
    df["harmonic_mean_auc"] = np.where(
        (site > 0) & (species > 0),
        2 * site * species / (site + species),
        0.0,
    )

    param_cols = [c for c in df.columns if c.startswith("param_")]
    meta_cols = ["method", "test_number", "option", "distance", "train_size", "test_size"]
    metric_cols = ["best_epoch", "avg_auc_site", "avg_auc_species", "mean_auc", "harmonic_mean_auc"]

    ordered = [c for c in param_cols + meta_cols + metric_cols if c in df.columns]
    return df[ordered].sort_values(param_cols + ["distance"])


def best_result_per_loss_and_option(
    summary_df: pd.DataFrame,
    loss_cols: list[str],
) -> pd.DataFrame:
    """
    Best param combo per (loss combo, option), ranked by harmonic_mean_auc.
    PER-DISTANCE best params, not averaged across distances.
    """
    missing = [c for c in loss_cols if c not in summary_df.columns]
    if missing:
        raise ValueError(f"loss_cols not found in summary: {missing}")

    param_cols = [c for c in summary_df.columns if c.startswith("param_")]

    best = (
        summary_df
        .sort_values("harmonic_mean_auc", ascending=False)
        .groupby(loss_cols + ["option"], as_index=False)
        .head(1)
        .sort_values(["option", "harmonic_mean_auc"], ascending=[True, False])
    )

    keep_cols = param_cols + [
        "option", "distance",
        "avg_auc_site", "avg_auc_species", "mean_auc", "harmonic_mean_auc",
    ]
    return best[[c for c in keep_cols if c in best.columns]]


def best_result_avg_over_distance(
    summary_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """
    Best param combo, AVERAGED across all distances/options, with one
    best row returned PER group in group_cols (e.g. per loss combo).
    """
    missing = [c for c in group_cols if c not in summary_df.columns]
    if missing:
        raise ValueError(f"group_cols not found in summary: {missing}")

    param_cols = [c for c in summary_df.columns if c.startswith("param_")]

    agg = (
        summary_df
        .groupby(param_cols, dropna=False, as_index=False)
        .agg(
            avg_auc_site=("avg_auc_site", "mean"),
            avg_auc_species=("avg_auc_species", "mean"),
            best_epoch=("best_epoch", "mean"),
            n_splits=("avg_auc_site", "size"),
        )
    )
    agg["mean_auc"] = agg[["avg_auc_site", "avg_auc_species"]].mean(axis=1)
    site, species = agg["avg_auc_site"], agg["avg_auc_species"]
    agg["harmonic_mean_auc"] = np.where(
        (site > 0) & (species > 0), 2 * site * species / (site + species), 0.0
    )

    best = (
        agg.sort_values("harmonic_mean_auc", ascending=False)
           .groupby(group_cols, as_index=False)
           .head(1)
           .sort_values(group_cols + ["harmonic_mean_auc"], ascending=[True] * len(group_cols) + [False])
    )

    keep_cols = param_cols + ["n_splits", "best_epoch", "avg_auc_site", "avg_auc_species", "mean_auc", "harmonic_mean_auc"]
    return best[[c for c in keep_cols if c in best.columns]]

def aggregate_method(tune_dir: Path, method: str, group_cols: list[str]) -> pd.DataFrame | None:
    trials_dir = tune_dir / f"trials_{method}"
    results = load_all_trial_results(trials_dir)
    if not results:
        print(f"[{method}] no trial results found under {trials_dir}, skipping.")
        return None

    n_errors = sum(1 for r in results if "error" in r)
    if n_errors:
        print(f"[{method}] WARNING: {n_errors}/{len(results)} trials recorded an error.")

    results = [r for r in results if "error" not in r]
    if not results:
        print(f"[{method}] all trials errored, nothing to summarize.")
        return None

    summary = build_summary(results, method=method)
    summary.to_csv(tune_dir / f"summary_{method}.csv", index=False)
    print(f"[{method}] {len(results)} trials -> summary_{method}.csv")

    best_avg = best_result_avg_over_distance(summary, group_cols=group_cols)
    best_avg.to_csv(tune_dir / f"best_{method}_avg_over_distance.csv", index=False)
    print(f"[{method}] -> best_{method}_avg_over_distance.csv")

    if method == "popa":
        best_per_option = best_result_per_loss_and_option(summary, loss_cols=group_cols)
        best_per_option.to_csv(tune_dir / "best_popa.csv", index=False)
        print(f"[{method}] -> best_popa.csv")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune_dir", type=Path, required=True)
    args = parser.parse_args()

    summaries = {}
    summaries["pa"] = aggregate_method(args.tune_dir, "pa", group_cols=["param_loss_name"])
    summaries["po"] = aggregate_method(args.tune_dir, "po", group_cols=["param_loss_name"])
    summaries["popa"] = aggregate_method(args.tune_dir, "popa", group_cols=LOSS_COLS_POPA)

    present = [s for s in summaries.values() if s is not None]
    if len(present) > 1:
        combined = pd.concat(present, ignore_index=True)
        combined.to_csv(args.tune_dir / "summary_combined.csv", index=False)
        print(f"-> summary_combined.csv ({len(present)} methods combined)")

    for method, summary in summaries.items():
        if summary is not None:
            print(f"\n{'=' * 60}\n  {method.upper()} SUMMARY\n{'=' * 60}")
            print(summary.sort_values("distance").to_string(index=False))
    
    


if __name__ == "__main__":
    main()