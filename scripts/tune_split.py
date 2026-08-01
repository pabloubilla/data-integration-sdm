"""
Hyperparameter grid search over a single test_number.

For a given test_number, every split option (distance variant) is
run for every param combo, for both PA-only and POPA methods.

Results are saved as:
    outputs/tune/<test_number>/
        results_pa.json
        results_popa.json
        summary_pa.csv
        summary_popa.csv
        summary_combined.csv   ← the main analysis table

Usage:
    python scripts/tune_split.py --test_number 0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from isdm.load_data import load_geoplant_processed
from isdm.splits import load_split_specs, run_one_split_popa_tunable
from isdm.utils import set_all_seeds
from isdm.tune import run_grid_search, make_param_grid
from isdm.splits import run_one_split_pa, run_one_split_popa  # reuse existing fns


# ─────────────────────────────────────────────
#  Hyperparameter grid — edit this to taste
# ─────────────────────────────────────────────
PARAM_GRID_PA = {
    "lr":           [1e-4],
    "weight_decay": [1e-4, 1e-3],
    "hidden_dim":   [128],
    "hidden_layers": [2],
    "batch_size": [500], 
    "epochs": [10, 20, 30], 
}

PARAM_GRID_PO_PA = {
    "lr":            [1e-4],
    "weight_decay":  [1e-3],
    "hidden_dim":    [128],
    "hidden_layers": [2],
    "batch_size":    [126],
    "epochs":        [10, 20, 30],
    "w_pa":          [.5, 2],
    "loss_po_name":  ["deep_maxent", "balanced_bce"],          # add more names once you have X2 etc.
    "loss_pa_name": ["deep_maxent", "balanced_bce"],        # uncomment to sweep PA loss too
}

# Fixed across all trials
FIXED = {
    # "batch_size":  500,
    # "epochs":      20,
    # "lr": 1e-4,
    # "weight_decay": 1e-3,
    # "hidden_dim": 250,
    "seed":        42,
    # "hidden_layers": 2,
}


def build_summary(results: list[dict], method: str) -> pd.DataFrame:
    """
    Build a structured summary DataFrame from raw result dicts.

    Columns:
        param_*         — the hyperparameter values for this trial
        option          — split option label (e.g. "A", "B", "C")
        distance        — spatial distance of the split
        avg_auc         — main metric
        method          — "pa" or "popa"
    """
    df = pd.DataFrame(results)
    df["method"] = method

    param_cols = [c for c in df.columns if c.startswith("param_")]
    meta_cols  = ["method", "test_number", "option", "distance", "train_size", "test_size"]
    metric_cols = ["avg_auc"]

    # Put param cols first for readability
    ordered = param_cols + meta_cols + metric_cols
    ordered = [c for c in ordered if c in df.columns]  # only keep what exists
    return df[ordered].sort_values(param_cols + ["distance"])


def pivot_auc_by_option(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot so each row = one param combo, columns = option AUC values.

    Useful for seeing at a glance which combo is best across all distances.

        param_lr | param_wd | ... | AUC_option_A | AUC_option_B | AUC_option_C | mean_auc
    """
    param_cols = [c for c in df.columns if c.startswith("param_")]

    pivot = df.pivot_table(
        index=param_cols,
        columns="option",
        values="avg_auc",
        aggfunc="mean",   # in case of duplicates
    ).reset_index()

    # Rename option columns to be explicit
    option_cols = [c for c in pivot.columns if c not in param_cols]
    pivot.rename(columns={c: f"AUC_option_{c}" for c in option_cols}, inplace=True)

    auc_cols = [c for c in pivot.columns if c.startswith("AUC_")]
    pivot["mean_auc"] = pivot[auc_cols].mean(axis=1)
    pivot = pivot.sort_values("mean_auc", ascending=False)

    return pivot


def main(test_number: int):

    run_pa = False
    run_popa = True

    set_all_seeds(FIXED["seed"])

    data_path  = "data/processed/GeoPlant/france"
    split_dir  = Path("outputs/splits/GeoPlant/france_bands/environmental") # TODO: This might need a builder as we can take different paths
    output_dir = Path(f"outputs/tune/test_{test_number}")
    output_dir.mkdir(parents=True, exist_ok=True)

    project_name = "sdm-integration-tune"

    data        = load_geoplant_processed(data_path)
    split_table = load_split_specs(split_dir)

    # ── Keep only the 3 options for the requested test_number ──────────
    mask = split_table["test_number"] == test_number
    splits_for_test = split_table[mask].reset_index(drop=True)


    if len(splits_for_test) == 0:
        raise ValueError(f"No splits found for test_number={test_number}")

    options_found = splits_for_test["option"].unique().tolist()
    print(
        f"test_number={test_number} | "
        f"{len(splits_for_test)} splits | "
        f"options={options_found}"
    )

    # kwargs forwarded unchanged to every run_fn call
    shared_fixed = dict(
        split_dir=split_dir,
        data=data,
        output_dir=output_dir,
        project_name=project_name,
        **FIXED,
    )

    # param keys that both run functions accept
    tunable_keys = list(PARAM_GRID_PA.keys())

    # ── Pretty label for console output ────────────────────────────────
    def combo_label(combo):
        return "  ".join(f"{k}={v}" for k, v in combo.items())

    # ── PA grid search ──────────────────────────────────────────────────
    if run_pa:
        print("\n\n" + "█"*60)
        print("  PA-ONLY GRID SEARCH")
        print("█"*60)

        results_pa = run_grid_search(
            param_grid=PARAM_GRID_PA,
            splits=[row for _, row in splits_for_test.iterrows()],
            run_fn=run_one_split_pa,
            fixed_kwargs=shared_fixed,
            param_keys=tunable_keys,
            results_path=output_dir / "results_pa.json",
            combo_label_fn=combo_label,
        )

    # ── POPA grid search ────────────────────────────────────────────────
    if run_popa:
        print("\n\n" + "█"*60)
        print("  PO+PA GRID SEARCH")
        print("█"*60)

        tunable_keys_popa = list(PARAM_GRID_PO_PA.keys())

        results_popa = run_grid_search(
            param_grid=PARAM_GRID_PO_PA,
            splits=[row for _, row in splits_for_test.iterrows()],
            run_fn=run_one_split_popa_tunable,
            fixed_kwargs=shared_fixed,
            param_keys=tunable_keys_popa,
            results_path=output_dir / "results_popa.json",
            combo_label_fn=combo_label,
        )

    # ── Build summary tables ─────────────────────────────────────────────
    if run_pa: summary_pa   = build_summary(results_pa,   method="pa")
    if run_popa: summary_popa = build_summary(results_popa, method="popa")

    if run_pa: summary_pa.to_csv(output_dir / "summary_pa.csv", index=False)
    if run_popa: summary_popa.to_csv(output_dir / "summary_popa.csv", index=False)

    # Combined flat table (both methods together)
    if run_pa and run_popa:
        combined = pd.concat([summary_pa, summary_popa], ignore_index=True)
        combined.to_csv(output_dir / "summary_combined.csv", index=False)

    # Pivot tables: one row per combo, AUC per option as separate columns
    if run_pa: pivot_pa   = pivot_auc_by_option(summary_pa)
    if run_popa: pivot_popa = pivot_auc_by_option(summary_popa)

    if run_pa: pivot_pa.to_csv(output_dir / "pivot_pa.csv", index=False)
    if run_popa: pivot_popa.to_csv(output_dir / "pivot_popa.csv", index=False)
    # ── Console report ───────────────────────────────────────────────────
    if run_pa:
        print("\n\n" + "="*60)
        print("  PA-ONLY SUMMARY")
        print("="*60)
        print(summary_pa.sort_values("distance").to_string(index=False))
    if run_popa:
        print("\n\n" + "="*60)
        print("  POPA SUMMARY")
        print("="*60)
        print(summary_popa.sort_values("distance").to_string(index=False))


    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_number",
        type=int,
        default=0,
        help="Which test_number to run the grid search on (default: 0)",
    )
    args = parser.parse_args()
    main(args.test_number)