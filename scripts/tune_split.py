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
        summary_combined.csv   ← both methods together (only if both run)
        best_popa.csv          ← best param combo per (loss combo, option)

Usage:
    python scripts/tune_split.py --test_number 0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyparsing import Any

from isdm.load_data import load_geoplant_processed
from isdm.splits import load_split_specs, run_one_split_pa, run_one_split_popa_tunable
from isdm.utils import set_all_seeds
from isdm.tune import run_grid_search

from typing import Optional, List, Dict
import itertools


# ─────────────────────────────────────────────
#  Which methods to run
# ─────────────────────────────────────────────
RUN_PA = False
RUN_POPA = True

# ─────────────────────────────────────────────
#  Hyperparameter grids — edit these to taste
# ─────────────────────────────────────────────
PARAM_GRID_PA = {
    "lr":            [1e-4],
    "weight_decay":  [1e-4, 1e-3],
    "hidden_dim":    [128],
    "hidden_layers": [2],
    "batch_size":    [500],
    "epochs":        [10, 20, 30],
}

PARAM_GRID_PO_PA = {
    "lr":            [0.001, 0.0001],
    "weight_decay":  [1e-3, 1e-4],
    "hidden_dim":    [128],
    "hidden_layers": [2,3],
    "batch_size":    [126,512],
    "epochs":        [15]
}

LINKED_PARAM_GRID_PO_PA = {
    ("loss_po_name", "loss_pa_name"): [
        ("deep_maxent", "balanced_bce"),
        ("balanced_bce", "balanced_bce"),
    ],
    ("w_pa", "w_po",): [   # a "group" of size 1 works too, if you just want an
        (0.3,0.7),
        (0.5,0.5),    
        (0.7,0.3),
    ],
}

# Fixed across all trials
FIXED = {
    "seed": 42,
}

# Which param_* columns identify a "loss combo" for best-result grouping
LOSS_COLS = ["param_loss_po_name", "param_loss_pa_name"]







# ─────────────────────────────────────────────
#  Summary building
# ─────────────────────────────────────────────
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
    metric_cols = ["avg_auc_site", "avg_auc_species", "mean_auc", "harmonic_mean_auc"]

    ordered = [c for c in param_cols + meta_cols + metric_cols if c in df.columns]
    return df[ordered].sort_values(param_cols + ["distance"])


def best_result_per_loss_and_option(
    summary_df: pd.DataFrame,
    loss_cols: list[str] = LOSS_COLS,
) -> pd.DataFrame:
    """
    Best param combo per (loss combo, option), ranked by harmonic_mean_auc.

    This is PER-DISTANCE best params, not averaged across distances.
    Assumes summary_df already has mean_auc / harmonic_mean_auc columns
    (i.e. this is called after build_summary).
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
    keep_cols = [c for c in keep_cols if c in best.columns]
    return best[keep_cols]


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main(test_number: int):
    set_all_seeds(FIXED["seed"])

    data_path = "data/processed/GeoPlant/france"
    split_dir = Path("outputs/splits/GeoPlant/france_bands/environmental")
    output_dir = Path(f"outputs/tune/test_{test_number}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_geoplant_processed(data_path)
    split_table = load_split_specs(split_dir)

    splits_for_test = split_table[split_table["test_number"] == test_number].reset_index(drop=True)
    if len(splits_for_test) == 0:
        raise ValueError(f"No splits found for test_number={test_number}")

    options_found = splits_for_test["option"].unique().tolist()
    print(f"test_number={test_number} | {len(splits_for_test)} splits | options={options_found}")

    split_rows = [row for _, row in splits_for_test.iterrows()]

    shared_fixed = dict(
        split_dir=split_dir,
        data=data,
        output_dir=output_dir,
        project_name="sdm-integration-tune",
        **FIXED,
    )

    def combo_label(combo):
        return "  ".join(f"{k}={v}" for k, v in combo.items())

    summary_pa = None
    summary_popa = None

    # ── PA grid search ──────────────────────────────────────────────────
    if RUN_PA:
        print("\n\n" + "█" * 60)
        print("  PA-ONLY GRID SEARCH")
        print("█" * 60)

        results_pa = run_grid_search(
            param_grid=PARAM_GRID_PA,
            splits=split_rows,
            run_fn=run_one_split_pa,
            fixed_kwargs=shared_fixed,
            param_keys=list(PARAM_GRID_PA.keys()),
            results_path=output_dir / "results_pa.json",
            combo_label_fn=combo_label,
        )
        summary_pa = build_summary(results_pa, method="pa")
        summary_pa.to_csv(output_dir / "summary_pa.csv", index=False)

    # ── POPA grid search ────────────────────────────────────────────────
    if RUN_POPA:
        print("\n\n" + "█" * 60)
        print("  PO+PA GRID SEARCH")
        print("█" * 60)

        tunable_keys_popa = (
            list(PARAM_GRID_PO_PA.keys())
            + [k for group in LINKED_PARAM_GRID_PO_PA.keys() for k in group]
        )

        results_popa = run_grid_search(
            param_grid=PARAM_GRID_PO_PA,
            splits=split_rows,
            run_fn=run_one_split_popa_tunable,
            fixed_kwargs=shared_fixed,
            param_keys=tunable_keys_popa,
            linked_param_grid=LINKED_PARAM_GRID_PO_PA,
            results_path=output_dir / "results_popa.json",
            combo_label_fn=combo_label,
        )
        summary_popa = build_summary(results_popa, method="popa")
        summary_popa.to_csv(output_dir / "summary_popa.csv", index=False)

        best_popa = best_result_per_loss_and_option(summary_popa)
        best_popa.to_csv(output_dir / "best_popa.csv", index=False)

    # ── Combined table (only if both ran) ──────────────────────────────
    if summary_pa is not None and summary_popa is not None:
        combined = pd.concat([summary_pa, summary_popa], ignore_index=True)
        combined.to_csv(output_dir / "summary_combined.csv", index=False)

    # ── Console report ───────────────────────────────────────────────────
    if summary_pa is not None:
        print("\n\n" + "=" * 60)
        print("  PA-ONLY SUMMARY")
        print("=" * 60)
        print(summary_pa.sort_values("distance").to_string(index=False))

    if summary_popa is not None:
        print("\n\n" + "=" * 60)
        print("  POPA SUMMARY")
        print("=" * 60)
        print(summary_popa.sort_values("distance").to_string(index=False))

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_number",
        type=int,
        default=1,
        help="Which test_number to run the grid search on (default: 1)",
    )
    args = parser.parse_args()
    main(args.test_number)