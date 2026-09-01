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
from isdm.splits import load_split_specs, run_one_split_po_or_pa, run_one_split_popa_tunable, run_one_split_po_or_pa_tunable
from isdm.utils import set_all_seeds
from isdm.tune import run_grid_search, build_trial_combos
from isdm.tune_parallel import run_grid_search_parallel

from typing import Optional, List, Dict
import itertools

import torch.multiprocessing as torch_mp
torch_mp.set_sharing_strategy('file_system')

# time 
from time import time


RUN_PARALLEL = False  # if True, uses multiprocessing to run trials in parallel (faster, but more memory-intensive)

# ─────────────────────────────────────────────
#  Which methods to run
# ─────────────────────────────────────────────
RUN_PA = True
RUN_PO = True
RUN_POPA = True

# ─────────────────────────────────────────────
#  Hyperparameter grids — edit these to taste
# ─────────────────────────────────────────────

# constants for the validation-based epoch discovery
EPOCHS_CEILING = 100
VAL_PATIENCE = 5

### Values for PA ####
N_TRIALS_PA = 5
PARAM_GRID_PA = {
    "lr":            [1e-3, 1e-4],
    "weight_decay":  [1e-3, 1e-4],
    "hidden_dim":    [128, 256],
    "hidden_layers": [2, 3],
    "batch_size":    [100, 500, 1000],
}
LOSS_NAMES_PA = ["bce", "balanced_bce"]

#### Values for PO ####
N_TRIALS_PO = 5
PARAM_GRID_PO = {
    "lr":            [1e-3, 1e-4],
    "weight_decay":  [1e-3, 1e-4],
    "hidden_dim":    [128, 256],
    "hidden_layers": [2, 3],
    "batch_size":    [100, 500, 1000],
}
LOSS_NAMES_PO = ["deep_maxent", "balanced_bce"]

#### Values for PO+PA ####
N_TRIALS_POPA = 5
PARAM_GRID_POPA = {
    "lr":            [1e-3, 1e-4],
    "weight_decay":  [1e-3, 1e-4],
    "hidden_dim":    [128, 256],
    "hidden_layers": [2, 3],
    "batch_size":    [100, 500, 1000],
}
LOSS_COMBOS_POPA = [
    ("deep_maxent", "balanced_bce_ippp"),
    ("deep_maxent", "balanced_bce"),
    ("balanced_bce", "balanced_bce"),
    ("deep_maxent", "deep_maxent"),
]
W_PA_PO_PAIRS = [(0.3, 0.7),
                 (0.5, 0.5), 
                 (0.7, 0.3)]
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
    metric_cols = ["best_epoch", "avg_auc_site", "avg_auc_species", "mean_auc", "harmonic_mean_auc"]

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
    agg["harmonic_mean_auc"] = np.where((site > 0) & (species > 0), 2 * site * species / (site + species), 0.0)

    best = (
        agg.sort_values("harmonic_mean_auc", ascending=False)
           .groupby(group_cols, as_index=False)
           .head(1)
           .sort_values(group_cols + ["harmonic_mean_auc"], ascending=[True] * len(group_cols) + [False])
    )

    keep_cols = param_cols + ["n_splits", "best_epoch", "avg_auc_site", "avg_auc_species", "mean_auc", "harmonic_mean_auc"]
    return best[[c for c in keep_cols if c in best.columns]]

# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main(test_number: int, dataset: str, split_type: str, region: str):
    set_all_seeds(FIXED["seed"])


    data_path = f"data/processed/{dataset}/{region}"
    split_dir = Path(f"outputs/splits/{dataset}/{region}_bands/{split_type}")
    output_dir = Path(f"outputs/tune/{dataset}/{region}/{split_type}/test_{test_number}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_geoplant_processed(data_path)
    split_table = load_split_specs(split_dir)
    splits_for_test = split_table[split_table["test_number"] == test_number].reset_index(drop=True)

    # exclusively against the held-out validation set
    val_options = ("closest_val", "middle_val", "farthest_val")
    splits_for_test = splits_for_test[splits_for_test["option"].isin(val_options)].reset_index(drop=True)
    if len(splits_for_test) == 0:
        raise ValueError(f"No validation splits found for test_number={test_number}")

    options_found = splits_for_test["option"].unique().tolist()
    print(f"test_number={test_number} | {len(splits_for_test)} splits | options={options_found}")

    split_rows = [row for _, row in splits_for_test.iterrows()]

    split_rows_close = [split_rows.copy()[0]]

    shared_fixed = dict(
        split_dir=split_dir,
        data=data,
        output_dir=output_dir,
        project_name="sdm-integration-tune",
        # For Validation: fixed (not swept) — epochs is a ceiling, validation picks the real count
        epochs=EPOCHS_CEILING,
        validate_during_training=True,
        val_patience=VAL_PATIENCE,
        **FIXED,
    )

    def combo_label(combo):
        return "  ".join(f"{k}={v}" for k, v in combo.items())

    summary_pa = None
    summary_po = None
    summary_popa = None

    # ── PA grid search ──────────────────────────────────────────────────
    if RUN_PA:
        print("\n\n" + "█" * 60)
        print("  PA-ONLY GRID SEARCH")
        print("█" * 60)

        # if RUN_PARALLEL:
        #     results_popa = run_grid_search_parallel(
        #         param_grid=PARAM_GRID_PA,
        #         splits=split_rows,
        #         run_fn=run_one_split_po_or_pa_tunable,
        #         fixed_kwargs={**shared_fixed, "source": "pa"},
        #         param_keys=list(PARAM_GRID_PA.keys()) + ["loss_name"],
        #         results_path=output_dir / "results_pa.json",
        #         combo_label_fn=combo_label,
        #         max_workers=16,   # tune this — see below
        #     )   

        combos_pa = build_trial_combos(
            method="pa",
            param_grid=PARAM_GRID_PA,
            n_trials=N_TRIALS_PA,
            loss_names=LOSS_NAMES_PA,
            seed=FIXED["seed"],
        )

        results_pa  = run_grid_search(
            combos=combos_pa,
            splits=split_rows,
            run_fn=run_one_split_po_or_pa_tunable,
            fixed_kwargs={**shared_fixed, "source": "pa"},
            param_keys=list(PARAM_GRID_PA.keys()) + ["loss_name"],
            results_path=output_dir / "results_pa.json",
            combo_label_fn=combo_label
        )




        summary_pa = build_summary(results_pa, method="pa")
        summary_pa.to_csv(output_dir / "summary_pa.csv", index=False)

        best_pa_avg = best_result_avg_over_distance(summary_pa, group_cols=["param_loss_name"])
        best_pa_avg.to_csv(output_dir / "best_pa_avg_over_distance.csv", index=False)



    # ── PO grid search ──────────────────────────────────────────────────
    if RUN_PO:
        print("\n\n" + "█" * 60)
        print("  PO-ONLY GRID SEARCH")
        print("█" * 60)

        combos_po = build_trial_combos(
            method="po",
            param_grid=PARAM_GRID_PO,
            n_trials=N_TRIALS_PO,
            loss_names=LOSS_NAMES_PO,
            seed=FIXED["seed"],
        )

        results_po = run_grid_search(
            combos=combos_po,
            splits=split_rows_close, # only run the closest_val split for PO-only tuning
            run_fn=run_one_split_po_or_pa_tunable,
            fixed_kwargs={**shared_fixed, "source": "po"},
            param_keys=list(PARAM_GRID_PO.keys()) + ["loss_name"],
            results_path=output_dir / "results_po.json",
            combo_label_fn=combo_label,
        )
        summary_po = build_summary(results_po, method="po")
        summary_po.to_csv(output_dir / "summary_po.csv", index=False)

        # PO
        best_po_avg = best_result_avg_over_distance(summary_po, group_cols=["param_loss_name"])
        best_po_avg.to_csv(output_dir / "best_po_avg_over_distance.csv", index=False)

    # ── POPA grid search ────────────────────────────────────────────────
    if RUN_POPA:
        print("\n\n" + "█" * 60)
        print("  PO+PA GRID SEARCH")
        print("█" * 60)

        combos_popa = build_trial_combos(
            method="popa",
            param_grid=PARAM_GRID_POPA,
            n_trials=N_TRIALS_POPA,
            loss_combos=LOSS_COMBOS_POPA,
            w_pa_po_pairs=W_PA_PO_PAIRS,
            seed=FIXED["seed"],
        )

        tunable_keys_popa = (
            list(PARAM_GRID_POPA.keys())
            + ["loss_po_name", "loss_pa_name", "w_pa", "w_po"]
        )

        results_popa = run_grid_search(
            combos=combos_popa,
            splits=split_rows,
            run_fn=run_one_split_popa_tunable,
            fixed_kwargs=shared_fixed,
            param_keys=tunable_keys_popa,
            results_path=output_dir / "results_popa.json",
            combo_label_fn=combo_label,
        )
        summary_popa = build_summary(results_popa, method="popa")
        summary_popa.to_csv(output_dir / "summary_popa.csv", index=False)

        best_popa = best_result_per_loss_and_option(summary_popa)
        best_popa.to_csv(output_dir / "best_popa.csv", index=False)

        best_popa_avg = best_result_avg_over_distance(summary_popa, group_cols = LOSS_COLS)
        best_popa_avg.to_csv(output_dir / "best_popa_avg_over_distance.csv", index=False)

    # ── Combined table (only if more than one method ran) ──────────────
    all_summaries = [s for s in (summary_pa, summary_po, summary_popa) if s is not None]
    if len(all_summaries) > 1:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(output_dir / "summary_combined.csv", index=False)

    # ── Console report ───────────────────────────────────────────────────
    if summary_pa is not None:
        print("\n\n" + "=" * 60)
        print("  PA-ONLY SUMMARY")
        print("=" * 60)
        print(summary_pa.sort_values("distance").to_string(index=False))

    if summary_po is not None:
        print("\n\n" + "=" * 60)
        print("  PO-ONLY SUMMARY")
        print("=" * 60)
        print(summary_po.sort_values("distance").to_string(index=False))

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
        default=0,
        help="Which test_number to run the grid search on (default: 1)",
    )
    # dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="GeoPlant",
        help="Which dataset to use (default: GeoPlant)",
    )
    # split type
    parser.add_argument(
        "--split_type",
        type=str,
        default="geographical",
        help="Which split type to use (default: geographical)",
    )
    # region
    parser.add_argument(
        "--region",
        type=str,
        default="france",
        help="Which region to use (default: france)",
    )

    args = parser.parse_args()
    initial_time = time()
    main(args.test_number, args.dataset, args.split_type, args.region)
    elapsed_time = time() - initial_time
    print(f"\nTotal elapsed time: {elapsed_time:.2f} seconds")