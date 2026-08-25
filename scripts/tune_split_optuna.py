"""
Optuna-based hyperparameter search over a single test_number.
Replaces the exhaustive grid with TPE sampling + pruning.

Usage:
    python scripts/tune_split.py --test_number 0 --n_trials 60
"""

import argparse
from pathlib import Path

import optuna
import numpy as np
import pandas as pd

from isdm.load_data import load_geoplant_processed
from isdm.splits import load_split_specs
from isdm.utils import set_all_seeds
from isdm.splits import run_one_split_po_or_pa, run_one_split_popa

FIXED = {"seed": 42}


# ── Objective factories ──────────────────────────────────────────────────────

def make_objective_pa(splits, shared_fixed):
    def objective(trial):
        combo = {
            "lr":            trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay":  trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),
            "hidden_dim":    trial.suggest_categorical("hidden_dim", [128, 256, 512]),
            "hidden_layers": trial.suggest_int("hidden_layers", 1, 3),
            "batch_size":    trial.suggest_categorical("batch_size", [128, 256, 500]),
            "epochs":        trial.suggest_categorical("epochs", [10, 50, 100]),
        }

        aucs = []
        for split_row in splits:
            result = run_one_split_po_or_pa(split_row=split_row, **shared_fixed, **combo)
            aucs.append(result["avg_auc"])

        return float(np.mean(aucs))
    return objective


def make_objective_popa(splits, shared_fixed):
    def objective(trial):
        combo = {
            "lr":            trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay":  trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),
            "hidden_dim":    trial.suggest_categorical("hidden_dim", [128, 256, 512]),
            "hidden_layers": trial.suggest_int("hidden_layers", 1, 3),
            "batch_size":    trial.suggest_categorical("batch_size", [128, 256, 500]),
            "epochs":        trial.suggest_categorical("epochs", [10, 50, 100]),
            "w_pa":          trial.suggest_float("w_pa", 0.25, 4.0, log=True),
        }

        aucs = []
        for split_row in splits:
            result = run_one_split_popa(split_row=split_row, **shared_fixed, **combo)
            aucs.append(result["avg_auc"])

        return float(np.mean(aucs))
    return objective


# ── Summary helpers (unchanged from your original) ───────────────────────────

def study_to_df(study: optuna.Study, method: str) -> pd.DataFrame:
    """Convert Optuna study trials to the same flat format as before."""
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {f"param_{k}": v for k, v in t.params.items()}
        row["method"] = method
        row["mean_auc"] = t.value
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_auc", ascending=False)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(test_number: int, n_trials: int):
    run_pa   = False
    run_popa = True

    set_all_seeds(FIXED["seed"])

    split_dir  = Path("outputs/splits/france")
    output_dir = Path(f"outputs/tune_optuna/test_{test_number}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data       = load_geoplant_processed("data/processed/geoplant/france")
    split_table = load_split_specs(split_dir)

    mask = split_table["test_number"] == test_number
    splits_for_test = split_table[mask].reset_index(drop=True)
    if len(splits_for_test) == 0:
        raise ValueError(f"No splits found for test_number={test_number}")

    splits = [row for _, row in splits_for_test.iterrows()]
    print(f"test_number={test_number} | {len(splits)} splits | options={splits_for_test['option'].unique().tolist()}")

    shared_fixed = dict(
        split_dir=split_dir,
        data=data,
        output_dir=output_dir,
        project_name="sdm-integration-tune",
        **FIXED,
    )

    # Persist studies to SQLite — crash-safe, and you can query/resume later
    storage = f"sqlite:///{output_dir}/optuna.db"

    # ── PA ───────────────────────────────────────────────────────────────
    if run_pa:
        print("\n" + "█"*60 + "\n  PA-ONLY OPTUNA SEARCH\n" + "█"*60)
        study_pa = optuna.create_study(
            study_name=f"pa_test{test_number}",
            direction="maximize",
            storage=storage,
            load_if_exists=True,   # resume if interrupted
        )
        study_pa.optimize(
            make_objective_pa(splits, shared_fixed),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        df_pa = study_to_df(study_pa, "pa")
        df_pa.to_csv(output_dir / "summary_pa.csv", index=False)
        print(f"\nBest PA:  AUC={study_pa.best_value:.4f}")
        print(study_pa.best_params)

    # ── POPA ─────────────────────────────────────────────────────────────
    if run_popa:
        print("\n" + "█"*60 + "\n  PO+PA OPTUNA SEARCH\n" + "█"*60)
        study_popa = optuna.create_study(
            study_name=f"popa_test{test_number}",
            direction="maximize",
            storage=storage,
            load_if_exists=True,
        )
        study_popa.optimize(
            make_objective_popa(splits, shared_fixed),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        df_popa = study_to_df(study_popa, "popa")
        df_popa.to_csv(output_dir / "summary_popa.csv", index=False)
        print(f"\nBest POPA: AUC={study_popa.best_value:.4f}")
        print(study_popa.best_params)

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_number", type=int, default=0)
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of Optuna trials (replaces exhaustive grid)")
    args = parser.parse_args()
    main(args.test_number, args.n_trials)