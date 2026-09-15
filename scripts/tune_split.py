"""
Runs the validation-tuning grid search for PA, PO, and/or POPA, saving
ONE JSON PER TRIAL under output_dir/trials_<method>/. This script only
RUNS trials — it does not build summary CSVs. Run
scripts/aggregate_tune_results.py afterwards (or anytime, even while this
is still running on the cluster) to build those.

Resumable: re-running this exact command skips any trial that already
has a saved result file.

Usage:
    python scripts/tune_split.py --test_number 0
"""
import os
import argparse
from pathlib import Path

import torch.multiprocessing as torch_mp
torch_mp.set_sharing_strategy('file_system')

from isdm.load_data import load_geoplant_processed
from isdm.splits import load_split_specs, run_one_split_popa_tunable, run_one_split_po_or_pa_tunable
from isdm.utils import set_all_seeds
from isdm.tune import run_grid_search, build_trial_combos


# ─────────────────────────────────────────────
#  Which methods to run
# ─────────────────────────────────────────────
RUN_PA = True
RUN_PO = True
RUN_POPA = True

# MAX_WORKERS = 16   # 1 = sequential. Set based on what timing experiment showed.
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 4))


EPOCHS_CEILING = 100
VAL_PATIENCE = 5

# ─────────────────────────────────────────────
#  PA
# ─────────────────────────────────────────────
PARAM_GRID_PA = {
    "lr":            [5e-3, 1e-3, 5e-4, 1e-4],
    "weight_decay":  [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 0],
    "hidden_dim":    [100,250,500],
    "hidden_layers": [1,2,3],
    "batch_size":    [100,200,500,1000],
}
LOSS_NAMES_PA = ["bce", "balanced_bce"]
N_TRIALS_PA = 100

# ─────────────────────────────────────────────
#  PO
# ─────────────────────────────────────────────
PARAM_GRID_PO = {
    "lr":            [5e-3, 1e-3, 5e-4, 1e-4],
    "weight_decay":  [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 0],
    "hidden_dim":    [100,250,500],
    "hidden_layers": [1,2,3],
    "batch_size":    [100,200,500,1000],
}
LOSS_NAMES_PO = ["deep_maxent", "balanced_bce", "bce"]
N_TRIALS_PO = 100

# ─────────────────────────────────────────────
#  POPA
# ─────────────────────────────────────────────
PARAM_GRID_POPA = {
    "lr":            [5e-3, 1e-3, 5e-4, 1e-4],
    "weight_decay":  [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 0],
    "hidden_dim":    [100,250,500],
    "hidden_layers": [1,2,3],
    "batch_size":    [100,200,500,1000],
}
LOSS_COMBOS_POPA = [
    ("deep_maxent", "balanced_bce_ippp"),
    ("deep_maxent", "balanced_bce"),
    ("deep_maxent", "bce"),
    ("deep_maxent", "bce_ippp"),
    ("balanced_bce", "balanced_bce"),
    ("deep_maxent", "deep_maxent"),
]
W_PA_PO_PAIRS = [
    (0.0, 1.0),
    (0.1, 0.9),
    (0.2, 0.8),
    (0.3, 0.7),
    (0.4, 0.6),
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1),
    (1.0, 0.0),
]  # (w_pa, w_po) pairs from (0.0,1.0) to (1.0,0.0)
N_TRIALS_POPA = 100

FIXED = {"seed": 42}


def main(test_number: int, dataset: str, split_type: str, region: str):
    set_all_seeds(FIXED["seed"])

    data_path = f"data/processed/{dataset}/{region}"
    split_dir = Path(f"outputs/splits/{dataset}/{region}_bands/{split_type}")
    output_dir = Path(f"outputs/tune/{dataset}/{region}/{split_type}/test_{test_number}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_geoplant_processed(data_path)
    split_table = load_split_specs(split_dir)
    splits_for_test = split_table[split_table["test_number"] == test_number].reset_index(drop=True)

    val_options = ("closest_val", "middle_val", "farthest_val")
    splits_for_test = splits_for_test[splits_for_test["option"].isin(val_options)].reset_index(drop=True)
    if len(splits_for_test) == 0:
        raise ValueError(f"No validation splits found for test_number={test_number}")

    print(f"test_number={test_number} | {len(splits_for_test)} splits | "
          f"options={splits_for_test['option'].unique().tolist()}")

    split_rows = [row for _, row in splits_for_test.iterrows()]
    split_rows_close = [split_rows[0]]  # PO train/test don't vary by option — one split is enough

    shared_fixed = dict(
        split_dir=split_dir,
        data=data,
        output_dir=output_dir,
        project_name="sdm-integration-tune",
        epochs=EPOCHS_CEILING,
        validate_during_training=True,
        val_patience=VAL_PATIENCE,
        **FIXED,
    )

    def combo_label(combo):
        return "  ".join(f"{k}={v}" for k, v in combo.items())

    if RUN_PA:
        print("\n" + "█" * 60 + "\n  PA-ONLY\n" + "█" * 60)
        combos_pa = build_trial_combos(
            method="pa", param_grid=PARAM_GRID_PA, n_trials=N_TRIALS_PA,
            loss_names=LOSS_NAMES_PA, seed=FIXED["seed"],
        )
        run_grid_search(
            combos=combos_pa,
            splits=split_rows,
            run_fn=run_one_split_po_or_pa_tunable,
            fixed_kwargs={**shared_fixed, "source": "pa"},
            param_keys=list(PARAM_GRID_PA.keys()) + ["loss_name"],
            results_dir=output_dir / "trials_pa",
            combo_label_fn=combo_label,
            max_workers=MAX_WORKERS,
        )

    if RUN_PO:
        print("\n" + "█" * 60 + "\n  PO-ONLY\n" + "█" * 60)
        combos_po = build_trial_combos(
            method="po", param_grid=PARAM_GRID_PO, n_trials=N_TRIALS_PO,
            loss_names=LOSS_NAMES_PO, seed=FIXED["seed"],
        )
        run_grid_search(
            combos=combos_po,
            splits=split_rows_close,
            run_fn=run_one_split_po_or_pa_tunable,
            fixed_kwargs={**shared_fixed, "source": "po"},
            param_keys=list(PARAM_GRID_PO.keys()) + ["loss_name"],
            results_dir=output_dir / "trials_po",
            combo_label_fn=combo_label,
            max_workers=MAX_WORKERS,
        )

    if RUN_POPA:
        print("\n" + "█" * 60 + "\n  PO+PA\n" + "█" * 60)
        combos_popa = build_trial_combos(
            method="popa", param_grid=PARAM_GRID_POPA, n_trials=N_TRIALS_POPA,
            loss_combos=LOSS_COMBOS_POPA, w_pa_po_pairs=W_PA_PO_PAIRS, seed=FIXED["seed"],
        )
        tunable_keys_popa = list(PARAM_GRID_POPA.keys()) + ["loss_po_name", "loss_pa_name", "w_pa", "w_po"]
        run_grid_search(
            combos=combos_popa,
            splits=split_rows,
            run_fn=run_one_split_popa_tunable,
            fixed_kwargs=shared_fixed,
            param_keys=tunable_keys_popa,
            results_dir=output_dir / "trials_popa",
            combo_label_fn=combo_label,
            max_workers=MAX_WORKERS,
        )

    print(f"\nAll trial results saved under: {output_dir}")
    print("Run scripts/aggregate_tune_results.py to build summary CSVs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_number", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="GeoPlant")
    parser.add_argument("--split_type", type=str, default="geographical")
    parser.add_argument("--region", type=str, default="france")
    args = parser.parse_args()
    main(args.test_number, args.dataset, args.split_type, args.region)