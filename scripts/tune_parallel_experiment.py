"""
Timing experiment: measures wall-clock time for a fixed batch of trials
(same split, varying lr) run sequentially vs. via the process pool at
several max_workers values. Run this on the cluster, with real data/GPU.

BEFORE RUNNING — checklist from the earlier file-descriptor debugging:
  1. torch.multiprocessing.set_sharing_strategy('file_system') is set
     (already done below).
  2. DataLoader num_workers=0 wherever run_one_split_popa builds loaders
     — check isdm/splits.py's make_loader calls.
  3. `ulimit -n 65536` was run in your shell / job script before this.

CONFOUND CONTROLS (added after the first sweep showed parallel getting
WORSE as max_workers grew — before concluding that's a real GPU limit,
these three common confounds need to be ruled out first):
  - CPU thread oversubscription: NumPy/sklearn/PyTorch each try to use
    ALL cores by default, PER PROCESS. With N worker processes, that's
    N x all-cores worth of threads fighting over the same physical
    cores — gets worse as N grows, matching the pattern we saw. Fixed
    below by pinning each library to 1 thread; must be set before
    numpy/torch are imported in each (spawned) worker, hence the
    os.environ lines are the very first thing in this file.
  - wandb.init() contention: many processes calling this near-
    simultaneously can serialize on local file locks / network calls.
    Disabled for this timing test only.
  - Per-worker spawn startup cost: each worker re-imports torch/pandas/
    sklearn/isdm from scratch and unpickles the full fixed_kwargs
    (including the loaded dataset) via the pool initializer. With few,
    short trials this fixed cost can dominate. Now printed explicitly
    so it's visible in the output rather than hidden inside the total.

Usage:
    python scripts/time_parallel_experiment.py --test_number 0 --n_trials 8 \
        --worker_counts 1 2 4 8
"""

import os

# MUST be set before numpy/torch/sklearn are imported anywhere, including
# in spawned worker processes (which inherit this environment at creation).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["WANDB_MODE"] = "disabled"

import argparse
import time
from pathlib import Path

import torch.multiprocessing as torch_mp
torch_mp.set_sharing_strategy('file_system')

from isdm.load_data import load_geoplant_processed
from isdm.splits import load_split_specs, run_one_split_popa
from isdm.tune_parallel_v2 import run_grid_search_parallel
from isdm.utils import set_all_seeds


def run_sequential(combos, splits, run_fn, fixed_kwargs, param_keys):
    all_results = []
    for combo in combos:
        param_kwargs = {k: combo[k] for k in param_keys if k in combo}
        for split_row in splits:
            result = run_fn(split_row=split_row, **fixed_kwargs, **param_kwargs)
            result.update({f"param_{k}": v for k, v in combo.items()})
            all_results.append(result)
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_number", type=int, default=0)
    parser.add_argument("--n_trials", type=int, default=8,
                         help="Number of trials (different lr) to run on the SAME split.")
    parser.add_argument("--worker_counts", type=int, nargs="+", default=[16],
                         help="max_workers values to test, one after another.")
    parser.add_argument("--data_path", type=str, default="data/processed/GeoPlant/france")
    parser.add_argument("--split_dir", type=str,
                         default="outputs/splits/GeoPlant/france_bands/geographical")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    set_all_seeds(42)

    print("Loading data (once, shared by all conditions)...")
    data = load_geoplant_processed(args.data_path)

    split_table = load_split_specs(args.split_dir)
    splits_for_test = split_table[
        (split_table["test_number"] == args.test_number)
        & (split_table["option"] == "closest_val")
    ]
    if len(splits_for_test) == 0:
        raise ValueError(f"No closest_val split found for test_number={args.test_number}")
    split_row = splits_for_test.iloc[0]
    splits = [split_row]
    print(f"Using split: {split_row['split_id']}")

    output_dir = Path("outputs/time_parallel_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    combos = [{"lr": 1e-4 * (i + 1)} for i in range(args.n_trials)]
    param_keys = ["lr"]

    fixed_kwargs = dict(
        split_dir=Path(args.split_dir),
        data=data,
        output_dir=output_dir,
        project_name="time-parallel-experiment",
        seed=42,
        batch_size=500,
        epochs=args.epochs,
        weight_decay=1e-3,
        hidden_dim=128,
        hidden_layers=2,
        validate_during_training=False,
    )

    results = {}

    # ── sequential baseline ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"SEQUENTIAL: {args.n_trials} trials, one after another")
    print("=" * 60)
    t0 = time.time()
    run_sequential(combos, splits, run_one_split_popa, fixed_kwargs, param_keys)
    t_sequential = time.time() - t0
    print(f"Sequential total: {t_sequential:.2f}s")
    results["sequential"] = t_sequential

    # ── parallel, sweeping max_workers ──────────────────────────────────
    for w in args.worker_counts:
        print("\n" + "=" * 60)
        print(f"PARALLEL: {args.n_trials} trials, max_workers={w}")
        print("=" * 60)
        t0 = time.time()
        run_grid_search_parallel(
            combos=combos,
            splits=splits,
            run_fn=run_one_split_popa,
            fixed_kwargs=fixed_kwargs,
            param_keys=param_keys,
            max_workers=w,
        )
        t_parallel = time.time() - t0
        print(f"max_workers={w} total: {t_parallel:.2f}s  (speedup vs sequential: {t_sequential / t_parallel:.2f}x)")
        results[f"parallel_w{w}"] = t_parallel

    # ── report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'condition':<20}{'total (s)':<15}{'speedup':<10}")
    for name, t in results.items():
        speedup = t_sequential / t
        print(f"{name:<20}{t:<15.2f}{speedup:<10.2f}")

    print("\nLook for where speedup stops increasing as max_workers grows —")
    print("that's your GPU's effective concurrency ceiling for this model size.")


if __name__ == "__main__":
    main()