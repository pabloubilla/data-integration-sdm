"""
Generic grid search runner for split-based experiments.

Combos are built separately (see make_param_grid / make_random_param_grid /
build_trial_combos) — this module only executes them.

Results are saved ONE FILE PER TRIAL, not one growing file for the whole
run. This makes the run resumable (already-done trials are detected and
skipped) and crash-safe (a killed process leaves only complete, valid
per-trial files behind — never a corrupted combined file).

run_grid_search runs sequentially when max_workers=1 (the default) and
via a process pool when max_workers > 1 — same function either way, same
per-trial saving/resume behavior in both modes.
"""

import hashlib
import itertools
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ─────────────────────────────────────────────
#  Combo construction (unchanged from before)
# ─────────────────────────────────────────────

def make_param_grid(
    grid: Dict[str, List[Any]],
    linked: Optional[Dict[tuple, List[tuple]]] = None,
) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    values = list(grid.values())
    base_combos = (
        [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        if keys else [{}]
    )
    if not linked:
        return base_combos

    linked_key_groups = list(linked.keys())
    linked_value_lists = list(linked.values())
    final_combos = []
    for base in base_combos:
        for linked_choice in itertools.product(*linked_value_lists):
            combo = dict(base)
            for key_group, values_tuple in zip(linked_key_groups, linked_choice):
                combo.update(dict(zip(key_group, values_tuple)))
            final_combos.append(combo)
    return final_combos


def make_random_param_grid(
    grid: dict,
    n_trials: int,
    linked_random: Optional[dict] = None,
    linked_deterministic: Optional[dict] = None,
    seed: int = 42,
) -> List[dict]:
    import numpy as np

    rng = np.random.default_rng(seed)
    linked_random = linked_random or {}
    linked_deterministic = linked_deterministic or {}

    base_combos = []
    for _ in range(n_trials):
        combo = {k: v[rng.integers(len(v))] for k, v in grid.items()}
        for key_group, choices in linked_random.items():
            chosen_tuple = choices[rng.integers(len(choices))]
            combo.update(dict(zip(key_group, chosen_tuple)))
        base_combos.append(combo)

    if not linked_deterministic:
        return base_combos

    det_key_groups = list(linked_deterministic.keys())
    det_value_lists = list(linked_deterministic.values())
    final_combos = []
    for combo in base_combos:
        for det_choice in itertools.product(*det_value_lists):
            full_combo = dict(combo)
            for key_group, values_tuple in zip(det_key_groups, det_choice):
                full_combo.update(dict(zip(key_group, values_tuple)))
            final_combos.append(full_combo)
    return final_combos


def build_trial_combos(
    *,
    method: str,
    param_grid: dict,
    n_trials: int,
    loss_names: Optional[List[str]] = None,
    loss_combos: Optional[List[tuple]] = None,
    w_pa_po_pairs: Optional[List[tuple]] = None,
    seed: int = 42,
) -> List[dict]:
    if method in ("pa", "po"):
        if not loss_names:
            raise ValueError(f"loss_names is required for method='{method}'")
        linked_deterministic = {("loss_name",): [(name,) for name in loss_names]}
        return make_random_param_grid(
            param_grid, n_trials=n_trials,
            linked_deterministic=linked_deterministic, seed=seed,
        )

    if method == "popa":
        if not loss_combos:
            raise ValueError("loss_combos is required for method='popa'")
        linked_deterministic = {("loss_po_name", "loss_pa_name"): loss_combos}
        linked_random = {("w_pa", "w_po"): w_pa_po_pairs} if w_pa_po_pairs else None
        return make_random_param_grid(
            param_grid, n_trials=n_trials,
            linked_random=linked_random,
            linked_deterministic=linked_deterministic, seed=seed,
        )

    raise ValueError(f"Unknown method: {method!r}")


# ─────────────────────────────────────────────
#  Per-trial resume-safe saving
# ─────────────────────────────────────────────

def trial_id_for(combo: Dict[str, Any], split_row) -> str:
    """Deterministic id for a (combo, split) pair. Same inputs -> same id,
    so a result file's presence on disk IS the resume state."""
    key = json.dumps(
        {"combo": combo, "split_id": split_row["split_id"]},
        sort_keys=True, default=str,
    )
    return hashlib.md5(key.encode()).hexdigest()[:12]


def save_trial_result(results_dir: Path, trial_id: str, result: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = results_dir / f"{trial_id}.json.tmp"
    final_path = results_dir / f"{trial_id}.json"
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    os.replace(tmp_path, final_path)  # atomic on POSIX — no partial-write corruption


def load_existing_trial(results_dir: Path, trial_id: str) -> Optional[dict]:
    p = results_dir / f"{trial_id}.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def load_all_trial_results(results_dir: Path) -> List[dict]:
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]


# ─────────────────────────────────────────────
#  Core per-trial execution (shared by sequential + parallel paths)
# ─────────────────────────────────────────────

def _execute_trial(
    run_fn: Callable,
    fixed_kwargs: Dict[str, Any],
    param_keys: List[str],
    combo: Dict[str, Any],
    split_row,
    results_dir: Optional[Path],
) -> dict:
    tid = trial_id_for(combo, split_row) if results_dir is not None else None

    if results_dir is not None:
        existing = load_existing_trial(results_dir, tid)
        if existing is not None:
            return existing

    param_kwargs = {k: combo[k] for k in param_keys if k in combo}
    try:
        result = run_fn(split_row=split_row, **fixed_kwargs, **param_kwargs)
    except Exception as e:
        result = {"error": str(e), "test_number": split_row.get("test_number")}

    result.update({f"param_{k}": v for k, v in combo.items()})

    if results_dir is not None:
        save_trial_result(results_dir, tid, result)

    return result


# ─────────────────────────────────────────────
#  Parallel worker plumbing (spawn context, one-time initializer)
# ─────────────────────────────────────────────

_WORKER_RUN_FN = None
_WORKER_FIXED_KWARGS = None
_WORKER_PARAM_KEYS = None
_WORKER_RESULTS_DIR = None


def _worker_init(run_fn, fixed_kwargs, param_keys, results_dir, pool_start_time):
    global _WORKER_RUN_FN, _WORKER_FIXED_KWARGS, _WORKER_PARAM_KEYS, _WORKER_RESULTS_DIR
    _WORKER_RUN_FN = run_fn
    _WORKER_FIXED_KWARGS = fixed_kwargs
    _WORKER_PARAM_KEYS = param_keys
    _WORKER_RESULTS_DIR = results_dir
    print(f"[worker init] ready {time.time() - pool_start_time:.2f}s after pool creation "
          f"(pid={os.getpid()})")


def _worker_task(combo, split_row):
    return _execute_trial(
        _WORKER_RUN_FN, _WORKER_FIXED_KWARGS, _WORKER_PARAM_KEYS,
        combo, split_row, _WORKER_RESULTS_DIR,
    )


# ─────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────

def run_grid_search(
    *,
    combos: List[Dict[str, Any]],
    splits: List[Any],
    run_fn: Callable[..., Dict[str, Any]],
    fixed_kwargs: Dict[str, Any],
    param_keys: List[str],
    results_dir: Optional[Path] = None,
    combo_label_fn: Optional[Callable[[Dict], str]] = None,
    max_workers: int = 1,
) -> List[Dict[str, Any]]:
    """
    Run `run_fn` for every (combo × split) pair.

    results_dir: if given, each trial's result is saved as its own JSON
    file (named by a deterministic hash of (combo, split_id)), and any
    trial already present is skipped rather than recomputed. This makes
    the run resumable and crash-safe. Aggregate afterwards with
    load_all_trial_results(results_dir).

    max_workers: 1 (default) runs sequentially in this process. >1 runs
    via a process pool (spawn context — safe for CUDA), with run_fn and
    fixed_kwargs sent to each worker ONCE via the pool initializer rather
    than re-pickled per trial.
    """
    tasks = [(combo, split_row) for combo in combos for split_row in splits]
    total = len(tasks)

    if results_dir is not None:
        results_dir = Path(results_dir)

    if max_workers <= 1:
        print(f"Running {total} trials sequentially...")
        all_results = []
        for i, (combo, split_row) in enumerate(tasks, 1):
            label = combo_label_fn(combo) if combo_label_fn else str(combo)
            print(f"[{i}/{total}] {label}")
            result = _execute_trial(run_fn, fixed_kwargs, param_keys, combo, split_row, results_dir)
            all_results.append(result)
        return all_results

    print(f"Launching {total} trials across {max_workers} worker processes (spawn)...")
    all_results = []
    ctx = multiprocessing.get_context("spawn")
    pool_start_time = time.time()

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(run_fn, fixed_kwargs, param_keys, results_dir, pool_start_time),
    ) as executor:
        future_to_combo = {
            executor.submit(_worker_task, combo, split_row): combo
            for combo, split_row in tasks
        }
        for i, future in enumerate(as_completed(future_to_combo), 1):
            combo = future_to_combo[future]
            result = future.result()
            all_results.append(result)
            label = combo_label_fn(combo) if combo_label_fn else str(combo)
            print(f"[{i}/{total}] done: {label}")

    return all_results