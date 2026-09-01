"""
Process-parallel version of run_grid_search.

Key design point: `fixed_kwargs` typically contains the loaded dataset
(potentially large — e.g. ~1M PO records). If passed to executor.submit()
per task, it would be pickled and sent through IPC once per trial, which
would dominate runtime. Instead, run_fn and fixed_kwargs are sent ONCE per
worker process via the pool's initializer, and each submitted task only
carries the small per-trial (combo, split_row) — cheap to pickle.

Uses the "spawn" start method explicitly rather than the Linux default
"fork", because forking a process that has already touched CUDA can hang
or crash. spawn is slower to start workers but is the safe, PyTorch-
recommended choice for CUDA multiprocessing.
"""

import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from isdm.tune import make_param_grid  # reuse existing combo expansion


_WORKER_RUN_FN = None
_WORKER_FIXED_KWARGS = None


def _worker_init(run_fn: Callable, fixed_kwargs: Dict[str, Any]):
    """Runs once per worker process, at pool startup."""
    global _WORKER_RUN_FN, _WORKER_FIXED_KWARGS
    _WORKER_RUN_FN = run_fn
    _WORKER_FIXED_KWARGS = fixed_kwargs


def _worker_task(combo: Dict[str, Any], split_row, param_keys: List[str]) -> Dict[str, Any]:
    """Runs once per (combo, split) trial. Only combo/split_row cross the
    process boundary here — run_fn and fixed_kwargs already live in this
    worker's memory from _worker_init."""
    param_kwargs = {k: combo[k] for k in param_keys if k in combo}
    result = _WORKER_RUN_FN(split_row=split_row, **_WORKER_FIXED_KWARGS, **param_kwargs)
    result.update({f"param_{k}": v for k, v in combo.items()})
    return result


def run_grid_search_parallel(
    *,
    param_grid: Dict[str, List[Any]],
    splits: List[Any],
    run_fn: Callable[..., Dict[str, Any]],
    fixed_kwargs: Dict[str, Any],
    param_keys: List[str],
    linked_param_grid: Optional[Dict[tuple, List[tuple]]] = None,
    results_path: Optional[Path] = None,
    combo_label_fn: Optional[Callable[[Dict], str]] = None,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    combos = make_param_grid(param_grid, linked=linked_param_grid)
    tasks = [(combo, split_row) for combo in combos for split_row in splits]
    total = len(tasks)
    print(f"Launching {total} trials across {max_workers} worker processes (spawn)...")

    all_results: List[Dict[str, Any]] = []
    ctx = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(run_fn, fixed_kwargs),
    ) as executor:
        future_to_combo = {
            executor.submit(_worker_task, combo, split_row, param_keys): combo
            for combo, split_row in tasks
        }

        for i, future in enumerate(as_completed(future_to_combo), 1):
            combo = future_to_combo[future]
            result = future.result()
            all_results.append(result)

            label = combo_label_fn(combo) if combo_label_fn else str(combo)
            print(f"[{i}/{total}] done: {label}")

            if results_path is not None:
                results_path.parent.mkdir(parents=True, exist_ok=True)
                with open(results_path, "w") as f:
                    json.dump(all_results, f, indent=2)

    return all_results