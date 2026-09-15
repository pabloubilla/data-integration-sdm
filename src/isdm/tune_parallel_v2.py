"""
Process-parallel grid search: runs multiple trials concurrently, each in
its own process, sharing one GPU.

Key points (from earlier debugging):
  - Uses "spawn", not the Linux default "fork" — forking a process that
    has touched CUDA can hang/crash. spawn is the PyTorch-recommended,
    safe choice for CUDA multiprocessing.
  - run_fn and fixed_kwargs (which include the large loaded dataset) are
    sent to each worker ONCE via the pool's initializer, not per task —
    per-task would re-pickle the full dataset through IPC every trial.
  - IMPORTANT: make sure DataLoader num_workers=0 wherever run_fn builds
    loaders (e.g. inside run_one_split_popa's make_loader calls) when
    running under this parallel path. Nested multiprocessing (grid-search
    workers each spawning their own DataLoader workers) exhausts the OS
    file-descriptor limit — see the "Too many open files" issue from
    earlier. Also keep `torch.multiprocessing.set_sharing_strategy
    ('file_system')` set at the top of the calling script, and
    `ulimit -n 65536` in the job script as a backstop.
"""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


_WORKER_RUN_FN = None
_WORKER_FIXED_KWARGS = None


def _worker_init(run_fn: Callable, fixed_kwargs: Dict[str, Any], pool_start_time: float):
    # pool_start_time is the parent's clock reading from right before the
    # pool was created — the gap between that and "now" (inside this
    # worker, after spawn + reimporting torch/pandas/sklearn/isdm +
    # unpickling fixed_kwargs) is the REAL per-worker startup cost.
    import time
    global _WORKER_RUN_FN, _WORKER_FIXED_KWARGS
    _WORKER_RUN_FN = run_fn
    _WORKER_FIXED_KWARGS = fixed_kwargs
    print(f"[worker init] ready {time.time() - pool_start_time:.2f}s after pool creation "
          f"(pid={__import__('os').getpid()})")


def _worker_task(combo: Dict[str, Any], split_row, param_keys: List[str]) -> Dict[str, Any]:
    param_kwargs = {k: combo[k] for k in param_keys if k in combo}
    result = _WORKER_RUN_FN(split_row=split_row, **_WORKER_FIXED_KWARGS, **param_kwargs)
    result.update({f"param_{k}": v for k, v in combo.items()})
    return result


def run_grid_search_parallel(
    *,
    combos: List[Dict[str, Any]],
    splits: List[Any],
    run_fn: Callable[..., Dict[str, Any]],
    fixed_kwargs: Dict[str, Any],
    param_keys: List[str],
    results_path: Optional[Path] = None,
    combo_label_fn: Optional[Callable[[Dict], str]] = None,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    tasks = [(combo, split_row) for combo in combos for split_row in splits]
    total = len(tasks)
    print(f"Launching {total} trials across {max_workers} worker processes (spawn)...")

    all_results: List[Dict[str, Any]] = []
    ctx = multiprocessing.get_context("spawn")

    import time
    pool_start_time = time.time()

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(run_fn, fixed_kwargs, pool_start_time),
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