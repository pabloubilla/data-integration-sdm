"""
Generic grid search runner for split-based experiments.

Results are accumulated into a flat list of dicts — one row per
(param_combo × split_option × method) — so they can be turned into
a DataFrame for analysis.
"""

import itertools
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def make_param_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Expand a dict-of-lists into a list of all combinations.

    Example:
        make_param_grid({"lr": [1e-3, 1e-4], "wd": [0, 1e-4]})
        → [{"lr": 1e-3, "wd": 0}, {"lr": 1e-3, "wd": 1e-4},
           {"lr": 1e-4, "wd": 0}, {"lr": 1e-4, "wd": 1e-4}]
    """
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_grid_search(
    *,
    param_grid: Dict[str, List[Any]],
    splits: List[Any],                        # list of split_row (pd.Series or dict)
    run_fn: Callable[..., Dict[str, Any]],    # run_one_split_pa / run_one_split_popa
    fixed_kwargs: Dict[str, Any],             # everything that doesn't change (data, dirs, etc.)
    param_keys: List[str],                    # which keys to forward from each combo to run_fn
    results_path: Optional[Path] = None,      # if set, saves running JSON after every trial
    combo_label_fn: Optional[Callable[[Dict], str]] = None,  # for pretty printing
) -> List[Dict[str, Any]]:
    """
    Run `run_fn` for every (param_combo × split) pair.

    Each call to run_fn must return a dict with at least:
        - test_number, option, distance, avg_auc
    The returned combo params are merged into that dict so every row
    is self-contained.

    Args:
        param_grid:     dict of param_name → list of values to try
        splits:         iterable of split rows to pass to run_fn
        run_fn:         function with signature run_fn(*, split_row, **fixed_kwargs, **param_kwargs)
        fixed_kwargs:   constant kwargs forwarded to every run_fn call
        param_keys:     subset of param_grid keys that run_fn actually accepts
                        (lets you include "label-only" entries in the grid)
        results_path:   optional path to stream results to JSON incrementally
        combo_label_fn: optional fn(combo_dict) → str for progress printing

    Returns:
        List of result dicts, one per (combo, split) trial.
    """

    combos = make_param_grid(param_grid)
    all_results: List[Dict[str, Any]] = []

    total = len(combos) * len(splits)
    trial_idx = 0

    for combo in combos:
        label = combo_label_fn(combo) if combo_label_fn else str(combo)
        print(f"\n{'='*60}")
        print(f"Param combo: {label}")
        print(f"{'='*60}")

        # Only forward keys that run_fn actually accepts
        param_kwargs = {k: combo[k] for k in param_keys if k in combo}

        for split_row in splits:

            trial_idx += 1
            print(split_row)
            print(
                f"\n  [Trial {trial_idx}/{total}] "
                f"split={split_row['test_number']}  option={split_row['option']}  "
                f"distance={split_row['distance']:.4f}"
            )

            # try:
            result = run_fn(
                split_row=split_row,
                **fixed_kwargs,
                **param_kwargs,
            )
            # except Exception as e:
            #     print(f"  !! Trial failed: {e}")
            #     result = {
            #         "test_number": split_row["test_number"],
            #         "option": split_row["option"],
            #         "distance": float(split_row["distance"]),
            #         "avg_auc": None,
            #         "error": str(e),
            #     }

            # Merge combo params into result row for full traceability
            result.update({f"param_{k}": v for k, v in combo.items()})
            all_results.append(result)

            # Stream to disk so a crash doesn't lose everything
            if results_path is not None:
                results_path.parent.mkdir(parents=True, exist_ok=True)
                with open(results_path, "w") as f:
                    json.dump(all_results, f, indent=2)

    return all_results