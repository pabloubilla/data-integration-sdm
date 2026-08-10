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



def make_param_grid(
    grid: Dict[str, List[Any]],
    linked: Optional[Dict[tuple, List[tuple]]] = None,
) -> List[Dict[str, Any]]:
    """
    Expand param combos, with optional "linked" groups of params that move
    together as fixed tuples instead of being cross-produced independently.

    `grid` behaves exactly as before — full cross product of every key.

    `linked` lets you say "these N params only take these specific
    combinations" instead of the full cross product of their individual
    values. Each entry is:
        (key1, key2, ...) -> [(val1, val2, ...), (val1, val2, ...), ...]

    Multiple linked groups are independent of each other and DO cross-
    multiply against each other and against `grid` — only params within
    the same group are locked together.

    Example:
        make_param_grid(
            grid={"epochs": [10, 20]},
            linked={
                ("loss_po_name", "loss_pa_name"): [
                    ("deep_maxent", "balanced_bce"),
                    ("balanced_bce", "balanced_bce"),
                ],
                ("lr", "weight_decay"): [
                    (1e-3, 1e-2),   # high lr paired with high wd
                    (1e-4, 1e-4),   # low lr paired with low wd
                ],
            },
        )
        → 2 epochs × 2 loss-pairs × 2 (lr,wd)-pairs = 8 combos total,
          but NEVER e.g. lr=1e-3 paired with weight_decay=1e-4 —
          only the pairs you explicitly listed.
    """
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

def run_grid_search(
    *,
    param_grid: Dict[str, List[Any]],
    splits: List[Any],                        # list of split_row (pd.Series or dict)
    run_fn: Callable[..., Dict[str, Any]],    # run_one_split_pa / run_one_split_popa
    fixed_kwargs: Dict[str, Any],             # everything that doesn't change (data, dirs, etc.)
    param_keys: List[str],                    # which keys to forward from each combo to run_fn
    linked_param_grid: Optional[Dict[tuple, List[tuple]]] = None, # Grid can be linked (meaning some parameters are always used together)
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

    combos = make_param_grid(param_grid, linked_param_grid)
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