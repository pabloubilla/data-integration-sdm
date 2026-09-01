"""
Generic grid search runner for split-based experiments. 

(also random option)

Results are accumulated into a flat list of dicts — one row per
(param_combo × split_option × method) — so they can be turned into
a DataFrame for analysis.
"""

import itertools
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np



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


def build_trial_combos(
    *,
    method: str,          # "pa", "po", or "popa"
    param_grid: dict,
    n_trials: int,
    loss_names: list[str] | None = None,               # for "pa" / "po"
    loss_combos: list[tuple[str, str]] | None = None,   # for "popa"
    w_pa_po_pairs: list[tuple[float, float]] | None = None,  # for "popa", sampled randomly
    seed: int = 42,
) -> list[dict]:
    """
    Generic entry point for building trial combos regardless of method.
    Wraps the per-method linked-group shape so callers never need to know
    that a single loss_name is a 1-tuple key under the hood.
    """
    if method in ("pa", "po"):
        if not loss_names:
            raise ValueError(f"loss_names is required for method='{method}'")
        linked_deterministic = {("loss_name",): [(name,) for name in loss_names]}
        return make_random_param_grid(
            param_grid, n_trials=n_trials,
            linked_deterministic=linked_deterministic,
            seed=seed,
        )

    if method == "popa":
        if not loss_combos:
            raise ValueError("loss_combos is required for method='popa'")
        linked_deterministic = {("loss_po_name", "loss_pa_name"): loss_combos}
        linked_random = {("w_pa", "w_po"): w_pa_po_pairs} if w_pa_po_pairs else None
        return make_random_param_grid(
            param_grid, n_trials=n_trials,
            linked_random=linked_random,
            linked_deterministic=linked_deterministic,
            seed=seed,
        )

    raise ValueError(f"Unknown method: {method!r}")

def make_random_param_grid(
    grid: dict,
    n_trials: int,
    linked_random: dict | None = None,
    linked_deterministic: dict | None = None,
    seed: int = 42,
) -> list[dict]:
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

def run_grid_search(
    *,
    combos: List[Dict[str, Any]],             # pre-built list of combo dicts
    splits: List[Any],
    run_fn: Callable[..., Dict[str, Any]],
    fixed_kwargs: Dict[str, Any],
    param_keys: List[str],
    results_path: Optional[Path] = None,
    combo_label_fn: Optional[Callable[[Dict], str]] = None,
) -> List[Dict[str, Any]]:
    """
    Run `run_fn` for every (combo × split) pair.

    combos: list of param-combo dicts, e.g. from build_trial_combos or
    make_param_grid — this function doesn't build them, just runs them.
    """
    all_results: List[Dict[str, Any]] = []
    total = len(combos) * len(splits)
    trial_idx = 0

    for combo in combos:
        label = combo_label_fn(combo) if combo_label_fn else str(combo)
        print(f"\n{'='*60}")
        print(f"Param combo: {label}")
        print(f"{'='*60}")

        param_kwargs = {k: combo[k] for k in param_keys if k in combo}

        for split_row in splits:
            trial_idx += 1
            print(
                f"\n  [Trial {trial_idx}/{total}] "
                f"split={split_row['test_number']}  option={split_row['option']}  "
                f"distance={split_row['distance']:.4f}"
            )

            result = run_fn(
                split_row=split_row,
                **fixed_kwargs,
                **param_kwargs,
            )

            result.update({f"param_{k}": v for k, v in combo.items()})
            all_results.append(result)

            if results_path is not None:
                results_path.parent.mkdir(parents=True, exist_ok=True)
                with open(results_path, "w") as f:
                    json.dump(all_results, f, indent=2)

    return all_results