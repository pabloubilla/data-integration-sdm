"""
Runs the FINAL evaluation sweep — using hyperparameters already selected
by tune_split.py / aggregate_tune_results.py — across every real test
split (excludes the *_val splits used for tuning).

Parallelizable via MAX_WORKERS (env var), and resumable: each (config,
split) trial is saved as its own JSON under output_dir/trials/<config>/,
so an interrupted run can be safely re-launched.

Usage:
    MAX_WORKERS=4 python scripts/run_split_sweep.py --dataset_name GeoPlant
"""

import os
from pathlib import Path

import pandas as pd

import argparse

from isdm.losses import LOSS_REGISTRY
from isdm.load_data import load_geoplant_processed
from isdm.splits import (
    load_split_specs, train_po_for_splits, evaluate_model_on_all_pa_split_tests,
    set_all_seeds, run_one_split_po_or_pa_tunable, run_one_split_popa_tunable,
)
from isdm.tune import run_grid_search as run_batch_over_splits 

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class RunConfig:
    """One fixed hyperparameter setting — run once per real split."""
    name: str
    fn: Callable
    enabled: bool
    kwargs: dict = field(default_factory=dict)


@dataclass
class GlobalRunConfig:
    """Train-once + evaluate-all runner — called after the split loop.
    Not parallelized: there's a single expensive step (training), not one
    per split, so there's nothing here to distribute across workers."""
    name: str
    train_fn: Callable
    eval_fn: Callable
    enabled: bool
    train_kwargs: dict = field(default_factory=dict)
    eval_kwargs: dict = field(default_factory=dict)


def load_best_params(
    tune_dir: Path,
    method: str,
    loss_name: str | None = None,
    loss_po_name: str | None = None,
    loss_pa_name: str | None = None,
) -> dict:
    df = pd.read_csv(tune_dir / f"best_{method}_avg_over_distance.csv")

    if method == "popa":
        if loss_po_name is None or loss_pa_name is None:
            raise ValueError("loss_po_name and loss_pa_name are required for method='popa'")
        mask = (df["param_loss_po_name"] == loss_po_name) & (df["param_loss_pa_name"] == loss_pa_name)
    else:
        if loss_name is None:
            raise ValueError(f"loss_name is required for method='{method}'")
        mask = df["param_loss_name"] == loss_name

    matches = df[mask]
    if matches.empty:
        raise ValueError(f"No tuned result for method={method}, loss_name={loss_name}, "
                          f"loss_po_name={loss_po_name}, loss_pa_name={loss_pa_name}")
    best = matches.iloc[0]

    print(f"Best hyperparameters for {method.upper()}:")
    for col in df.columns:
        if col.startswith("param_"):
            print(f"  {col}: {best[col]}")
    print(f"  best_epoch (mean over distance): {best['best_epoch']:.2f}")

    params = {
        "lr": float(best["param_lr"]),
        "weight_decay": float(best["param_weight_decay"]),
        "hidden_dim": int(best["param_hidden_dim"]),
        "batch_size": int(best["param_batch_size"]),
        "hidden_layers": int(best["param_hidden_layers"]),
        "epochs": int(round(best["best_epoch"])),
    }
    if method == "popa":
        params["w_pa"] = float(best["param_w_pa"])
        params["w_po"] = float(best["param_w_po"])
    return params


def get_tuned_loss_names(tune_dir: Path, method: str) -> list[str]:
    df = pd.read_csv(tune_dir / f"best_{method}_avg_over_distance.csv")
    return sorted(df["param_loss_name"].unique().tolist())


def get_tuned_loss_combos(tune_dir: Path) -> list[tuple[str, str]]:
    df = pd.read_csv(tune_dir / "best_popa_avg_over_distance.csv")
    combos = df[["param_loss_po_name", "param_loss_pa_name"]].drop_duplicates()
    return list(combos.itertuples(index=False, name=None))


def main(dataset_name: str = "GeoPlant", split_type: str = "geographical", region: str = "france",
         use_overlapping_species: bool = True, tune_dir: str | None = None):
    seed = 42
    set_all_seeds(seed)

    run_pa = True
    run_popa = True
    run_po = True

    max_workers = int(os.environ.get("MAX_WORKERS", 1))
    print(f"MAX_WORKERS={max_workers}")

    overlap_name = "intersect" if use_overlapping_species else "union"

    data_path = f"data/processed/{dataset_name}/{region}"
    split_dir = Path(f"outputs/splits/{dataset_name}/{region}_bands/{split_type}")
    output_dir = Path(f"outputs/split_sweep/{dataset_name}/{region}_bands/{split_type}/{overlap_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    project_name = "sdm-integration"
    if tune_dir is None:
        tune_dir = Path(f"outputs/tune/{dataset_name}/{region}/{split_type}/test_0")
    else:
        tune_dir = Path(tune_dir)

    data = load_geoplant_processed(data_path)
    split_table = load_split_specs(split_dir)
    split_table = split_table[~split_table["option"].str.contains("val", case=False, na=False)]
    split_rows = [row for _, row in split_table.iterrows()]

    shared = dict(
        data=data,
        output_dir=output_dir,
        project_name=project_name,
        seed=seed,
        use_overlapping_species=use_overlapping_species,
    )

    shared_with_split_dir = dict(shared, split_dir=split_dir)

    # ── per-split configs, one per tuned loss choice — plain strings only,
    # NOT instantiated loss objects: run_grid_search hashes each combo via
    # json.dumps for resume/dedup, and a live object's default repr
    # includes its memory address, which differs every run and would
    # silently break resume. The _tunable wrappers resolve loss_name /
    # loss_po_name / loss_pa_name into real loss objects internally. ──
    pa_loss_names = get_tuned_loss_names(tune_dir, method="pa")
    popa_loss_combos = get_tuned_loss_combos(tune_dir)

    configs = []

    for loss_name in pa_loss_names:
        p = load_best_params(tune_dir, method="pa", loss_name=loss_name)
        configs.append(RunConfig(
            name=f"pa_{loss_name}",
            fn=run_one_split_po_or_pa_tunable,
            enabled=run_pa,
            kwargs=dict(
                lr=p["lr"], weight_decay=p["weight_decay"], hidden_dim=p["hidden_dim"],
                batch_size=p["batch_size"], hidden_layers=p["hidden_layers"], epochs=p["epochs"],
                loss_name=loss_name, source="pa",
            ),
        ))

    for loss_po_name, loss_pa_name in popa_loss_combos:
        p = load_best_params(tune_dir, method="popa", loss_po_name=loss_po_name, loss_pa_name=loss_pa_name)
        configs.append(RunConfig(
            name=f"po_{loss_po_name}_pa_{loss_pa_name}",
            fn=run_one_split_popa_tunable,
            enabled=run_popa,
            kwargs=dict(
                lr=p["lr"], weight_decay=p["weight_decay"], hidden_dim=p["hidden_dim"],
                batch_size=p["batch_size"], hidden_layers=p["hidden_layers"], epochs=p["epochs"],
                w_po=p["w_po"], w_pa=p["w_pa"],
                loss_po_name=loss_po_name, loss_pa_name=loss_pa_name,
                return_logits=False, concat_sources=False,
            ),
        ))

    # ── global (train-once) configs, one per tuned PO loss — NOT
    # parallelized: one training run per loss, not one per split, so
    # there's no per-split cost here to distribute across workers ──
    po_loss_names = get_tuned_loss_names(tune_dir, method="po")

    global_configs = []
    for loss_name in po_loss_names:
        p = load_best_params(tune_dir, method="po", loss_name=loss_name)
        global_configs.append(GlobalRunConfig(
            name=f"po_{loss_name}",
            train_fn=train_po_for_splits,
            eval_fn=evaluate_model_on_all_pa_split_tests,
            enabled=run_po,
            train_kwargs=dict(
                lr=p["lr"], weight_decay=p["weight_decay"], hidden_dim=p["hidden_dim"],
                batch_size=p["batch_size"], hidden_layers=p["hidden_layers"], epochs=p["epochs"],
                criterion=LOSS_REGISTRY[loss_name](),
            ),
            eval_kwargs=dict(
                split_dir=split_dir,
                source_name="po",
                use_overlapping_species=use_overlapping_species,
            ),
        ))

    # ── per-split configs: run via run_batch_over_splits (parallel + resumable) ──
    active_configs = [c for c in configs if c.enabled]
    results: dict[str, list] = {}

    for cfg in active_configs:
        print(f"\n{'=' * 60}\n=== {cfg.name}: {len(split_rows)} splits, max_workers={max_workers} ===\n{'=' * 60}")
        results[cfg.name] = run_batch_over_splits(
            combos=[cfg.kwargs],   # ONE fixed config — already tuned, nothing to search here
            splits=split_rows,
            run_fn=cfg.fn,
            fixed_kwargs=shared_with_split_dir,
            param_keys=list(cfg.kwargs.keys()),
            results_dir=output_dir / "trials" / cfg.name,
            combo_label_fn=lambda c, name=cfg.name: name,
            max_workers=max_workers,
        )

    # ── global loop: unchanged, not parallelized ────────────────────────
    for cfg in global_configs:
        if not cfg.enabled:
            continue
        print(f"\n{'=' * 60}\n=== {cfg.name} (train-once + evaluate-all) ===\n{'=' * 60}")
        model, scaler = cfg.train_fn(**shared, **cfg.train_kwargs)
        summary, _ = cfg.eval_fn(
            model=model, scaler=scaler, split_table=split_table, data=data,
            output_dir=output_dir, project_name=project_name, **cfg.eval_kwargs,
        )
        results[cfg.name] = summary

    # ── save, print, common summary ────────────────────────────────────────────────
    SORT_COL = "distance"
    summaries: dict[str, pd.DataFrame] = {}

    all_configs = active_configs + [c for c in global_configs if c.enabled]
    for cfg in all_configs:
        raw = results[cfg.name]
        df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
        df.to_csv(output_dir / f"summary_{cfg.name}.csv", index=False)
        summaries[cfg.name] = df
        print(f"\n=== {cfg.name} ===")
        print(df.sort_values(SORT_COL) if SORT_COL in df.columns else df)

    shared_cols = sorted(set.intersection(*(set(df.columns) for df in summaries.values())))
    common_summary = pd.concat(
        [df[shared_cols].assign(run=name) for name, df in summaries.items()],
        ignore_index=True,
    )
    common_summary.to_csv(output_dir / "summary_common.csv", index=False)
    print("\n=== common summary ===")
    print(common_summary.sort_values(["run", SORT_COL] if SORT_COL in shared_cols else ["run"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the final split sweep for presence-absence models.")
    parser.add_argument("--dataset_name", type=str, default="GeoPlant")
    parser.add_argument("--split_type", type=str, default="geographical")
    parser.add_argument("--region", type=str, default="france")
    parser.add_argument("--use_overlapping_species", action="store_true")
    parser.add_argument("--tune_dir", type=str, default=None, help="Path to the directory containing tuned hyperparameters. If not provided, defaults to outputs/tune/{dataset_name}/{region}/{split_type}/test_0.")
    args = parser.parse_args()

    main(dataset_name=args.dataset_name, split_type=args.split_type,
         region=args.region, use_overlapping_species=args.use_overlapping_species,
         tune_dir=args.tune_dir)