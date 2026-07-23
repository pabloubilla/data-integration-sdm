from pathlib import Path
from functools import partial
import json
import pickle

import numpy as np
import pandas as pd

import argparse

from isdm.models import BalancedBCELoss, BernoulliFromLogRateLoss, DeepMaxEntLoss, IntegratedLoss

from isdm.load_data import load_geoplant_processed

from isdm.splits import load_split_specs, load_split, train_po_for_splits, evaluate_model_on_all_pa_split_tests, set_all_seeds, run_one_split_pa, run_one_split_popa
from isdm.evaluation import LogitsStore

from dataclasses import dataclass, field
from typing import Callable

from torch.nn import BCEWithLogitsLoss



@dataclass
class RunConfig:
    """Per-split runner — called once per row in the loop."""
    name: str
    fn: Callable
    enabled: bool
    kwargs: dict = field(default_factory=dict)

@dataclass
class GlobalRunConfig:
    """Train-once + evaluate-all runner — called after the split loop."""
    name: str
    train_fn: Callable
    eval_fn: Callable
    enabled: bool
    train_kwargs: dict = field(default_factory=dict)
    eval_kwargs: dict = field(default_factory=dict)


def load_best_params(tune_dir: Path, method: str = "pa") -> dict:
    """Load best hyperparams from a completed tune run."""
    pivot = pd.read_csv(tune_dir / f"pivot_{method}.csv")
    # Already sorted by mean_auc descending, so top row is best
    best = pivot.iloc[0]

    # print
    print(f"Best hyperparameters for {method.upper()}:")
    for param, value in best.items():
        if param.startswith("param_"):
            print(f"{param}: {value}")

    return {
        "lr":           best["param_lr"],
        "weight_decay": best["param_weight_decay"],
        "hidden_dim":   int(best["param_hidden_dim"]),
        "batch_size":   int(best["param_batch_size"]),
        "epochs":       int(best["param_epochs"]),
        "hidden_layers": int(best["param_hidden_layers"]),
        "w_pa": best.get("param_w_pa", None),  # only for popa
    }


def main(dataset_name: str = "GeoPlant", split_type: str = "geographical", use_overlapping_species: bool = True):
    seed = 42
    set_all_seeds(seed)

    run_pa = True
    run_popa = True
    run_po = True

    overlap_name = "intersect" if use_overlapping_species else "union"

    sweep_subsample = None  # set to an integer to subsample splits for quick testing

    data_path = f"data/processed/{dataset_name}/france"
    split_dir = Path(f"outputs/splits/{dataset_name}/france_bands/{split_type}")
    output_dir = Path(f"outputs/split_sweep/{dataset_name}/france_bands/{split_type}/{overlap_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_dir_logits = output_dir / "logits"
    output_dir_logits.mkdir(parents=True, exist_ok=True)

    project_name = "sdm-integration"

    batch_size_po = 500
    epochs_po = 20
    lr_po = 1e-3
    weight_decay_po = 3e-4
    hidden_dim_po = 250
    hidden_layers_po = 2

    # HYPERPARAMS FOR PA MODEL (fixed across all trials in this sweep)
   # params_pa = load_best_params(Path("outputs/tune/test_0"), method="pa")
    batch_size_pa = 500 #params_pa["batch_size"]
    epochs_pa = 20 #params_pa["epochs"]
    lr_pa = 0.001 #params_pa["lr"]
    weight_decay_pa = 3e-4 #params_pa["weight_decay"]
    hidden_dim_pa = 250 #params_pa["hidden_dim"]
    hidden_layers_pa = 2 #params_pa["hidden_layers"]

    # HYPERPARAMS FOR POPA MODEL (fixed across all trials in this sweep)
    # params_popa = load_best_params(Path("outputs/tune/test_0"), method="popa")
    batch_size_popa = 500 #params_popa["batch_size"]
    epochs_popa = 20 #params_popa["epochs"]
    lr_popa = .001 # params_popa["lr"]
    weight_decay_popa = 3e-4 #params_popa["weight_decay"]
    hidden_dim_popa = 250 #params_popa["hidden_dim"]
    hidden_layers_popa = 2 #params_popa["hidden_layers"]
    w_pa_popa = 1.0 #params_popa["w_pa"]


    data = load_geoplant_processed(data_path)
    split_table = load_split_specs(split_dir)

    # ── shared kwargs ──────────────────────────────────────────────────────────────
    shared = dict(
        data=data,
        output_dir=output_dir,
        project_name=project_name,
        seed=seed,
        use_overlapping_species=use_overlapping_species,
    )

    # ── per-split configs ──────────────────────────────────────────────────────────
    configs = [
        #### Classic Experiment ###
        RunConfig(
            name="pa_bbce",
            fn=run_one_split_pa,
            enabled=run_pa,
            kwargs=dict(
                batch_size=batch_size_pa, epochs=epochs_pa, lr=lr_pa,
                weight_decay=weight_decay_pa, hidden_dim=hidden_dim_pa,
                hidden_layers=hidden_layers_pa, criterion=BalancedBCELoss(),
            ),
        ),
        RunConfig(
            name="pa_bce",
            fn=run_one_split_pa,
            enabled=run_pa,
            kwargs=dict(
                batch_size=batch_size_pa, epochs=epochs_pa, lr=lr_pa,
                weight_decay=weight_decay_pa, hidden_dim=hidden_dim_pa,
                hidden_layers=hidden_layers_pa, criterion=BCEWithLogitsLoss(),
            ),
        ),
        RunConfig(
            name="po_dme_pa_dme",
            fn=run_one_split_popa,
            enabled=run_popa,
            kwargs=dict(
                batch_size=batch_size_popa, epochs=epochs_popa, lr=lr_popa,
                weight_decay=weight_decay_popa, hidden_dim=hidden_dim_popa,
                hidden_layers=hidden_layers_popa, w_pa=w_pa_popa, return_logits=False,
                criterion_po=DeepMaxEntLoss(), criterion_pa=DeepMaxEntLoss(), concat_sources=False,
            ),
        ),
        RunConfig(
            name="po_bbce_pa_bbce",
            fn=run_one_split_popa,
            enabled=run_popa,
            kwargs=dict(
                batch_size=batch_size_popa, epochs=epochs_popa, lr=lr_popa,
                weight_decay=weight_decay_popa, hidden_dim=hidden_dim_popa,
                hidden_layers=hidden_layers_popa, w_pa=w_pa_popa, return_logits=False,
                criterion_po=BalancedBCELoss(), criterion_pa=BalancedBCELoss(), concat_sources=False,
            ),
        ),
        # RunConfig(
        #     name="po_dme_pa_bipp",
        #     fn=run_one_split_popa,
        #     enabled=run_popa,
        #     kwargs=dict(
        #         batch_size=batch_size_popa, epochs=epochs_popa, lr=lr_popa,
        #         weight_decay=weight_decay_popa, hidden_dim=hidden_dim_popa,
        #         hidden_layers=hidden_layers_popa, w_pa=w_pa_popa, return_logits=False,
        #         criterion_po=DeepMaxEntLoss(), criterion_pa=BernoulliFromLogRateLoss(balance_pos=True), concat_sources=False,
        #     ),
        # ),
        RunConfig(
            name="po_dme_pa_bbce",
            fn=run_one_split_popa,
            enabled=run_popa,
            kwargs=dict(
                batch_size=batch_size_popa, epochs=epochs_popa, lr=lr_popa,
                weight_decay=weight_decay_popa, hidden_dim=hidden_dim_popa,
                hidden_layers=hidden_layers_popa, w_pa=w_pa_popa, return_logits=False,
                criterion_po=DeepMaxEntLoss(), criterion_pa=BalancedBCELoss(), concat_sources=False,
            ),
        ),
        
        # ### ver with COVS
        # RunConfig(
        #     name="po_bbce_pa_bbce_wpocov",
        #     fn=run_one_split_popa,
        #     enabled=run_popa,
        #     kwargs=dict(
        #         batch_size=batch_size_popa, epochs=epochs_popa, lr=lr_popa,
        #         weight_decay=weight_decay_popa, hidden_dim=hidden_dim_popa,
        #         hidden_layers=hidden_layers_popa, w_pa=w_pa_popa, return_logits=False,
        #         criterion_po=BalancedBCELoss(), criterion_pa=BalancedBCELoss(), concat_sources=False,
        #         add_po_cov=True,
        #     ),
        # ),
        # RunConfig(
        #     name="po_dme_pa_bbce_wpocov",
        #     fn=run_one_split_popa,
        #     enabled=run_popa,
        #     kwargs=dict(
        #         batch_size=batch_size_popa, epochs=epochs_popa, lr=lr_popa,
        #         weight_decay=weight_decay_popa, hidden_dim=hidden_dim_popa,
        #         hidden_layers=hidden_layers_popa, w_pa=w_pa_popa, return_logits=False,
        #         criterion_po=DeepMaxEntLoss(), criterion_pa=BalancedBCELoss(), concat_sources=False,
        #         add_po_cov=True,
        #     ),
        # ),
        # RunConfig(
        #     name="po_dme_pa_dme_wpocov",
        #     fn=run_one_split_popa,
        #     enabled=run_popa,
        #     kwargs=dict(
        #         batch_size=batch_size_popa, epochs=epochs_popa, lr=lr_popa,
        #         weight_decay=weight_decay_popa, hidden_dim=hidden_dim_popa,
        #         hidden_layers=hidden_layers_popa, w_pa=w_pa_popa, return_logits=False,
        #         criterion_po=DeepMaxEntLoss(), criterion_pa=DeepMaxEntLoss(), concat_sources=False,
        #         add_po_cov=True,
        #     ),
        # ),
        
    ]

    # ── global (train-once) configs ────────────────────────────────────────────────
    global_configs = [
        GlobalRunConfig(
            name="po_dme",
            train_fn=train_po_for_splits,
            eval_fn=evaluate_model_on_all_pa_split_tests,
            enabled=run_po,
            train_kwargs=dict(
                batch_size=batch_size_po, epochs=epochs_po, lr=lr_po,
                weight_decay=weight_decay_po, hidden_dim=hidden_dim_po,
                hidden_layers=hidden_layers_po,
            ),
            eval_kwargs=dict(
                split_dir=split_dir,
                source_name="po",
                use_overlapping_species=use_overlapping_species,
            ),
        ),
        GlobalRunConfig(
            name="po_bbce",
            train_fn=train_po_for_splits,
            eval_fn=evaluate_model_on_all_pa_split_tests,
            enabled=run_po,
            train_kwargs=dict(
                batch_size=batch_size_po, epochs=epochs_po, lr=lr_po,
                weight_decay=weight_decay_po, hidden_dim=hidden_dim_po,
                hidden_layers=hidden_layers_po, criterion=BalancedBCELoss()
            ),
            eval_kwargs=dict(
                split_dir=split_dir,
                source_name="po",
                use_overlapping_species=use_overlapping_species,
            ),
        ),
    ]

    # ── per-split loop ─────────────────────────────────────────────────────────────
    active_configs = [c for c in configs if c.enabled]
    results: dict[str, list] = {c.name: [] for c in active_configs}

    for _, row in split_table.iterrows():
        for cfg in active_configs:
            results[cfg.name].append(
                cfg.fn(split_row=row, split_dir=split_dir, **shared, **cfg.kwargs)
            )

    # ── global loop ────────────────────────────────────────────────────────────────
    for cfg in global_configs:
        if not cfg.enabled:
            continue
        model, scaler = cfg.train_fn(**shared, **cfg.train_kwargs)
        summary, _ = cfg.eval_fn(
            model=model,
            scaler=scaler,
            split_table=split_table,
            data=data,
            output_dir=output_dir,
            project_name=project_name,
            **cfg.eval_kwargs,
        )
        results[cfg.name] = summary  # df directly, not a list

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
        ignore_index=True
    )
    common_summary.to_csv(output_dir / "summary_common.csv", index=False)
    print("\n=== common summary ===")
    print(common_summary.sort_values(["run", SORT_COL] if SORT_COL in shared_cols else ["run"]))

if __name__ == "__main__":

    # dataset name from command line if provided, else default to "GeoPlant"
    parser = argparse.ArgumentParser(description="Run split sweep for presence-absence models.")
    parser.add_argument("--dataset_name", type=str, default="GeoPlant")
    parser.add_argument("--split_type", type=str, default="geographical")
    parser.add_argument("--use_overlapping_species", action="store_true", help="If set, will filter out species not in the overlapping set for each split.")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    split_type = args.split_type
    use_overlapping_species = args.use_overlapping_species
    main(dataset_name=dataset_name, split_type=split_type, use_overlapping_species=use_overlapping_species)