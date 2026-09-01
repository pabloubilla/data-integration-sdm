from pathlib import Path
from functools import partial
import json
import pickle

import numpy as np
import pandas as pd

import argparse

from isdm.losses import BalancedBCELoss, BernoulliFromLogRateLoss, DeepMaxEntLoss, IntegratedLoss, LOSS_REGISTRY

from isdm.load_data import load_geoplant_processed

from isdm.splits import load_split_specs, load_split, train_po_for_splits, evaluate_model_on_all_pa_split_tests, set_all_seeds, run_one_split_po_or_pa, run_one_split_popa
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


def load_best_params(
    tune_dir: Path,
    method: str,                      # "pa", "po", or "popa"
    loss_name: str | None = None,     # required for pa/po
    loss_po_name: str | None = None,  # required for popa
    loss_pa_name: str | None = None,  # required for popa
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
    """All loss names present in a tuned best_{method}_avg_over_distance.csv."""
    df = pd.read_csv(tune_dir / f"best_{method}_avg_over_distance.csv")
    return sorted(df["param_loss_name"].unique().tolist())


def get_tuned_loss_combos(tune_dir: Path) -> list[tuple[str, str]]:
    """All (loss_po_name, loss_pa_name) combos present in a tuned best_popa_avg_over_distance.csv."""
    df = pd.read_csv(tune_dir / "best_popa_avg_over_distance.csv")
    combos = df[["param_loss_po_name", "param_loss_pa_name"]].drop_duplicates()
    return list(combos.itertuples(index=False, name=None))

def main(dataset_name: str = "GeoPlant", split_type: str = "geographical", region: str = "france", use_overlapping_species: bool = True):
    seed = 42
    set_all_seeds(seed)

    run_pa = True
    run_popa = True
    run_po = True

    overlap_name = "intersect" if use_overlapping_species else "union"

    sweep_subsample = None  # set to an integer to subsample splits for quick testing

    data_path = f"data/processed/{dataset_name}/{region}"
    split_dir = Path(f"outputs/splits/{dataset_name}/{region}_bands/{split_type}")
    output_dir = Path(f"outputs/split_sweep/{dataset_name}/{region}_bands/{split_type}/{overlap_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_dir_logits = output_dir / "logits"
    output_dir_logits.mkdir(parents=True, exist_ok=True)

    project_name = "sdm-integration"

    tune_dir = Path(f"outputs/tune/{dataset_name}/{region}/{split_type}/test_0")  # the validation-tuning run; reused across all real splits



    data = load_geoplant_processed(data_path)
    split_table = load_split_specs(split_dir)

    # remove val splits from the sweep table, since we only want to run on test splits
    # done using option column and removing those containing "val" in their name
    split_table = split_table[~split_table["option"].str.contains("val", case=False, na=False)]

    # ── shared kwargs ──────────────────────────────────────────────────────────────
    shared = dict(
        data=data,
        output_dir=output_dir,
        project_name=project_name,
        seed=seed,
        use_overlapping_species=use_overlapping_species,
    )


    # ── per-split configs, one per tuned loss choice ────────────────────────────────
    pa_loss_names = get_tuned_loss_names(tune_dir, method="pa")
    popa_loss_combos = get_tuned_loss_combos(tune_dir)

    configs = []

    for loss_name in pa_loss_names:
        p = load_best_params(tune_dir, method="pa", loss_name=loss_name)
        configs.append(RunConfig(
            name=f"pa_{loss_name}",
            fn=run_one_split_po_or_pa,
            enabled=run_pa,
            kwargs=dict(
                lr=p["lr"], weight_decay=p["weight_decay"], hidden_dim=p["hidden_dim"],
                batch_size=p["batch_size"], hidden_layers=p["hidden_layers"], epochs=p["epochs"],
                criterion=LOSS_REGISTRY[loss_name](), source="pa",
            ),
        ))

    for loss_po_name, loss_pa_name in popa_loss_combos:
        p = load_best_params(tune_dir, method="popa", loss_po_name=loss_po_name, loss_pa_name=loss_pa_name)
        configs.append(RunConfig(
            name=f"po_{loss_po_name}_pa_{loss_pa_name}",
            fn=run_one_split_popa,
            enabled=run_popa,
            kwargs=dict(
                lr=p["lr"], weight_decay=p["weight_decay"], hidden_dim=p["hidden_dim"],
                batch_size=p["batch_size"], hidden_layers=p["hidden_layers"], epochs=p["epochs"],
                w_po=p["w_po"], w_pa=p["w_pa"],
                criterion_po=LOSS_REGISTRY[loss_po_name](), criterion_pa=LOSS_REGISTRY[loss_pa_name](),
                return_logits=False, concat_sources=False,
            ),
        ))

    # ── global (train-once) configs, one per tuned PO loss ──────────────────────────
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
        

    # ── global (train-once) configs, one per tuned PO loss ──────────────────────────
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
    parser.add_argument("--region", type=str, default="france", help="Region to run the split sweep on (default: france).")
    parser.add_argument("--use_overlapping_species", action="store_true", help="If set, will filter out species not in the overlapping set for each split.")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    split_type = args.split_type
    region = args.region
    use_overlapping_species = args.use_overlapping_species
    main(dataset_name=dataset_name, split_type=split_type, region=region, use_overlapping_species=use_overlapping_species)