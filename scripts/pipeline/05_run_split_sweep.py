
"""
Runs the FINAL evaluation sweep — using hyperparameters already selected
by tune_split.py / aggregate_tune_results.py — across every real test
split (excludes the *_val splits used for tuning).

Parallelizable via MAX_WORKERS (env var), and resumable: each (config,
split) trial is saved as its own JSON under output_dir/trials/<config>/.
Each config also gets its own output_dir subfolder, so different loss
combos never overwrite each other's saved models.

Usage:
    MAX_WORKERS=4 python scripts/run_split_sweep.py --dataset_name GeoPlant
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from isdm.losses import LOSS_REGISTRY
from isdm.load_data import load_geoplant_processed
from isdm.splits import (
    load_split_specs, train_po_for_splits, evaluate_model_on_all_pa_split_tests,
    set_all_seeds, run_one_split_po_or_pa_tunable, run_one_split_popa_tunable,
)
from isdm.tune import run_grid_search as run_batch_over_splits

RUN_PA = True
RUN_POPA = True
RUN_PO = True

def load_best_params(tune_dir, method, loss_name=None, loss_po_name=None, loss_pa_name=None):
    df = pd.read_csv(tune_dir / f"best_{method}_avg_over_distance.csv")
    mask = (df["param_loss_po_name"] == loss_po_name) & (df["param_loss_pa_name"] == loss_pa_name) \
        if method == "popa" else df["param_loss_name"] == loss_name
    best = df[mask].iloc[0]

    params = {
        "lr": float(best["param_lr"]), "weight_decay": float(best["param_weight_decay"]),
        "hidden_dim": int(best["param_hidden_dim"]), "batch_size": int(best["param_batch_size"]),
        "hidden_layers": int(best["param_hidden_layers"]), "epochs": int(round(best["best_epoch"])),
    }
    if method == "popa":
        params["w_pa"], params["w_po"] = float(best["param_w_pa"]), float(best["param_w_po"])
    return params


def tuned_loss_names(tune_dir, method):
    return sorted(pd.read_csv(tune_dir / f"best_{method}_avg_over_distance.csv")["param_loss_name"].unique())


def tuned_loss_combos(tune_dir):
    df = pd.read_csv(tune_dir / "best_popa_avg_over_distance.csv")
    return list(df[["param_loss_po_name", "param_loss_pa_name"]].drop_duplicates().itertuples(index=False, name=None))


def main(dataset_name="GeoPlant", split_type="geographical", region="france",
         use_overlapping_species=True, tune_dir=None):
    seed = 42
    set_all_seeds(seed)
    max_workers = int(os.environ.get("MAX_WORKERS", 1))
    print(f"MAX_WORKERS={max_workers}")

    overlap_name = "intersect" if use_overlapping_species else "union"
    split_dir = Path(f"outputs/splits/{dataset_name}/{region}_bands/{split_type}")
    output_dir = Path(f"outputs/split_sweep/{dataset_name}/{region}_bands/{split_type}/{overlap_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tune_dir = Path(tune_dir) if tune_dir else Path(f"outputs/tune/{dataset_name}/{region}/{split_type}/test_0")

    data = load_geoplant_processed(f"data/processed/{dataset_name}/{region}")
    split_table = load_split_specs(split_dir)
    split_table = split_table[~split_table["option"].str.contains("val", case=False, na=False)]
    split_rows = [row for _, row in split_table.iterrows()]

    shared = dict(data=data, project_name="sdm-integration", seed=seed,
                  use_overlapping_species=use_overlapping_species, split_dir=split_dir)
    # output_dir deliberately NOT in `shared` — each config below gets its own subfolder

    summaries = {}

    # ── PA + POPA: per-split, parallel, resumable ───────────────────────
    per_split = []
    if RUN_PA:
        per_split += [(f"pa_{ln}", run_one_split_po_or_pa_tunable,
                       {**load_best_params(tune_dir, "pa", loss_name=ln), "loss_name": ln, "source": "pa"})
                      for ln in tuned_loss_names(tune_dir, "pa")]
    if RUN_POPA:
        per_split += [(f"po_{po}_pa_{pa}", run_one_split_popa_tunable,
                       {**load_best_params(tune_dir, "popa", loss_po_name=po, loss_pa_name=pa),
                    "loss_po_name": po, "loss_pa_name": pa, "return_logits": False, "concat_sources": False})
                  for po, pa in tuned_loss_combos(tune_dir)]

    for name, fn, kwargs in per_split:
        print(f"\n=== {name}: {len(split_rows)} splits, max_workers={max_workers} ===")
        results = run_batch_over_splits(
            combos=[kwargs],
            splits=split_rows, run_fn=fn,
            fixed_kwargs={**shared, "output_dir": output_dir / name},
            param_keys=list(kwargs.keys()),
            results_dir=output_dir / "trials" / name,
            combo_label_fn=lambda c, n=name: n,
            max_workers=max_workers,
        )
        summaries[name] = pd.DataFrame(results)

    # ── PO: train once, evaluate on all splits — not parallelized ──────
    if RUN_PO:
        for loss_name in tuned_loss_names(tune_dir, "po"):
            name = f"po_{loss_name}"
            print(f"\n=== {name} (train-once + evaluate-all) ===")
            p = load_best_params(tune_dir, "po", loss_name=loss_name)
            model, scaler = train_po_for_splits(**p,
                                                data=data, 
                                                project_name="sdm-integration", 
                                                seed=seed,
                                                use_overlapping_species=use_overlapping_species,
                                                criterion=LOSS_REGISTRY[loss_name](),
                                                output_dir=output_dir / name)
            summary, _ = evaluate_model_on_all_pa_split_tests(
                model=model, scaler=scaler, split_table=split_table, data=data,
                output_dir=output_dir / name, project_name="sdm-integration", split_dir=split_dir,
                source_name="po", use_overlapping_species=use_overlapping_species,
            )
            summaries[name] = summary if isinstance(summary, pd.DataFrame) else pd.DataFrame(summary)

    # ── save + summarize ─────────────────────────────────────────────────
    for name, df in summaries.items():
        df.to_csv(output_dir / f"summary_{name}.csv", index=False)
        print(f"\n=== {name} ===")
        print(df.sort_values("distance") if "distance" in df.columns else df)

    shared_cols = sorted(set.intersection(*(set(df.columns) for df in summaries.values())))
    common = pd.concat([df[shared_cols].assign(run=n) for n, df in summaries.items()], ignore_index=True)
    common.to_csv(output_dir / "summary_common.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="GeoPlant")
    p.add_argument("--split_type", default="geographical")
    p.add_argument("--region", default="france")
    p.add_argument("--use_overlapping_species", action="store_true")
    p.add_argument("--tune_dir", default=None)
    args = p.parse_args()

    # for use_overlapping_species we will keep as True always as we won't use the union case anymore
    main(dataset_name=args.dataset_name, split_type=args.split_type, region=args.region,
         use_overlapping_species=True, tune_dir=args.tune_dir)
