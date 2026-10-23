# data-integration-sdm

Code to reproduce the experiments of the article:

> **Data Integration Improves Deep Species Distribution Models When Evaluated Far From Presence–Absence Data**

## Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies. All versions are pinned in `uv.lock`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv (one-time)
cd data-integration-sdm
uv sync --locked
```

This creates a local `.venv/` with Python 3.11, all dependencies, and the `isdm` package installed in editable mode. Prefix commands with `uv run` to run them inside the environment.

Experiment tracking with Weights & Biases is optional. To run without an account:

```bash
export WANDB_MODE=disabled
```

## Data

We use the GeoPlant dataset (GeoLifeCLEF 2024), available through [Kaggle](https://www.kaggle.com/competitions/geolifeclef-2024/data) or directly from [Seafile](https://seafile.plantnet.org/d/59325675470447b38add/?p=%2F&mode=list).

Download the following folders in full and place them in `data/raw/GeoPlant/`:

```
data/raw/GeoPlant/
├── PresenceAbsenceSurveys/
├── PresenceOnlyOccurrences/
└── values/
```

## Repository structure

- **`scripts/`**: everything needed to run the experiments.
  - `shell/`: orchestrator scripts that run the full pipeline.
  - `pipeline/`: individual pipeline steps, numbered in execution order.
  - `additional/`: post-processing steps needed by some figures.
  - `figures/`: scripts to produce the figures and tables of the article.
- **`src/isdm/`**: library with the shared code (datasets, models, losses, splits, training, evaluation).
- **`data/raw/`**: input data (not tracked by git; see [Data](#data)).
- **`outputs/`**: all results (splits, tuning, trained models, metrics, figures, tables) are written here.

## Running the experiments

### 1. Tuning region (Continental France)

```bash
bash scripts/shell/full_pipeline.sh
```

Runs the full pipeline for France over both split types (geographical and environmental): preprocessing, split generation, hyperparameter tuning, selection of the best configuration per loss, and the final sweep over all splits. Each stage can be switched on or off with the `RUN_*` flags at the top of the script.

### 2. Other regions (Denmark, BeNe, CEE)

```bash
bash scripts/shell/full_pipeline_extra_regions.sh
```

Same pipeline, but **without tuning**: it reuses the hyperparameters selected on France (`outputs/tune/GeoPlant/france/...`), so step 1 must be run first.

Both scripts were written for an OAR cluster with 2 GPUs (one pipeline per split type). The `#OAR` lines are ignored when running locally. With a single GPU, the second pipeline falls back to CPU.

### 3. Post-processing

```bash
uv run scripts/additional/add_distance_to_splits.py
uv run scripts/additional/aggregate_regions_output.py --use_overlapping_species
```

The first adds the train–test distance (km) to each split's results; the second combines all regions into a single result set (`agg_regions`). Run them in this order: the aggregated results must already include the distances.

### Pipeline steps

The steps in `scripts/pipeline/` can also be run individually with `uv run`:

1. `01_preprocess_geoplant.py`: preprocess the GeoPlant PO and PA data with BioClim covariates.
2. `02_generate_pa_splits.py`: generate the PA test and train (close, medium, far) splits.
3. `03_tune_split.py`: run the hyperparameter random search on the tuning split.
4. `04_aggregate_tune_split.py`: aggregate tuning results and select the best configuration per loss.
5. `05_run_split_sweep.py`: train and evaluate the selected configurations on all splits.

Run any script with `--help` to see all options.

### Figures and tables

Scripts in `scripts/figures/` produce the figures and tables of the article (written to `outputs/figures/` and `outputs/tables/`):

| Script | Output | Requires |
|---|---|---|
| `geoplant_map.py` | Fig. 1 | data only |
| `splits_maps.py` | Fig. 4, Figs S3–S6 | splits (steps 1–2) |
| `split_sweep_results.py` | Figs 5–6, Tables 2 and S2–S5 | steps 1–3 |
| `tune_results.py` | Fig. 7, Figs S1–S2 | tuning (step 1) |
| `distance_to_po_effect.py` | Fig. S7 | steps 1–2 |


### Using the benchmark with other methods

The splits generated in step 2 are saved to `outputs/splits/` and do not depend on any model, so they can be reused to evaluate other SDM approaches under the same close/medium/far scenarios. Step 5 (`05_run_split_sweep.py`) shows how a method is trained and evaluated across all splits; new models or losses can be added in `src/isdm/` (`models.py`, `losses.py`) and run through the same script with some modification.

We plan to provide a simpler, standalone script for evaluating new methods on this benchmark.