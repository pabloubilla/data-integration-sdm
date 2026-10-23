# data-integration-sdm

Code to reproduce the experiments of the article:

> **Data Integration Improves Deep Species Distribution Models When Evaluated Far From Presence–Absence Data**

## Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies. All versions are pinned in `uv.lock`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv (one-time)
git clone <repo-url>
cd data-integration-sdm
uv sync --locked
```

This creates a local `.venv/` with Python 3.11, all dependencies, and the `isdm` package installed in editable mode. Prefix commands with `uv run` to run them inside the environment.

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
  - `run_full_experiment.sh`: orchestrator that runs the whole pipeline.
  - `pipeline/`: individual pipeline steps, numbered in execution order.
  - `figures/`: scripts to produce the figures of the article.
- **`src/isdm/`**: library with the shared code (datasets, models, losses, splits, training, evaluation).
- **`data/raw/`**: input data (not tracked by git; see [Data](#data)).
- **`outputs/`**: all results (splits, trained models, metrics, figures) are written here.

## Running the experiments

A single shell script orchestrates the whole pipeline over both split types (geographical and environmental):

```bash
bash scripts/run_full_experiment.sh
```

The script was written for an OAR cluster; the `#OAR` directives are commented out and can be ignored when running locally.

### Pipeline steps

The steps in `scripts/pipeline/` can also be run individually with `uv run`:

1. `01_preprocess_geoplant.py`: preprocess the GeoPlant PO and PA data with BioClim covariates.
2. `02_generate_pa_splits.py`: generate the PA test and train (close, medium, far) splits.
3. `03_tune_split.py`: run the hyperparameter random search on the tuning split.
4. `04_aggregate_tune_split.py`: aggregate tuning results and select the best configuration per loss.
5. `05_run_split_sweep.py`: train and evaluate the selected configurations on all splits.

Use `--split_type geographical` or `--split_type environmental` where applicable. Run any script with `--help` to see all options.

### Figures

Scripts in `scripts/figures/` produce the figures of the article:

| Script | Output |
|---|---|
| `geoplant_map.py` | Fig. 1: GeoPlant data and distance to nearest PA plot |
| `splits_maps.py` | Fig. 4 and Figs. S1–S4: split maps per region |
| `split_sweep_results.py` | Fig. 5 and Tables 2, S2–S5: performance per loss and distance |
| `distance_to_po_effect.py` | Fig. 6: integration gain vs. distance per region |
| `tune_results.py` | Fig. 7: sensitivity to the PO weight |