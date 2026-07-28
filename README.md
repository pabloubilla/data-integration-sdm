# data-integration-sdm

This repository contains the code necessary to replicate the experiments for the article: "DATA INTEGRATION FOR DEEP SPECIES DISTRIBUTION MODELS UNDER SPATIAL AND ENVIRONMENTAL SEPARATION"

## Structure

- **`notebooks/`** — exploratory analysis (spatial analysis, data integration results, clustering, split visualizations, etc.). Not part of the main pipeline, used for inspection and figures.
- **`scripts/`** — main runnable files for experiments
- **`src/isdm/`** — core package: all shared modules (datasets, preprocessing, models, splits, training, evaluation, tuning).
- **`Rscripts/`**, **`src_legacy/`**, **`src_old/`** — older/alternative implementations, not part of the current pipeline.

- **`data/raw`** - the main data folder, it can be downloaded from [here](https://seafile.plantnet.org/d/59325675470447b38add/?p=%2F&mode=list). (TODO: explain which files are relevant and where should they be put)

## Setup

```bash
pip install -r requirements.txt
```

The `isdm` package (`src/isdm`) can be installed in editable mode via `pyproject.toml`:

```bash
pip install -e .
```

## Pipeline & main scripts

The general pipeline is: **preprocess data → generate splits → run experiment sweep → plot results**.

- `scripts/preprocess_geoplant.py` / `scripts/preprocess_geoplant_alphaearth.py` — preprocess the GeoPlant dataset (BioClim or AlphaEarth features). Key arg: `--vocab-mode` (e.g. `intersection_po_pa`).
- `scripts/generate_pa_splits_v2.py` — generates presence-absence splits.
  - `--dataset_name`: dataset to use (default `GeoPlant`)
  - `--split_type`: `geographical` or `environmental` (default `geographical`)
- `scripts/run_split_sweep.py` — runs the experiment sweep over a given split.
  - `--dataset_name`, `--split_type`: same as above
  - `--use_overlapping_species`: if set, restricts to species present in both PO and PA sources
- `scripts/plots/splits_boxplot.py` — generates result plots (e.g. `--metric avg_auc_site`, `--add_average`).

### `.sh` files

- **`scripts/run_full_experiment.sh`** — the main experiment runner. Sweeps over datasets (`GeoPlant`), split types (`environmental`, `geographical`) and overlap settings, running preprocessing → split generation → sweep → plotting for each combination. Written for an OAR cluster (`#OAR` directives at the top); adjust the `PYTHON` path; the OAR commands can be ignored as they are commented out.