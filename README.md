# data-integration-sdm

Code to reproduce the experiments of the article:

> **Data Integration for Deep Species Distribution Models under Spatial and Environmental Separation**
> <Authors> (<Year>). <Venue>. <DOI / arXiv link>

## Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies. All package versions are pinned in `uv.lock`, so everyone gets the exact same environment.

1. Install uv (one-time):

```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository and create the environment:

```bash
   git clone <repo-url>
   cd data-integration-sdm
   uv sync --locked
```

This installs Python 3.11 (if needed), all dependencies, and the `isdm` package (`src/isdm`) in editable mode, into a local `.venv/`. You don't need to activate it: prefix commands with `uv run` and they run inside the environment.

```bash
uv run python -c "import isdm; print('ok')"   # quick check
```

## Data

Download the data from [here](https://seafile.plantnet.org/d/59325675470447b38add/?p=%2F&mode=list) and place it in `data/raw/`.

TODO: list the required files and the expected layout, e.g.

```
data/raw/
├── <file_1>
└── <file_2>
```

## Pipeline

The general pipeline is: **preprocess data → generate splits → run experiment sweep → plot results**.

### Run everything

```bash
bash scripts/run_full_experiment.sh
```

This sweeps over datasets (`GeoPlant`), split types (`environmental`, `geographical`) and overlap settings, running preprocessing → split generation → sweep → plotting for each combination. It was written for an OAR cluster; the `#OAR` directives are commented out and can be ignored when running locally.

### Individual steps

1. **Preprocess** the GeoPlant dataset (BioClim or AlphaEarth features):

```bash
   uv run scripts/pipeline/01_preprocess_geoplant.py --vocab-mode intersection_po_pa
   uv run scripts/preprocess_geoplant_alphaearth.py --vocab-mode intersection_po_pa
```

2. **Generate presence-absence splits**:

```bash
   uv run scripts/pipeline/02_generate_pa_splits.py --dataset_name GeoPlant --split_type geographical
```
   - `--dataset_name`: dataset to use (default `GeoPlant`)
   - `--split_type`: `geographical` or `environmental` (default `geographical`)

3. **Run the experiment sweep** over a split:

```bash
   uv run scripts/run_split_sweep.py --dataset_name GeoPlant --split_type geographical
```
   - `--use_overlapping_species`: restrict to species present in both PO and PA sources

4. **Plot results**:

```bash
   uv run scripts/plots/splits_boxplot.py --metric avg_auc_site --add_average
```

TODO: map outputs to the paper (e.g. "Figure 3 ← `splits_boxplot.py --metric ...`") and note rough runtimes and hardware (GPU used, time per sweep).

## Repository structure

- **`src/isdm/`**: core package with all shared modules (datasets, preprocessing, models, splits, training, evaluation, tuning).
- **`scripts/`**: main runnable files for the experiments.
- **`notebooks/`**: exploratory analysis (spatial analysis, data integration results, clustering, split visualizations). Not part of the main pipeline; used for inspection and figures.
- **`Rscripts/`**, **`src_legacy/`**, **`src_old/`**: older or alternative implementations, not part of the current pipeline.
- **`data/raw/`**: input data (not tracked by git; see [Data](#data)).

## Citation

```bibtex
@article{<key>,
  title   = {Data Integration for Deep Species Distribution Models under Spatial and Environmental Separation},
  author  = {<Authors>},
  journal = {<Venue>},
  year    = {<Year>}
}
```