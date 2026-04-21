# minimal_popa_alphaearth_pipeline.py
#
# A configurable, end-to-end pipeline to compare:
#   - COVS only
#   - AlphaEarth embeddings at points (64D)
#   - AlphaEarth neighborhood stats (e.g., mean/std over buffers) (64 * stats * radii)
#   - CONCATENATION: COVS + (point and/or neighborhood embeddings)
#
# And to run tasks:
#   - PO-only
#   - PA-only
#   - POPA (PO + PA_train)
#
# Notes:
# - Earth Engine "features" = rows (one per point/region); "bands" = columns.
# - AlphaEarth embedding has fixed 64 bands: A00..A63.
# - This script adds *derived* features via neighborhood reducers / multi-radius.
#
# Requirements:
# - ee.Initialize(project=EE_PROJECT) must work for your account/project.
# - Your load_po_pa_nceas / split_pa_train_test_spatially / scale_features / DeepMaxEntModel / deepmaxent_loss exist.

import os
import random
from time import time
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import ee
from sklearn.preprocessing import StandardScaler

from src.load_data import load_po_pa_nceas
from src.utils import split_pa_train_test_spatially, scale_features
from src.models import DeepMaxEntModel, DeepMaxEntLoss


# ============================================================
# CONFIG
# ============================================================
REGIONS = ["SA"]
GROUPS_BY_REGION = {
    "AWT": ["_plant", "_bird"],
    "CAN": [""],
    "NSW": ["_bat", "_bird", "_plant", "_reptile"],
    "SA":  [""],
    "SWI": [""],
    "NZ":  [""],
}

SEED = 42
TEST_PA_FRACTION = 0.5
K_SPLIT = 100

# ---- Which tasks to run
RUN_TASKS = {
    "PO": True,
    "PA": True,
    "POPA": True,
}

# ---- Which feature sets to run
# You can run multiple and they’ll all be logged into the final summary.
RUN_FEATURESETS = {
    "COVS": True,
    "AEF_POINT": True,
    "AEF_NEIGH": True,
    "CONCAT_COVS_AEFPOINT": True,
    "CONCAT_COVS_AEFNEIGH": True,
}

# ---- EE / AlphaEarth dataset
EE_PROJECT = "alpha-earth-test-483513"
AEF_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
AEF_YEAR = 2019
AEF_SCALE_M = 10  # dataset is 10m

# ---- NaN policy for EE sampling
# "raise" -> crash if missing values
# "fill0" -> fill missing with 0.0
# "drop"  -> drop rows with any missing features (and align labels)
NAN_POLICY = "drop"

# ---- Neighborhood configuration
# If AEF_NEIGH is enabled, we compute stats over buffers around each point.
# You can stack multi-radius: e.g. [100, 250, 1000] meters.
NEIGHBORHOOD = {
    "enabled": True,
    "radii_m": [250],          # e.g. [100, 250, 1000]
    "stats": ["mean", "std"],  # any subset of ["mean", "std", "min", "max"]
}

# ---- Model configs per feature set (you can tune independently)
MODEL_CFG = {
    "COVS": dict(hidden_size=250, hidden_layers=2, lr=1e-4, epochs=200, batch_size=256),
    "AEF_POINT": dict(hidden_size=512, hidden_layers=2, lr=2e-4, epochs=100, batch_size=512),
    "AEF_NEIGH": dict(hidden_size=768, hidden_layers=2, lr=2e-4, epochs=100, batch_size=512),
    "CONCAT_COVS_AEFPOINT": dict(hidden_size=768, hidden_layers=2, lr=2e-4, epochs=500, batch_size=512),
    "CONCAT_COVS_AEFNEIGH": dict(hidden_size=1024, hidden_layers=2, lr=2e-4, epochs=700, batch_size=512),
}

# ---- Loss per task (keep your logic)
CRITERION = {
    "PO": "deepmaxent",
    "PA": "bce",
    "POPA": "bce",
}

# ---- Output
OUTPUT_ROOT = os.path.join("output", "alphaearth_pipeline")


# ============================================================
# UTILITIES
# ============================================================
def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class XYDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self) -> int:
        return self.X.shape[0]
    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx], torch.tensor(idx, dtype=torch.long)

def make_model(input_dim: int, output_dim: int, hidden_size: int, hidden_layers: int) -> nn.Module:
    return DeepMaxEntModel(input_size=input_dim, hidden_size=hidden_size, output_size=output_dim, hidden_nbr=hidden_layers)

def train_model(model: nn.Module, train_loader: DataLoader, criterion: str, epochs: int, lr: float, dev: Optional[torch.device] = None) -> None:
    dev = dev or device()
    model.to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-4)

    if criterion == "bce":
        loss_f = torch.nn.BCEWithLogitsLoss()
    elif criterion == "deepmaxent":
        loss_f = DeepMaxEntLoss()
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    model.train()
    for _ in range(epochs):
        for xb, yb, _ in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            optim.zero_grad()
            out = model(xb)
            loss = loss_f(out, yb)
            loss.backward()
            optim.step()

@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, dev: Optional[torch.device] = None) -> np.ndarray:
    dev = dev or device()
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    out = model(Xt)
    return out.detach().cpu().numpy()

def per_species_auc(y_true: pd.DataFrame, y_score: np.ndarray, species: List[str]) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score
    out = {}
    for i, sp in enumerate(species):
        try:
            out[sp] = roc_auc_score(y_true[sp].values, y_score[:, i])
        except ValueError:
            out[sp] = np.nan
    return out

def avg_auc_ignore_nan(aucs: Dict[str, float]) -> float:
    vals = np.array(list(aucs.values()), dtype=float)
    return float(np.nanmean(vals))

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# ============================================================
# FEATURE BUILDERS
# ============================================================
def _aef_band_names() -> List[str]:
    return [f"A{i:02d}" for i in range(64)]

def _build_aef_image(year: int, geometry: ee.Geometry) -> ee.Image:
    """Mosaic all tiles intersecting geometry for the selected year; keep only A00..A63."""
    bands = _aef_band_names()
    col = (ee.ImageCollection(AEF_COLLECTION)
           .filterDate(f"{year}-01-01", f"{year+1}-01-01")
           .filterBounds(geometry))
    # mosaic gives a single image covering the region where tiles overlap
    img = col.mosaic().select(bands)
    return img

def _df_points_to_fc(df: pd.DataFrame, id_col: str = "__rid", geom_buffer_m: Optional[int] = None) -> ee.FeatureCollection:
    """
    df must contain columns x (lon), y (lat). We create a stable id __rid for join.
    If geom_buffer_m is set, geometry becomes a buffer polygon (for reduceRegions).
    """
    df2 = df.copy()
    feats = []
    for _, r in df2.iterrows():
        lon = float(r["x"]); lat = float(r["y"])
        geom = ee.Geometry.Point([lon, lat])
        if geom_buffer_m is not None:
            geom = geom.buffer(int(geom_buffer_m))
        feats.append(ee.Feature(geom, {id_col: int(r[id_col])}))
    return ee.FeatureCollection(feats)

def _apply_nan_policy(X: np.ndarray, nan_policy: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X2: possibly modified X
      keep_mask: boolean mask of rows kept (all True unless nan_policy=='drop')
    """
    nan_mask = np.isnan(X).any(axis=1)
    if not nan_mask.any():
        return X, np.ones(len(X), dtype=bool)

    if nan_policy == "raise":
        idx = np.where(nan_mask)[0][:10]
        raise ValueError(f"Found NaNs in features at rows {idx.tolist()} (showing up to 10).")
    elif nan_policy == "fill0":
        X2 = X.copy()
        X2[nan_mask] = np.nan_to_num(X2[nan_mask], nan=0.0)
        return X2, np.ones(len(X2), dtype=bool)
    elif nan_policy == "drop":
        keep = ~nan_mask
        return X[keep], keep
    else:
        raise ValueError(f"Unknown nan_policy: {nan_policy}")

def build_covs_features_triplet(
    X_po: pd.DataFrame,
    X_pa_tr: pd.DataFrame,
    X_pa_te: pd.DataFrame,
    covs: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale covariates (fit on PO + PA_train to keep POPA consistent),
    then return numpy arrays for PO, PA_train, PA_test.
    """
    # fit scaler on PO + PA train
    scaler = StandardScaler().fit(pd.concat([X_po[covs], X_pa_tr[covs]], axis=0).values)

    def _transform(df: pd.DataFrame) -> np.ndarray:
        return scaler.transform(df[covs].values).astype(np.float32)

    return _transform(X_po), _transform(X_pa_tr), _transform(X_pa_te)

def build_aef_point_features_triplet(
    X_po: pd.DataFrame,
    X_pa_tr: pd.DataFrame,
    X_pa_te: pd.DataFrame,
    year: int,
    nan_policy: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample AEF at point for PO, PA_train, PA_test.
    Returns:
      Xpo, Xpa_tr, Xpa_te, feat_names,
      keep masks for each split (all True unless nan_policy='drop')
    """
    bands = _aef_band_names()

    # Create stable row ids for each split
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        d = df.reset_index(drop=False).rename(columns={"index": "__orig_index"}).copy()
        d["__rid"] = np.arange(len(d), dtype=int)
        return d

    po = _prep(X_po)
    pa_tr = _prep(X_pa_tr)
    pa_te = _prep(X_pa_te)

    # Build geometry union to fetch correct mosaic (important!)
    geom = ee.Geometry.MultiPoint(
        po[["x","y"]].values.tolist() +
        pa_tr[["x","y"]].values.tolist() +
        pa_te[["x","y"]].values.tolist()
    ).bounds()

    img = _build_aef_image(year=year, geometry=geom)

    def _sample(dfprep: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        fc = _df_points_to_fc(dfprep, id_col="__rid", geom_buffer_m=None)
        samp = img.sampleRegions(collection=fc, properties=["__rid"], scale=AEF_SCALE_M, geometries=False)
        info = samp.getInfo()
        feats = info.get("features", [])
        # Allocate and fill by __rid
        X = np.full((len(dfprep), len(bands)), np.nan, dtype=np.float32)
        for f in feats:
            props = f.get("properties", {})
            rid = int(props["__rid"])
            # guard masked pixels / missing props
            row = [props.get(b, np.nan) for b in bands]
            X[rid, :] = np.array(row, dtype=np.float32)
        X2, keep = _apply_nan_policy(X, nan_policy)
        return X2, keep

    Xpo, keep_po = _sample(po)
    Xtr, keep_tr = _sample(pa_tr)
    Xte, keep_te = _sample(pa_te)

    # Standardize features using PO + PA_train (after dropping, if any)
    scaler = StandardScaler().fit(np.vstack([Xpo, Xtr]))
    Xpo = scaler.transform(Xpo).astype(np.float32)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)

    return Xpo, Xtr, Xte, bands, keep_po, keep_tr, keep_te

def build_aef_neigh_features_triplet(
    X_po: pd.DataFrame,
    X_pa_tr: pd.DataFrame,
    X_pa_te: pd.DataFrame,
    year: int,
    radii_m: List[int],
    stats: List[str],
    nan_policy: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Neighborhood features: reduceRegions over buffered geometries for each radius.
    For each band and stat, returns columns like:
      A00_mean_r250, A00_std_r250, ...
    Supports multi-radius.
    """
    bands = _aef_band_names()
    stats = [s.lower() for s in stats]
    allowed = {"mean", "std", "min", "max"}
    if any(s not in allowed for s in stats):
        raise ValueError(f"stats must be subset of {sorted(allowed)}")

    # stable ids
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        d = df.reset_index(drop=False).rename(columns={"index": "__orig_index"}).copy()
        d["__rid"] = np.arange(len(d), dtype=int)
        return d

    po = _prep(X_po)
    pa_tr = _prep(X_pa_tr)
    pa_te = _prep(X_pa_te)

    geom = ee.Geometry.MultiPoint(
        po[["x","y"]].values.tolist() +
        pa_tr[["x","y"]].values.tolist() +
        pa_te[["x","y"]].values.tolist()
    ).bounds()

    img = _build_aef_image(year=year, geometry=geom) 

    # Build combined output columns
    feat_names: List[str] = []
    for r in radii_m:
        for b in bands:
            if "mean" in stats: feat_names.append(f"{b}_mean_r{r}")
            if "std"  in stats: feat_names.append(f"{b}_std_r{r}")
            if "min"  in stats: feat_names.append(f"{b}_min_r{r}")
            if "max"  in stats: feat_names.append(f"{b}_max_r{r}")

    def _reducer_for_stats() -> ee.Reducer:
        red = None
        # Build one reducer that outputs band_mean, band_stdDev, band_min, band_max
        if "mean" in stats:
            red = ee.Reducer.mean()
        if "std" in stats:
            red = ee.Reducer.stdDev() if red is None else red.combine(ee.Reducer.stdDev(), sharedInputs=True)
        if "min" in stats:
            red = ee.Reducer.min() if red is None else red.combine(ee.Reducer.min(), sharedInputs=True)
        if "max" in stats:
            red = ee.Reducer.max() if red is None else red.combine(ee.Reducer.max(), sharedInputs=True)
        return red

    reducer = _reducer_for_stats()

    def _reduce_one_radius(dfprep: pd.DataFrame, radius: int) -> np.ndarray:
        fc = _df_points_to_fc(dfprep, id_col="__rid", geom_buffer_m=int(radius))
        # reduceRegions returns per-feature properties: e.g. A00_mean, A00_stdDev, ...
        rr = img.reduceRegions(collection=fc, reducer=reducer, scale=AEF_SCALE_M)
        info = rr.getInfo()
        feats = info.get("features", [])

        # Map stat name to EE suffix
        # mean -> "_mean", std -> "_stdDev", min -> "_min", max -> "_max"
        suffix = {"mean": "mean", "std": "stdDev", "min": "min", "max": "max"}

        X = np.full((len(dfprep), len(bands) * len(stats)), np.nan, dtype=np.float32)

        for f in feats:
            props = f.get("properties", {})
            rid = int(props["__rid"])
            row_vals = []
            for b in bands:
                for s in stats:
                    key = f"{b}_{suffix[s]}"
                    row_vals.append(props.get(key, np.nan))
            X[rid, :] = np.array(row_vals, dtype=np.float32)

        return X

    def _sample(dfprep: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        blocks = []
        for r in radii_m:
            blocks.append(_reduce_one_radius(dfprep, r))
        X = np.concatenate(blocks, axis=1)  # stack radii
        X2, keep = _apply_nan_policy(X, nan_policy)
        return X2, keep

    Xpo, keep_po = _sample(po)
    Xtr, keep_tr = _sample(pa_tr)
    Xte, keep_te = _sample(pa_te)

    # Standardize using PO + PA_train
    scaler = StandardScaler().fit(np.vstack([Xpo, Xtr]))
    Xpo = scaler.transform(Xpo).astype(np.float32)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)

    # Expand feat_names for multi-radius ordering:
    # Our construction was [r][band][stat], which matches the blocks stacking.
    # But inside _reduce_one_radius we did [band][stat], and we stacked radii in order. So:
    # feat_names already matches.
    return Xpo, Xtr, Xte, feat_names, keep_po, keep_tr, keep_te


# ============================================================
# TRAIN/EVAL for tasks
# ============================================================
def fit_eval_one(
    *,
    task: str,               # "PO" | "PA" | "POPA"
    feature_tag: str,        # e.g. "COVS" | "AEF_POINT"
    X_train: np.ndarray,
    Y_train: pd.DataFrame,
    X_test: np.ndarray,
    Y_test: pd.DataFrame,
    species: List[str],
    out_dir: str,
):
    cfg = MODEL_CFG[feature_tag]
    crit = CRITERION[task]

    model = make_model(
        input_dim=X_train.shape[1],
        output_dim=len(species),
        hidden_size=cfg["hidden_size"],
        hidden_layers=cfg["hidden_layers"],
    )

    bs = min(cfg["batch_size"], max(1, len(X_train)))
    loader = DataLoader(
        XYDataset(X_train, Y_train[species].values.astype(np.float32)),
        batch_size=bs,
        shuffle=True,
        drop_last=False,
    )

    train_model(model, loader, criterion=crit, epochs=cfg["epochs"], lr=cfg["lr"], dev=device())
    scores = predict(model, X_test, dev=device())

    aucs = per_species_auc(Y_test[species], scores, species)
    avg_auc = avg_auc_ignore_nan(aucs)

    ensure_dir(out_dir)
    model_path = os.path.join(out_dir, f"{feature_tag}_{task}.pt")
    torch.save(model, model_path)

    return dict(avg_auc=avg_auc, model_path=model_path)


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time()
    set_all_seeds(SEED)

    ensure_dir(OUTPUT_ROOT)
    ee.Initialize(project=EE_PROJECT)

    summary_rows: List[Dict[str, Any]] = []

    for region in REGIONS:
        for group in GROUPS_BY_REGION[region]:
            print(f"\n=== REGION: {region}, GROUP: {group or '(all)'} ===")

            # Load data (must include columns x=lon, y=lat; and index 'siteid')
            X_po, Y_po, X_pa, Y_pa, species, covs = load_po_pa_nceas(
                region=region,
                group_filter=group,
                add_po_var=False,
                keep_xy=True,
                index_col=["siteid"],
            )

            # Spatial split PA -> train/test
            X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te = split_pa_train_test_spatially(
                X_pa, Y_pa, test_frac=TEST_PA_FRACTION, seed=SEED, K=K_SPLIT
            )
            print(f"PA split → train: {len(X_pa_tr)}, test: {len(X_pa_te)}")

            exp_dir = os.path.join(OUTPUT_ROOT, f"{region}{group}")
            ensure_dir(exp_dir)

            # Define covariates baseline (drop x,y,PO)
            env_covs = [c for c in covs if c not in ["x", "y", "PO"]]

            # Prepare a dict to collect results for this region/group
            row: Dict[str, Any] = {
                "region": region,
                "group": group or "(all)",
                "pa_test_frac": TEST_PA_FRACTION,
                "aef_year": AEF_YEAR,
                "nan_policy": NAN_POLICY,
            }

            # ------------------------------------------------------------
            # Build feature matrices for each feature set requested
            # We'll compute and then run requested tasks.
            # ------------------------------------------------------------

            features: Dict[str, Dict[str, Any]] = {}

            # ---- COVS
            if RUN_FEATURESETS.get("COVS", False):
                Xpo_cov, Xtr_cov, Xte_cov = build_covs_features_triplet(X_po, X_pa_tr, X_pa_te, env_covs)
                features["COVS"] = dict(
                    Xpo=Xpo_cov, Xtr=Xtr_cov, Xte=Xte_cov,
                    Ypo=Y_po, Ytr=Y_pa_tr, Yte=Y_pa_te,
                )

            # ---- AEF point (64D)
            if RUN_FEATURESETS.get("AEF_POINT", False):
                Xpo, Xtr, Xte, fnames, keep_po, keep_tr, keep_te = build_aef_point_features_triplet(
                    X_po, X_pa_tr, X_pa_te, year=AEF_YEAR, nan_policy=NAN_POLICY
                )
                # align labels if dropping
                Ypo = Y_po.iloc[keep_po] if NAN_POLICY == "drop" else Y_po
                Ytr = Y_pa_tr.iloc[keep_tr] if NAN_POLICY == "drop" else Y_pa_tr
                Yte = Y_pa_te.iloc[keep_te] if NAN_POLICY == "drop" else Y_pa_te
                features["AEF_POINT"] = dict(Xpo=Xpo, Xtr=Xtr, Xte=Xte, Ypo=Ypo, Ytr=Ytr, Yte=Yte)

            # ---- AEF neighborhood (multi-radius * stats)
            if RUN_FEATURESETS.get("AEF_NEIGH", False) and NEIGHBORHOOD.get("enabled", True):
                Xpo, Xtr, Xte, fnames, keep_po, keep_tr, keep_te = build_aef_neigh_features_triplet(
                    X_po, X_pa_tr, X_pa_te,
                    year=AEF_YEAR,
                    radii_m=NEIGHBORHOOD["radii_m"],
                    stats=NEIGHBORHOOD["stats"],
                    nan_policy=NAN_POLICY
                )
                Ypo = Y_po.iloc[keep_po] if NAN_POLICY == "drop" else Y_po
                Ytr = Y_pa_tr.iloc[keep_tr] if NAN_POLICY == "drop" else Y_pa_tr
                Yte = Y_pa_te.iloc[keep_te] if NAN_POLICY == "drop" else Y_pa_te
                features["AEF_NEIGH"] = dict(Xpo=Xpo, Xtr=Xtr, Xte=Xte, Ypo=Ypo, Ytr=Ytr, Yte=Yte)

            # ---- CONCAT: COVS + AEF_POINT
            if RUN_FEATURESETS.get("CONCAT_COVS_AEFPOINT", False):
                if "COVS" in features and "AEF_POINT" in features:
                    # Need to ensure same row counts per split. If AEF dropped rows, we must drop cov rows to match.
                    cov = features["COVS"]
                    aef = features["AEF_POINT"]

                    # If NAN_POLICY=="drop", aef labels were already aligned to dropped rows by iloc masks.
                    # But cov still includes full data. We match by taking the same iloc mask used by AEF sampling.
                    # We stored only arrays, so easiest is: rebuild cov features with the same *filtered DataFrames*.
                    # Here we reconstruct those filtered DFs from labels indices:
                    X_po_f = X_po.loc[aef["Ypo"].index]
                    X_tr_f = X_pa_tr.loc[aef["Ytr"].index]
                    X_te_f = X_pa_te.loc[aef["Yte"].index]
                    Xpo_cov, Xtr_cov, Xte_cov = build_covs_features_triplet(X_po_f, X_tr_f, X_te_f, env_covs)

                    Xpo_cat = np.concatenate([Xpo_cov, aef["Xpo"]], axis=1)
                    Xtr_cat = np.concatenate([Xtr_cov, aef["Xtr"]], axis=1)
                    Xte_cat = np.concatenate([Xte_cov, aef["Xte"]], axis=1)

                    features["CONCAT_COVS_AEFPOINT"] = dict(
                        Xpo=Xpo_cat, Xtr=Xtr_cat, Xte=Xte_cat,
                        Ypo=aef["Ypo"], Ytr=aef["Ytr"], Yte=aef["Yte"]
                    )

            # ---- CONCAT: COVS + AEF_NEIGH
            if RUN_FEATURESETS.get("CONCAT_COVS_AEFNEIGH", False):
                if "COVS" in features and "AEF_NEIGH" in features:
                    cov = features["COVS"]
                    aef = features["AEF_NEIGH"]

                    X_po_f = X_po.loc[aef["Ypo"].index]
                    X_tr_f = X_pa_tr.loc[aef["Ytr"].index]
                    X_te_f = X_pa_te.loc[aef["Yte"].index]
                    Xpo_cov, Xtr_cov, Xte_cov = build_covs_features_triplet(X_po_f, X_tr_f, X_te_f, env_covs)

                    Xpo_cat = np.concatenate([Xpo_cov, aef["Xpo"]], axis=1)
                    Xtr_cat = np.concatenate([Xtr_cov, aef["Xtr"]], axis=1)
                    Xte_cat = np.concatenate([Xte_cov, aef["Xte"]], axis=1)

                    features["CONCAT_COVS_AEFNEIGH"] = dict(
                        Xpo=Xpo_cat, Xtr=Xtr_cat, Xte=Xte_cat,
                        Ypo=aef["Ypo"], Ytr=aef["Ytr"], Yte=aef["Yte"]
                    )

            # ------------------------------------------------------------
            # Run tasks over all requested feature sets
            # ------------------------------------------------------------
            for ftag, pack in features.items():
                print(f"\n--- FeatureSet: {ftag} | dims = {pack['Xpo'].shape[1]} ---")

                # PO-only: train on PO, test on PA_test
                if RUN_TASKS.get("PO", False):
                    res = fit_eval_one(
                        task="PO",
                        feature_tag=ftag,
                        X_train=pack["Xpo"],
                        Y_train=pack["Ypo"],
                        X_test=pack["Xte"],
                        Y_test=pack["Yte"],
                        species=species,
                        out_dir=os.path.join(exp_dir, ftag),
                    )
                    row[f"{ftag}_PO_auc"] = res["avg_auc"]
                    print(f"[{ftag}][PO]   avg AUC = {res['avg_auc']:.4f}")

                # PA-only: train on PA_train, test on PA_test
                if RUN_TASKS.get("PA", False):
                    res = fit_eval_one(
                        task="PA",
                        feature_tag=ftag,
                        X_train=pack["Xtr"],
                        Y_train=pack["Ytr"],
                        X_test=pack["Xte"],
                        Y_test=pack["Yte"],
                        species=species,
                        out_dir=os.path.join(exp_dir, ftag),
                    )
                    row[f"{ftag}_PA_auc"] = res["avg_auc"]
                    print(f"[{ftag}][PA]   avg AUC = {res['avg_auc']:.4f}")

                # POPA: train on PO + PA_train, test on PA_test
                if RUN_TASKS.get("POPA", False):
                    Xmix = np.vstack([pack["Xpo"], pack["Xtr"]])
                    Ymix = pd.concat([pack["Ypo"], pack["Ytr"]], axis=0, ignore_index=True)

                    res = fit_eval_one(
                        task="POPA",
                        feature_tag=ftag,
                        X_train=Xmix,
                        Y_train=Ymix,
                        X_test=pack["Xte"],
                        Y_test=pack["Yte"],
                        species=species,
                        out_dir=os.path.join(exp_dir, ftag),
                    )
                    row[f"{ftag}_POPA_auc"] = res["avg_auc"]
                    print(f"[{ftag}][POPA] avg AUC = {res['avg_auc']:.4f}")

            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(OUTPUT_ROOT, "results.csv"), index=False)
    print(f"\nDone in {(time() - t0):.1f}s")

if __name__ == "__main__":
    main()



# # minimal_popa_covs_vs_aef_flexible.py

# import os
# import random
# from time import time
# from typing import List, Dict, Optional, Any

# import numpy as np
# import pandas as pd

# import torch
# from torch.utils.data import Dataset, DataLoader
# import torch.nn as nn

# import ee
# from sklearn.preprocessing import StandardScaler

# from src.load_data import load_po_pa_nceas
# from src.utils import split_pa_train_test_spatially, scale_features
# from src.models import DeepMaxEntModel, deepmaxent_loss


# # =========================
# # Config (high level)
# # =========================
# REGIONS = ["SA"]
# GROUPS_BY_REGION = {
#     "AWT": ["_plant", "_bird"],
#     "CAN": [""],
#     "NSW": ["_bat", "_bird", "_plant", "_reptile"],
#     "SA":  [""],
#     "SWI": [""],
#     "NZ":  [""],
# }

# TEST_PA_FRACTION = .99
# SEED = 42

# # Which tasks to run
# RUN_TASKS = {
#     "PO": True,
#     "PA": False,
#     "POPA": False,
# }

# # Which feature sets to compare
# RUN_FEATURESETS = {
#     "COVS": True,
#     "AEF": True,
# }

# # Earth Engine / AlphaEarth
# EE_PROJECT = "alpha-earth-test-483513"
# AEF_YEAR = 2019

# # Embedding NaN handling:
# #  - "raise": stop if any NaNs
# #  - "fill0": replace NaNs with 0
# #  - "drop": drop rows with any NaN embeddings (recommended for correctness)
# AEF_NAN_POLICY = "drop"


# # =========================
# # Model/training configs per feature set
# # =========================
# # You can tune these independently.
# MODEL_CFG = {
#     "COVS": {
#         "hidden_size": 250,
#         "hidden_layers": 2,
#         "lr": 1e-4,
#         "epochs": 200,
#         "batch_size": 250,
#         "max_batch_pct": 1.0,
#         "criterion": {
#             "PO": "deepmaxent",
#             "PA": "bce",
#             "POPA": "bce",
#         },
#     },
#     "AEF": {
#         # embeddings are 64D -> usually benefit from larger model
#         "hidden_size": 200,
#         "hidden_layers": 3,
#         "lr": 2e-4,
#         "epochs": 500,
#         "batch_size": 512,
#         "max_batch_pct": 1.0,
#         "criterion": {
#             "PO": "deepmaxent",
#             "PA": "bce",
#             "POPA": "bce",
#         },
#     },
# }


# # =========================
# # Utils
# # =========================
# def set_all_seeds(seed: int = 42) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def device() -> torch.device:
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class XYDataset(Dataset):
#     def __init__(self, X: np.ndarray, Y: np.ndarray):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.Y = torch.tensor(Y, dtype=torch.float32)

#     def __len__(self) -> int:
#         return self.X.shape[0]

#     def __getitem__(self, idx: int):
#         return self.X[idx], self.Y[idx], torch.tensor(idx, dtype=torch.long)


# def make_model(input_dim: int, output_dim: int, hidden_size: int, hidden_layers: int) -> nn.Module:
#     return DeepMaxEntModel(
#         input_size=input_dim,
#         hidden_size=hidden_size,
#         output_size=output_dim,
#         hidden_nbr=hidden_layers
#     )


# def train_model(
#     model: nn.Module,
#     train_loader: DataLoader,
#     criterion: str,
#     epochs: int,
#     lr: float,
#     dev: Optional[torch.device] = None,
# ):
#     dev = dev or device()
#     model.to(dev)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-4)

#     if criterion == "bce":
#         loss_f = torch.nn.BCEWithLogitsLoss()
#     elif criterion == "deepmaxent":
#         loss_f = deepmaxent_loss()
#     else:
#         raise ValueError(f"Unknown criterion: {criterion}")

#     model.train()
#     for _ in range(epochs):
#         for xb, yb, _ in train_loader:
#             xb = xb.to(dev)
#             yb = yb.to(dev)

#             optimizer.zero_grad()
#             out = model(xb)
#             loss = loss_f(out, yb)
#             loss.backward()
#             optimizer.step()


# @torch.no_grad()
# def predict(model: nn.Module, X: np.ndarray, dev: Optional[torch.device] = None) -> np.ndarray:
#     dev = dev or device()
#     model.eval()
#     X_t = torch.tensor(X, dtype=torch.float32, device=dev)
#     out = model(X_t)
#     return out.detach().cpu().numpy()


# def per_species_auc(y_true: pd.DataFrame, y_score: np.ndarray, species: List[str]) -> Dict[str, float]:
#     from sklearn.metrics import roc_auc_score
#     scores = {}
#     for i, sp in enumerate(species):
#         try:
#             scores[sp] = roc_auc_score(y_true[sp].values, y_score[:, i])
#         except ValueError:
#             scores[sp] = np.nan
#     return scores

# def per_site_auc(y_true: pd.DataFrame, y_score: np.ndarray) -> Dict[str, float]:
#     from sklearn.metrics import roc_auc_score
#     scores = {}
#     sites = y_true.index.tolist()
#     for i, site in enumerate(sites):
#         try:
#             scores[site] = roc_auc_score(y_true.iloc[i].values, y_score[i, :])
#         except ValueError:
#             scores[site] = np.nan
#     return scores


# # =========================
# # Feature builders
# # =========================
# def build_covariate_features(
#     X_train: pd.DataFrame,
#     X_test: pd.DataFrame,
#     covs: List[str],
# ):
#     # no scaler written (keeps this script standalone)
#     X_train_s, X_test_s, _ = scale_features(X_train, X_test, covs, output_path=None, verbose=False)
#     Xtr = X_train_s[covs].values.astype(np.float32)
#     Xte = X_test_s[covs].values.astype(np.float32)
#     return Xtr, Xte

# def df_to_fc(df: pd.DataFrame) -> ee.FeatureCollection:
#     """
#     Convert a dataframe with columns x (lon), y (lat) into an EE FeatureCollection,
#     with stable row id __rid and optional siteid.
#     """
#     df2 = df.reset_index(drop=False).copy()     # keep original index in a column
#     df2["__rid"] = np.arange(len(df2), dtype=int)

#     feats = []
#     for _, row in df2.iterrows():
#         lon = float(row["x"])
#         lat = float(row["y"])
#         props = {"__rid": int(row["__rid"])}
#         if "siteid" in df2.columns:
#             props["siteid"] = str(row["siteid"])
#         feats.append(ee.Feature(ee.Geometry.Point([lon, lat]), props))

#     return ee.FeatureCollection(feats)


# def aef_add_embeddings(df: pd.DataFrame, year: int, nan_policy: str = "raise") -> pd.DataFrame:
#     """
#     Adds A00..A63 sampled at (x=lon, y=lat).
#     NaN policy:
#       - raise: raise if any missing
#       - fill0: fill NaNs with 0
#       - drop: drop rows with missing embeddings
#     """
#     df = df.copy()

#     bands = [f"A{i:02d}" for i in range(64)]

#     col = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
#         .filterDate(f"{year}-01-01", f"{year+1}-01-01"))

#     fc = df_to_fc(df)

#     # Mosaic tiles that intersect the points
#     img = col.filterBounds(fc.geometry()).mosaic().select(bands)

#     samples = img.sampleRegions(collection=fc, properties=["__rid", "siteid"], scale=10, geometries=False)

#     info = samples.getInfo()
#     rows = info["features"]

#     # Keep stable join key
#     df = df.reset_index(drop=False).rename(columns={"index": "__orig_index"})
#     feats = []
#     for ridx, row in df.iterrows():
#         lon = float(row["x"]); lat = float(row["y"])
#         feats.append(ee.Feature(ee.Geometry.Point([lon, lat]), {"__rid": int(ridx)}))
#     fc = ee.FeatureCollection(feats)

#     samples = img.sampleRegions(collection=fc, properties=["__rid"], scale=10, geometries=False)
#     info = samples.getInfo()
#     rows = info["features"]



#     emb = np.full((len(df), 64), np.nan, dtype=np.float32)
#     for feat in rows:
#         props = feat["properties"]
#         rid = int(props["__rid"])
#         # Some points may miss bands if masked; guard:
#         if any(b not in props for b in bands):
#             continue
#         emb[rid, :] = np.array([props[b] for b in bands], dtype=np.float32)

#     for j, b in enumerate(bands):
#         df[b] = emb[:, j]

#     nan_mask = df[bands].isna().any(axis=1)
#     n_nan = int(nan_mask.sum())

#     if n_nan > 0:
#         # show first few problematic rows for debugging
#         cols_show = ["__orig_index", "x", "y"]
#         if "siteid" in df.columns:
#             cols_show = ["siteid"] + cols_show
#         print(f"[AEF] {n_nan}/{len(df)} rows have NaN embeddings. Examples:")
#         print(df.loc[nan_mask, cols_show].head(10).to_string(index=False))

#         if nan_policy == "raise":
#             raise ValueError("NaNs in AlphaEarth embeddings. Set AEF_NAN_POLICY to 'drop' or 'fill0'.")
#         elif nan_policy == "fill0":
#             df.loc[:, bands] = df[bands].fillna(0.0)
#         elif nan_policy == "drop":
#             df = df.loc[~nan_mask].copy()
#         else:
#             raise ValueError(f"Unknown nan_policy: {nan_policy}")

#     # restore original index for clean alignment with Y
#     df = df.set_index("__orig_index", drop=True)
#     return df

# def aef_add_embeddings_neighborhood(
#     df: pd.DataFrame,
#     year: int,
#     radius_m: int = 250,
#     reducers=("mean", "std"),
# ):
#     """
#     Sample AlphaEarth embeddings over a buffer around each point.
#     Returns df with A00_mean, A00_std, ..., A63_* columns.
#     """
#     bands = [f"A{i:02d}" for i in range(64)]

#     # build image
#     col = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
#            .filterDate(f"{year}-01-01", f"{year+1}-01-01"))

#     # build feature collection with stable ids
#     df0 = df.reset_index(drop=False).rename(columns={"index": "__orig_index"})
#     df0["__rid"] = np.arange(len(df0))

#     feats = []
#     for _, r in df0.iterrows():
#         geom = ee.Geometry.Point([float(r.x), float(r.y)]).buffer(radius_m)
#         feats.append(ee.Feature(geom, {"__rid": int(r.__rid)}))
#     fc = ee.FeatureCollection(feats)

#     img = col.filterBounds(fc.geometry()).mosaic().select(bands)

#     # build reducer
#     r = None
#     if "mean" in reducers:
#         r = ee.Reducer.mean()
#     if "std" in reducers:
#         r = r.combine(ee.Reducer.stdDev(), sharedInputs=True)

#     samples = img.reduceRegions(
#         collection=fc,
#         reducer=r,
#         scale=10
#     )

#     info = samples.getInfo()
#     rows = info["features"]

#     # prepare output
#     out_cols = []
#     for b in bands:
#         if "mean" in reducers:
#             out_cols.append(f"{b}_mean")
#         if "std" in reducers:
#             out_cols.append(f"{b}_std")

#     X = np.full((len(df), len(out_cols)), np.nan, dtype=np.float32)

#     for f in rows:
#         rid = int(f["properties"]["__rid"])
#         vals = []
#         for b in bands:
#             if "mean" in reducers:
#                 vals.append(f["properties"].get(f"{b}_mean"))
#             if "std" in reducers:
#                 vals.append(f["properties"].get(f"{b}_std"))
#         X[rid, :] = vals

#     # attach to df
#     df_out = df.copy()
#     for i, c in enumerate(out_cols):
#         df_out[c] = X[:, i]

#     return df_out, out_cols


# def build_aef_features_triplet(
#     X_po: pd.DataFrame,
#     X_pa_tr: pd.DataFrame,
#     X_pa_te: pd.DataFrame,
#     year: int,
#     nan_policy: str,
# ):
#     bands = [f"A{i:02d}" for i in range(64)]
#     X_po_e = aef_add_embeddings(X_po, year, nan_policy=nan_policy)
#     X_pa_tr_e = aef_add_embeddings(X_pa_tr, year, nan_policy=nan_policy)
#     X_pa_te_e = aef_add_embeddings(X_pa_te, year, nan_policy=nan_policy)

#     # Fit scaler on PO + PA_train (matches your POPA philosophy)
#     scaler = StandardScaler().fit(pd.concat([X_po_e[bands], X_pa_tr_e[bands]], axis=0).values)

#     X_po_e.loc[:, bands] = scaler.transform(X_po_e[bands].values)
#     X_pa_tr_e.loc[:, bands] = scaler.transform(X_pa_tr_e[bands].values)
#     X_pa_te_e.loc[:, bands] = scaler.transform(X_pa_te_e[bands].values)

#     return X_po_e, X_pa_tr_e, X_pa_te_e, bands


# # =========================
# # Task runner
# # =========================
# def fit_eval_task(
#     *,
#     task_name: str,                    # "PO" | "PA" | "POPA"
#     name_prefix: str,
#     X_train_np: np.ndarray,
#     Y_train_df: pd.DataFrame,
#     X_test_np: np.ndarray,
#     Y_test_df: pd.DataFrame,
#     species: List[str],
#     out_dir: str,
#     model_cfg: Dict[str, Any],
# ):
#     os.makedirs(out_dir, exist_ok=True)

#     crit = model_cfg["criterion"][task_name]
#     model = make_model(
#         input_dim=X_train_np.shape[1],
#         output_dim=len(species),
#         hidden_size=model_cfg["hidden_size"],
#         hidden_layers=model_cfg["hidden_layers"],
#     )

#     bs = max(1, min(model_cfg["batch_size"], int(len(X_train_np) * model_cfg["max_batch_pct"])))
#     loader = DataLoader(
#         XYDataset(X_train_np, Y_train_df[species].values.astype(np.float32)),
#         batch_size=bs,
#         shuffle=True,
#         drop_last=False
#     )

#     train_model(
#         model=model,
#         train_loader=loader,
#         criterion=crit,
#         epochs=model_cfg["epochs"],
#         lr=model_cfg["lr"],
#         dev=device(),
#     )

#     scores = predict(model, X_test_np, dev=device())
#     aucs = per_species_auc(Y_test_df[species], scores, species)
#     avg_auc = float(np.nanmean(list(aucs.values())))

#     # aucs_site = per_site_auc(Y_test_df[species], scores)
#     # avg_auc_site = float(np.nanmean(list(aucs_site.values())))

#     model_path = os.path.join(out_dir, f"{name_prefix}_{task_name}.pt")
#     torch.save(model, model_path)

#     return {
#         "avg_auc": avg_auc,
#         # "avg_auc_site": avg_auc_site,
#         "model_path": model_path,
#     }


# # =========================
# # Main
# # =========================
# def main():
#     start = time()
#     set_all_seeds(SEED)

#     ee.Initialize(project=EE_PROJECT)

#     output_root = os.path.join("output", "minimal_popa_compare_flexible")
#     os.makedirs(output_root, exist_ok=True)

#     summary_rows = []

#     for region in REGIONS:
#         for group in GROUPS_BY_REGION[region]:
#             print(f"\n=== REGION: {region}, GROUP: {group or '(all)'} ===")

#             # Load aligned PO/PA
#             X_po, Y_po, X_pa, Y_pa, species, covs = load_po_pa_nceas(
#                 region=region,
#                 group_filter=group,
#                 add_po_var=False,
#                 keep_xy=True,
#                 index_col=["siteid"],
#             )

#             # Split PA train/test
#             X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te = split_pa_train_test_spatially(
#                 X_pa, Y_pa, test_frac=TEST_PA_FRACTION, seed=SEED, K=100
#             )
#             print(f"PA split → train: {len(X_pa_tr)}, test: {len(X_pa_te)}")



#             exp_dir = os.path.join(output_root, f"{region}{group}")
#             os.makedirs(exp_dir, exist_ok=True)

#             # baseline covariates = all non-coordinate covs
#             env_covs = [c for c in covs if c not in ["x", "y", "PO"]]

#             # We will evaluate on PA_test for all tasks (as you did)
#             results = {}

#             # =========================
#             # FeatureSet: COVS
#             # =========================
#             if RUN_FEATURESETS["COVS"]:
#                 cfg = MODEL_CFG["COVS"]
#                 Xtr_po, Xte = build_covariate_features(X_po, X_pa_te, env_covs)
#                 Xtr_pa, _ = build_covariate_features(X_pa_tr, X_pa_te, env_covs)

#                 # Task: PO
#                 if RUN_TASKS["PO"]:
#                     r = fit_eval_task(
#                         task_name="PO",
#                         name_prefix=f"{region}{group}_COVS",
#                         X_train_np=Xtr_po,
#                         Y_train_df=Y_po,
#                         X_test_np=Xte,
#                         Y_test_df=Y_pa_te,
#                         species=species,
#                         out_dir=exp_dir,
#                         model_cfg=cfg,
#                     )
#                     results["COVS_PO_auc"] = r["avg_auc"]

#                 # Task: PA
#                 if RUN_TASKS["PA"]:
#                     r = fit_eval_task(
#                         task_name="PA",
#                         name_prefix=f"{region}{group}_COVS",
#                         X_train_np=Xtr_pa,
#                         Y_train_df=Y_pa_tr,
#                         X_test_np=Xte,
#                         Y_test_df=Y_pa_te,
#                         species=species,
#                         out_dir=exp_dir,
#                         model_cfg=cfg,
#                     )
#                     results["COVS_PA_auc"] = r["avg_auc"]

#                 # Task: POPA
#                 if RUN_TASKS["POPA"]:
#                     X_mix = np.vstack([Xtr_po, Xtr_pa])
#                     Y_mix = pd.concat([Y_po, Y_pa_tr], axis=0, ignore_index=True)
#                     r = fit_eval_task(
#                         task_name="POPA",
#                         name_prefix=f"{region}{group}_COVS",
#                         X_train_np=X_mix,
#                         Y_train_df=Y_mix,
#                         X_test_np=Xte,
#                         Y_test_df=Y_pa_te,
#                         species=species,
#                         out_dir=exp_dir,
#                         model_cfg=cfg,
#                     )
#                     results["COVS_POPA_auc"] = r["avg_auc"]

#             # =========================
#             # FeatureSet: AEF
#             # =========================
#             if RUN_FEATURESETS["AEF"]:
#                 cfg = MODEL_CFG["AEF"]

#                 # Build embeddings for all three sets, and align Y by index
#                 X_po_e, X_pa_tr_e, X_pa_te_e, emb_bands = build_aef_features_triplet(
#                     X_po, X_pa_tr, X_pa_te,
#                     year=AEF_YEAR,
#                     nan_policy=AEF_NAN_POLICY
#                 )

#                 # Align labels with potentially dropped rows
#                 Y_po_e = Y_po.loc[X_po_e.index]
#                 Y_pa_tr_e = Y_pa_tr.loc[X_pa_tr_e.index]
#                 Y_pa_te_e = Y_pa_te.loc[X_pa_te_e.index]


#                 Xtr_po = X_po_e[emb_bands].values.astype(np.float32)
#                 Xtr_pa = X_pa_tr_e[emb_bands].values.astype(np.float32)
#                 Xte = X_pa_te_e[emb_bands].values.astype(np.float32)

#                 # Task: PO
#                 if RUN_TASKS["PO"]:
#                     r = fit_eval_task(
#                         task_name="PO",
#                         name_prefix=f"{region}{group}_AEF{AEF_YEAR}",
#                         X_train_np=Xtr_po,
#                         Y_train_df=Y_po_e,
#                         X_test_np=Xte,
#                         Y_test_df=Y_pa_te_e,
#                         species=species,
#                         out_dir=exp_dir,
#                         model_cfg=cfg,
#                     )
#                     results["AEF_PO_auc"] = r["avg_auc"]

#                 # Task: PA
#                 if RUN_TASKS["PA"]:
#                     r = fit_eval_task(
#                         task_name="PA",
#                         name_prefix=f"{region}{group}_AEF{AEF_YEAR}",
#                         X_train_np=Xtr_pa,
#                         Y_train_df=Y_pa_tr_e,
#                         X_test_np=Xte,
#                         Y_test_df=Y_pa_te_e,
#                         species=species,
#                         out_dir=exp_dir,
#                         model_cfg=cfg,
#                     )
#                     results["AEF_PA_auc"] = r["avg_auc"]

#                 # Task: POPA
#                 if RUN_TASKS["POPA"]:
#                     X_mix = np.vstack([Xtr_po, Xtr_pa])
#                     Y_mix = pd.concat([Y_po_e, Y_pa_tr_e], axis=0, ignore_index=True)
#                     r = fit_eval_task(
#                         task_name="POPA",
#                         name_prefix=f"{region}{group}_AEF{AEF_YEAR}",
#                         X_train_np=X_mix,
#                         Y_train_df=Y_mix,
#                         X_test_np=Xte,
#                         Y_test_df=Y_pa_te_e,
#                         species=species,
#                         out_dir=exp_dir,
#                         model_cfg=cfg,
#                     )
#                     results["AEF_POPA_auc"] = r["avg_auc"]

#             # Record row
#             row = {
#                 "region": region,
#                 "group": group or "(all)",
#                 "pa_test_frac": TEST_PA_FRACTION,
#                 "aef_year": AEF_YEAR,
#                 "nan_policy": AEF_NAN_POLICY,
#                 **results
#             }
#             summary_rows.append(row)

#             # quick print
#             printable = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in results.items()}
#             print("Results:", printable)

#     summary = pd.DataFrame(summary_rows)
#     print("\n=== Summary ===")
#     print(summary.to_string(index=False))
#     summary.to_csv(os.path.join(output_root, "results.csv"), index=False)

#     print(f"\nDone in {(time() - start):.1f}s")


# if __name__ == "__main__":
#     main()
