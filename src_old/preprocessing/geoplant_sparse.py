import os
from functools import lru_cache
from typing import Optional, List, Tuple, Dict

import duckdb
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.prepared import prep
from scipy.sparse import coo_matrix, csr_matrix

import ee
from time import time
import tqdm

# -----------------------------
# Natural Earth country polygons
# -----------------------------
NE_URL = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"


@lru_cache(maxsize=1)
def _load_natural_earth_world() -> gpd.GeoDataFrame:
    # cached in-process so you don't re-download every call
    return gpd.read_file(NE_URL)[["NAME", "geometry"]]


def _get_country_geometry(region_name: str, mainland_only: bool = True):
    world = _load_natural_earth_world()
    region = world.loc[world["NAME"] == region_name]
    if region.empty:
        raise ValueError(f"Region '{region_name}' not found in Natural Earth.")

    geom = region.geometry.iloc[0]

    # Pragmatic "mainland-only": keep largest polygon by area (projected CRS)
    if mainland_only and geom.geom_type == "MultiPolygon":
        geom_m = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
        largest = max(geom_m.geoms, key=lambda g: g.area)
        geom = gpd.GeoSeries([largest], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]

    return geom


def region_filter_worldmap_sparse_df(
    df: pd.DataFrame,
    region_name: str,
    lon_col: str = "lon",
    lat_col: str = "lat",
    mainland_only: bool = True,
) -> pd.DataFrame:
    """
    Exact point-in-polygon filter using Natural Earth country geometry.
    Assumes df already got bbox-prefiltered (but works either way).
    """
    geom = _get_country_geometry(region_name, mainland_only=mainland_only)

    # bbox prefilter (cheap)
    minx, miny, maxx, maxy = geom.bounds
    bbox_mask = (
        (df[lon_col] >= minx) & (df[lon_col] <= maxx) &
        (df[lat_col] >= miny) & (df[lat_col] <= maxy)
    )
    df2 = df.loc[bbox_mask].copy()
    if df2.empty:
        return df2

    pts = gpd.GeoSeries(gpd.points_from_xy(df2[lon_col], df2[lat_col]), crs="EPSG:4326")
    prepared = prep(geom)
    inside = pts.apply(prepared.contains).to_numpy()

    return df2.loc[inside].copy()


# -----------------------------
# Sparse loader (CSR matrix)
# -----------------------------
def load_geoplant_dataset_sparse(
    filepath: str,
    *,
    region_filter: Optional[str] = None,        # filter on df["region"] if present
    worldmap_region: Optional[str] = None,      # Natural Earth NAME, e.g. "France"
    species_min_presence: int = 10,
    sample_fraction: float = 1.0,
    keep_survey_id: bool = True,
    meta_cols: Optional[List[str]] = None,      # columns to carry into meta (e.g. ["lon","lat","year"])
    lon_col: str = "lon",
    lat_col: str = "lat",
    mainland_only: bool = True,
    species_prefix: str = "sp_",
    subsample : Optional[float] = None,
) -> Tuple[csr_matrix, pd.DataFrame, List[str]]:
    """
    Returns:
      X: csr_matrix shape (n_rows, n_species) with 0/1
      meta: DataFrame with the index cols (surveyId + meta_cols) aligned to X rows
      species_cols: list of 'sp_<id>' aligned to X columns
    """
    meta_cols = meta_cols or []
    index_cols = (["surveyId"] if keep_survey_id else []) + meta_cols
    if not index_cols:
        raise ValueError("Need at least one index column (surveyId and/or meta_cols).")

    con = duckdb.connect()

    bbox_sql = ""
    if worldmap_region is not None:
        geom = _get_country_geometry(worldmap_region, mainland_only=mainland_only)
        minx, miny, maxx, maxy = geom.bounds
        bbox_sql = f" AND {lon_col} BETWEEN {minx} AND {maxx} AND {lat_col} BETWEEN {miny} AND {maxy} "

    # WHERE clause
    where = "WHERE speciesId IS NOT NULL"
    if region_filter is not None:
        # note: simple quoting; if region values can contain quotes, escape them
        where += f" AND region = '{region_filter}'"
    if bbox_sql:
        where += bbox_sql

    # Only read needed columns
    need_geo_cols = (worldmap_region is not None)
    select_cols = index_cols + ["speciesId"]
    if need_geo_cols:
        # ensure lon/lat exist for polygon filtering even if not in meta_cols
        if lon_col not in select_cols:
            select_cols.append(lon_col)
        if lat_col not in select_cols:
            select_cols.append(lat_col)

    cols_sql = ", ".join(select_cols)

    base = f"""
        SELECT {cols_sql}
        FROM read_csv_auto('{filepath}')
        {where}
    """

    # if sample_fraction < 1.0:
    #     base = f"SELECT * FROM ({base}) USING SAMPLE {sample_fraction*100:.6f}%"

    # Deduplicate presences: one row per (index_cols + speciesId)
    # (If lon/lat are included here, it can stop dedup from working; so dedup on index + species only)
    dedup_cols = ", ".join(index_cols + ["speciesId"] + ([lon_col, lat_col] if need_geo_cols else []))
    # If lon/lat are part of meta_cols, they are already in index_cols, so above may duplicate; harmless.
    dedup = f"SELECT DISTINCT {dedup_cols} FROM ({base})"

    pairs = con.execute(dedup).fetchdf()
    if pairs.empty:
        return csr_matrix((0, 0), dtype=np.uint8), pd.DataFrame(columns=index_cols), []

    # Exact polygon filter (on reduced subset)
    if worldmap_region is not None:
        if lon_col not in pairs.columns or lat_col not in pairs.columns:
            raise ValueError(f"Need '{lon_col}' and '{lat_col}' columns for worldmap filtering.")
        pairs = region_filter_worldmap_sparse_df(
            pairs,
            worldmap_region,
            lon_col=lon_col,
            lat_col=lat_col,
            mainland_only=mainland_only,
        )
        if pairs.empty:
            return csr_matrix((0, 0), dtype=np.uint8), pd.DataFrame(columns=index_cols), []

    # Keep only what we need for matrix construction
    pairs = pairs[index_cols + ["speciesId"]]

    # Meta rows (unique index)
    X = pairs[index_cols].drop_duplicates().reset_index(drop=True)
    X["_row_id"] = np.arange(len(X), dtype=np.int64)

    pairs = pairs.merge(X, on=index_cols, how="left", validate="many_to_one")

    # Map speciesId -> col_id
    species = pd.Index(pairs["speciesId"].unique()).sort_values()
    species_to_col = pd.Series(np.arange(len(species), dtype=np.int64), index=species)

    row_id = pairs["_row_id"].to_numpy()
    col_id = species_to_col.loc[pairs["speciesId"]].to_numpy()


    # # Filter rare species
    # counts = np.bincount(col_id, minlength=len(species))
    # keep = counts >= species_min_presence
    # if not keep.all():
    #     kept_cols = np.flatnonzero(keep)
    #     remap = -np.ones(len(species), dtype=np.int64)
    #     remap[kept_cols] = np.arange(len(kept_cols), dtype=np.int64)

    #     sel = keep[col_id]
    #     row_id = row_id[sel]
    #     col_id = remap[col_id[sel]]
    #     species = species[kept_cols]

    data = np.ones(len(row_id), dtype=np.uint8)
    Y = coo_matrix(
        (data, (row_id, col_id)),
        shape=(len(X), len(species)),
        dtype=np.uint8
    ).tocsr()

    species_cols = [f"{species_prefix}{int(s)}" for s in species]
    X = X.drop(columns=["_row_id"])


    if subsample is not None:
        n_subsample = int(len(X) * subsample)
        if n_subsample < len(X):
            chosen_indices = np.random.choice(len(X), size=n_subsample, replace=False)
            X = X.iloc[chosen_indices].reset_index(drop=True)
            Y = Y[chosen_indices, :]

    # min species again
    species_counts = np.array(Y.sum(axis=0)).flatten()
    keep_species = species_counts >= species_min_presence
    if not keep_species.all():
        Y = Y[:, keep_species]
        species_cols = [s for k, s in zip(keep_species, species_cols) if k]

    return Y, X, species_cols


# -----------------------------
# Sparse utilities
# -----------------------------
def intersect_species_columns_sparse(
    X1: csr_matrix, species_cols1: List[str],
    X2: csr_matrix, species_cols2: List[str],
) -> Tuple[csr_matrix, List[str], csr_matrix, List[str]]:
    """
    Intersect two sparse matrices on common species columns.
    Returns X1_common, common_species_cols, X2_common, common_species_cols
    """
    set1 = set(species_cols1)
    common = sorted(set1.intersection(species_cols2))
    if not common:
        # empty intersection
        return X1[:, :0], [], X2[:, :0], []

    idx1 = {s: i for i, s in enumerate(species_cols1)}
    idx2 = {s: i for i, s in enumerate(species_cols2)}
    cols1 = [idx1[s] for s in common]
    cols2 = [idx2[s] for s in common]

    return X1[:, cols1], common, X2[:, cols2], common


def get_species_column(X: csr_matrix, species_cols: List[str], sp_name: str):
    """
    Returns the sparse column vector for one species name, shape (n_rows, 1).
    """
    j = species_cols.index(sp_name)
    return X[:, j]


# -----------------------------
# Google Earth Engine covariates (to meta)
# -----------------------------
def add_covariates_geoplant_embedding(
    meta: pd.DataFrame,
    *,
    lon_col: str = "lon",
    lat_col: str = "lat",
    year_col: str = "year",
    scale: int = 10,
    chunk_size: int = 2000,
    project: str = "alpha-earth-test-483513",
) -> pd.DataFrame:
    """
    Adds 64-dim annual satellite embedding covariates (A00..A63) to `meta`.

    - Uses meta[year_col] so each row can request the proper year.
    - Chunked to keep Earth Engine responses manageable.
    - Returns meta with added columns A00..A63 (NA if missing/masked).
    """
    ee.Initialize(project=project)

    bands = [f"A{i:02d}" for i in range(64)]
    dataset = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    base = meta.reset_index(drop=True).copy()
    base["row_id"] = np.arange(len(base), dtype=np.int64)

    # Ensure needed columns exist
    for c in [lon_col, lat_col, year_col]:
        if c not in base.columns:
            raise ValueError(f"meta is missing required column '{c}'")

    all_parts = []
    total_missing = 0

    # group by year to avoid doing a separate ImageCollection filter per-point
    for yr, grp in base.groupby(year_col, sort=True):
        start_time_yr = time()
        print(f"Processing year {yr} with {len(grp)} points...")

        # GEE expects Python int
        yr_int = int(yr)

        year_ic = dataset.filterDate(f"{yr_int}-01-01", f"{yr_int+1}-01-01")

        # chunk within each year
        grp = grp[["row_id", lon_col, lat_col]].copy().reset_index(drop=True)

        for start in range(0, len(grp), chunk_size):
            part = grp.iloc[start:start + chunk_size].copy()

            feats = [
                ee.Feature(
                    ee.Geometry.Point([float(r[lon_col]), float(r[lat_col])]),
                    {"row_id": int(r["row_id"])}
                )
                for _, r in part.iterrows()
            ]
            fc = ee.FeatureCollection(feats)

            # Mosaic tiles that intersect this chunk (matches your earlier approach)
            chunk_img = (
                year_ic
                .filterBounds(fc.geometry())
                .mosaic()
                .select(bands)
            )

            sampled = chunk_img.sampleRegions(
                collection=fc,
                properties=["row_id"],
                scale=scale,
                geometries=False
            )

            info = sampled.getInfo()
            features = info.get("features", [])
            props = [f.get("properties", {}) for f in features]
            emb_df = pd.DataFrame(props)

            if "row_id" not in emb_df.columns:
                emb_df["row_id"] = pd.Series(dtype=int)

            for b in bands:
                if b not in emb_df.columns:
                    emb_df[b] = pd.NA

            got = emb_df["row_id"].nunique(dropna=True)
            missing = len(part) - got
            total_missing += missing

            all_parts.append(emb_df[["row_id"] + bands])

        print(f"  Year {yr} done in {time() - start_time_yr:.1f}s, missing {total_missing} points so far.")

    emb_all = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame(columns=["row_id"] + bands)

    out = (
        base
        .merge(emb_all, on="row_id", how="left")
        .drop(columns=["row_id"])
        .reset_index(drop=True)
    )

    if total_missing > 0:
        print(f"TOTAL missing embeddings: {total_missing}/{len(base)}")

    return out


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    start_time = time()

    country = "France"
    pa_path = "data/raw/GeoPlant/PresenceAbsenceSurveys/PA_metadata_train.csv"
    po_path = "data/raw/GeoPlant/PresenceOnlyOccurrences/PO_metadata_train.csv"

    subsample_pa = 1  # set to None to disable subsampling
    subsample_po = .2  # set to None to disable subsampling


    # IMPORTANT: keep lon/lat and year in meta so EE calls can be year-aware
    meta_cols_pa = ["lon", "lat", "year", "areaInM2"]
    meta_cols_po = ["lon", "lat", "year"]
    # subsample = 0.1  # set to None to disable subsampling


    # ---- Load sparse PA ----
    Y_pa, X_pa, sp_pa = load_geoplant_dataset_sparse(
        pa_path,
        worldmap_region=country,
        species_min_presence=1,
        meta_cols=meta_cols_pa,
        keep_survey_id=True,
        mainland_only=True,
        subsample=subsample_pa
    )
    print("PA:", Y_pa.shape, "species:", len(sp_pa), "meta rows:", len(X_pa))

    # ---- Load sparse PO ----
    Y_po, X_po, sp_po = load_geoplant_dataset_sparse(
        po_path,
        worldmap_region=country,
        species_min_presence=1,
        meta_cols=meta_cols_po,
        keep_survey_id=True,
        mainland_only=True,
        subsample=subsample_po
    )
    print("PO:", Y_po.shape, "species:", len(sp_po), "meta rows:", len(X_po))

    # ---- Intersect species columns ----
    Y_pa_i, common_species, Y_po_i, _ = intersect_species_columns_sparse(Y_pa, sp_pa, Y_po, sp_po)
    print("Common species:", len(common_species))
    print("PA intersected:", Y_pa_i.shape)
    print("PO intersected:", Y_po_i.shape)

    # ---- Add covariates to meta (year-aware) ----
    # (Do PA and PO separately because their rows are different surveys/points)
    X_pa_cov = add_covariates_geoplant_embedding(X_pa, year_col="year")
    X_po_cov = add_covariates_geoplant_embedding(X_po, year_col="year")

    # ---- Save outputs ----
    outdir = f"data/processed/geoplant/{country.lower()}_sparse_{subsample_pa}_{subsample_po}"
    os.makedirs(outdir, exist_ok=True)

    # Save matrices as .npz (recommended for sparse)
    from scipy.sparse import save_npz
    save_npz(os.path.join(outdir, f"Y_pa_{country.lower()}.npz"), Y_pa_i)
    save_npz(os.path.join(outdir, f"Y_po_{country.lower()}.npz"), Y_po_i)

    # if cols are small enough, can save as CSV too (dense)
    if len(common_species) <= 1000:
        pd.DataFrame.sparse.from_spmatrix(Y_pa_i, columns=common_species).to_csv(
            os.path.join(outdir, f"Y_pa_{country.lower()}.csv"), index=False
        )
        pd.DataFrame.sparse.from_spmatrix(Y_po_i, columns=common_species).to_csv(
            os.path.join(outdir, f"Y_po_{country.lower()}.csv"), index=False
        )

    # Save meta + covariates
    X_pa_cov.to_csv(os.path.join(outdir, f"X_pa_{country.lower()}_covs.csv"), index=False)
    X_po_cov.to_csv(os.path.join(outdir, f"X_po_{country.lower()}_covs.csv"), index=False)

    # Save species column names (aligned to matrix columns)
    pd.Series(common_species).to_csv(os.path.join(outdir, f"species_{country.lower()}.csv"), index=False, header=False)

    # Save covariate names
    cov_cols = [c for c in X_pa_cov.columns if c.startswith("A")]
    pd.Series(cov_cols).to_csv(os.path.join(outdir, f"covariates_{country.lower()}.csv"), index=False, header=False)

    # ---- Example: access first row / a species ----
    print("First PA row nonzeros:", Y_pa_i[0, :].nonzero()[1][:20])
    if common_species:
        sp = common_species[0]
        col = get_species_column(Y_pa_i, common_species, sp)
        print("Example species:", sp, "presence count:", int(col.sum()))

    print(f"Done in {time() - start_time:.1f} seconds.")