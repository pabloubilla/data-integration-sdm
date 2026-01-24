import pandas as pd
import numpy as np
import os
import ee
from tqdm import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt

from typing import Optional, List

import duckdb
import numpy as np
import pandas as pd
import geopandas as gpd
from functools import lru_cache
from shapely.prepared import prep
from scipy.sparse import coo_matrix


NE_URL = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"


@lru_cache(maxsize=1)
def _load_natural_earth_world():
    # cached in-process so you don't re-download every call
    return gpd.read_file(NE_URL)[["NAME", "geometry"]]


def region_filter_worldmap_v2(
    df: pd.DataFrame,
    region_name: str,
    lon_col: str = "lon",
    lat_col: str = "lat",
    mainland_only: bool = True,
    plot: bool = False,
):
    """
    Filters df to points contained in the Natural Earth country polygon `region_name`.
    If mainland_only=True and the country is MultiPolygon, keeps the largest polygon
    (pragmatic way to avoid overseas territories for countries like France).
    """
    world = _load_natural_earth_world()

    region = world.loc[world["NAME"] == region_name]
    if region.empty:
        raise ValueError(f"Region '{region_name}' not found in Natural Earth.")

    geom = region.geometry.iloc[0]

    # Keep only the largest polygon (works well for "metropolitan-ish" filtering)
    if mainland_only and geom.geom_type == "MultiPolygon":
        # area is better measured in a projected CRS
        geom_m = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
        largest = max(geom_m.geoms, key=lambda g: g.area)
        geom = gpd.GeoSeries([largest], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]

    # Quick bbox prefilter (cheap and speeds up contains())
    minx, miny, maxx, maxy = geom.bounds
    bbox_mask = (
        (df[lon_col] >= minx) & (df[lon_col] <= maxx) &
        (df[lat_col] >= miny) & (df[lat_col] <= maxy)
    )
    df2 = df.loc[bbox_mask].copy()
    if df2.empty:
        return df2

    # Exact point-in-polygon
    pts = gpd.GeoSeries(gpd.points_from_xy(df2[lon_col], df2[lat_col]), crs="EPSG:4326")
    prepared = prep(geom)
    inside = pts.apply(prepared.contains).to_numpy()

    out = df2.loc[inside].copy()

    if plot:
        import matplotlib.pyplot as plt

        region_gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")
        pts_gdf = gpd.GeoDataFrame(df2, geometry=pts, crs="EPSG:4326")

        fig, ax = plt.subplots()
        region_gdf.plot(ax=ax)
        pts_gdf.loc[inside].plot(ax=ax, color="red", markersize=5)
        ax.set_title(f"Points within {region_name} (n={len(out)})")
        plt.show()

        print(f"{len(out)} out of {len(df)} points are within {region_name}.")

    return out


def load_geoplant_dataset_sparse(
    filepath: str,
    region_filter: Optional[str] = None,
    worldmap_region: Optional[str] = None,
    species_min_presence: int = 10,
    sample_fraction: float = 1.0,
    meta_cols: Optional[List[str]] = None,
    keep_survey_id: bool = True,
    species_prefix: str = "sp_",
    lon_col: str = "lon",
    lat_col: str = "lat",
    mainland_only: bool = True,
):

    """
    Returns:
      X: scipy.sparse.csr_matrix (n_surveys x n_species) with 0/1 presences
      meta: pandas.DataFrame with row-aligned metadata (surveyId + meta_cols)
      species_cols: list of species column names (aligned to X columns)
    """
    meta_cols = meta_cols or []
    index_cols = (["surveyId"] if keep_survey_id else []) + meta_cols
    if not index_cols:
        raise ValueError("Need at least one index column (surveyId and/or meta_cols).")

    con = duckdb.connect()

    # --- If we have a worldmap_region, load its geometry bounds so DuckDB can bbox-filter ---
    bbox_sql = ""
    if worldmap_region is not None:
        world = _load_natural_earth_world()
        region = world.loc[world["NAME"] == worldmap_region]
        if region.empty:
            raise ValueError(f"Region '{worldmap_region}' not found in Natural Earth.")
        geom = region.geometry.iloc[0]
        if mainland_only and geom.geom_type == "MultiPolygon":
            geom_m = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
            largest = max(geom_m.geoms, key=lambda g: g.area)
            geom = gpd.GeoSeries([largest], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
        minx, miny, maxx, maxy = geom.bounds
        bbox_sql = f" AND {lon_col} BETWEEN {minx} AND {maxx} AND {lat_col} BETWEEN {miny} AND {maxy} "

    # --- Build WHERE clause ---
    where = "WHERE speciesId IS NOT NULL"
    if region_filter is not None:
        where += f" AND region = '{region_filter}'"
    if bbox_sql:
        where += bbox_sql

    # Only read needed columns from CSV
    cols_sql = ", ".join(index_cols + ["speciesId", lon_col, lat_col] if worldmap_region else index_cols + ["speciesId"])
    base = f"""
        SELECT {cols_sql}
        FROM read_csv_auto('{filepath}')
        {where}
    """

    # if sample_fraction < 1.0:
    #     base = f"SELECT * FROM ({base}) USING SAMPLE {sample_fraction*100:.6f}%"

    # Deduplicate survey×species presences
    dedup_cols = ", ".join(index_cols + ["speciesId"] + ([lon_col, lat_col] if worldmap_region else []))
    dedup = f"SELECT DISTINCT {dedup_cols} FROM ({base})"

    pairs = con.execute(dedup).fetchdf()

    if pairs.empty:
        from scipy.sparse import csr_matrix
        return csr_matrix((0, 0), dtype=np.uint8), pd.DataFrame(columns=index_cols), []

    # If polygon filtering is requested, apply exact GeoPandas containment now (on reduced subset)
    if worldmap_region is not None:
        pairs = region_filter_worldmap_v2(
            pairs,
            worldmap_region,
            lon_col=lon_col,
            lat_col=lat_col,
            mainland_only=mainland_only,
            plot=False,
        )
        if pairs.empty:
            from scipy.sparse import csr_matrix
            return csr_matrix((0, 0), dtype=np.uint8), pd.DataFrame(columns=index_cols), []

    # Now we only need index_cols + speciesId
    pairs = pairs[index_cols + ["speciesId"]]

    # Build meta rows (unique surveys)
    Y = pairs[index_cols].drop_duplicates().reset_index(drop=True)
    Y["_row_id"] = np.arange(len(Y), dtype=np.int64)

    pairs = pairs.merge(Y, on=index_cols, how="left", validate="many_to_one")

    # Map speciesId -> col_id
    species = pd.Index(pairs["speciesId"].unique()).sort_values()
    species_to_col = pd.Series(np.arange(len(species), dtype=np.int64), index=species)

    row_id = pairs["_row_id"].to_numpy()
    col_id = species_to_col.loc[pairs["speciesId"]].to_numpy()

    # Filter rare species by presence count
    counts = np.bincount(col_id, minlength=len(species))
    keep = counts >= species_min_presence
    if not keep.all():
        kept_cols = np.flatnonzero(keep)
        remap = -np.ones(len(species), dtype=np.int64)
        remap[kept_cols] = np.arange(len(kept_cols), dtype=np.int64)

        sel = keep[col_id]
        row_id = row_id[sel]
        col_id = remap[col_id[sel]]
        species = species[kept_cols]

    data = np.ones(len(row_id), dtype=np.uint8)
    X = coo_matrix((data, (row_id, col_id)), shape=(len(Y), len(species)), dtype=np.uint8).tocsr()

    species_cols = [f"{species_prefix}{int(s)}" for s in species]

    # meta = meta.drop(columns=["_row_id"])
    return X, Y, species_cols


def region_filter_worldmap(df, region_name, lon_col="lon", lat_col="lat", continental_only=True):
    world = gpd.read_file(
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    )

    region = world[world["NAME"] == region_name]
    if region.empty:
        raise ValueError(f"Region '{region_name}' not found in Natural Earth.")

    # Keep only metropolitan France
    if continental_only and region_name in ["France", "Denmark", "Spain", "Italy", 
                                            "Portugal", "Greece", "Netherlands", "Belgium", 
                                            "Germany", "Sweden", "Norway", "Finland"]:
        region = region[region["CONTINENT"] == "Europe"]

    pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )

    region = region.to_crs(pts.crs)
    mask = pts.within(region.geometry.iloc[0])

    # plot to check (single figure)
    fig, ax = plt.subplots()
    region.plot(ax=ax)
    pts.loc[mask].plot(ax=ax, color="red", markersize=5)

    how_many = int(mask.sum())
    ax.set_title(f"Points within {region_name} (n={how_many})")
    plt.show()

    # OUT OF THE TOTAL POINTS HOW MANY ARE IN THE REGION
    total_points = len(df)
    print(f"{how_many} out of {total_points} points are within {region_name}.")

    return df.loc[mask.values].copy()
    

def load_geoplant_dataset(
    filepath,
    region_filter=None,
    worldmap_region=None,
    species_min_presence=10,
    sample_fraction=1.0,
    meta_cols=None,
    keep_survey_id=True,
    species_prefix="sp_",
):
    df = pd.read_csv(filepath)
    if region_filter is not None:
        df = df[df["region"] == region_filter].copy()
    if worldmap_region is not None:
        df = region_filter_worldmap(df, worldmap_region, lon_col="lon", lat_col="lat")


    # meta_cols = meta_cols or []
    index_cols = (["surveyId"] if keep_survey_id else []) + meta_cols

    # Pivot to presence/absence
    pivot = (
        df.assign(present=1)
          .pivot_table(
              index=index_cols,
              columns="speciesId",
              values="present",
              fill_value=0,
              aggfunc="max"
          )
    )

    # Species counts + filtering
    species_counts = pivot.sum(axis=0)
    selected_species = species_counts[species_counts >= species_min_presence].index

    pivot = pivot.loc[:, selected_species]

    # Turn into a normal tabular dataframe (meta cols become columns)
    data = pivot.reset_index()

    # Keep species column names for later (after we rename them)
    # Rename species columns to strings with a prefix to avoid int column pitfalls
    species_cols = [f"{species_prefix}{int(s)}" for s in pivot.columns]
    rename_map = {old: new for old, new in zip(pivot.columns, species_cols)}
    data = data.rename(columns=rename_map)

    return data, species_cols

def add_covariates_geoplant(df, lon_col="lon", lat_col="lat", year=2023):
    ee.Initialize(project='alpha-earth-test-483513')

    dataset = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    bands = [f"A{i:02d}" for i in range(64)]

    def fetch_embedding(row):
        lon, lat = row[lon_col], row[lat_col]
        pt = ee.Geometry.Point([lon, lat])

        img = (dataset
               .filterDate(f"{year}-01-01", f"{year+1}-01-01")
               .filterBounds(pt)
               .first())

        d = img.select(bands).reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=pt,
            scale=10,
            maxPixels=1e9
        ).getInfo()

        embedding = [d[b] for b in bands]
        return embedding

    tqdm.pandas()
    embeddings = df.progress_apply(fetch_embedding, axis=1)
    embedding_df = pd.DataFrame(embeddings.tolist(), columns=bands)

    df_with_covs = pd.concat([df.reset_index(drop=True), embedding_df], axis=1)
    return df_with_covs





def add_covariates_geoplant_fast(df, year=2023, scale=10):
    """
    Adds 64-dim annual satellite embedding covariates (A00..A63) to a dataframe
    containing 'lat' and 'lon' columns.

    Mask-aware:
      - Handles points with masked/no-data pixels (they may return no sample)
      - Reports how many points did not get embeddings
      - Guarantees output has all band columns (filled with NA where missing)
    """
    ee.Initialize(project="alpha-earth-test-483513")

    bands = [f"A{i:02d}" for i in range(64)]
    dataset = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    # Keep original row order via an id derived from index (guaranteed to exist)
    base = df.reset_index(drop=False).rename(columns={"index": "row_id"}).copy()

    # Select the annual image once
    img = (
        dataset
        .filterDate(f"{year}-01-01", f"{year+1}-01-01")
        .first()
        .select(bands)
    )

    # Build a FeatureCollection of all points with row_id property
    feats = [
        ee.Feature(
            ee.Geometry.Point([r.lon, r.lat]),
            {"row_id": int(r.row_id)}
        )
        for r in base[["row_id", "lon", "lat"]].itertuples(index=False)
    ]
    fc = ee.FeatureCollection(feats)

    # Sample all points in one server-side call
    sampled = img.sampleRegions(
        collection=fc,
        properties=["row_id"],
        scale=scale,
        geometries=False
    )

    # Pull results (may be empty or missing row_id if everything is masked)
    info = sampled.getInfo()
    features = info.get("features", [])
    props = [f.get("properties", {}) for f in features]
    emb_df = pd.DataFrame(props)

    # Ensure row_id exists so merge is safe
    if "row_id" not in emb_df.columns:
        emb_df["row_id"] = pd.Series(dtype=int)

    # Ensure all band columns exist (masked pixels -> missing keys)
    for b in bands:
        if b not in emb_df.columns:
            emb_df[b] = pd.NA

    # Detection: how many original rows didn't get embeddings
    got = emb_df["row_id"].nunique(dropna=True)
    total = len(base)
    missing = total - got
    if missing > 0:
        print(f"⚠️ {missing}/{total} points had no embedding (masked / no-data / outside coverage).")

    # Merge back (safe even if some samples are missing)
    out = (
        base
        .merge(emb_df[["row_id"] + bands], on="row_id", how="left")
        .drop(columns=["row_id"])
        .reset_index(drop=True)
    )

    return out



def add_covariates_geoplant_fast(
    df,
    year=2023,
    scale=10,
    chunk_size=2000,
    project="alpha-earth-test-483513",
):
    """
    Fast version for tiled ImageCollections like GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL.

    - Assumes df has columns: 'lon', 'lat'
    - Preserves row order via row_id
    - Uses filterBounds(chunk_geometry).mosaic() to match your per-point filterBounds(pt).first()
    - Chunked to avoid huge geometries / getInfo payloads
    - Reports how many points got no embedding (masked / outside coverage)
    """
    ee.Initialize(project=project)

    bands = [f"A{i:02d}" for i in range(64)]
    dataset = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate(
        f"{year}-01-01", f"{year+1}-01-01"
    )

    base = df.reset_index(drop=False).rename(columns={"index": "row_id"}).copy()

    all_parts = []
    total_missing = 0

    for start in range(0, len(base), chunk_size):
        part = base.iloc[start : start + chunk_size][["row_id", "lon", "lat"]].copy()

        feats = [
            ee.Feature(
                ee.Geometry.Point([r.lon, r.lat]),
                {"row_id": int(r.row_id)}
            )
            for r in part.itertuples(index=False)
        ]
        fc = ee.FeatureCollection(feats)

        # IMPORTANT: match your slow logic
        # Find all tiles intersecting these points, then mosaic into one image
        chunk_img = (
            dataset
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

        # Guarantee columns exist
        if "row_id" not in emb_df.columns:
            emb_df["row_id"] = pd.Series(dtype=int)
        for b in bands:
            if b not in emb_df.columns:
                emb_df[b] = pd.NA

        got = emb_df["row_id"].nunique(dropna=True)
        missing = len(part) - got
        total_missing += missing
        if missing > 0:
            print(f"⚠️ chunk {start}:{start+len(part)} -> {missing}/{len(part)} points missing (masked/outside coverage)")

        all_parts.append(emb_df[["row_id"] + bands])

    emb_all = pd.concat(all_parts, ignore_index=True)

    out = (
        base
        .merge(emb_all, on="row_id", how="left")
        .drop(columns=["row_id"])
        .reset_index(drop=True)
    )

    if total_missing > 0:
        print(f"⚠️ TOTAL missing embeddings: {total_missing}/{len(base)}")

    return out



# intersect datasets
def intersect_datasets(df1, df2):
    common_cols = df1.columns.intersection(df2.columns)
    df1_filtered = df1[common_cols]
    df2_filtered = df2[common_cols]
    return df1_filtered, df2_filtered

if __name__ == "__main__":

    filter = "France"
    pa_data_dir = "data/raw/GeoPlant/PresenceAbsenceSurveys/PA_metadata_train.csv"
    po_data_dir = "data/raw/GeoPlant/PresenceOnlyOccurrences/PO_metadata_train.csv"

    meta_cols = [
        "lon", "lat"
    ]

    # pa_df, pa_species_cols = load_geoplant_dataset(pa_data_dir, species_min_presence=20, meta_cols=meta_cols, sample_fraction=1, worldmap_region=filter)
    # po_df, po_species_cols = load_geoplant_dataset(po_data_dir, species_min_presence=20, meta_cols=meta_cols, sample_fraction=1, worldmap_region=filter)


    # print(f"PA data shape: {pa_df.shape}")
    # print(f"PO data shape: {po_df.shape}")
    # exit()
    import time
    start = time.time()
    pa_df, meta, pa_species_cols = load_geoplant_dataset_sparse(
        pa_data_dir,
        region_filter=None,
        worldmap_region=filter,
        species_min_presence=20,
        sample_fraction=1.0,
        meta_cols=meta_cols,
        keep_survey_id=True,
        species_prefix="sp_",
    )
    print(pa_df)
    print(meta)
    print(pa_species_cols) 

    # access the column of first species
    print(pa_df[0,:].toarray())

    exit()
    print(f"Loaded PA sparse in {time.time() - start:.2f} seconds")
    print(f"PA data shape: {pa_df.shape}")
    print(f"Number of PA species columns: {len(pa_species_cols)}")

    start = time.time()
    po_df, _, po_species_cols = load_geoplant_dataset_sparse(
        po_data_dir,
        region_filter=None,
        worldmap_region=filter,
        species_min_presence=20,
        sample_fraction=1.0,
        meta_cols=meta_cols,
        keep_survey_id=True,
        species_prefix="sp_",
    )
    print(f"Loaded PO sparse in {time.time() - start:.2f} seconds")
    print(f"PO data shape: {po_df.shape}")
    print(f"Number of PO species columns: {len(po_species_cols)}")
    print(f"Total time for loading sparse: {time.time() - start:.2f} seconds")

    # data types for pa_df and po_df
    print(f"PA data types:\n{pa_df.dtypes}")
    print(f"PO data types:\n{po_df.dtypes}")

    exit()
    
    # start = time.time()
    # pa_df, pa_species_cols = load_geoplant_dataset(pa_data_dir, species_min_presence=20, meta_cols=meta_cols, sample_fraction=1, worldmap_region=filter)
    # print(f"Loaded PA non-sparse in {time.time() - start:.2f} seconds")
    # print(f"PA data shape (non-sparse): {pa_df.shape}")
    # print(f"Number of PA species columns (non-sparse): {len(pa_species_cols)}")
    # exit()


    pa_df_int, po_df_int = intersect_datasets(pa_df, po_df)

    # after intersect print shapes
    print(f"PA data shape after intersect: {pa_df_int.shape}")
    print(f"PO data shape after intersect: {po_df_int.shape}")

    # make outdir data/processed/geoplant
    # modify this part for different experiments (it could be a more standarized name)
    outdir = "data/processed/geoplant"
    os.makedirs(outdir, exist_ok=True)
    pa_df_int.to_csv(os.path.join(outdir, f"geoplant_pa_{filter.lower()}.csv"), index=False)
    po_df_int.to_csv(os.path.join(outdir, f"geoplant_po_{filter.lower()}.csv"), index=False) 


    # add covariates
    pa_df_covs = add_covariates_geoplant_fast(pa_df_int, year=2023)
    po_df_covs = add_covariates_geoplant_fast(po_df_int, year=2023)
    # pa_df_covs = add_covariates_geoplant(pa_df_int, lon_col
    #                                          ="lon", lat_col="lat", year=2023)
    # po_df_covs = add_covariates_geoplant(po_df_int, lon_col="lon", lat_col="lat", year=2023)

    pa_df_covs.to_csv(os.path.join(outdir, f"geoplant_pa_{filter.lower()}_withcovs.csv"), index=False)
    po_df_covs.to_csv(os.path.join(outdir, f"geoplant_po_{filter.lower()}_withcovs.csv"), index=False)