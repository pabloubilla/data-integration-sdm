import pandas as pd
import numpy as np
import os
import ee
from tqdm import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt

def region_filter_worldmap(df, region_name, lon_col="lon", lat_col="lat", continental_only=True):
    world = gpd.read_file(
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    )

    region = world[world["NAME"] == region_name]
    if region.empty:
        raise ValueError(f"Region '{region_name}' not found in Natural Earth.")

    # Keep only metropolitan France
    if continental_only and region_name == "France":
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
    exit()

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
    df = df.sample(frac=sample_fraction, random_state=42)
    if region_filter is not None:
        df = df[df["region"] == region_filter].copy()
    if worldmap_region is not None:
        df = region_filter_worldmap(df, worldmap_region, lon_col="lon", lat_col="lat")


    meta_cols = meta_cols or []
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

    pa_df, pa_species_cols = load_geoplant_dataset(pa_data_dir, species_min_presence=20, meta_cols=meta_cols, sample_fraction=0.1, worldmap_region=filter)
    po_df, po_species_cols = load_geoplant_dataset(po_data_dir, species_min_presence=20, meta_cols=meta_cols, sample_fraction=0.02, worldmap_region=filter)

    exit()

    print(f"PA data shape: {pa_df.shape}")
    print(f"PO data shape: {po_df.shape}")


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