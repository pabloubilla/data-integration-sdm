# scripts/preprocess_geoplant.py

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


NON_COVARIATE_COLS = {
    "surveyId",
    "speciesId",
    "predictions",
    "lon",
    "lat",
    "x",
    "y",
    "areaInM2",
    "country",
    "region",
}


DEFAULT_METADATA_COLS = ["lon", "lat", "areaInM2"]


# -------------------------
# basic helpers
# -------------------------

def coerce_int_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col]).copy()
    df[col] = df[col].astype("int64")
    return df


def normalize_lon_lat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure consistent lon/lat names if possible.
    """
    df = df.copy()

    rename = {}
    if "longitude" in df.columns and "lon" not in df.columns:
        rename["longitude"] = "lon"
    if "latitude" in df.columns and "lat" not in df.columns:
        rename["latitude"] = "lat"

    df = df.rename(columns=rename)
    return df


def filter_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """
    Region filtering happens before species vocab creation.

    For France:
    - if country/region column exists, use it
    - otherwise use a rough metropolitan-France lon/lat bounding box

    NOTE:
    This bounding box excludes overseas France.
    Use polygon filtering later if overseas territories matter.
    """
    region = region.lower()

    if region in {"full", "all", "global", "none"}:
        return df.copy()

    df = normalize_lon_lat(df)

    if region == "france":
        # for col in ["country", "region"]:
        #     if col in df.columns:
        #         vals = df[col].astype(str).str.lower()
        #         mask = vals.isin({"france", "fr", "fra", "metropolitan france"})
        #         return df[mask].copy()

        if {"lon", "lat"}.issubset(df.columns):
            lon = pd.to_numeric(df["lon"], errors="coerce")
            lat = pd.to_numeric(df["lat"], errors="coerce")

            mask = (
                lon.between(-5.5, 10.0)
                & lat.between(41.0, 52.0)
            )
            return df[mask].copy()

        raise ValueError(
            "Cannot filter region='france'. Need country/region column or lon/lat columns."
        )

    raise ValueError(f"Unknown region: {region}")


def extract_unique_species_from_column(series: pd.Series) -> np.ndarray:
    return (
        pd.to_numeric(series, errors="coerce")
        .dropna()
        .astype("int64")
        .unique()
    )


def extract_unique_species_from_prediction_strings(series: pd.Series) -> np.ndarray:
    exploded = series.dropna().astype("string").str.split().explode()
    return (
        pd.to_numeric(exploded, errors="coerce")
        .dropna()
        .astype("int64")
        .unique()
    )


def make_species_string_table_from_rows(
    df: pd.DataFrame,
    species_col: str = "speciesId",
    metadata_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Convert row-level species observations into one row per surveyId:

        surveyId | speciesId = "12 45 900" | metadata...

    Used for PO and PA train, where species usually appear one row per observation.
    """
    metadata_cols = metadata_cols or []
    metadata_cols = [c for c in metadata_cols if c in df.columns]

    tmp = df[["surveyId", species_col] + metadata_cols].copy()
    tmp = tmp.dropna(subset=["surveyId", species_col])

    tmp["surveyId"] = pd.to_numeric(tmp["surveyId"], errors="coerce")
    tmp[species_col] = pd.to_numeric(tmp[species_col], errors="coerce")
    tmp = tmp.dropna(subset=["surveyId", species_col]).copy()

    tmp["surveyId"] = tmp["surveyId"].astype("int64")
    tmp[species_col] = tmp[species_col].astype("int64")

    out = (
        tmp.sort_values(["surveyId", species_col])
        .groupby("surveyId")
        .agg(
            {
                species_col: lambda s: " ".join(map(str, sorted(set(s)))),
                **{col: "first" for col in metadata_cols},
            }
        )
        .reset_index()
    )

    return out


def make_species_string_table_from_predictions(
    df: pd.DataFrame,
    prediction_col: str = "predictions",
    metadata_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Convert PA test labels with prediction strings into standard format:

        surveyId | speciesId = "12 45 900" | metadata...
    """
    metadata_cols = metadata_cols or []
    metadata_cols = [c for c in metadata_cols if c in df.columns]

    out = (
        df[["surveyId", prediction_col] + metadata_cols]
        .rename(columns={prediction_col: "speciesId"})
        .dropna(subset=["surveyId"])
        .drop_duplicates(subset=["surveyId"], keep="first")
        .copy()
    )

    out["surveyId"] = pd.to_numeric(out["surveyId"], errors="coerce")
    out = out.dropna(subset=["surveyId"]).copy()
    out["surveyId"] = out["surveyId"].astype("int64")
    out["speciesId"] = out["speciesId"].fillna("").astype("string")

    return out


def remap_species_string(s: str, mapping: dict[int, int]) -> str:
    """
    Convert original species IDs to contiguous class indices.
    Drops species not in mapping.
    """
    if pd.isna(s) or s is None:
        return ""

    s = str(s).strip()
    if not s:
        return ""

    remapped = []
    for tok in s.split():
        sid = int(tok)
        if sid in mapping:
            remapped.append(str(mapping[sid]))

    return " ".join(remapped)


def build_vocab(
    po_df: pd.DataFrame,
    pa_train_df: pd.DataFrame,
    pa_test_df: pd.DataFrame,
    mode: str = "intersection_po_pa",
) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    """
    Build species vocabulary after region filtering.

    Modes:
    - intersection_po_pa:
        species must occur in PO and PA train
    - intersection_po_pa_test:
        species must occur in PO, PA train, and PA test
    - union:
        species from PO, PA train, or PA test
    - pa_only:
        species from PA train and PA test
    """
    po_u = set(extract_unique_species_from_column(po_df["speciesId"]).tolist())
    pa_u = set(extract_unique_species_from_column(pa_train_df["speciesId"]).tolist())

    if "predictions" in pa_test_df.columns:
        te_u = set(
            extract_unique_species_from_prediction_strings(
                pa_test_df["predictions"]
            ).tolist()
        )
    else:
        te_u = set()

    if mode == "intersection_po_pa":
        ids = po_u & pa_u
    elif mode == "intersection_po_pa_test":
        ids = po_u & pa_u & te_u
    elif mode == "union":
        ids = po_u | pa_u | te_u
    elif mode == "pa_only":
        ids = pa_u | te_u
    else:
        raise ValueError(
            "mode must be one of: "
            "'intersection_po_pa', 'intersection_po_pa_test', 'union', 'pa_only'"
        )

    vocab = np.array(sorted(ids), dtype=np.int64)

    species_id_to_idx = {int(sid): int(i) for i, sid in enumerate(vocab)}
    idx_to_species_id = {int(i): int(sid) for i, sid in enumerate(vocab)}

    return vocab, species_id_to_idx, idx_to_species_id


def infer_covariates(df: pd.DataFrame, metadata_cols: list[str]) -> list[str]:
    excluded = set(NON_COVARIATE_COLS) | set(metadata_cols)

    covariates = [
        c for c in df.columns
        if c not in excluded
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    return covariates


def safe_select_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df[cols].copy()


def save_pickle(df: pd.DataFrame, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(df, f)


def save_species_and_covariates(
    merged: pd.DataFrame,
    prefix: str,
    species_dir: Path,
    covariates_dir: Path,
    metadata_cols: list[str],
    covariates: list[str],
) -> dict:
    """
    Save one split as:
    - species/{prefix}_species.csv
    - covariates/{prefix}_covariates.pkl
    """
    metadata_cols = [c for c in metadata_cols if c in merged.columns]

    species_cols = ["surveyId", "speciesId"] + metadata_cols
    cov_cols = ["surveyId"] + metadata_cols + covariates

    species_df = safe_select_columns(merged, species_cols)
    cov_df = safe_select_columns(merged, cov_cols)

    species_path = species_dir / f"{prefix}_species.csv"
    cov_path = covariates_dir / f"{prefix}_covariates.pkl"

    species_df.to_csv(species_path, index=False)
    save_pickle(cov_df, cov_path)

    return {
        f"{prefix}_rows": int(len(merged)),
        f"{prefix}_species_path": str(species_path),
        f"{prefix}_covariates_path": str(cov_path),
    }


# -------------------------
# main preprocessing
# -------------------------

def preprocess_geoplant(
    raw_root: str,
    output_root: str,
    region: str = "france",
    vocab_mode: str = "intersection_po_pa",
    metadata_cols: Optional[list[str]] = None,
    drop_empty_surveys: bool = True,
) -> None:
    metadata_cols = metadata_cols or DEFAULT_METADATA_COLS

    raw_root = Path(raw_root)
    output_root = Path(output_root)

    po_path = raw_root / "PresenceOnlyOccurrences"
    pa_path = raw_root / "PresenceAbsenceSurveys"
    values_path = raw_root / "values"

    processed_root = output_root / region
    species_dir = processed_root / "species"
    covariates_dir = processed_root / "covariates"

    species_dir.mkdir(parents=True, exist_ok=True)
    covariates_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Preprocessing GeoPlant ===")
    print(f"Region     : {region}")
    print(f"Vocab mode : {vocab_mode}")
    print(f"Output     : {processed_root}")

    # -------------------------
    # load raw metadata/labels
    # -------------------------
    po_df = pd.read_csv(po_path / "PO_metadata_train.csv")
    pa_train_df = pd.read_csv(pa_path / "PA_metadata_train.csv")
    pa_test_df = pd.read_csv(pa_path / "test_labels.csv")
    pa_test_metadata = pd.read_csv(pa_path / "PA_metadata_test.csv")

    po_df = normalize_lon_lat(po_df)
    pa_train_df = normalize_lon_lat(pa_train_df)
    pa_test_metadata = normalize_lon_lat(pa_test_metadata)

    pa_test_df = pa_test_df.merge(
        pa_test_metadata,
        on="surveyId",
        how="left",
    )

    po_df = coerce_int_col(po_df, "surveyId")
    pa_train_df = coerce_int_col(pa_train_df, "surveyId")
    pa_test_df = coerce_int_col(pa_test_df, "surveyId")

    # -------------------------
    # filter region BEFORE vocab
    # -------------------------
    n_raw = {
        "po_raw": len(po_df),
        "pa_train_raw": len(pa_train_df),
        "pa_test_raw": len(pa_test_df),
    }


    po_df = filter_region(po_df, region)
    pa_train_df = filter_region(pa_train_df, region)
    pa_test_df = filter_region(pa_test_df, region)

    n_region = {
        "po_region": len(po_df),
        "pa_train_region": len(pa_train_df),
        "pa_test_region": len(pa_test_df),
    }

    print("Rows after region filtering:")
    print(f"  PO       : {n_raw['po_raw']:,} -> {n_region['po_region']:,}")
    print(f"  PA train : {n_raw['pa_train_raw']:,} -> {n_region['pa_train_region']:,}")
    print(f"  PA test  : {n_raw['pa_test_raw']:,} -> {n_region['pa_test_region']:,}")

    if len(po_df) == 0 or len(pa_train_df) == 0 or len(pa_test_df) == 0:
        raise ValueError("At least one split is empty after region filtering.")

    # -------------------------
    # build species vocab
    # -------------------------
    vocab, species_id_to_idx, idx_to_species_id = build_vocab(
        po_df=po_df,
        pa_train_df=pa_train_df,
        pa_test_df=pa_test_df,
        mode=vocab_mode,
    )

    if len(vocab) == 0:
        raise ValueError("Species vocabulary is empty. Check region/vocab_mode.")

    print(f"Num classes: {len(vocab):,}")
    print("First 10 original species IDs:", vocab[:10].tolist())

    np.save(species_dir / "species_vocab.npy", vocab)

    with open(species_dir / "species_id_to_idx.pkl", "wb") as f:
        pickle.dump(species_id_to_idx, f)

    with open(species_dir / "idx_to_species_id.pkl", "wb") as f:
        pickle.dump(idx_to_species_id, f)

    with open(species_dir / "all_species_list_original_ids.txt", "w") as f:
        for sid in vocab.tolist():
            f.write(f"{int(sid)}\n")

    # -------------------------
    # species string tables
    # -------------------------
    po_species_tbl = make_species_string_table_from_rows(
        po_df,
        species_col="speciesId",
        metadata_cols=metadata_cols,
    )

    pa_train_species_tbl = make_species_string_table_from_rows(
        pa_train_df,
        species_col="speciesId",
        metadata_cols=metadata_cols,
    )

    pa_test_species_tbl = make_species_string_table_from_predictions(
        pa_test_df,
        prediction_col="predictions",
        metadata_cols=metadata_cols,
    )

    # remap original species IDs -> class indices
    for tbl in [po_species_tbl, pa_train_species_tbl, pa_test_species_tbl]:
        tbl["speciesId"] = tbl["speciesId"].map(
            lambda s: remap_species_string(s, species_id_to_idx)
        )

    if drop_empty_surveys:
        before = {
            "po": len(po_species_tbl),
            "pa_train": len(pa_train_species_tbl),
            "pa_test": len(pa_test_species_tbl),
        }

        po_species_tbl = po_species_tbl[po_species_tbl["speciesId"].str.len() > 0].copy()
        pa_train_species_tbl = pa_train_species_tbl[
            pa_train_species_tbl["speciesId"].str.len() > 0
        ].copy()
        pa_test_species_tbl = pa_test_species_tbl[
            pa_test_species_tbl["speciesId"].str.len() > 0
        ].copy()

        print("Dropped empty-label surveys after remapping:")
        print(f"  PO       : {before['po']:,} -> {len(po_species_tbl):,}")
        print(f"  PA train : {before['pa_train']:,} -> {len(pa_train_species_tbl):,}")
        print(f"  PA test  : {before['pa_test']:,} -> {len(pa_test_species_tbl):,}")

    # -------------------------
    # load bioclimatic covariates
    # -------------------------
    pa_test_bioclim = coerce_int_col(
        pd.read_csv(values_path / "PA-test-bioclimatic-average.csv"),
        "surveyId",
    )
    pa_train_bioclim = coerce_int_col(
        pd.read_csv(values_path / "PA-train-bioclimatic-average.csv"),
        "surveyId",
    )
    po_train_bioclim = coerce_int_col(
        pd.read_csv(values_path / "PO-train-bioclimatic-average.csv"),
        "surveyId",
    )

    # -------------------------
    # merge labels/metadata with covariates
    # -------------------------
    po_merged = po_species_tbl.merge(po_train_bioclim, on="surveyId", how="inner")
    pa_train_merged = pa_train_species_tbl.merge(pa_train_bioclim, on="surveyId", how="inner")
    pa_test_merged = pa_test_species_tbl.merge(pa_test_bioclim, on="surveyId", how="inner")

    print("Merged shapes:")
    print("  PO       :", po_merged.shape)
    print("  PA train :", pa_train_merged.shape)
    print("  PA test  :", pa_test_merged.shape)

    # -------------------------
    # infer covariates robustly
    # -------------------------
    covariates_po = set(infer_covariates(po_merged, metadata_cols))
    covariates_pa_train = set(infer_covariates(pa_train_merged, metadata_cols))
    covariates_pa_test = set(infer_covariates(pa_test_merged, metadata_cols))

    covariates = sorted(
        covariates_po
        & covariates_pa_train
        & covariates_pa_test
    )

    if len(covariates) == 0:
        raise ValueError("No common covariates found across PO/PA train/PA test.")

    print(f"Num common covariates: {len(covariates)}")
    print("First 10 covariates:", covariates[:10])

    # -------------------------
    # save splits
    # -------------------------
    split_stats = {}
    split_stats.update(
        save_species_and_covariates(
            po_merged,
            "po",
            species_dir,
            covariates_dir,
            metadata_cols,
            covariates,
        )
    )
    split_stats.update(
        save_species_and_covariates(
            pa_train_merged,
            "pa_train",
            species_dir,
            covariates_dir,
            metadata_cols,
            covariates,
        )
    )
    split_stats.update(
        save_species_and_covariates(
            pa_test_merged,
            "pa_test",
            species_dir,
            covariates_dir,
            metadata_cols,
            covariates,
        )
    )

    # -------------------------
    # metadata
    # -------------------------
    metadata = {
        "dataset": "GeoPlant",
        "region": region,
        "vocab_mode": vocab_mode,
        "drop_empty_surveys": drop_empty_surveys,
        "num_classes": int(len(vocab)),
        "num_covariates": int(len(covariates)),
        "covariates": covariates,
        "metadata_cols": metadata_cols,
        **n_raw,
        **n_region,
        "po_rows_final": int(len(po_merged)),
        "pa_train_rows_final": int(len(pa_train_merged)),
        "pa_test_rows_final": int(len(pa_test_merged)),
        **split_stats,
    }

    with open(processed_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved processed dataset:")
    print(processed_root)
    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-root",
        type=str,
        default="data/raw/GeoPlant",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/processed/GeoPlant",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="france",
        choices=["france", "full", "all"],
    )
    parser.add_argument(
        "--vocab-mode",
        type=str,
        default="intersection_po_pa",
        choices=[
            "intersection_po_pa",
            "intersection_po_pa_test",
            "union",
            "pa_only",
        ],
    )
    parser.add_argument(
        "--keep-empty-surveys",
        action="store_true",
        help="Keep surveys whose labels become empty after species remapping.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    region = "full" if args.region == "all" else args.region

    preprocess_geoplant(
        raw_root=args.raw_root,
        output_root=args.output_root,
        region=region,
        vocab_mode=args.vocab_mode,
        metadata_cols=DEFAULT_METADATA_COLS,
        drop_empty_surveys=not args.keep_empty_surveys,
    )


if __name__ == "__main__":
    main()