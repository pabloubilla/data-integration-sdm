import os
import pickle
import numpy as np
import pandas as pd


USE_INTERSECTION = True  # True = intersection, False = union

def coerce_int_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col]).copy()
    df[col] = df[col].astype("int64")
    return df


def extract_unique_species_from_column(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").dropna().astype("int64").unique()


def extract_unique_species_from_prediction_strings(series: pd.Series) -> np.ndarray:
    exploded = series.dropna().astype("string").str.split().explode()
    return pd.to_numeric(exploded, errors="coerce").dropna().astype("int64").unique()


def make_species_string_table_from_rows(df: pd.DataFrame, species_col: str = "speciesId",
                                        metadata_cols: list[str] = None) -> pd.DataFrame:
    # keep metadata that exists already
    metadata_cols = [col for col in metadata_cols if col in df.columns.to_list()] if metadata_cols is not None else []
    tmp = df[["surveyId", species_col] + metadata_cols].copy()
    tmp = tmp.dropna(subset=["surveyId", species_col])
    tmp["surveyId"] = pd.to_numeric(tmp["surveyId"], errors="coerce")
    tmp[species_col] = pd.to_numeric(tmp[species_col], errors="coerce")
    tmp = tmp.dropna(subset=["surveyId", species_col]).copy()
    tmp["surveyId"] = tmp["surveyId"].astype("int64")
    tmp[species_col] = tmp[species_col].astype("int64")

    # out = (
    #     tmp.sort_values(["surveyId", species_col])
    #        .groupby("surveyId")[[species_col] + metadata_cols]
    #        .apply(lambda s: " ".join(map(str, s.tolist())))
    #        .reset_index(name="speciesId")
    # )

    out = (
        tmp.sort_values(["surveyId", species_col])
        .groupby("surveyId")
        .agg({
            species_col: lambda s: " ".join(map(str, s)),
            **{col: "first" for col in metadata_cols}
        })
        .reset_index()
    )


    return out


def remap_species_string(s: str, mapping: dict[int, int]) -> str:
    if pd.isna(s) or s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
    # keep only ids that exist in mapping (important for intersection mode)
    remapped = []
    for tok in s.split():
        sid = int(tok)
        if sid in mapping:
            remapped.append(str(mapping[sid]))
    return " ".join(remapped)


def build_vocab(po_df: pd.DataFrame, pa_train_df: pd.DataFrame, pa_test_df: pd.DataFrame, use_intersection: bool):
    po_u = set(extract_unique_species_from_column(po_df["speciesId"]).tolist())
    pa_u = set(extract_unique_species_from_column(pa_train_df["speciesId"]).tolist())
    te_u = set(extract_unique_species_from_prediction_strings(pa_test_df["predictions"]).tolist())


    if False: # to test using all PA
        ids  = pa_u | te_u
    elif use_intersection:
        # ids = po_u & pa_u & te_u
        ids = po_u & pa_u 
    else:
        ids = po_u | pa_u | te_u

    vocab = np.array(sorted(ids), dtype=np.int64)
    species_id_to_idx = {int(sid): int(i) for i, sid in enumerate(vocab)}
    idx_to_species_id = {int(i): int(sid) for i, sid in enumerate(vocab)}
    return vocab, species_id_to_idx, idx_to_species_id

def make_metadata_table(df: pd.DataFrame, metadata_cols: list[str]) -> pd.DataFrame:
    # keep surveyId + requested metadata columns
    # missing metadata columns are created automatically and filled with NaN
    cols = ["surveyId"] + metadata_cols
    meta = df.reindex(columns=cols).copy()

    meta = meta.dropna(subset=["surveyId"]).copy()
    meta["surveyId"] = pd.to_numeric(meta["surveyId"], errors="coerce")
    meta = meta.dropna(subset=["surveyId"]).copy()
    meta["surveyId"] = meta["surveyId"].astype("int64")

    # one row per surveyId
    meta = meta.drop_duplicates(subset=["surveyId"], keep="first").copy()
    return meta

def save_split(df_merged: pd.DataFrame, prefix: str, metadata_cols: list, output_path_species: str, output_path_covariates: str):
    metadata_cols = [col for col in metadata_cols if col in df_merged.columns.to_list()] if metadata_cols is not None else []
    species_df = df_merged[["surveyId", "speciesId"] + metadata_cols].copy()
    cov_df = df_merged.drop(columns=["speciesId"]).copy()

    species_csv_path = os.path.join(output_path_species, f"{prefix}_species.csv")
    cov_pkl_path = os.path.join(output_path_covariates, f"{prefix}_covariates.pkl")

    species_df.to_csv(species_csv_path, index=False)
    with open(cov_pkl_path, "wb") as f:
        pickle.dump(cov_df, f)

    print(f"Saved {prefix}: {species_csv_path} | {cov_pkl_path}")

def main():

    metadata_cols = ["lon", "lat", 'areaInM2']

    data_path = "data/raw/GeoPlant"
    po_path = os.path.join(data_path, "PresenceOnlyOccurrences")
    pa_path = os.path.join(data_path, "PresenceAbsenceSurveys")
    values_path = os.path.join(data_path, "values")

    output_path_species = "data/processed/GeoPlant/full_data/"
    output_path_covariates = "data/processed/GeoPlant/climatic/"
    os.makedirs(output_path_species, exist_ok=True)
    os.makedirs(output_path_covariates, exist_ok=True)

    po_df = pd.read_csv(os.path.join(po_path, "PO_metadata_train.csv"))
    pa_train_df = pd.read_csv(os.path.join(pa_path, "PA_metadata_train.csv"))
    pa_test_df = pd.read_csv(os.path.join(pa_path, "test_labels.csv"))
    pa_test_metadata = pd.read_csv(os.path.join(pa_path, "PA_metadata_test.csv"))
    # merge metadata into pa_test_df
    pa_test_df = pa_test_df.merge(pa_test_metadata, on="surveyId", how="left")

    po_df = coerce_int_col(po_df, "surveyId")
    pa_train_df = coerce_int_col(pa_train_df, "surveyId")
    pa_test_df = coerce_int_col(pa_test_df, "surveyId")

    vocab, species_id_to_idx, idx_to_species_id = build_vocab(
        po_df, pa_train_df, pa_test_df, use_intersection=USE_INTERSECTION
    )
    C = len(vocab)
    mode = "INTERSECTION" if USE_INTERSECTION else "UNION"
    print(f"Species vocab mode: {mode}")
    print("Num classes (C):", C)
    print("First 10 original species IDs in vocab:", vocab[:10].tolist())

    np.save(os.path.join(output_path_species, "species_vocab.npy"), vocab)
    with open(os.path.join(output_path_species, "species_id_to_idx.pkl"), "wb") as f:
        pickle.dump(species_id_to_idx, f)
    with open(os.path.join(output_path_species, "idx_to_species_id.pkl"), "wb") as f:
        pickle.dump(idx_to_species_id, f)

    # TODO: review this part, metadata filter is not working
    po_species_tbl = make_species_string_table_from_rows(po_df, "speciesId", metadata_cols=metadata_cols)
    pa_train_species_tbl = make_species_string_table_from_rows(pa_train_df, "speciesId", metadata_cols=metadata_cols)


    metadata_cols_pa_test = [col for col in metadata_cols if col in pa_test_df.columns.to_list()]
    pa_test_species_tbl = (
        pa_test_df[["surveyId", "predictions"] + metadata_cols_pa_test]
        .rename(columns={"predictions": "speciesId"})
        .dropna(subset=["surveyId"])
        .drop_duplicates(subset=["surveyId"], keep="first")
        .copy()
    )
    pa_test_species_tbl["surveyId"] = pa_test_species_tbl["surveyId"].astype("int64")
    pa_test_species_tbl["speciesId"] = pa_test_species_tbl["speciesId"].astype("string")


    # # also make common MetaData: surveyId -> metadata_cols
    # ### add here (TODO: review for PA test as it might be a different dataset that contains the metadata)
    # po_metadata_tbl = make_metadata_table(po_df, metadata_cols)
    # pa_train_metadata_tbl = make_metadata_table(pa_train_df, metadata_cols)
    # pa_test_metadata_tbl = make_metadata_table(pa_test_df, metadata_cols)


    

    # Remap; intersection mode will DROP ids not in vocab
    po_species_tbl["speciesId"] = po_species_tbl["speciesId"].map(lambda s: remap_species_string(s, species_id_to_idx))
    pa_train_species_tbl["speciesId"] = pa_train_species_tbl["speciesId"].map(lambda s: remap_species_string(s, species_id_to_idx))
    pa_test_species_tbl["speciesId"] = pa_test_species_tbl["speciesId"].map(lambda s: remap_species_string(s, species_id_to_idx))

    # Optionally drop surveys that became empty after intersection filtering
    # (you can comment this out if you want to keep them with empty labels)
    po_species_tbl = po_species_tbl[po_species_tbl["speciesId"].str.len() > 0]
    pa_train_species_tbl = pa_train_species_tbl[pa_train_species_tbl["speciesId"].str.len() > 0]
    pa_test_species_tbl = pa_test_species_tbl[pa_test_species_tbl["speciesId"].str.len() > 0]

    pa_test_bioclim = coerce_int_col(pd.read_csv(os.path.join(values_path, "PA-test-bioclimatic-average.csv")), "surveyId")
    pa_train_bioclim = coerce_int_col(pd.read_csv(os.path.join(values_path, "PA-train-bioclimatic-average.csv")), "surveyId")
    po_train_bioclim = coerce_int_col(pd.read_csv(os.path.join(values_path, "PO-train-bioclimatic-average.csv")), "surveyId")

    po_merged = po_species_tbl.merge(po_train_bioclim, on="surveyId", how="inner")
    pa_train_merged = pa_train_species_tbl.merge(pa_train_bioclim, on="surveyId", how="inner")
    pa_test_merged = pa_test_species_tbl.merge(pa_test_bioclim, on="surveyId", how="inner")

    print("Merged shapes:")
    print("PO:", po_merged.shape)
    print("PA train:", pa_train_merged.shape)
    print("PA test:", pa_test_merged.shape)


    save_split(po_merged, "po", metadata_cols, output_path_species, output_path_covariates)
    save_split(pa_train_merged, "pa_train", metadata_cols, output_path_species, output_path_covariates)
    save_split(pa_test_merged, "pa_test", metadata_cols, output_path_species, output_path_covariates)

    txt_path = os.path.join(output_path_species, "all_species_list_original_ids.txt")
    with open(txt_path, "w") as f:
        for sid in vocab.tolist():
            f.write(f"{int(sid)}\n")
    print("Saved original-ID vocab list ->", txt_path)


if __name__ == "__main__":
    main()