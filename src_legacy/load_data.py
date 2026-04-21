import os
from typing import List, Tuple, Dict
import pandas as pd
from utils import safe_reindex_columns
import numpy as np
from scipy.sparse import load_npz

def build_paths_nceas(region: str, group_filter: str) -> Tuple[str, str, str]:
    ## this can be useful if we aggregate groups
    if region == 'NSW':
        data_state_env_and_pa = 'processed'
    else:
        data_state_env_and_pa = 'raw'
    po_path = os.path.join("data", "processed", "NCEAS", "Records", "train_po", f"{region}train_po{group_filter}.csv")
    pa_path = os.path.join("data", data_state_env_and_pa, "NCEAS", "Records", "test_pa", f"{region}test_pa{group_filter}.csv")
    env_path = os.path.join("data", data_state_env_and_pa, "NCEAS", "Records", "test_env", f"{region}test_env{group_filter}.csv")
    return po_path, pa_path, env_path

def load_po_pa_nceas(region: str, group_filter: str, add_po_var: bool, keep_xy: bool = False,
                     categorical_vars: list = ['ontveg', 'vegsys'], index_col: list = ['x', 'y']) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]    :
    """
    Load PO (aggregated by (x,y)) and PA (full table with covariates & labels),
    return aligned X/Y for PO and PA, with unified species columns and covariates.
    """
    po_path, pa_path, env_path = build_paths_nceas(region, group_filter)

    # Presence-only (PO) records
    df_po = pd.read_csv(po_path)

    # PA labels + environmental covariates
    df_pa = pd.read_csv(pa_path)
    df_env = pd.read_csv(env_path)

    # covariates from 5th col onward in env
    covariates = df_env.columns[4:].tolist()

    # Species 
    species = sorted(df_po["spid"].unique().tolist()) 

    # final lists of covariates
    covs = covariates.copy()
    if add_po_var and "PO" not in covs:
        covs.append("PO")
    if keep_xy:
        covs = ["x", "y"] + covs


    # --- One-hot encode categorical variables with a shared category universe ---
    # Only consider categorical vars that actually exist in the data
    cat_vars_present = [c for c in (categorical_vars or []) if c in covariates]

    # # Helper: add one-hot columns for a single variable given a fixed, shared category list
    # def _one_hot_inplace(df, var, categories, prefix=None):
    #     pref = var if prefix is None else prefix
    #     for cat in categories:
    #         col = f"{pref}__{cat}"
    #         df[col] = (df[var] == cat).astype(int)
    #     # Ensure the original column exists before dropping
    #     if var in df.columns:
    #         df.drop(columns=[var], inplace=True, errors="ignore")
    #     return [f"{pref}__{cat}" for cat in categories]
    def _one_hot_inplace(df, var, categories, prefix=None, drop_last=True):
        pref = var if prefix is None else prefix
        cats = list(categories)

        # choose a baseline category consistently across PO/ENV
        if drop_last and len(cats) > 0:
            baseline = cats[-1]
            cats_to_encode = cats[:-1]
        else:
            cats_to_encode = cats

        new_cols = []
        for cat in cats_to_encode:
            col = f"{pref}__{cat}"
            df[col] = (df[var] == cat).astype(int)
            new_cols.append(col)

        if var in df.columns:
            df.drop(columns=[var], inplace=True, errors="ignore")
        return new_cols


    # Build a union of categories across PO & ENV for each categorical var
    dummy_columns_by_var = {}
    for var in cat_vars_present:
        po_vals = set(df_po[var].dropna().unique()) if var in df_po.columns else set()
        env_vals = set(df_env[var].dropna().unique()) if var in df_env.columns else set()
        union_cats = sorted(po_vals | env_vals)
        # If no categories found (column empty/missing), skip safely
        if len(union_cats) == 0:
            continue
        # Create identical dummy columns across both frames
        po_dummies = _one_hot_inplace(df_po, var, union_cats, prefix=var)
        env_dummies = _one_hot_inplace(df_env, var, union_cats, prefix=var)
        # Sanity check: same names
        dummy_columns_by_var[var] = po_dummies  # equals env_dummies

    # --- Build final covariate column list (replace categorical vars with their dummies) ---
    all_dummy_cols = []
    for var, dummies in dummy_columns_by_var.items():
        if var in covs:
            covs.remove(var)
        covs.extend(dummies)
        all_dummy_cols.extend(dummies)

    # site_index_cols = ['siteid']
    # site_index_cols = ['x', 'y']

    # --- Build Y_po: pivot to multi-label per (x, y)
    Y_po = (
        df_po.pivot_table(index=index_col, columns="spid", aggfunc="size", fill_value=0)
            .reset_index()
    )
    Y_po = safe_reindex_columns(Y_po, index_col + species)
    
    
    # Y_po_sid = (
    #     df_po.pivot_table(index=index_col, columns="spid", aggfunc="size", fill_value=0)
    #         .reset_index()
    # )
    # Y_po_sid = safe_reindex_columns(Y_po_sid, ["siteid"] + species)



    # --- X_po: mean covariates at (x, y)
    X_po = (
        df_po.groupby(index_col)[covs]
            .mean()
            .reset_index()
    )
    if add_po_var:
        X_po["PO"] = 1



    # --- PA: labels and covariates
    Y_pa = safe_reindex_columns(df_pa.copy(), species)
    if add_po_var:
        df_env["PO"] = 0
    X_pa = df_env[covs].copy()



    # Align PO by (x,y) join
    XY_po = pd.merge(X_po, Y_po, on=index_col, how="inner")


    # Split X and Y for PO
    X_po_final = XY_po[covs].copy()
    Y_po_final = XY_po[species].copy()


    # # for 30 random rows print Y_pa fully
    # rand_indices = np.random.choice(Y_pa.index, size=30, replace=False)
    # print('Random sample of Y_pa:')
    # print(Y_pa.loc[rand_indices])


    ### IMPORTANT TO CHECK: Should values be clipped, or exploded, depending on the model
    Y_po_final[species] = np.clip(Y_po_final[species], 0, 1)
    Y_pa[species] = np.clip(Y_pa[species], 0, 1)
    print('Values clipped to 0,1')
    print(np.unique(Y_pa[species].values),np.unique(Y_po_final[species].values))

    


    # ensure same order
    Y_pa = Y_pa[species]
    Y_po_final = Y_po_final[species]


    # For PA we already have X_pa, Y_pa aligned to species
    return X_po_final, Y_po_final, X_pa, Y_pa, species, covs




## FREEZE THIS FOR A SECOND
def load_po_pa_nceas_OLD(region: str, group_filter: str, add_po_var: bool, keep_xy: bool = True):
    """
    Load PO (aggregated by (x,y)) and PA (full table with covariates & labels),
    return aligned X/Y for PO and PA, with unified species columns and covariates.
    """
    po_path, pa_path, env_path = build_paths_nceas(region, group_filter)

    # Presence-only (PO) records
    df_po = pd.read_csv(po_path)

    # PA labels + environmental covariates
    df_pa = pd.read_csv(pa_path)
    df_env = pd.read_csv(env_path)

    # covariates from 5th col onward in env
    covariates = df_env.columns[4:].tolist()

    # Species 
    species = sorted(df_po["spid"].unique().tolist()) 

    # --- Build Y_po: pivot to multi-label per (x, y)
    Y_po = (
        df_po.pivot_table(index=["x", "y"], columns="spid", aggfunc="size", fill_value=0)
            .reset_index()
    )
    Y_po = safe_reindex_columns(Y_po, ["x", "y"] + species)
    
    
    Y_po_sid = (
        df_po.pivot_table(index=['siteid'], columns="spid", aggfunc="size", fill_value=0)
            .reset_index()
    )
    Y_po_sid = safe_reindex_columns(Y_po_sid, ["siteid"] + species)



    # --- X_po: mean covariates at (x, y)
    X_po = (
        df_po.groupby(["x", "y"])[covariates]
            .mean()
            .reset_index()
    )
    if add_po_var:
        X_po["PO"] = 1

    # final lists of covariates
    covs = covariates.copy()
    if add_po_var and "PO" not in covs:
        covs.append("PO")
    if keep_xy:
        covs = ["x", "y"] + covs

    # --- PA: labels and covariates
    Y_pa = safe_reindex_columns(df_pa.copy(), species)
    if add_po_var:
        df_env["PO"] = 0
    X_pa = df_env[covs].copy()



    # Align PO by (x,y) join
    XY_po = pd.merge(X_po, Y_po, on=["x", "y"], how="inner")


    # Split X and Y for PO
    X_po_final = XY_po[covs].copy()
    Y_po_final = XY_po[species].copy()



    ### IMPORTANT TO CHECK: Should values be clipped, or exploded, depending on the model
    Y_po_final[species] = np.clip(Y_po_final[species], 0, 1)
    Y_pa[species] = np.clip(Y_pa[species], 0, 1)
    print('Values clipped to 0,1')
    print(np.unique(Y_pa[species].values),np.unique(Y_po_final[species].values))



    # For PA we already have X_pa, Y_pa aligned to species
    return X_po_final, Y_po_final, X_pa, Y_pa, species, covs



def filter_geoplant_species(X_po: pd.DataFrame, Y_po: pd.DataFrame,
                            X_pa: pd.DataFrame, Y_pa: pd.DataFrame,
                            species: List[str], min_occurrence: int = 20, min_percentage: float = None):
    
    species_count_po = Y_po.sum(axis=0)
    species_count_pa = Y_pa.sum(axis=0)

    if min_percentage is not None:
        total_pa = len(Y_pa)
        total_po = len(Y_po)
        species_to_keep = [sp for sp in species if (species_count_po[sp] >= min_percentage * total_po) or (species_count_pa[sp] >= min_percentage * total_pa)]
    if min_occurrence is not None:
        species_to_keep = [sp for sp in species if (species_count_po[sp] >= min_occurrence) and (species_count_pa[sp] >= min_occurrence)]

    # if both are not None raise warning and say order
    if min_percentage is not None and min_occurrence is not None:
        print("Warning: Both min_percentage and min_occurrence are set. First filtering by min_percentage, then by min_occurrence.")

    X_po_filtered = X_po.copy()
    Y_po_filtered = Y_po[species_to_keep].copy()
    X_pa_filtered = X_pa.copy()
    Y_pa_filtered = Y_pa[species_to_keep].copy()

    print(f"Filtered species: kept {len(species_to_keep)} out of {len(species)} species with at least {min_occurrence} occurrences in PO or PA.")

    return X_po_filtered, Y_po_filtered, X_pa_filtered, Y_pa_filtered, species_to_keep


def aggregate_po_to_grid(
    X_po: pd.DataFrame,
    Y_po: pd.DataFrame,
    grid_size: float = 1000.0,
    y_agg: str = "sum",          # "sum" or "max"
    cov_agg: str = "mean",
    keep_cell_center: bool = True,
    verbose: bool = True,
):
    # ---- stats BEFORE ----
    n_before = len(X_po)
    avg_sp_before = Y_po.sum(axis=1).mean()

    X = X_po.copy()
    Y = Y_po.copy()

    gx = np.floor(X["x"].to_numpy() / grid_size).astype(np.int64)
    gy = np.floor(X["y"].to_numpy() / grid_size).astype(np.int64)

    # grouping key
    cell_index = pd.MultiIndex.from_arrays([gx, gy], names=["gx", "gy"])

    # ---- aggregate Y ----
    if y_agg == "sum":
        Yg = Y.groupby(cell_index).sum()
    elif y_agg == "max":
        Yg = Y.groupby(cell_index).max()
    else:
        raise ValueError("y_agg must be 'sum' or 'max'")

    # ---- aggregate X ----
    Xg = X.groupby(cell_index).agg(cov_agg)

    # Ensure index is MultiIndex + set names (some pandas paths drop them)
    if not isinstance(Xg.index, pd.MultiIndex):
        Xg.index = pd.MultiIndex.from_tuples(Xg.index)
    Xg.index = Xg.index.set_names(["gx", "gy"])

    if not isinstance(Yg.index, pd.MultiIndex):
        Yg.index = pd.MultiIndex.from_tuples(Yg.index)
    Yg.index = Yg.index.set_names(["gx", "gy"])

    if keep_cell_center:
        gx_vals = Xg.index.get_level_values(0).to_numpy()
        gy_vals = Xg.index.get_level_values(1).to_numpy()
        Xg["x"] = (gx_vals + 0.5) * grid_size
        Xg["y"] = (gy_vals + 0.5) * grid_size

    # ---- stats AFTER ----
    n_after = len(Xg)
    avg_sp_after = Yg.sum(axis=1).mean()
    reduction = n_before / n_after

    if verbose:
        print("PO grid aggregation")
        print(f"  Grid size           : {grid_size:g}")
        print(f"  Y aggregation       : {y_agg}")
        print(f"  Sites before        : {n_before:,}")
        print(f"  Sites after         : {n_after:,}")
        print(f"  Reduction factor    : {reduction:.2f}×")
        print(f"  Avg spp / site      : {avg_sp_before:.2f} → {avg_sp_after:.2f}")

    # align
    Xg = Xg.loc[Yg.index]

    return Xg, Yg


def load_po_pa_geoplant(data_path, region, 
                        filter_species: bool = True, min_occurrence: int = 20, min_percentage: float = None,
                        npz_file: bool = False, subsample_po: float = None, subsample_pa: float = None):
    X_po = pd.read_csv(os.path.join(data_path, f"X_po_{region}_covs.csv"), index_col=0)
    X_pa = pd.read_csv(os.path.join(data_path, f"X_pa_{region}_covs.csv"), index_col=0)

    
    
    if npz_file:
        Y_pa = load_npz(os.path.join(data_path, f"Y_pa_{region}.npz")).toarray()
        Y_pa = pd.DataFrame(Y_pa, index=X_pa.index)
        Y_po = load_npz(os.path.join(data_path, f"Y_po_{region}.npz")).toarray()
        Y_po = pd.DataFrame(Y_po, index=X_po.index)
        # add columns of species


    else:
        Y_po = pd.read_csv(os.path.join(data_path, f"Y_po_{region}.csv"))
        Y_pa = pd.read_csv(os.path.join(data_path, f"Y_pa_{region}.csv"))
    
    
    species = pd.read_csv(os.path.join(data_path, f"species_{region}.csv"), header=None).iloc[:,0].tolist()

    covs = pd.read_csv(os.path.join(data_path, f"covariates_{region}.csv"), header=None).iloc[:,0].tolist()

    # if npz_file add species
    if npz_file:
        Y_pa.columns = species
        Y_po.columns = species


    # change lon, lat to x, y
    X_po = X_po.rename(columns={"lon": "x", "lat": "y"})
    X_pa = X_pa.rename(columns={"lon": "x", "lat": "y"})
    # make index of Y to be X indexes
    Y_po.index = X_po.index
    Y_pa.index = X_pa.index

    # covs = X_pa.columns.tolist()[3:]  

    if subsample_po is not None:
        selected_indices = np.random.choice(X_po.index, size=int(len(X_po) * subsample_po), replace=False)
        X_po, Y_po = X_po.loc[selected_indices], Y_po.loc[selected_indices]

    if subsample_pa is not None:
        selected_indices = np.random.choice(X_pa.index, size=int(len(X_pa) * subsample_pa), replace=False)
        X_pa, Y_pa = X_pa.loc[selected_indices], Y_pa.loc[selected_indices]

    if filter_species:
        X_po, Y_po, X_pa, Y_pa, species = filter_geoplant_species(
            X_po, Y_po, X_pa, Y_pa, species, min_occurrence=min_occurrence, min_percentage=min_percentage
        )




    # X_po, Y_po = aggregate_po_to_grid(X_po, Y_po, grid_size=.01, y_agg="sum")


    # drop rows with all zeros in Y_po
    nonzero_indices = Y_po.index[Y_po.sum(axis=1) > 0]
    print(f"Kept {len(nonzero_indices)} out of {len(Y_po)} PO sites with non-zero species after aggregation.")
    X_po = X_po.loc[nonzero_indices]
    Y_po = Y_po.loc[nonzero_indices] 

    # same for Y_pa
    nonzero_indices_pa = Y_pa.index[Y_pa.sum(axis=1) > 0]
    print(f"Kept {len(nonzero_indices_pa)} out of {len(Y_pa)} PA sites with non-zero species.")
    X_pa = X_pa.loc[nonzero_indices_pa]
    Y_pa = Y_pa.loc[nonzero_indices_pa]

    # exit()
          



    # iter over Y_po rows
    break_point = 0
    for idx, row in Y_po.iterrows():
        species_count = row.sum()
        print(f"PO site {idx} has {species_count} species.")
        break_point += 1
        if break_point >= 5:
            break
        
    

    ## for column areaInM2 calculate mean without -inf and nan, then impute
    # if 'areaInM2' in X_pa.columns:
    #     area_mean = X_pa['areaInM2'].replace([np.inf, -np.inf], np.nan).mean()
    #     X_pa['areaInM2'] = X_pa['areaInM2'].replace([np.inf, -np.inf], np.nan).fillna(area_mean)
    #     print(f"Imputed areaInM2 with mean value: {area_mean}")

    # impute with KNN using x,y
    from sklearn.impute import KNNImputer
    if 'areaInM2' in X_pa.columns:
        # replace -inf and inf with NaN for imputation
        X_pa['areaInM2'] = X_pa['areaInM2'].replace([np.inf, -np.inf], np.nan)
        imputer = KNNImputer(n_neighbors=5)
        X_pa_imputed = imputer.fit_transform(X_pa)
        X_pa = pd.DataFrame(X_pa_imputed, columns=X_pa.columns, index=X_pa.index)
        print("Imputed areaInM2 using KNN imputer based on nearest neighbors in covariate space.")
        # new mean, how many nans
        print(f"After imputation, areaInM2 mean: {X_pa['areaInM2'].mean()}, number of NaNs: {X_pa['areaInM2'].isna().sum()}")
        # exit()
    return X_po, Y_po, X_pa, Y_pa, species, covs




def load_po_pa_geoplant_full(species_path, covariates_path):
    # data/processed/geoplant/climatic/pa_train_covariates.pkl is like loading a pickle, load as numpy directly
    X_pa_train = pd.read_pickle(os.path.join(covariates_path, "pa_train_covariates.pkl"))
    X_pa_test = pd.read_pickle(os.path.join(covariates_path, "pa_test_covariates.pkl"))
    X_po = pd.read_pickle(os.path.join(covariates_path, "po_covariates.pkl"))

    # for species, load as csv
    Y_pa_train = pd.read_csv(os.path.join(species_path, "pa_train_species.csv"))
    Y_pa_test = pd.read_csv(os.path.join(species_path, "pa_test_species.csv"))
    Y_po = pd.read_csv(os.path.join(species_path, "po_species.csv"))

    # species list (is inside species_path as all_species_list.txt)
    with open(os.path.join(species_path, "all_species_list_original_ids.txt"), "r") as f:
        species = [line.strip() for line in f.readlines()]


    # print headers
    print("X_po columns:", X_po.columns.tolist())
    print("X_pa_train columns:", X_pa_train.columns.tolist())
    print("X_pa_test columns:", X_pa_test.columns.tolist())


    def parse_species_column(df, col="speciesId"):
        """Convert '12 45 900' → [12,45,900]"""
        return (
            df[col]
            .fillna("")
            .astype(str)
            .str.split()
            .apply(lambda xs: [int(x) for x in xs if x != ""])
            .tolist()
        )
    
    Y_po = parse_species_column(Y_po)
    Y_pa_train = parse_species_column(Y_pa_train)
    Y_pa_test = parse_species_column(Y_pa_test)

    print("Sample Y_po species lists:")
    print(Y_po[:3])
    print("Sample Y_pa_train species lists:")
    print(Y_pa_train[:3])
    print("Sample Y_pa_test species lists:")
    print(Y_pa_test[:3])


    # covariates are columns of X_pa_train without first column
    covariates = X_pa_train.columns.tolist()[1:]
    print("Covariates:", covariates)

    return X_po, Y_po, X_pa_train, Y_pa_train, X_pa_test, Y_pa_test, species, covariates