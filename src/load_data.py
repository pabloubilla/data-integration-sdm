import os
from typing import List, Tuple, Dict
import pandas as pd
from src.utils import safe_reindex_columns
import numpy as np

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

def load_po_pa_nceas(region: str, group_filter: str, add_po_var: bool, keep_xy: bool = True,
                     categorical_vars: list = []):
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

    # Helper: add one-hot columns for a single variable given a fixed, shared category list
    def _one_hot_inplace(df, var, categories, prefix=None):
        pref = var if prefix is None else prefix
        for cat in categories:
            col = f"{pref}__{cat}"
            df[col] = (df[var] == cat).astype(int)
        # Ensure the original column exists before dropping
        if var in df.columns:
            df.drop(columns=[var], inplace=True, errors="ignore")
        return [f"{pref}__{cat}" for cat in categories]

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


    # --- Build Y_po: pivot to multi-label per (x, y)
    Y_po = (
        df_po.pivot_table(index=['siteid'], columns="spid", aggfunc="size", fill_value=0)
            .reset_index()
    )
    Y_po = safe_reindex_columns(Y_po, ['siteid'] + species)
    
    
    Y_po_sid = (
        df_po.pivot_table(index=['siteid'], columns="spid", aggfunc="size", fill_value=0)
            .reset_index()
    )
    Y_po_sid = safe_reindex_columns(Y_po_sid, ["siteid"] + species)



    # --- X_po: mean covariates at (x, y)
    X_po = (
        df_po.groupby(['siteid'])[covs]
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
    XY_po = pd.merge(X_po, Y_po, on=['siteid'], how="inner")


    # Split X and Y for PO
    X_po_final = XY_po[covs].copy()
    Y_po_final = XY_po[species].copy()


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
    # Y_po_final[species] = np.clip(Y_po_final[species], 0, 1)
    # Y_pa[species] = np.clip(Y_pa[species], 0, 1)
    # print('Values clipped to 0,1')
    # print(np.unique(Y_pa[species].values),np.unique(Y_po_final[species].values))

    # For PA we already have X_pa, Y_pa aligned to species
    return X_po_final, Y_po_final, X_pa, Y_pa, species, covs