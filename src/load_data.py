import os
from typing import List, Tuple, Dict
import pandas as pd
from utils import safe_reindex_columns

def build_paths_nceas(region: str, group_filter: str) -> Tuple[str, str, str]:
    po_path = os.path.join("data", "processed", "NCEAS", "Records", "train_po", f"{region}train_po{group_filter}.csv")
    pa_path = os.path.join("data", "raw", "NCEAS", "Records", "test_pa", f"{region}test_pa{group_filter}.csv")
    env_path = os.path.join("data", "raw", "NCEAS", "Records", "test_env", f"{region}test_env{group_filter}.csv")
    return po_path, pa_path, env_path

def load_po_pa_nceas(region: str, group_filter: str, add_po_var: bool, keep_xy: bool = True):
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


    # For PA we already have X_pa, Y_pa aligned to species
    return X_po_final, Y_po_final, X_pa, Y_pa, species, covs