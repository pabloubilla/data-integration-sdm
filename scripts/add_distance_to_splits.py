from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.metrics import pairwise_distances

from isdm.load_data import load_geoplant_processed
from isdm.splits_bands import load_split_specs, load_split


def project_to_lambert93(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Project lon/lat (EPSG:4326) to Lambert-93 (EPSG:2154), in meters.
    Mainland France + Corsica only — do not use for overseas territories.
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    x_m, y_m = transformer.transform(lon, lat)
    return x_m, y_m


def add_distance_km(
    split_dir: str | Path,
    X_pa: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    out_csv: str | Path | None = None,
    split_sweep_dir: str | Path | None = None,
    add_to_sweep: bool = True,
) -> pd.DataFrame:
    """
    Load splits.csv from split_dir (via load_split_specs), compute a physical
    distance_km per split (mean min distance from test points to nearest train
    point, in km, projected to Lambert-93), and write it back.

    Parameters
    ----------
    split_dir : directory containing splits.csv and the per-split .npz files
    X_pa      : dataframe used to generate the splits (data.X_pa_train, loaded
                with add_coordinates=True), in the SAME row order as when the
                splits were generated
    out_csv   : where to save the updated csv. Defaults to overwriting
                splits.csv in split_dir (a .bak backup is written first).

    Returns
    -------
    The updated splits dataframe (also written to disk).
    """
    split_dir = Path(split_dir)
    df = load_split_specs(split_dir)

    if len(X_pa) == 0:
        raise ValueError("X_pa is empty.")
    for col in (lat_col, lon_col):
        if col not in X_pa.columns:
            raise ValueError(
                f"Column '{col}' not found in X_pa — did you load it with "
                "add_coordinates=True?"
            )

    x_m, y_m = project_to_lambert93(
        X_pa[lon_col].to_numpy(), X_pa[lat_col].to_numpy()
    )
    coords_m = np.column_stack([x_m, y_m])
    n_coords = len(coords_m)

    distances_km = []
    for _, row in df.iterrows():
        split = load_split(split_dir, row["split_file"])
        train_idx, test_idx = split["train_idx"], split["test_idx"]

        if train_idx.max(initial=-1) >= n_coords or test_idx.max(initial=-1) >= n_coords:
            raise ValueError(
                f"Index out of range for split {row['split_id']}: "
                f"X_pa has {n_coords} rows but indices go higher. "
                "X_pa is likely not in the same order used to generate the splits."
                f"Train idx max: {train_idx.max(initial=-1)}, test idx max: {test_idx.max(initial=-1)}"
            )
            

        if len(train_idx) == 0 or len(test_idx) == 0:
            distances_km.append(np.nan)
            continue

        D = pairwise_distances(coords_m[test_idx], coords_m[train_idx], metric="euclidean")
        distances_km.append(float(D.min(axis=1).mean()) / 1000.0)

    df["distance_km"] = distances_km

    out_csv = Path(out_csv) if out_csv else split_dir / "splits.csv"
    if out_csv == split_dir / "splits.csv":
        backup = split_dir / "splits.csv.bak"
        if not backup.exists():
            load_split_specs(split_dir).to_csv(backup, index=False)
            print(f"Backed up original splits.csv to {backup}")

    df.to_csv(out_csv, index=False)
    print(f"Wrote distance_km for {len(df)} splits to {out_csv}")

    if split_sweep_dir is not None and add_to_sweep:
        split_sweep_dir = Path(split_sweep_dir)
        sweep_csv = split_sweep_dir / "summary_common.csv"
        if sweep_csv.exists():
            sweep_df = pd.read_csv(sweep_csv)
            sweep_df = sweep_df.merge(df[["split_id", "distance_km"]], on="split_id", how="left")
            # if _x and _y check they are the same and keep only distance_km
            if "distance_km_x" in sweep_df.columns and "distance_km_y" in sweep_df.columns:
                if not np.allclose(sweep_df["distance_km_x"], sweep_df["distance_km_y"], equal_nan=True):
                    raise ValueError("distance_km values in sweep and splits do not match.")
                sweep_df["distance_km"] = sweep_df["distance_km_x"]
                sweep_df = sweep_df.drop(columns=["distance_km_x", "distance_km_y"])

            sweep_df.to_csv(sweep_csv, index=False)
            print(f"Added distance_km to {sweep_csv} ({len(sweep_df)} rows)")
    return df


if __name__ == "__main__":
    # add_coordinates=True is required — lat/lon get dropped from the
    # covariates dataframe otherwise. X_pa_train is what the splits were
    # generated from (the PA pool carved into train/test bands via
    # partition_sweep_bands / partition_sweep_ranges_v2_indices);
    # X_pa_test is the held-out set and was not part of split generation.


    countries_to_run = ['france', 'denmark', 'netherlands', 'sparse_pa']
    split_types = ["geographical", "environmental"]
    for country in countries_to_run:
        for split_type in split_types:
            data = load_geoplant_processed(
                    processed_root=f"data/processed/GeoPlant/{country}",
                    add_coordinates=True,
                )
            print(f"Adding distance_km to splits for {country} ({split_type})...")
            add_distance_km(
                split_dir=f"outputs/splits/GeoPlant/{country}_bands/{split_type}",
                X_pa=data.X_pa_train,
                lat_col="lat",
                lon_col="lon",
                split_sweep_dir =f"outputs/split_sweep/GeoPlant/{country}_bands/{split_type}/intersect",
                add_to_sweep=True,
            )
