"""
Add distance_km to each split: mean great-circle distance (km) from each
test point to its nearest train point. Writes it into splits.csv and,
if present, into the split sweep's summary_common.csv.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from isdm.utils import haversine_tree, nearest_km
from isdm.load_data import load_geoplant_processed
from isdm.splits_bands import load_split, load_split_specs

COORD_COLS = ["lon", "lat"]


def split_distance_km(lonlat, train_idx, test_idx):
    if len(train_idx) == 0 or len(test_idx) == 0:
        return np.nan
    tree = haversine_tree(lonlat[train_idx])
    return float(nearest_km(tree, lonlat[test_idx]).mean())


def add_distance_km(split_dir, X_pa, sweep_csv=None):
    """X_pa must be in the same row order used to generate the splits (data.X_pa_train)."""
    split_dir = Path(split_dir)
    specs = load_split_specs(split_dir)
    lonlat = X_pa[COORD_COLS].to_numpy()

    distances = []
    for _, row in specs.iterrows():
        split = load_split(split_dir, row["split_file"])
        train_idx, test_idx = split["train_idx"], split["test_idx"]
        if max(train_idx.max(initial=-1), test_idx.max(initial=-1)) >= len(lonlat):
            raise ValueError(f"{row['split_id']}: indices exceed X_pa rows ({len(lonlat)}). Wrong row order?")
        distances.append(split_distance_km(lonlat, train_idx, test_idx))
    specs["distance_km"] = distances

    splits_csv = split_dir / "splits.csv"
    backup = split_dir / "splits.csv.bak"
    if not backup.exists():
        shutil.copy(splits_csv, backup)
    specs.to_csv(splits_csv, index=False)
    print(f"  wrote distance_km for {len(specs)} splits -> {splits_csv}")

    if sweep_csv is not None and Path(sweep_csv).exists():
        sweep = pd.read_csv(sweep_csv).drop(columns="distance_km", errors="ignore")
        sweep = sweep.merge(specs[["split_id", "distance_km"]], on="split_id", how="left")
        sweep.to_csv(sweep_csv, index=False)
        print(f"  updated distance_km in {sweep_csv}")

    return specs


if __name__ == "__main__":
    regions = ["france", "denmark", "bene", "sparse_pa"]
    split_types = ["geographical", "environmental"]

    for region in regions:
        data = load_geoplant_processed(processed_root=f"data/processed/GeoPlant/{region}", add_coordinates=True)
        for split_type in split_types:
            print(f"{region} / {split_type}")
            add_distance_km(
                split_dir=f"outputs/splits/GeoPlant/{region}_bands/{split_type}",
                X_pa=data.X_pa_train,
                sweep_csv=f"outputs/split_sweep/GeoPlant/{region}_bands/{split_type}/intersect/summary_common.csv",
            )