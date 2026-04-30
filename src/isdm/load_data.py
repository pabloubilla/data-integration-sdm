# src/load_data.py

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class GeoPlantData:
    X_po: pd.DataFrame
    y_po: list[list[int]]
    X_pa_train: pd.DataFrame
    y_pa_train: list[list[int]]
    X_pa_test: pd.DataFrame
    y_pa_test: list[list[int]]
    species: list[str]
    covariates: list[str]
    metadata: dict


def parse_species_column(df: pd.DataFrame, col: str = "speciesId") -> list[list[int]]:
    return (
        df[col]
        .fillna("")
        .astype(str)
        .str.split()
        .apply(lambda xs: [int(x) for x in xs if x != ""])
        .tolist()
    )


def load_geoplant_processed(
    processed_root: str = "data/processed/GeoPlant/france",
    verbose: bool = True,
    add_coordinates: bool = False
) -> GeoPlantData:
    root = Path(processed_root)
    species_dir = root / "species"
    covariates_dir = root / "covariates"

    with open(root / "metadata.json", "r") as f:
        metadata = json.load(f)

    covariates = metadata["covariates"]

    X_po = pd.read_pickle(covariates_dir / "po_covariates.pkl")
    X_pa_train = pd.read_pickle(covariates_dir / "pa_train_covariates.pkl")
    X_pa_test = pd.read_pickle(covariates_dir / "pa_test_covariates.pkl")

    Y_po = pd.read_csv(species_dir / "po_species.csv")
    Y_pa_train = pd.read_csv(species_dir / "pa_train_species.csv")
    Y_pa_test = pd.read_csv(species_dir / "pa_test_species.csv")

    y_po = parse_species_column(Y_po)
    y_pa_train = parse_species_column(Y_pa_train)
    y_pa_test = parse_species_column(Y_pa_test)

    vocab = np.load(species_dir / "species_vocab.npy")
    species = [str(int(x)) for x in vocab.tolist()]

    if add_coordinates:
        covariates = covariates + ["lon", "lat"]

    X_po = X_po[covariates].copy()
    X_pa_train = X_pa_train[covariates].copy()
    X_pa_test = X_pa_test[covariates].copy()

    if verbose:
        print(f"Loaded GeoPlant processed dataset: {root}")
        print(f"  Region       : {metadata.get('region')}")
        print(f"  Vocab mode   : {metadata.get('vocab_mode')}")
        print(f"  PO rows      : {len(X_po):,}")
        print(f"  PA train rows: {len(X_pa_train):,}")
        print(f"  PA test rows : {len(X_pa_test):,}")
        print(f"  Num species  : {len(species):,}")
        print(f"  Num covs     : {len(covariates):,}")

    return GeoPlantData(
        X_po=X_po,
        y_po=y_po,
        X_pa_train=X_pa_train,
        y_pa_train=y_pa_train,
        X_pa_test=X_pa_test,
        y_pa_test=y_pa_test,
        species=species,
        covariates=covariates,
        metadata=metadata,
    )