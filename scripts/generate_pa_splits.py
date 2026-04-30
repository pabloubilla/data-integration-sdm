from pathlib import Path
import json

from isdm.load_data import load_geoplant_processed
from isdm.splits import partition_sweep_ranges_v2_indices, save_split_specs


def main():
    data_path = "data/processed/GeoPlant/france"
    output_dir = Path("outputs/splits/france")

    data = load_geoplant_processed(data_path, add_coordinates=True)

    X_pa = data.X_pa_train
    covariates = data.covariates

    # Choose clustering and distance covariates.
    # If x/y exist in processed X, use them for spatial clusters.
    # In your new loader you may have removed metadata from X, so this may need adjustment.
    covs_cluster = [c for c in ["x", "y", "lon", "lat"] if c in X_pa.columns]
    if not covs_cluster:
        covs_cluster = covariates

    covs_distance = covariates

    print('covs_cluster:', covs_cluster)
    print('covs_distance:', covs_distance)

    specs = partition_sweep_ranges_v2_indices(
        X_pa=X_pa,
        covs_cluster=covs_cluster,
        covs_distance=covs_distance,
        k_clusters=50,
        select_subset=20,
        train_proportion=0.4,
        distance_metric="mahalanobis",
        seed=42,
        options=("closest", "middle", "farthest"),
    )

    save_split_specs(specs, output_dir)

    metadata = {
        "data_path": data_path,
        "method": "partition_sweep_ranges_v2",
        "region": data.metadata.get("region"),
        "vocab_mode": data.metadata.get("vocab_mode"),
        "covs_cluster": covs_cluster,
        "covs_distance": covs_distance,
        "k_clusters": 50,
        "select_subset": 20,
        "train_proportion": 0.4,
        "distance_metric": "mahalanobis",
        "seed": 42,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved split metadata.")


if __name__ == "__main__":
    main()