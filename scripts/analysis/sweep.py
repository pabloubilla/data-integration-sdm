from isdm.evaluation import per_species_auc_sparse, LogitsStore
from isdm.load_data import load_geoplant_processed
from isdm.splits import set_all_seeds
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def main():
    seed = 42
    set_all_seeds(seed)

    data_path  = "data/processed/GeoPlant/france"
    logits_dir = Path("outputs/split_sweep/logits")
    output_dir = Path("outputs/split_sweep")

    data        = load_geoplant_processed(data_path, add_coordinates=True)
    num_classes = len(data.species)

    coordinates = data.X_pa_train[["lon", "lat"]].values



    # print(data.X_pa_train)
    # exit(0)



    groups = defaultdict(lambda: {"logits": [], "y_lists": []})

    for model_name in ["pa", "popa", "po"]:
        store = LogitsStore.load(logits_dir / f"logits_{model_name}.npz")
        for rec in store._records:
            key = (model_name, rec["option"])   # drop test_number from key
            groups[key]["logits"].append(rec["logits"])
            groups[key]["y_lists"].extend(rec["y_lists"])

    records = []
    for (model_name, option), group in groups.items():
        pooled_logits = np.concatenate(group["logits"], axis=0)
        pooled_y      = group["y_lists"]

        aucs = per_species_auc_sparse(
            logits=pooled_logits,
            y_lists=pooled_y,
            num_classes=num_classes,
        )

        records.append({
            "model":           model_name,
            "option":          option,
            "avg_auc":         float(np.nanmean(list(aucs.values()))),
            "n_valid_species": int(sum(~np.isnan(list(aucs.values())))),
        })

    df = pd.DataFrame(records).sort_values(["model", "option"])
    # save
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "concatenated_auc.csv", index=False)

    print(df)


if __name__ == "__main__":
    main()