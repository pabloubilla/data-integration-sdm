import argparse
import os
import pandas as pd


def output_dir_for(dataset_name, region, split_type, spec_name):
    return f"outputs/split_sweep/{dataset_name}/{region}_bands/{split_type}/{spec_name}"


def aggregated_dir_for(dataset_name, split_type, spec_name, agg_name="agg_regions"):
    return f"outputs/split_sweep/{dataset_name}/{agg_name}_bands/{split_type}/{spec_name}"


def aggregate(dataset_name, regions, split_types, spec_name="intersect", agg_name="agg_regions", remove_nan=False):
    for split_type in split_types:
        dfs = []
        for region in regions:
            path = os.path.join(output_dir_for(dataset_name, region, split_type, spec_name), "summary_common.csv")
            df = pd.read_csv(path)
            df["country"] = region
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        if remove_nan:
            # mention which columns have NaN values
            print("Columns with NaN values:")
            for col in combined.columns:
                if combined[col].isnull().any():
                    print(f"  - {col}")
            combined = combined.dropna()

        # add region before test_number in test_number
        combined["test_number"] = combined["country"] + "_" + combined["test_number"].astype(str)


        out_dir = aggregated_dir_for(dataset_name, split_type, spec_name, agg_name)
        os.makedirs(out_dir, exist_ok=True)

        combined.to_csv(os.path.join(out_dir, "summary_common.csv"), index=False)
        with open(os.path.join(out_dir, "countries.txt"), "w") as f:
            f.write("\n".join(regions) + "\n")

        print(f"[{split_type}] Aggregated {len(regions)} regions ({', '.join(regions)}) "
              f"→ {out_dir} ({len(combined)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate summary_common.csv across multiple regions.")
    parser.add_argument("--dataset_name", default="GeoPlant", type=str)
    parser.add_argument("--regions", type=str, default="france,netherlands,sparse_pa,denmark")
    parser.add_argument("--split_types", default="geographical,environmental", type=str)
    parser.add_argument("--use_overlapping_species", action="store_true")
    parser.add_argument("--agg_name", default="agg_regions", type=str,
                         help="Name used in place of '<region>' in the output path.")
    parser.add_argument("--remove_nan", action="store_true")
    args = parser.parse_args()

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    split_types = [s.strip() for s in args.split_types.split(",") if s.strip()]
    spec_name = "intersect" if args.use_overlapping_species else "intersect"

    aggregate(args.dataset_name, regions, split_types, spec_name, args.agg_name, args.remove_nan)