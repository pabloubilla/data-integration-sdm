#!/bin/bash
#OAR -q abaca
# #OAR -t besteffort
#OAR -l host=1/gpu=1,walltime=12:00:00
#OAR -p cluster IN ('esterel23','esterel24','esterel28','esterel30','esterel41')
# #OAR -p esterel41
#OAR -O jobs/OAR_%jobid%.out
#OAR -E jobs/OAR_%jobid%.err

# set -euo pipefail

PYTHON=/home/pubillap/.conda/envs/eco/bin/python

echo "===== Job info ====="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
echo "OAR_JOB_ID: ${OAR_JOB_ID:-unknown}"
echo

# ---- Define the axes of variation ----
datasets=("GeoPlant")
split_types=("geographical") # environmental optimal is missing
overlap_flags=("--use_overlapping_species")
regions=("full") # "france" "europe" "world"

# ---- Main sweep over dataset x split_type x overlap ----
for dataset in "${datasets[@]}"; do
    echo "===== Dataset: $dataset ====="

    # --- Preprocessing (once per dataset) ---
    if [ "$dataset" == "GeoPlant" ]; then
        echo "Preprocessing GeoPlant (BioClim)..."
        $PYTHON -u scripts/preprocess_geoplant.py --vocab-mode intersection_po_pa --region "${regions[0]}"
    elif [ "$dataset" == "GeoPlant_AE" ]; then
        echo "Preprocessing GeoPlant (AlphaEarth)..."
        $PYTHON -u scripts/preprocess_geoplant_alphaearth.py --vocab-mode intersection_po_pa
    fi

    for split_type in "${split_types[@]}"; do
        echo "--- Split type: $split_type ---"

        # --- Generate splits (once per dataset x split_type) ---
        $PYTHON -u scripts/generate_pa_splits_v2.py \
            --dataset_name "$dataset" \
            --split_type "$split_type" \
            --region "${regions[0]}"

        # # --- Tune split (once per dataset x split_type) ---
        # $PYTHON -u scripts/tune_split.py \
        #     --dataset "$dataset" \
        #     --split_type "$split_type" \


        for overlap in "${overlap_flags[@]}"; do
            echo "Running sweep (overlap flag: '${overlap:-none}')"

            # $PYTHON -u scripts/run_split_sweep.py \
            #     --dataset_name "$dataset" \
            #     --split_type "$split_type" \
            #     $overlap

            # $PYTHON -u scripts/plots/splits_boxplot.py \
            #     --dataset_name "$dataset" \
            #     --split_type "$split_type" \
            #     --add_average \
            #     $overlap

            # $PYTHON -u scripts/plots/splits_boxplot.py \
            #     --dataset_name "$dataset" \
            #     --split_type "$split_type" \
            #     --metric avg_auc_site \
            #     --add_average
            #     $overlap
        done
    done
    echo
done