#!/bin/bash
#OAR -q abaca
# #OAR -t besteffort
#OAR -l host=1/gpu=1,walltime=12:00:00
#OAR -p cluster IN ('esterel23','esterel24','esterel28','esterel30','esterel41')
# #OAR -p esterel41
#OAR -O jobs/OAR_%jobid%.out
#OAR -E jobs/OAR_%jobid%.err

# set -euo pipefail

echo "===== Job info ====="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
echo "OAR_JOB_ID: ${OAR_JOB_ID:-unknown}"
echo

# # If you expect a GPU on esterel41, keep this; otherwise remove it.
# echo "===== GPU info (optional) ====="
# command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found"
# echo

# echo "===== Python env ====="
# if [ -f "venv/bin/activate" ]; then
#   # shellcheck disable=SC1091
#   source "venv/bin/activate"
#   echo "Activated venv: venv/"
# fi
# module load conda
# conda activate venv

# this is the direct path to the conda environment's python binary
# /home/pubillap/.conda/envs/eco/bin/python -u scripts/generate_pa_splits.py

# /home/pubillap/.conda/envs/eco/bin/python -u scripts/generate_pa_splits_gaussian.py --dataset_name GeoPlant 
# /home/pubillap/.conda/envs/eco/bin/python -u scripts/run_integration_experiment.py 

# /home/pubillap/.conda/envs/eco/bin/python -u scripts/run_split_sweep.py --dataset_name GeoPlant 
# /home/pubillap/.conda/envs/eco/bin/python -u scripts/run_integration_experiment.py 
/home/pubillap/.conda/envs/eco/bin/python -u scripts/tune_split.py 

# -c 'import torch; print("Test run");print("cuda available:", torch.cuda.is_available()); print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
echo

# echo "===== Run your command ====="
# # Put your real command here (what was inside script_lancement_Atos_train.sh)
# bash ./script_lancement_Atos_train.sh
# # OR replace the line above with: python3 -u train.py ...

# echo "===== Done ====="
# date
