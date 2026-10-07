#!/bin/bash
#OAR -q abaca
# #OAR -t besteffort
#OAR -l host=1/gpu=2,walltime=48:00:00
# #OAR -p cluster IN ('esterel23','esterel24','esterel28','esterel30','esterel41')
#OAR -p esterel41
#OAR -O jobs/OAR_%jobid%.out
#OAR -E jobs/OAR_%jobid%.err

# set -euo pipefail


# PYTHON=/home/pubillap/.conda/envs/eco/bin/python
PYTHON="uv run --no-sync python"

echo "===== Job info ====="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
echo "OAR_JOB_ID: ${OAR_JOB_ID:-unknown}"
echo

LOG_DIR="logs/${OAR_JOB_ID:-local}"
mkdir -p "$LOG_DIR"
echo "Per-pipeline logs: $LOG_DIR/<split_type>.log"
echo

# ─────────────────────────────────────────────
#  Which stages to run
# ─────────────────────────────────────────────
RUN_PREPROCESS=false
RUN_GENERATE_SPLITS=false
RUN_TUNE=true
RUN_AGGREGATE_TUNE=false
RUN_FINAL_SWEEP=false

# ─────────────────────────────────────────────
#  Axes of variation
# ─────────────────────────────────────────────
datasets=("GeoPlant")
split_types=("environmental" "geographical")   # one pipeline per entry, each on its own GPU
regions=("france")
overlap_flags=("--use_overlapping_species")
n_anchors=10
n_trials=100

N_GPUS=2
N_CONCURRENT=${#split_types[@]}
if [ "$N_CONCURRENT" -gt "$N_GPUS" ]; then
    echo "WARNING: ${#split_types[@]} split_types but only $N_GPUS GPUs — some will share a GPU."
fi

# ─────────────────────────────────────────────
#  Parallelization settings — divided across concurrent pipelines, since
#  they now share this node's CPU cores simultaneously
# ─────────────────────────────────────────────
export MAX_WORKERS=12   # per-pipeline worker count; halve this again if you add more split_types
export WANDB_MODE=disabled

TOTAL_CORES=$(nproc)
THREADS_PER_WORKER=$(( TOTAL_CORES / (MAX_WORKERS * N_CONCURRENT) ))
if [ "$THREADS_PER_WORKER" -lt 1 ]; then THREADS_PER_WORKER=1; fi

export OMP_NUM_THREADS=$THREADS_PER_WORKER
export MKL_NUM_THREADS=$THREADS_PER_WORKER
export OPENBLAS_NUM_THREADS=$THREADS_PER_WORKER
export NUMEXPR_NUM_THREADS=$THREADS_PER_WORKER

ulimit -n 65536

echo "===== Resource settings ====="
echo "N_CONCURRENT_PIPELINES=$N_CONCURRENT"
echo "MAX_WORKERS (per pipeline)=$MAX_WORKERS"
echo "TOTAL_CORES(nproc)=$TOTAL_CORES"
echo "THREADS_PER_WORKER=$THREADS_PER_WORKER"
echo "ulimit -n: $(ulimit -n)"
echo

echo "===== GPU info ====="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found"
echo

echo "===== Stages enabled ====="
echo "preprocess=$RUN_PREPROCESS  generate_splits=$RUN_GENERATE_SPLITS  tune=$RUN_TUNE"
echo "aggregate_tune=$RUN_AGGREGATE_TUNE  final_sweep=$RUN_FINAL_SWEEP  plots=$RUN_PLOTS"
echo

# ─────────────────────────────────────────────
#  Preprocessing — CPU-only, independent of split_type, run once up front
# ─────────────────────────────────────────────
for dataset in "${datasets[@]}"; do
  if [ "$RUN_PREPROCESS" = true ]; then
    echo "[RUN][preprocess] START $dataset at $(date +%H:%M:%S)"
    if [ "$dataset" == "GeoPlant" ]; then
      $PYTHON -u scripts/pipeline/01_preprocess_geoplant.py --vocab-mode intersection_po_pa --region "${regions[0]}"
    elif [ "$dataset" == "GeoPlant_AE" ]; then
      $PYTHON -u scripts/preprocess_geoplant_alphaearth.py --vocab-mode intersection_po_pa
    fi
    echo "[RUN][preprocess] END $dataset (rc=$?) at $(date +%H:%M:%S)"
  fi
done
echo

# ─────────────────────────────────────────────
#  Per-split_type pipeline, pinned to one GPU, run in the background.
#  Each python call's full output goes to its own log file; only short
#  start/end/rc lines (tagged, greppable) go to the main OAR output.
# ─────────────────────────────────────────────
run_pipeline() {
    local dataset=$1 region=$2 split_type=$3 gpu_id=$4 n_anchors=$5 n_trials=$6
    export CUDA_VISIBLE_DEVICES=$gpu_id
    local LOGFILE="$LOG_DIR/${split_type}.log"
    local TAG="[RUN][$split_type][gpu=$gpu_id]"

    echo "$TAG PIPELINE START at $(date +%H:%M:%S) (pid=$$, log=$LOGFILE)"

    if [ "$RUN_GENERATE_SPLITS" = true ]; then
        echo "$TAG START generate_splits at $(date +%H:%M:%S)"
        $PYTHON -u scripts/pipeline/02_generate_pa_splits.py \
          --dataset_name "$dataset" --split_type "$split_type" --region "$region" --n_anchors "$n_anchors" \
          >> "$LOGFILE" 2>&1
        echo "$TAG END generate_splits (rc=$?) at $(date +%H:%M:%S)"
    fi

    local TUNE_DIR="outputs/tune/$dataset/$region/$split_type/test_0"

    if [ "$RUN_TUNE" = true ]; then
        echo "$TAG START tune -> $TUNE_DIR at $(date +%H:%M:%S)"
        $PYTHON -u scripts/pipeline/03_tune_split.py \
          --dataset "$dataset" --split_type "$split_type" --region "$region" --test_number 0 --n_trials "$n_trials" \
          >> "$LOGFILE" 2>&1
        echo "$TAG END tune (rc=$?) at $(date +%H:%M:%S)"
    fi

    if [ "$RUN_AGGREGATE_TUNE" = true ]; then
        echo "$TAG START aggregate_tune at $(date +%H:%M:%S)"
        $PYTHON -u scripts/pipeline/04_aggregate_tune_split.py --tune_dir "$TUNE_DIR" \
          >> "$LOGFILE" 2>&1
        echo "$TAG END aggregate_tune (rc=$?) at $(date +%H:%M:%S)"
    fi

    for overlap in "${overlap_flags[@]}"; do
        if [ "$RUN_FINAL_SWEEP" = true ]; then
            echo "$TAG START final_sweep (overlap='${overlap:-none}') at $(date +%H:%M:%S)"
            $PYTHON -u scripts/pipeline/05_run_split_sweep.py \
              --dataset_name "$dataset" --split_type "$split_type" --region "$region" $overlap \
              >> "$LOGFILE" 2>&1
            echo "$TAG END final_sweep (rc=$?) at $(date +%H:%M:%S)"
        fi

    done

    echo "$TAG PIPELINE DONE at $(date +%H:%M:%S)"
}

# ─────────────────────────────────────────────
#  Launch one pipeline per split_type, each on its own GPU, in parallel
# ─────────────────────────────────────────────
for dataset in "${datasets[@]}"; do
  for region in "${regions[@]}"; do
    pids=()
    for i in "${!split_types[@]}"; do
        split_type=${split_types[$i]}
        gpu_id=$(( i % N_GPUS ))
        run_pipeline "$dataset" "$region" "$split_type" "$gpu_id" "$n_anchors" "$n_trials"&
        pids+=($!)
    done

    echo "Launched ${#pids[@]} pipelines: ${pids[*]} — waiting..."
    wait "${pids[@]}"
  done
done

echo
echo "===== Done ====="
date