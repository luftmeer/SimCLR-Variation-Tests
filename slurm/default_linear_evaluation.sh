#!/bin/bash
#SBATCH --job-name=linear-eval
#SBATCH -o ./runs/%j/%x.%j.out
#SBATCH -e ./runs/%j/%x.%j.err
#SBATCH -D ./
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=NvidiaAll
# Notification and type
#SBATCH --mail-type=NONE
# Wall clock limit:
#SBATCH --no-requeue

# Activate Environment
. ${PYENV_ROOT}/versions/aaml-simclr/bin/activate

ga_flag=""

# --- Parse arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --checkpoint) checkpoint="$2"; shift ;;
        --id) job_id="$2"; shift ;;
        --config) config="$2"; shift ;;
        --batch-size) batch_size="$2"; shift ;;
        --optimizer) optimizer="$2"; shift ;;
        --lr) lr="$2"; shift ;;
        --weight-decay) weight_decay="$2"; shift ;;
        --dataset-name) dataset_name="$2"; shift ;;
        --n_classes) n_classes="$2"; shift ;;
        --seed) seed="$2"; shift ;;
        --save-every-epoch) save_every_epoch="$2"; shift ;;
        --gradient-accumulation) ga_flag="--ga"; shift ;;
        --ga-count) ga_count="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# --- Validate required flags ---
if [[ -z "$job_id" || -z "$checkpoint" ]]; then
    echo "❌ Error: --id, --checkpoint are required"
    exit 1
fi

# --- Defaults ---
config="${config:-config/linear_evaluation_default.yaml}"
batch_size="${batch_size:-256}"
optimizer="${optimizer:-SGD}"
lr="${lr:-0.01}"
weight_decay="${weight_decay:-0}"
dataset_name="${dataset_name:-CIFAR10}"
n_classes="${n_classes:-10}"
seed="${seed:-42}"
save_every_epoch="${save_every_epoch:-10}"
ga_count="${ga_count:-8}"

# --- Log info ---
echo "======== SIMCLR LINEAR EVALUATION ========"
echo "SLURM_JOB_ID:        $job_id"
echo "Config file:         $config"
echo "Batch size:          $batch_size"
echo "Optimizer:           $optimizer"
echo "Learning rate:       $lr"
echo "Weight decay:        $weight_decay"
echo "Dataset:             $dataset_name"
echo "Checkpoint:          $checkpoint"
echo "Gradient Accum:      $ga_flag"
echo "Accum steps:         $ga_count"
echo "==================================="

# --- Define output path ---
output_dir="./runs/${job_id}/checkpoints/cpt_epoch${epoch}"
mkdir -p "$output_dir"

# --- Run evaluation ---
python linear_evaluation.py \
    --slurm_job_id "$job_id" \
    --checkpoint "$ckpt_file" \
    --batch_size "$batch_size" \
    --optimizer "$optimizer" \
    --lr "$lr" \
    --weight_decay "$weight_decay" \
    --dataset_name "$dataset_name" \
    --n_classes "$n_classes" \
    --seed "$seed" \
    --save_every_epoch "$save_every_epoch" \
    --ga_count "$ga_count" \
    $ga_flag