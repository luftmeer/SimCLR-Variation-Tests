#!/bin/bash

ga_flag=""

# --- Parse arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --id) job_id="$2"; shift ;;
        --config) config="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --optimizer) optimizer="$2"; shift ;;
        --lr) lr="$2"; shift ;;
        --weight-decay) weight_decay="$2"; shift ;;
        --dataset-name) dataset_name="$2"; shift ;;
        --n_classes) n_classes="$2"; shift ;;
        --seed) seed="$2"; shift ;;
        --save-every-epoch) save_every_epoch="$2"; shift ;;
        --gradient-accumulation) ga_flag="--gradient--accumulation"; shift ;;
        --ga-count) ga_count="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# --- Required: job_id ---
if [[ -z "$job_id" ]]; then
    echo "Error: Missing required --id <slurm_job_id>"
    exit 1
fi

# --- Defaults ---
config="${./config/linear_evaluation_default.yaml}"
batch_size="${batch_size:-256}"
optimizer="${optimizer:-SGD}"
lr="${lr:-0.01}"
weight_decay="${weight_decay:-0}"
dataset_name="${dataset_name:-CIFAR10}"
n_classes="${n_classes:-10}"
seed="${seed:-42}"
save_every_epoch="${save_every_epoch:-10}"
ga_count="${ga_count:-8}"

# --- Paths ---
checkpoint_dir="./runs/${job_id}/checkpoints"

if [[ ! -d "$checkpoint_dir" ]]; then
    echo "No checkpoint directory found at: $checkpoint_dir"
    exit 1
fi

# --- Find missing subfolders ---
echo "Scanning checkpoint directory for unprocessed epochs..."

for ckpt_file in "$checkpoint_dir"/*.cpt; do
    [[ -e "$ckpt_file" ]] || continue

    filename=$(basename "$ckpt_file" .cpt)

    # Extract last underscore-separated field as epoch number
    epoch_num=$(echo "$filename" | awk -F_ '{print $NF}')

    # Check target folder
    target_folder="${checkpoint_dir}/cpt_epoch${epoch_num}"
    if [[ ! -d "$target_folder" ]]; then
        echo "Submitting job for checkpoint: epoch ${epoch_num} (File: ${ckpt_file})"

        slurm_id=$(sbatch --parsable slurm/default_linear_evaluation.sh \
            --id "$job_id" \
            --checkpoint "$ckpt_file" \
            --batch-size "$batch_size" \
            --optimizer "$optimizer" \
            --lr "$lr" \
            --weight-decay "$weight_decay" \
            --dataset-name "$dataset_name" \
            --n_classes "$n_classes" \
            --seed "$seed" \
            --save-every-epoch "$save_every_epoch" \
            --ga-count "$ga_count" \
            $ga_flag
        )
        mkdir -p "./runs/$slurm_id"
        
    else
        echo "✅ Already processed: epoch ${epoch_num}"
    fi
done
