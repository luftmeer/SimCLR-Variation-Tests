#!/bin/zsh
#b Name and Files (also --job-name)
#SBATCH -J simclr
#Output and error (also --output, --error):
#SBATCH -o ./runs/%j/%x.%j.out
#SBATCH -e ./runs/%j/%x.%j.err
#Initial working directory (also --chdir):
#SBATCH -D ./
#Notification and type
#SBATCH --mail-type=NONE
# Wall clock limit:
#SBATCH --no-requeue

#SBATCH --partition=NvidiaAll
#Number of nodes and tasks per node:
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCh --cpus-per-task=8

resume_flag=""
ga_flag=""

# --- Argument parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) config="$2"; shift ;;
        # Encoder
        --encoder) encoder="$2"; shift ;;
        --epochs) epochs="$2"; shift ;;
        --batch-size) batch_size="$2"; shift ;;
        --widening) widening="$2"; shift ;;
        # Optimizer
        --optimizer) optimizer="$2"; shift ;;
        --lr) lr="$2"; shift ;;
        --weight-decay) weight_decay="$2"; shift ;;
        --eps) eps="$2"; shift ;;
        # Dataset
        --dataset-name) dataset_name="$2"; shift ;;
        # SimCLR
        --augmentations) augmentations="$2"; shift ;;
        --resize) resize="$2"; shift ;;
        --projection-dim) projection_dim="$2"; shift ;;
        --temperature) temperature="$2"; shift ;;
        # Resume & Checkpoint Loading
        --resume) resume_flag="--resume" ;;
        --checkpoint) checkpoint="$2"; shift ;;
        # Gradient Acuumulation
        --gradient-accumulation) ga_flag="--ga" ;;
        --ga-count) ga_count="$2"; shift ;;
        
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Set defaults if not passed ---
config="${config:-config/default.yaml}"
encoder="${encoder:-resnet18}"
epochs="${epochs:-1000}"
batch_size="${batch_size:-128}"
widening="${widening:-1}"
optimizer="${optimizer:-Adam}"
lr="${lr:-0.001}"
weight_decay="${weight_decay:-0.}"
eps="${eps:-1e-08}"
dataset_name="${dataset_name:-CIFAR10}"
augmentations="${augmentations:-2}"
resize="${resize:-224}"
projection_dim="${projection_dim:-64}"
temperature="${temperature:-0.5}"
checkpoint="${checkpoint:-NULL}"
ga_count="${ga_count:-8}"


# Activate Environment
. ${PYENV_ROOT}/versions/aaml-simclr/bin/activate

# Obtain all nodes for the slurm job and set a master node
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
nodes_array=($nodes)
head_node=${nodes_array[1]}

# Print out the head node (master)
echo Node Ip: $head_node

# Logging and set eth0 as main communication port
export LOGLEVEL_INFO
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0

# Print Config
echo "======== SIMCLR JOB CONFIG ========"
echo "SLURM_JOB_ID:        $SLURM_JOB_ID"
echo "Config file:         $config"
echo "Encoder:             $encoder"
echo "Epochs:              $epochs"
echo "Batch size:          $batch_size"
echo "Widening:             $widening"
echo "Optimizer:           $optimizer"
echo "Learning rate:       $lr"
echo "Weight decay:        $weight_decay"
echo "Epsilon:             $eps"
echo "Dataset:             $dataset_name"
echo "Augmentations:       $augmentations"
echo "Resize:              $resize"
echo "Projection dim:      $projection_dim"
echo "Temperature:         $temperature"
echo "Resume:              $resume_flag"
echo "Checkpoint:          $checkpoint"
echo "Gradient Accum:      $ga_flag"
echo "Accum steps:         $ga_count"
echo "Head node:           $head_node"
echo "NCCL IFACE:          ${NCCL_SOCKET_IFNAME:-eth0}"
echo "==================================="

# Store Config
timestamp=$(date +'%Y-%m-%d_%H-%M-%S')
output_dir="./runs/${SLURM_JOB_ID}"
mkdir -p "$output_dir"

cat <<EOF > "$output_dir/config_snapshot.yaml"
slurm_job_id: $SLURM_JOB_ID
config: $config
encoder: $encoder
epochs: $epochs
batch_size: $batch_size
widening: $widening
optimizer: $optimizer
lr: $lr
weight_decay: $weight_decay
eps: $eps
dataset_name: $dataset_name
augmentations: $augmentations
resize: $resize
projection_dim: $projection_dim
temperature: $temperature
resume: $resume
checkpoint: $checkpoint
gradient_accumulation: $gradient_accumulation
ga_count: $ga_count
launch_time: $timestamp
EOF



# Execute
srun torchrun --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node:29603 \
    train_ddp.py \
    --slurm_job_id=$SLURM_JOB_ID \
    --config "$config" \
    --encoder "$encoder" \
    --epochs "$epochs" \
    --batch_size "$batch_size" \
    --widening "$widening" \
    --optimizer "$optimizer" \
    --lr "$lr"  \
    --weight_decay "$weight_decay" \
    --eps "$eps" \
    --dataset_name "$dataset_name" \
    --augmentations "$augmentations" \
    --resize "$resize" \
    --projection_dim "$projection_dim" \
    --temperature "$temperature" \
    --checkpoint "$checkpoint" \
    --ga_count "$ga_count" \
    $resume_flag \
    $ga_flag