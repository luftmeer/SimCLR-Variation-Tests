#!/bin/zsh
#b Name and Files (also --job-name)
#SBATCH -J simclr-multinode
#Output and error (also --output, --error):
#SBATCH -o ./%x.%j.out
#SBATCH -e ./%x.%j.err
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
#SBATCh --cpus-per-task=4


. ${PYENV_ROOT}/versions/aaml-simclr/bin/activate

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
nodes_array=($nodes)
head_node=${nodes_array[1]}

echo Node Ip: $head_node

export LOGLEVEL_INFO
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0

srun torchrun --nnodes=4 --nproc_per_node=1 --rdzv_id=$RANDOM --rdzv_backend=c10d --rdzv_endpoint=$head_node:29603 train_ddp.py --slurm_job_id=$SLURM_JOB_ID --config=./config/resnet50/CIFAR10_nowidening_bs64_halfprecision_2augments_4nodes.yaml