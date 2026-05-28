#!/bin/bash

#SBATCH --job-name=pangu_scaling         # short name for the job
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1         # run 1 task per node
#SBATCH --gpus-per-node=4           # GPUs per node
#SBATCH -c 72                       # CPU cores per task
#SBATCH --exclusive
#SBATCH --time=12:00:00             # total run time (HH:MM:SS)
#SBATCH --account=your_account
#SBATCH --partition=normal
#SBATCH --output=logs/train_pangu_%j_n4_ddp.out  # output log file

# ------------------------------------------------------------------------------
# Environment setup
# (Adjust or remove lines to match your environment)
# ------------------------------------------------------------------------------
#export WANDB_KEY="$WANDB_KEY"
#export WANDB__SERVICE_WAIT=300
export OMP_NUM_THREADS=4
ulimit -c 0  # disable core dumps
ulimit -t unlimited
export NCCL_CROSS_NIC=1
# export NCCL_DEBUG="INFO"
# ------------------------------------------------------------------------------
# Rendezvous setup
# ------------------------------------------------------------------------------
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR="$master_addr"
export MASTER_PORT=29501
export WORLD_SIZE=$SLURM_NTASKS_PER_NODE

node_rank=$SLURM_NODEID
nnodes=$SLURM_JOB_NUM_NODES

echo "================= SLURM Info ================="
echo "Run started at:    $(date)"
echo "SLURM_JOB_ID:      $SLURM_JOB_ID"
echo "SLURM_NODELIST:    $SLURM_JOB_NODELIST"
echo "MASTER_ADDR:       $MASTER_ADDR"
echo "MASTER_PORT:       $MASTER_PORT"
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "node_rank:         $node_rank"
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "================================================"
set -x


export HTTP_PROXY="your_proxy"
export HTTPS_PROXY="your_proxy"


# Set the working directory (users can modify this environment variable)
export WORKDIR="/data"

echo "Run started at:" $(date)

# Set the embed_dim variable (users can modify this environment variable)
export net_type=$1 # scaling laws analysis, for example: scaling_pretrain, deep_scaling_pretrain, wide_scaling_pretrain, large_scaling_pretrain, shallow_scaling_pretrain, narrow_scaling_pretrain, small_scaling_pretrain

srun --ntasks=$nnodes \
     --ntasks-per-node=1 \
     --export=ALL \
     --environment=pangu_scaling \
     -u -l \
  bash -c '
    cd $WORKDIR
    
    echo "[Node ${SLURM_PROCID}] Starting torchrun ..."
    echo "[Node ${SLURM_PROCID}] Environment:"
    echo "  MASTER_ADDR=$MASTER_ADDR"
    echo "  MASTER_PORT=$MASTER_PORT"
    echo "  SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
    echo "  SLURM_PROCID=$SLURM_PROCID"
    echo "  PWD=$(pwd)"
    echo "  PYTHONPATH=$PYTHONPATH"

    set -x  # Enable command printing
    python3 -c "import torch; print(f\"Torch version: {torch.__version__}\")"
    python3 -c "import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")"
    pip3 install torchdata

    torchrun \
      --nnodes=$SLURM_JOB_NUM_NODES \
      --node_rank=$SLURM_PROCID \
      --nproc_per_node=4 \
      --rdzv_id=42 \
      --rdzv_backend=c10d \
      --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      finetune/scaling_pretrain.py \
      --resume True \
      --type_net $net_type \
         2>&1
'

echo "Run finished at: $(date)"
