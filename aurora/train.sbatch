#!/bin/bash

#SBATCH --job-name=aurora_train          # short name for the job
#SBATCH --nodes=4                   # number of nodes
#SBATCH --ntasks-per-node=1         # run 1 task per node
#SBATCH --gpus-per-node=4           # GPUs per node
#SBATCH --exclusive
#SBATCH --time=8:00:00             # total run time (HH:MM:SS)
#SBATCH --account=your_account
#SBATCH --partition=normal
#SBATCH --output=logs/aurora_train_%j_n4.out  # output log file

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
export NCCL_DEBUG="INFO"
# ------------------------------------------------------------------------------
# Rendezvous setup
# ------------------------------------------------------------------------------
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR="$master_addr"
export MASTER_PORT=29501

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
-------------------------------


export HTTP_PROXY="your_proxy"
export HTTPS_PROXY="your_proxy"


# Set the working directory (users can modify this environment variable)
export WORKDIR="/workspace"

echo "Run started at:" $(date)

# Set the embed_dim variable (users can modify this environment variable)
export embed_dim=$1
export depth_scaling_factor=$2
export learning_rate=$3
export max_norm=$4
echo "embed_dim: $embed_dim"
echo "depth_scaling_factor: $depth_scaling_factor"
echo "learning_rate: $learning_rate"
echo "max_norm: $max_norm"

srun --ntasks=$nnodes \
     --ntasks-per-node=1 \
     --cpus-per-task=72 \
     --export=ALL \
     --environment=your_environment \
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
    
    torchrun \
      --nnodes=$SLURM_JOB_NUM_NODES \
      --node_rank=$SLURM_PROCID \
      --nproc_per_node=4 \
      --rdzv_id=42 \
      --rdzv_backend=c10d \
      --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      train_fsdp.py \
        --seed 42 \
        --batch_size 1 \
        --num_workers 1 \
        --epochs 40 \
        --devices 4 \
        --embed_dim ${embed_dim} \
        --learning_rate ${learning_rate} \
        --constant_lr_after_warmup \
        --depth_scaling_factor ${depth_scaling_factor} \
        --max_norm ${max_norm} \
        --num_nodes $SLURM_JOB_NUM_NODES \
        --wnb_mode disabled  \
        --resume \
        --no-load_aurora_pretrain_weights \
        --log_dir  /scratch/aurora_checkpoints/n${SLURM_JOB_NUM_NODES}_0525_dim${embed_dim}_dsf${depth_scaling_factor}_newgclip${max_norm}_constafterwarm_gcloss_lr${learning_rate}_distsamp-new \
        2>&1
'

echo "Run finished at: $(date)"
