#!/bin/bash
#SBATCH --job-name=aifs          # short name for the job
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1         # run 1 srun task per node (srun launches 1 torchrun which spawns 4 processes)
#SBATCH --gpus-per-node=4           # GPUs per node
#SBATCH --cpus-per-task=72          # CPU cores per task (all cores for the single task)
#SBATCH --exclusive
#SBATCH --time=12:00:00             # total run time (HH:MM:SS)
#SBATCH --account=your_account
#SBATCH --partition=normal
#SBATCH --output=logs/aifs_scaling_%j.out  # output log file

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
export NGPUS_PER_NODE=4  # Match #SBATCH --gpus-per-node

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

echo "Run started at:" $(date)

export width=${1:-512} # scaling laws analysis: width
export depth=${2:-16} # scaling laws analysis: depth
export lr=${3:-6.25e-6}
export num_heads=$((width/64))


export run_id="null"

# Launch one task per node; each task runs torchrun which spawns NGPUS_PER_NODE processes
# check latest environment
srun --ntasks=$nnodes \
     --ntasks-per-node=1 \
     --export=ALL \
     --environment=aifs_scaling \
     -u -l \
  bash -c '
    echo "[Node ${SLURM_NODEID}] Starting torchrun ..."
    echo "[Node ${SLURM_NODEID}] Environment:"
    echo "  MASTER_ADDR=$MASTER_ADDR"
    echo "  MASTER_PORT=$MASTER_PORT"
    echo "  SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
    echo "  SLURM_NODEID=$SLURM_NODEID"
    echo "  PWD=$(pwd)"
    echo "  PYTHONPATH=$PYTHONPATH"

    set -x  # Enable command printing
    python3 -c "import torch; print(f\"Torch version: {torch.__version__}\")"
    python3 -c "import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")"
    
    # pip install --no-cache-dir --src /tmp/ ./graphs ./models ./training pytorch-lightning==2.5.5


    torchrun --nproc_per_node=$NGPUS_PER_NODE \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --node_rank=$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        training/src/anemoi/training/train/train.py \
        training.run_id=$run_id \
        training.lr.rate=$lr \
        model.num_channels=$width \
        model.processor.num_layers=$depth \
        model.processor.num_heads=$num_heads \
        model.encoder.num_heads=$num_heads \
        model.decoder.num_heads=$num_heads 
'

echo "Run finished at: $(date)"
