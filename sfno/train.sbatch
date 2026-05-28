#!/bin/bash

#SBATCH --job-name=sfno_scaling          # short name for the job
#SBATCH --nodes=4                  # number of nodes
#SBATCH --ntasks-per-node=1         # run 1 task per node
#SBATCH --gpus-per-node=4           # GPUs per node
#SBATCH --exclusive
#SBATCH --time=8:00:00             # total run time (HH:MM:SS)
#SBATCH --account=your_account
#SBATCH --partition=normal
#SBATCH --output=logs/output_%j.out  # output log file


# ------------------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------------------
export OMP_NUM_THREADS=4
ulimit -c 0  # disable core dumps
ulimit -t unlimited
export NCCL_CROSS_NIC=1
export NCCL_DEBUG="WARN"
export TORCH_DISTRIBUTED_TIMEOUT=1800

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

# ------------------------------------------------------------------------------
# Set environment variables
# ------------------------------------------------------------------------------
export HTTP_PROXY="your_proxy"
export HTTPS_PROXY="your_proxy"

# Set the working directory
export WORKDIR="/workspace"

# Training parameters
export BATCH_SIZE=16 # this will override the batch size in the yaml config
export RUN_NUM="00"
WIDTH=256 # scaling laws analysis
DEPTH=4 # scaling laws analysis
export CONFIG="base_69scaling_scale3_wid${WIDTH}_layers${DEPTH}_lr1e4"
if [ $DEPTH -eq 4 ]; then
  export CONFIG="base_69scaling_scale3_wid${WIDTH}_lr1e4"
fi
export YAML_CONFIG="./config/sfno_scaling.yaml"

echo "Run started at:" $(date)

# Set the embed_dim variable (users can modify this environment variable)
export WANDB_KEY="$WANDB_KEY"

srun --ntasks=$nnodes \
     --ntasks-per-node=1 \
     --cpus-per-task=72 \
     --export=ALL \
     --environment=sfno_scaling \
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
      makani/train.py \
        --yaml_config ${YAML_CONFIG} \
        --config ${CONFIG} \
        --run_num ${RUN_NUM} \
        --batch_size ${BATCH_SIZE} \
        --parameters_reduction_buffer_count 1 \
        --enable_odirect \
        --checkpointing_level 0 \
        --print_timings_frequency 100 \
        --validation_interval 40 \
        --resuming \
        2>&1
'

echo "Run finished at: $(date)"
