#!/bin/bash
#SBATCH -A your_account
#SBATCH --exclusive
#SBATCH -t 6:00:00
#SBATCH -J gcjax_scaling
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=4
#SBATCH --partition=normal
#SBATCH -o logs/gcjax_scaling-%j.out

export NGPU_PER_NODE=4
##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "GPUs per node:= "  $NGPU_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "RANK="$RANK

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
# ******************************************************************************************

export OMP_NUM_THREADS=4

echo "Run started at:- " 
date
export WANDB_KEY=$WANDB_KEY
export XLA_FLAGS="--xla_dump_to=logs/xla_dump"

# add other parameters and pass to srun
latent_size=32 # scaling laws analysis: width
gnn_msg_steps=4 # scaling laws analysis: depth
mesh_size=4
nodes=${SLURM_JOB_NUM_NODES:-1}
checkpoint_dir="/scratch/gcjax_checkpoint_n${nodes}_mesh${mesh_size}_lat${latent_size}_gstep${gnn_msg_steps}"

# automatically set checkpoint_dir to have all parameter
srun --mpi=pmi2 --environment=graphcast0326 bash -c "cd /graphcast &&  \
unset HTTPS_PROXY HTTP_PROXY http_proxy https_proxy no_proxy NO_PROXY && \
free -h >> node_log && nvidia-smi >> node_log && \
mkdir ${checkpoint_dir} && echo $SLURM_JOB_ID >> ${checkpoint_dir}/job_id_list && \
python3 -u graphcast/graphcast_trainer.py --resume-training --distribute \
    --normalization-dir ./stats --checkpoint-dir $checkpoint_dir \
    --latent-size $latent_size --gnn-msg-steps $gnn_msg_steps --mesh-size $mesh_size \
    --steps-per-epoch 91 --epochs 10 --warmup-steps 200 \
    --cosine-steps 710 \
    --zarr-path /ERA5/weatherbench2_original --wandb-mode offline"
