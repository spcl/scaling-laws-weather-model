# Scaling laws for GraphCast

This repo is based on the official JAX implementation from [Google DeepMind](https://github.com/google-deepmind/graphcast). Significant debugging and reimplementation of the training pipeline were required, including replacing the Colab-based demonstration with a full Python script, adding explicit gradient all-reduce, checkpointing with optimizer state, and support for multi-node training.

## Installation

Install using the Dockerfile:

```bash
docker build -f container/graphcast.Dockerfile -t graphcast:latest .
```

The Dockerfile sets up:
- CUDA 12.3.1 runtime environment
- Python dependencies including JAX with CUDA support
- GraphCast package from DeepMind's repository
- Additional dependencies: `google-cloud-storage`, `gcsfs`, `optax`, `wandb`, `importlib_resources`

## Running Training

Run training using the SLURM batch script:

```bash
sbatch graphcast/train.sbatch
```

### Configuration

The training script (`graphcast/train.sbatch`) can be customized with the following scaling parameters:

- **Width**: `latent_size` (default: 32)
- **Depth**: `gnn_msg_steps` (default: 4)

### Example

To modify scaling parameters, edit the variables in `graphcast/train.sbatch`:

```bash
latent_size=64  # Change width
gnn_msg_steps=8  # Change depth
```

The checkpoint directory is automatically generated based on these parameters and the number of nodes used.

### Requirements

- SLURM workload manager
- Access to GPU nodes (4 GPUs per node recommended)
- ERA5 data in Zarr format following WeatherBench2 conventions
- Normalization statistics files in `graphcast/stats/` directory

