# Scaling Laws of Global Weather Models

This repository contains training code for reproducing the scaling results of the following weather models:

- **Aurora** - A foundation model of the atmosphere
- **AIFS** - AI Forecasting System
- **Pangu** - Pangu-Weather model
- **SFNO** - Spherical Fourier Neural Operator
- **GraphCast** - Graph-based neural weather model

**Note:** The codes in this repository are modified from public repositories.

## Implementation Overview

All models in this work are implemented with PyTorch or JAX backends. To ensure reproducibility, we containerize the environment setup with Dockerfiles specifying CUDA, NCCL, and model-specific dependencies.

## Model-Specific Modifications

### GraphCast
We adopt the official JAX implementation from [Google DeepMind](https://github.com/google-deepmind/graphcast), but find that significant debugging and reimplementation of the training pipeline are required. In particular, we replace the Colab-based demonstration with a full Python script, add explicit gradient all-reduce, checkpointing with optimizer state, and support for multi-node training.

### Aurora
Aurora is based on the official [Microsoft implementation](https://github.com/microsoft/aurora) and [ESFM](https://github.com/swiss-ai/ESFM). ESFM includes a re-implementation of the training loop in PyTorch Lightning and a new stateful dataloader to enable flexible checkpointing and reliable resumption.

### Pangu
Pangu relies on a PyTorch reimplementation ([pangu-pytorch](https://github.com/zhaoshan2/pangu-pytorch)), with its earlier pseudo-code version ([Pangu-Weather](https://github.com/198808xc/Pangu-Weather)) consulted but not sufficient for reproducibility. To ensure experimental consistency, we use the same stateful dataloader as Aurora that supports mid-epoch resumption, alongside modifications for distributed training.

### SFNO
SFNO is reproduced using [NVIDIA's Makani repository](https://github.com/NVIDIA/makani); although the base implementation is stable, we re-implement its dataloader in PyTorch to ensure statefulness and consistency with the other models.

### AIFS
AIFS is provided in [Anemoi package](https://github.com/ecmwf/anemoi-core) from ECMWF.

## Data Pipeline

We developed a unified data pipeline across all models using ERA5 reanalysis data in Zarr format. Following WeatherBench conventions, we use a 6-hour timestep and restrict the training set to the 0, 6, 12, and 18 UTC hours of each day. This choice balances temporal coverage with computational feasibility while ensuring comparability across models.

## Checkpointing

Checkpointing is implemented in all models to save both parameters and optimizer states at regular intervals. All PyTorch-based models (Aurora, AIFS, Pangu, SFNO) require custom stateful dataloaders or saving the index of used data batches to support reliable checkpointing, while GraphCast's JAX pipeline demands a full reimplementation of training to achieve consistent multi-node behavior.

## Scaling Behavior

To explore scaling behavior, models are modified in width and depth, while holding other factors constant:

- **GraphCast**: We vary latent size and message-passing steps.
- **Aurora**: We systematically adjust width and depth for every layer in the encoder and decoder of the Swin Transformer backbone.
- **Pangu**: We modify width and depth within the constraints of the official implementation, which splits surface and upper-air variables.
- **SFNO**: We change the embedding dimension and number of operator layers.
- **AIFS**: We vary the latent size as width, and the number of transformer blocks in the Transformer backbone as depth.