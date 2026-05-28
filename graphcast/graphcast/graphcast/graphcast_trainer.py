#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphCast Distributed Training

This script trains a GraphCast model using distributed data-parallel training
across multiple devices (GPUs/TPUs), with checkpoint support.
"""

import os
import time
import datetime
import dataclasses
import pickle
import re
import functools
import argparse
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union, List, Callable, NamedTuple
from copy import deepcopy
import json

import numpy as np
import xarray as xr
import optax  # Optimizer library for JAX

import jax
jax.config.update("jax_compilation_cache_dir", "/jax_cache")
# jax.config.update("jax_mock_gpu_topology", "2x4x1")

import jax.numpy as jnp
from jax.sharding import PositionalSharding
from jax.experimental import mesh_utils
import haiku as hk
from jax.tree_util import tree_map, tree_flatten, tree_unflatten, tree_leaves, tree_map_with_path

# Import WandB for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: Weights & Biases (wandb) not available. Install with: pip install wandb")

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree


class TrainState(NamedTuple):
  params: Any
#   state: Any
  opt_state: Any
  

# ------------------------
# Learning Rate Schedules
# ------------------------

def create_sequential_lr_schedule(
    base_learning_rate: float,
    num_iters_step1: int,  # Linear warmup
    num_iters_step2: int,  # Cosine annealing
    step3_lr: float = 0.1,  # Fixed LR for step 3
    start_factor: float = 1e-3,    # Start factor for linear warmup
    cosine_eta_min: float = 0.0    # Minimum value for cosine annealing
) -> Callable[[int], float]:
    """Create a sequential learning rate schedule with three phases:
    1. Linear warmup from base_lr * start_factor to base_lr
    2. Cosine annealing from base_lr to cosine_eta_min
    3. Fixed learning rate at base_lr * step3_lr_factor
    
    Uses optax schedulers for JAX compatibility.
    """
    # Phase 1: Linear warmup scheduler
    warmup_schedule = optax.linear_schedule(
        init_value=base_learning_rate * start_factor,
        end_value=base_learning_rate,
        transition_steps=num_iters_step1
    )
    
    # Phase 2: Cosine annealing scheduler
    if num_iters_step2 > 0:
        cosine_schedule = optax.cosine_decay_schedule(
            init_value=base_learning_rate,
            decay_steps=num_iters_step2,
            alpha=cosine_eta_min / base_learning_rate  # Alpha is the final value as a fraction of init_value
        )
    else:
        cosine_schedule = optax.constant_schedule(base_learning_rate)
    
    # Phase 3: Constant scheduler for fixed learning rate
    fixed_schedule = optax.constant_schedule(step3_lr)
    
    # Combine the three schedulers with appropriate boundaries
    return optax.join_schedules(
        schedules=[warmup_schedule, cosine_schedule, fixed_schedule],
        boundaries=[num_iters_step1, num_iters_step1 + num_iters_step2]
    )

# ------------------------
# Checkpoint Classes
# ------------------------

@dataclasses.dataclass(frozen=True)
class GraphCastTrainingCheckpoint:
    """Training checkpoint for GraphCast that preserves optimizer state."""
    params: hk.Params
    opt_state: optax.OptState
    task_config: graphcast.TaskConfig
    model_config: graphcast.ModelConfig
    global_step: int
    rng: jax.Array
    created_at: str  # Added created_at field
    step_count: int = 0
    best_loss: Optional[float] = None
    learning_rate: float = 1e-4
    metadata: Optional[Dict[str, Any]] = None


def save_checkpoint(directory: Path, ckpt: GraphCastTrainingCheckpoint, global_rank: int = 0) -> None:
    """
    Stores a GraphCast training checkpoint at the given directory with the name
    directory / graphcast_{epoch}.pkl
    
    If the given directory does not exist, it will be created.
    If there already exists a checkpoint in the same directory with the same epoch,
    it will be overwritten.
    
    Only rank 0 process will save the checkpoint to avoid conflicts.
    
    Args:
        directory: Path to the directory where checkpoint will be saved
        ckpt: GraphCastTrainingCheckpoint object to save
        global_rank: Current process rank (default 0)
    """
    # Only rank 0 saves checkpoints
    if global_rank != 0:
        return
    
    directory.mkdir(parents=True, exist_ok=True)
    save_file = directory / f"graphcast_{ckpt.global_step}.pkl"
    
    # Save the checkpoint using pickle
    with open(save_file, mode="wb") as file:
        pickle.dump(ckpt, file)
    
    print(f"Saved checkpoint at global_step {ckpt.global_step} to {save_file}")

def load_checkpoint(directory: Path, global_step: int = -1) -> Union[GraphCastTrainingCheckpoint, None]:
    """
    Loads a GraphCast training checkpoint from the given directory.
    
    If a non-negative global_step is given, that specific checkpoint is loaded.
    Otherwise, the latest global_step is loaded.
    If no checkpoint is found, None is returned.
    """
    if not directory.exists():
        print(f"Checkpoint directory {directory} does not exist")
        return None
    
    # Define the pattern for GraphCast checkpoint files
    pattern = r"graphcast_(\d+)\.pkl"
    
    # Find all matching checkpoint files
    file_tuples = []
    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        if match:
            i = int(match.group(1))
            filepath = os.path.join(directory, filename)
            file_tuples.append((i, filepath))
    
    if len(file_tuples) == 0:
        print(f"No GraphCast checkpoints found in {directory}")
        return None
    
    # Sort checkpoints by epoch
    file_tuples.sort()
    
    ckpt_path = None
    if global_step < 0:
        # Load the latest checkpoint
        ckpt_path = file_tuples[-1][1]
        global_step_num = file_tuples[-1][0]
        print(f"Loading latest checkpoint (global_step {global_step_num})")
    else:
        # Load a specific global_step
        for (i, f) in file_tuples:
            if i == global_step:
                ckpt_path = f
                print(f"Loading checkpoint from global_step {i}")
                break
        if ckpt_path is None:
            print(f"No checkpoint found for global_step {global_step}")
            return None
    
    # Load the checkpoint using pickle
    try:
        with open(ckpt_path, "rb") as file:
            ckpt = pickle.load(file)
        print(f"Successfully loaded checkpoint from {ckpt_path}")
        return ckpt
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


# ------------------------
# Dataset Classes
# ------------------------

class WeatherBench2Dataset:
    """WeatherBench2 Dataset loader."""
    
    def __init__(self, start_year: int, end_year: int, steps: int, steps_per_input: int = 3, 
                 zarr_path: str = '/ERA5/weatherbench2_original'):
        """Initialize WeatherBench2 dataset.
        
        Args:
            start_year, end_year: Year to load data for
            steps: Number of time steps per batch
            steps_per_input: Number of steps per input (typically 3 for GraphCast)
            zarr_path: Path to the zarr dataset
        """
        self.vars = ["geopotential", "specific_humidity", "temperature",
                     "u_component_of_wind", "v_component_of_wind",
                     "vertical_velocity", "toa_incident_solar_radiation",
                     "10m_u_component_of_wind", "10m_v_component_of_wind",
                     "2m_temperature", "mean_sea_level_pressure",
                     "total_precipitation_6hr", "geopotential_at_surface", "land_sea_mask"]
        self.static_vars = ["geopotential_at_surface", "land_sea_mask"]
        
        print(f"Loading WeatherBench2 dataset for year {start_year}-{end_year} from {zarr_path}")
        self.ds = xr.open_zarr(zarr_path, consolidated=False)
        self.ds = self.ds[self.vars]
        self.ds = self.ds.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'))
        self.length = len(self.ds.time)
        self.steps = steps
        self.steps_per_input = steps_per_input
        self.coord_name_dict = dict(latitude="lat", longitude="lon")
        
        print(f"Dataset loaded with {self.length} time steps")

    def __len__(self):
        """Get number of batches in the dataset."""
        num_batches = self.length // self.steps
        if self.length % self.steps < self.steps_per_input - 1:
            num_batches -= 1
        return num_batches

    def __getitem__(self, item):
        """Get batch by index."""
        return self.get_data(item)

    def get_data(self, batch_idx):
        """Get data for a specific batch index."""
        batch_idx = int(batch_idx) % len(self)  # Ensure batch_idx is an integer scalar
        it_range = slice(batch_idx*self.steps, (batch_idx+1)*self.steps + self.steps_per_input - 1)
        static_data = self.ds[self.static_vars].rename(**self.coord_name_dict)
        data = self.ds.drop_vars(self.static_vars).isel(time=it_range)
        data = data.rename(**self.coord_name_dict)
        static_data = static_data.isel(lat=slice(None, None, -1))
        data = data.isel(lat=slice(None, None, -1))
        data = xr.merge([static_data, data.expand_dims({'batch': 1})])
        data = data.assign_coords(datetime=(["batch", "time"], data.time.data.reshape(1, -1)))
        data = data.assign_coords(time=("time", data.time.data - data.time.data[0]))
        return data.compute()

    def get_batch(self, batch_indices):
        """Get multiple batches as a single concatenated batch."""
        # Ensure all batch indices are scalar integers
        batch_indices = [int(idx) for idx in batch_indices.flatten()]
        batches = [self.get_data(idx) for idx in batch_indices]
        return xr.concat(batches, dim="batch")
    
    def get_data_by_date(self, date: str) -> xr.Dataset:
        """
        Get a sample batch from an exact datetime string (e.g., '2017-01-01T00:00').

        The function searches for the corresponding index and constructs the required window.
        """
        target_time = np.datetime64(date)
        time_index = self.ds.indexes['time'].get_loc(target_time)

        if isinstance(time_index, slice) or isinstance(time_index, np.ndarray):
            raise ValueError(f"Date '{date}' matched multiple entries.")

        # Ensure we have enough steps before and after
        required_window = self.steps + self.steps_per_input - 1
        start_index = max(0, time_index - (self.steps_per_input - 1))
        end_index = start_index + required_window

        if end_index > self.length:
            raise IndexError(f"Cannot get full input window for date {date} — not enough future data.")

        it_range = slice(start_index, end_index)

        static_data = self.ds[self.static_vars].rename(**self.coord_name_dict)
        data = self.ds.drop_vars(self.static_vars).isel(time=it_range).rename(**self.coord_name_dict)
        static_data = static_data.isel(lat=slice(None, None, -1))
        data = data.isel(lat=slice(None, None, -1))
        data = xr.merge([static_data, data.expand_dims({'batch': 1})])
        data = data.assign_coords(datetime=(["batch", "time"], data.time.data.reshape(1, -1)))
        data = data.assign_coords(time=("time", data.time.data - data.time.data[0]))

        return data.compute()


def load_normalization(path: str, downsample: bool = False):
    diffs_stddev_by_level = xr.load_dataset(f"{path}/diffs_stddev_by_level.nc").compute()
    mean_by_level = xr.load_dataset(f"{path}/mean_by_level.nc").compute()
    stddev_by_level = xr.load_dataset(f"{path}/stddev_by_level.nc").compute()
    return diffs_stddev_by_level, mean_by_level, stddev_by_level

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a GraphCast model with distributed training")
    parser.add_argument("--zarr-path", type=str, default='/ERA5/weatherbench2_original',
                        help="Path to the WeatherBench2 zarr dataset")
    parser.add_argument("--start-year", type=int, default=1979,
                        help="Year to use for training data")
    parser.add_argument("--end-year", type=int, default=2016,
                        help="Year to use for training data")
    parser.add_argument("--val-year", type=int, default=2017,
                        help="Year to use for validation")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--normalization-dir", type=str, required=True,
                        help="Directory containing normalization statistics")
    parser.add_argument("--mesh-size", type=int, default=6,
                        help="Mesh size (4-6)")
    parser.add_argument("--latent-size", type=int, default=512,
                        help="Latent size (16-512)")
    parser.add_argument("--gnn-msg-steps", type=int, default=16,
                        help="GNN message steps (1-32)")
    parser.add_argument("--pressure-levels", type=int, default=13, choices=[13, 37],
                        help="Number of pressure levels (13 or 37)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=4,
                        help="Number of batches per epoch")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--step3-learning-rate", type=float, default=3e-7,
                        help="Step 3 Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--cosine-steps", type=int, default=1000,
                        help="Number of cosine annealing steps for learning rate scheduler")
    parser.add_argument("--checkpoint-frequency", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--validation-frequency", type=int, default=1,
                        help="Validate every N epochs")
    parser.add_argument("--resume-training", action="store_true",
                        help="Resume training from the latest checkpoint")
    parser.add_argument("--resume-global-step", type=int, default=-1,
                        help="Resume from a specific global step (-1 for latest)")
    parser.add_argument("--distribute", action="store_true",
                        help="Enable distributed training")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Number of steps to accumulate gradients over")
    # WandB related arguments
    parser.add_argument("--wandb-project", type=str, default="Graphcast-Training",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="Weights & Biases entity (username or team name)")
    parser.add_argument("--wandb-mode", type=str, default="online",
                        help="Weights & Biases mode (online/offline)")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="Weights & Biases run name")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    return parser.parse_args()

# Global variable to store the pmap'd function
pmap_device_train_step = None

def update(optimizer, loss_fn, train_state, state, inputs, targets, forcings, n_devices, rng):
    """Execute a distributed training step."""
    # Define the individual device training function
    # pass config with partial from loss_fn
    def _loss_fn_aux(params, state, inputs, targets, forcings):
        (loss, diagnostics), next_state = loss_fn.apply(
            params, state, rng,
            inputs, targets, forcings)
        return loss, (diagnostics, next_state)
    
    def device_train_step_grad_and_update(train_state, state,
                        inputs, targets, forcings):
        """Training step for a single device."""
        # Compute loss and gradients
        (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
            _loss_fn_aux, has_aux=True)(train_state.params, state, inputs, targets, forcings)
        
        # print('loss from device_train_step, shape:', loss.shape, 'value:', loss)

        # Average gradients across devices, must be called in pmap function
        grads = jax.lax.pmean(grads, axis_name=pmap_dim)
        
        # Apply the averaged gradients to update parameters
        # print(train_state.opt_state, train_state.params)
        updates, new_optimizer_state = optimizer.update(grads, train_state.opt_state, train_state.params)
        new_params = optax.apply_updates(train_state.params, updates)
        
        return new_params, next_state, new_optimizer_state, loss, diagnostics
    
    global pmap_device_train_step
    
    pmap_dim = "device_count"
    if pmap_device_train_step is None:
        pmap_device_train_step = xarray_jax.pmap(device_train_step_grad_and_update, dim=pmap_dim, axis_name=pmap_dim)
    
    # Add leading dimension to training state and state components
    # train_state = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), train_state)
    # state = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), state)
    
    train_state = jax.tree_map(lambda x: jnp.stack([x] * n_devices), train_state)
    state = jax.tree_map(lambda x: jnp.stack([x] * n_devices), state)
    
    # rename xarray
    inputs = inputs.rename(batch=pmap_dim)
    targets = targets.rename(batch=pmap_dim)
    forcings = forcings.rename(batch=pmap_dim)

    # Now we have ('device_count':n_devices, 'batch':1, 'time': 2, 'lat': 721, 'lon': 1440, 'level': 37)
    inputs = inputs.expand_dims("batch", axis=1)
    targets = targets.expand_dims("batch", axis=1)
    forcings = forcings.expand_dims("batch", axis=1)

    # Run the distributed computation
    # before calling pmap_device_train_step, use expand_dims of pmap_dim
    new_params, next_state, new_optimizer_state, loss, diagnostics = pmap_device_train_step(
        train_state, state, 
        inputs, targets, forcings)
    
    # After pmap computation, select the right output for each device
    loss = jax.device_get(loss)
    # print(diagnostics.dims)
    # debug
    
    # always use the first rank
    rank = 0
    diagnostics = jax.tree_map(lambda x: x[rank], diagnostics)
    next_state = jax.tree_map(lambda x: x[rank], next_state)
    new_params = jax.tree_map(lambda x: x[rank], new_params)
    new_optimizer_state = jax.tree_map(lambda x: x[rank], new_optimizer_state)
    # print("grads.shape",grads.shape)
    
    new_train_state = TrainState(
        new_params,
        new_optimizer_state
    )
    # debug
    # print("print anything")
    print("new_params, next_state, new_optimizer_state",new_params, next_state, new_optimizer_state)
    
    return new_train_state, loss, next_state, diagnostics

def setup_devices():
    """Setup devices for distributed training."""
    
    # TODO: debug multi-node training
    jax.distributed.initialize(local_device_ids=[0,1,2,3])

    # Check the process configuration
    process_count = jax.process_count()
    process_index = jax.process_index()
    print(f"Process {process_index+1}/{process_count}")

    # Get all devices across all processes
    devices = jax.devices()
    local_devices = jax.local_devices()
    n_devices = len(local_devices)
    print(f"Total devices: {len(devices)}")
    print(f"Local devices: {len(local_devices)}")
    
    # Set up a positional sharding for data-parallel training
    sharding = PositionalSharding(devices)
    
    return devices, n_devices, sharding
  
def create_train_state(params, learning_rate, optimizer_state=None, optimizer=None):
    """Create training state dictionary."""
    if optimizer is None:
        optimizer = optax.adam(learning_rate=learning_rate)
    if optimizer_state is None:
        optimizer_state = optimizer.init(params)
    # print('optimizer_state',optimizer_state)
    return TrainState(
        params=params,
        opt_state=optimizer_state,
    )


def main():
    """Main training function."""
    args = parse_args()
    
    # Set up devices for distributed training
    devices, n_devices, sharding = setup_devices()
    is_distributed = args.distribute and n_devices > 1
    
    # Get the process/device rank for checkpoint saving
    process_rank = jax.process_index()
    
    # Set up WandB if enabled
    loss_log_path = ''
    loss_log = []
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        process_rank = jax.process_index()
        # Only the main process should log to WandB
        if process_rank == 0:
            # Generate a unique run name if not provided
            run_name = args.wandb_run_name
            if run_name is None:
                run_name = f"graphcast-{args.mesh_size}-{args.latent_size}-{args.gnn_msg_steps}-{args.start_year}-{args.end_year}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            wandb_id = f'{args.wandb_project}_{str(args.checkpoint_dir).split("/")[-1]}' #_{time.strftime("%Y%m%d-%H%M%S")}'

            wandb_key = os.getenv("WANDB_KEY")
            if wandb_key and args.wandb_mode=="online":
                wandb.login(key=wandb_key)
            else:
                print("Wandb not logged in.")
                
            # Setup WandB configuration
            config = {
                "start_year": args.start_year,
                "end_year": args.end_year,
                "val_year": args.val_year,
                "mesh_size": args.mesh_size,
                "latent_size": args.latent_size,
                "gnn_msg_steps": args.gnn_msg_steps,
                "pressure_levels": args.pressure_levels,
                "learning_rate": args.learning_rate,
                "warmup_steps": args.warmup_steps,
                "cosine_steps": args.cosine_steps,
                "epochs": args.epochs,
                "steps_per_epoch": args.steps_per_epoch,
                "distribute": args.distribute,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
            }
            
            # Initialize WandB
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=config,
                save_code=True,
                id=wandb_id,
                resume='allow',
                mode=args.wandb_mode
            )
            print(f"WandB initialized with run name: {run_name}")
            wandb_run_dir = wandb.run.dir
            # checkpoint_dir="/scratch/gcjax_checkpoint_n${nodes}_mesh${mesh_size}_lat${latent_size}_gstep${gnn_msg_steps}"
            loss_log_path = os.path.join(wandb_run_dir, "loss_log"+args.checkpoint_dir[25:]+".json")
            print(loss_log_path)
            if os.path.exists(loss_log_path):
                with open(loss_log_path, "r") as f:
                    loss_log = json.load(f)
            else:
                with open(loss_log_path, "w") as f:
                    json.dump(loss_log, f)
    
    if is_distributed:
        print(f"Using distributed training across {n_devices} devices")
        print(f"Current process rank: {process_rank}")
    else:
        print(f"Using single-device training (distributed={args.distribute}, devices={n_devices})")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    if process_rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure model
    model_config = graphcast.ModelConfig(
        resolution=0,  # Auto-detect from data
        mesh_size=args.mesh_size,
        latent_size=args.latent_size,
        gnn_msg_steps=args.gnn_msg_steps,
        hidden_layers=1,
        radius_query_fraction_edge_length=0.6)
    
    # Configure task
    task_config = graphcast.TaskConfig(
        input_variables=graphcast.TASK.input_variables,
        target_variables=graphcast.TASK.target_variables,
        forcing_variables=graphcast.TASK.forcing_variables,
        pressure_levels=graphcast.PRESSURE_LEVELS[args.pressure_levels],
        input_duration=graphcast.TASK.input_duration,
    )
    
    print(f"Task config pressure levels: {len(task_config.pressure_levels)}")
    
    # Load normalization statistics
    print(f"Loading normalization statistics from {args.normalization_dir}")
    diffs_stddev_by_level, mean_by_level, stddev_by_level = load_normalization(args.normalization_dir)

    # Initialize model parameters (or load from checkpoint)
    params = None
    state = {}
    optimizer_state = None
    start_step = 0
    start_epoch = 1
    best_loss = float('inf')
    step_count = 0
    rng = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
    
    # Check if we should resume training
    if args.resume_training:
        print(f"Attempting to resume training from {'latest' if args.resume_global_step < 0 else f'global step {args.resume_global_step}'}...")
        ckpt = load_checkpoint(checkpoint_dir, args.resume_global_step)
        
        if ckpt is not None:
            if ckpt.global_step is not None:
                print(f"Resuming from global_step {ckpt.global_step} (created at {ckpt.created_at})")
                assert args.steps_per_epoch == ckpt.metadata["steps_per_epoch"]
                start_step = ckpt.global_step % args.steps_per_epoch + 1
                start_epoch = ckpt.global_step // args.steps_per_epoch + 1
                step_count = getattr(ckpt, 'global_step', 0)
            else:
                print(f"Resuming from epoch {ckpt.epoch} (created at {ckpt.created_at})")
                assert args.steps_per_epoch == ckpt.metadata["steps_per_epoch"]
                start_step = ckpt.step_count % args.steps_per_epoch + 1
                start_epoch = ckpt.step_count // args.steps_per_epoch + 1
                step_count = getattr(ckpt, 'step_count', 0)
            params = ckpt.params
            optimizer_state = ckpt.opt_state
            task_config = ckpt.task_config
            model_config = ckpt.model_config
            if ckpt.best_loss is not None:
                best_loss = ckpt.best_loss
            rng = ckpt.rng
    
    # Use statics in closure
    def construct_wrapped_graphcast(
        model_config, task_config):
        """Constructs and wraps the GraphCast Predictor."""
        # Core GraphCast model
        predictor = graphcast.GraphCast(model_config, task_config)
        
        # Add BFloat16 casting
        predictor = casting.Bfloat16Cast(predictor)
        
        # Add normalization
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level)
        
        # Wrap for autoregressive prediction (without gradient checkpointing for training)
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=False)
        return predictor
    
    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        """Forward pass function."""
        
        predictor = construct_wrapped_graphcast(
            model_config, task_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    @hk.transform_with_state
    def loss_fn(inputs, targets, forcings):
        """Loss function."""
        
        predictor = construct_wrapped_graphcast(
            model_config, task_config)
        loss, diagnostics = predictor.loss(inputs, targets, forcings)
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            (loss, diagnostics))
        
    
    # Create learning rate schedule
    total_steps = args.epochs * args.steps_per_epoch
    
    # Set up learning rate schedule
    lr_schedule = create_sequential_lr_schedule(
        base_learning_rate=args.learning_rate,
        num_iters_step1=args.warmup_steps,
        num_iters_step2=args.cosine_steps,
        step3_lr=args.step3_learning_rate,
        start_factor=0.001
    )
    
    # Initialize optimizer with learning rate schedule
    learning_rate = args.learning_rate
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-1.0)  # Apply negative step (for gradient descent)
    )
    
    
    # Initialize the dataset
    print(f"Initializing WeatherBench2 dataset for year {args.start_year} to {args.end_year}...")
    try:
        steps_size = 6  # 6h
        dataset = WeatherBench2Dataset(
            start_year=args.start_year, 
            end_year=args.end_year,
            steps=steps_size, 
            steps_per_input=3,
            zarr_path=args.zarr_path
        )
        print(f"Training dataset initialized with {len(dataset)} batches")
        val_dataset = WeatherBench2Dataset(
            start_year=args.val_year, 
            end_year=args.val_year,
            steps=steps_size, 
            steps_per_input=3,
            zarr_path=args.zarr_path
        )
        print(f"Validation dataset initialized with {len(val_dataset)} batches")

        
        # Check if we have enough data
        if len(dataset) < args.steps_per_epoch:
            print(f"Warning: Only {len(dataset)} batches available, but {args.steps_per_epoch} requested.")
            args.steps_per_epoch = len(dataset)
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return
    
    # Get a sample batch to initialize the model
    print("Loading a sample batch for initialization...")
    try:
        sample_batch = dataset[0]
        print(f"Sample batch shape: {dict(sample_batch.dims)}")
        
        # Extract input and target data
        train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
            sample_batch, target_lead_times=slice("6h", "6h"),  # Single step prediction
            **dataclasses.asdict(task_config))
        
        print(f"Training data shapes:")
        for var in train_inputs.data_vars:
            print(f"  {var}: {train_inputs[var].shape}")
    except Exception as e:
        print(f"Error extracting data: {e}")
        print("This could be due to incorrect data format or missing variables.")
        return
    
    # Initialize model parameters if not loaded from checkpoint
    if params is None:
        print("Initializing model parameters...")
        init_fn = run_forward.init
        
        def count_params(params):
            return sum(jnp.size(p) for p in tree_leaves(params))
        def print_params(params):
            param_info = []
            
            def visit(path, value):
                if isinstance(value, jnp.ndarray):
                    path_str = [str(p) for p in path]
                    name = "/".join(path_str)
                    size = jnp.size(value)
                    param_info.append((name, size))
            
            tree_map_with_path(lambda p, v: visit(p, v), params)
            
            for name, size in param_info:
                print(f"Parameter: {name}, Size: {size}")
        
        try:
            params, state = init_fn(
                rng=rng,
                inputs=train_inputs,
                targets_template=train_targets,
                forcings=train_forcings)
            
            total_params = count_params(params)
            print(f"Model initialized successfully with {total_params:,} parameters")
            print_params(params)
        except Exception as e:
            print(f"Error initializing model: {e}")
            return
    
    if optimizer_state is None:
        optimizer_state = optimizer.init(params)
    
    # Setup for distributed training
    if not is_distributed:
        # For non-distributed training, create a simpler train step function
        def train_step(params, state, optimizer_state, inputs, targets, forcings, rng):
            # The non-distributed version is similar but doesn't use pmap
            def _loss_fn(params, state, inputs, targets, forcings):
                (loss, diagnostics), next_state = loss_fn.apply(
                    params, state, rng,
                    inputs, targets, forcings)
                return loss, (diagnostics, next_state)
            
            # Compute loss and gradients
            (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
                _loss_fn, has_aux=True)(params, state, inputs, targets, forcings)
            
            # Update parameters
            updates, new_optimizer_state = optimizer.update(grads, optimizer_state, params)
            new_params = optax.apply_updates(params, updates)
            
            return new_params, next_state, new_optimizer_state, loss, diagnostics
        
        # JIT compile for efficiency
        train_step = jax.jit(train_step)
        
    # Create training state
    train_state = create_train_state(params, learning_rate, optimizer_state, optimizer)
    
    # Training loop
    print(f"Starting training from epoch {start_epoch} step {start_step} to epoch {args.epochs} step {args.steps_per_epoch} "
          f"with {args.steps_per_epoch} steps per epoch")
    
    # Set up gradient accumulation if needed
    grad_accumulation_steps = max(1, args.gradient_accumulation_steps)
    if grad_accumulation_steps > 1:
        print(f"Using gradient accumulation over {grad_accumulation_steps} steps")
        
    # Set up partial
    update_fn = functools.partial(update, optimizer=optimizer)
    
    # Set up validation function
    rng_val = np.random.default_rng(seed=42)  # Create a random generator with fixed seed
    all_indices_val = rng_val.permutation(len(val_dataset))  # Reproducible permutation
    val_data = val_dataset.get_batch(all_indices_val[0])  # Get a fixed validation batch, default to 0
    inputs_val, targets_val, forcings_val = data_utils.extract_inputs_targets_forcings(
            val_data, target_lead_times=slice("6h", "6h"),
            **dataclasses.asdict(task_config))

    # Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        epoch_losses = []
        epoch_metrics = {}
        rng_epoch = jax.random.fold_in(rng, epoch)
        
        for step_idx in range(start_step, args.steps_per_epoch, grad_accumulation_steps):
            step_count += 1
            step_losses = []
            
            # Get batch indices for gradient accumulation
            remaining_steps = min(grad_accumulation_steps, args.steps_per_epoch - step_idx)
            
            # Get current learning rate
            current_lr = lr_schedule(step_count)
            
            if is_distributed:
                # For distributed training, we need one batch per device per accumulation step
                all_indices = np.random.permutation(len(dataset))
                batch_start = 0

                # In the training loop
                for acc_step in range(remaining_steps):
                    # For distributed training with sequential batching
                    batch_indices = []
                    
                    # Get the next sequential batch indices - one per device
                    for device in range(n_devices):
                        batch_idx = all_indices[batch_start % len(dataset)]
                        batch_indices.append(int(batch_idx))
                        batch_start += 1
                    
                    # Convert to array and get batches
                    batch_indices = np.array(batch_indices)
                    batch_data = dataset.get_batch(batch_indices)
                    
                    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                        batch_data, target_lead_times=slice("6h", "6h"),
                        **dataclasses.asdict(task_config))
                    print("inputs.dims", inputs.dims)
                    
                    # Perform distributed training step
                    start_time = time.time()

                    # Update train_state
                    train_state, loss, state, diagnostics = update_fn(
                        loss_fn=loss_fn,
                        train_state=train_state,
                        state=state,
                        inputs=inputs, targets=targets, forcings=forcings,
                        n_devices=n_devices,
                        rng=rng_epoch)
                    
                    # Track loss
                    # print('loss from main',loss)
                    loss_val = float(loss[0])
                    step_losses.append(loss_val)
                    
                    step_time = time.time() - start_time
                    print(f"  Step {step_idx + acc_step + 1}/{args.steps_per_epoch}, "
                          f"Training Loss: {loss_val:.4f}, LR: {current_lr:.6f}, Time: {step_time:.2f}s")
                    
                    # Log to WandB if enabled
                    if use_wandb and process_rank == 0:
                        wandb.log({
                            "train/loss": loss_val, 
                            "train/learning_rate": current_lr,
                            "train/step_time": step_time,
                            "train/epoch": epoch,
                            "train/step": step_idx + acc_step + 1,
                            "train/global_step": step_count
                        })
                        loss_log.append({
                            "train/loss": loss_val, 
                            "train/learning_rate": float(current_lr),
                            "train/step_time": float(step_time),
                            "train/epoch": epoch,
                            "train/step": step_idx + acc_step + 1,
                            "train/global_step": step_count
                        })
                        if step_count % args.checkpoint_frequency == 0:
                            with open(loss_log_path, "w") as f:
                                json.dump(loss_log, f)
                    
                    # Save checkpoint every N steps       
                    if step_count % args.checkpoint_frequency == 0 and process_rank == 0:

                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
                        params = jax.device_get(train_state.params)
                        optimizer_state = jax.device_get(train_state.opt_state)
                            
                        training_ckpt = GraphCastTrainingCheckpoint(
                            params=params,
                            opt_state=optimizer_state,
                            task_config=task_config,
                            model_config=model_config,
                            global_step=step_count,
                            rng=rng,
                            created_at=current_time,
                            best_loss=best_loss,
                            learning_rate=current_lr,
                            metadata={
                                "start_year": args.start_year, 
                                "end_year": args.end_year,
                                "steps_per_epoch": args.steps_per_epoch,
                                "warmup_steps": args.warmup_steps,
                                "cosine_steps": args.cosine_steps
                            }
                        )
                            
                        # Save checkpoint (only rank 0 will actually save)
                        save_checkpoint(checkpoint_dir, training_ckpt, process_rank)
                        
                    # Validation
                    if step_count % args.validation_frequency == 0 and process_rank == 0:
                        val_corrected_pred, _ = run_forward.apply(
                            train_state.params, state, rng,
                            inputs_val, targets_val, forcings_val)
                        
                        # (loss, diagnostics), next_state
                        (val_loss, _), _  = loss_fn.apply(
                            train_state.params, state, rng,
                            inputs_val, targets_val, forcings_val)
                        
                        val_corrected_pred = jax.device_get(val_corrected_pred)
                        # val_loss = jax.device_get(val_loss)
                        
                        diff = val_corrected_pred - targets_val

                        diff_500hPa = diff.sel(level=500)

                        val_loss = float(val_loss)
                        val_rmse = {f"val/rmse_500hPa/{k}": jnp.sqrt(xarray_jax.unwrap_data((v*v).mean())).item() for k,v in diff_500hPa.data_vars.items()}
        
                        
                        # Log validation metrics
                        print(f"\tValidation Loss: {val_loss:.4f}")
                        if use_wandb:
                            wandb.log({
                                "val/loss": float(val_loss),
                                "val/global_step": step_count,
                                **val_rmse
                            })
                        loss_log.append({
                            "val/loss": float(val_loss), 
                            "val/global_step": step_count,
                            **val_rmse
                        })
                        with open(loss_log_path, "w") as f:
                                json.dump(loss_log, f)
    


            else:
                # For non-distributed training, process one batch at a time
                for acc_step in range(remaining_steps):
                    start_time = time.time()
                    
                    # Get a random batch
                    batch_idx = np.random.randint(0, len(dataset))
                    batch_data = dataset[batch_idx]
                    
                    # Extract input and target data
                    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                        batch_data, target_lead_times=slice("6h", "6h"),
                        **dataclasses.asdict(task_config))
                    
                    # Perform training step
                    new_params, state, new_optimizer_state, loss, diagnostics = train_step(
                        train_state.params, 
                        state, 
                        train_state.opt_state, 
                        inputs, 
                        targets, 
                        forcings,
                        jax.random.fold_in(rng, step_count))

                    # Update training state
                    train_state = TrainState(
                        params=new_params,
                        opt_state=new_optimizer_state,
                    )
                    
                    step_time = time.time() - start_time
                    loss_val = float(loss)
                    step_losses.append(loss_val)
                    
                    print(f"  Step {step_idx + acc_step + 1}/{args.steps_per_epoch}, "
                          f"Loss: {loss_val:.4f}, LR: {current_lr:.6f}, Time: {step_time:.2f}s")
                    
                    # Log to WandB if enabled
                    if use_wandb and process_rank == 0:
                        wandb.log({
                            "train/loss": loss_val, 
                            "train/learning_rate": current_lr,
                            "train/step_time": step_time,
                            "train/epoch": epoch,
                            "train/step": step_idx + acc_step + 1
                        })
            
            # Add step losses to epoch losses
            epoch_losses.extend(step_losses)
            
            start_step = 0
        
        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start_time
        mean_loss = np.mean(epoch_losses)
        
        # Update best loss
        if mean_loss < best_loss:
            best_loss = mean_loss
            print(f"New best loss: {best_loss:.4f}")
        
        print(f"Epoch {epoch}/{args.epochs}, Avg Loss: {mean_loss:.4f}, "
              f"Best Loss: {best_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # Log epoch metrics to WandB
        if use_wandb and process_rank == 0:
            wandb.log({
                "train/epoch": epoch,
                "train/mean_loss": mean_loss,
                "train/best_loss": best_loss,
                "train/epoch_time": epoch_time
            })
    
    # Clean up WandB
    if use_wandb and process_rank == 0:
        wandb.finish()
    
    print("Training complete!")

if __name__ == "__main__":
    main()