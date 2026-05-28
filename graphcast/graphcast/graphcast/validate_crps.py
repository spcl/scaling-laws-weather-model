#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run GraphCast model using a checkpoint and a specific datetime from Zarr dataset
Apply Perlin noise to inputs and run ensemble predictions
"""

import argparse
import xarray as xr
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from pathlib import Path
import pickle
import dataclasses
from noise import pnoise2  # Perlin noise
from properscoring import crps_ensemble
import os
import json
import zarr

from graphcast import graphcast, autoregressive, casting, normalization, xarray_jax, xarray_tree, data_utils, model_utils
from graphcast_trainer import load_checkpoint, GraphCastTrainingCheckpoint, load_normalization, WeatherBench2Dataset

def construct_model(model_config, task_config, normalization_dir):
    diffs_stddev_by_level = xr.load_dataset(f"{normalization_dir}/diffs_stddev_by_level.nc").compute()
    mean_by_level = xr.load_dataset(f"{normalization_dir}/mean_by_level.nc").compute()
    stddev_by_level = xr.load_dataset(f"{normalization_dir}/stddev_by_level.nc").compute()

    model = graphcast.GraphCast(model_config, task_config)
    model = casting.Bfloat16Cast(model)
    model = normalization.InputsAndResiduals(
        model,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level
    )
    model = autoregressive.Predictor(model, gradient_checkpointing=False)
    return model


def generate_perlin_noise(shape, scale=10, octaves=1, seed=0):
    np.random.seed(seed)
    noise = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            noise[i, j] = pnoise2(i / scale, j / scale, octaves=octaves, repeatx=shape[0], repeaty=shape[1], base=seed)
    return noise

def apply_perlin_noise_to_input(inputs: xr.Dataset, scale=10, strength=0.05, seed=0):
    noisy_inputs = {}
    lat_dim = inputs.sizes['lat']
    lon_dim = inputs.sizes['lon']

    for var in inputs.data_vars:
        data = inputs[var]
        if {'lat', 'lon'}.issubset(data.dims):
            noise = generate_perlin_noise((lat_dim, lon_dim), scale=scale, seed=seed)
            noise_da = xr.DataArray(noise, coords={"lat": inputs.lat, "lon": inputs.lon}, dims=("lat", "lon"))

            # Match dimensions by expanding and broadcasting
            for dim in data.dims:
                if dim not in noise_da.dims:
                    noise_da = noise_da.expand_dims({dim: data.sizes[dim]})
            
            # Use broadcasting to match final shape exactly
            noise_da, data_broadcasted = xr.broadcast(noise_da, data)

            noisy_inputs[var] = data + strength * noise_da
        else:
            noisy_inputs[var] = data

    return xr.Dataset(noisy_inputs)

def main():

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
    
    parser = argparse.ArgumentParser(description="Run GraphCast on noisy inputs and save ensemble outputs")
    parser.add_argument('--zarr-path', required=True, help='Path to Zarr dataset')
    parser.add_argument('--date', required=True, help='Datetime to evaluate (e.g., 2017-01-01T00:00)')
    parser.add_argument('--ckpt-path', required=True, help='Path to model checkpoint')
    parser.add_argument('--normalization-dir', required=True, help='Path to normalization stats')
    parser.add_argument('--output-path', required=True, help='Where to save model prediction output')
    parser.add_argument('--ensemble-size', type=int, default=10, help='Number of ensemble members to generate')
    parser.add_argument('--noise-scale', type=float, default=10.0, help='Scale of Perlin noise')
    parser.add_argument('--noise-strength', type=float, default=0.05, help='Strength of noise')
    parser.add_argument('--noise-seed', type=int, default=0, help='Random seed for noise')
    args = parser.parse_args()

    print("Loading checkpoint...")
    ckpt = load_checkpoint(Path(args.ckpt_path))
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    params = ckpt.params
    state = {}

    print("Constructing model...")
    diffs_stddev_by_level, mean_by_level, stddev_by_level = load_normalization(args.normalization_dir)


    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        """Forward pass function."""
        
        predictor = construct_wrapped_graphcast(
            model_config, task_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)


    print("Loading Zarr dataset...")
    dataset = WeatherBench2Dataset(start_year=2017, end_year=2017, steps=6)
    batch = dataset.get_data_by_date("2017-01-01T00:00")

    print("Extracting inputs...")
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        batch,
        target_lead_times=slice("6h", "6h"),
        **dataclasses.asdict(task_config)
    )
    print(inputs.dims)

    print("Generating ensemble predictions with Perlin noise...")
    preds = []
    for i in range(args.ensemble_size):
        noisy_inputs = apply_perlin_noise_to_input(inputs, scale=args.noise_scale, strength=args.noise_strength, seed=args.noise_seed + i)
        rng = jax.random.PRNGKey(i)
        pred, _ = run_forward.apply(params, state, rng, noisy_inputs, targets, forcings)
        preds.append(pred)

    print("Stacking ensemble outputs...")
    ensemble_output = xr.concat(preds, dim='batch')
    print(f"Ensemble output shape: {ensemble_output.sizes}")

    targets = targets.isel(batch=0)

    print("Calculating CRPS...")
    crps_results = {}
    for var in ensemble_output.data_vars:
        if "level" in targets[var].dims:
            for level in targets[var].level.values:
                truth = targets[var].sel(level=level).values
                ens = ensemble_output[var].sel(level=level).values
                crps = crps_ensemble(truth, ens, axis=1)
                value = crps_results[f"{var}_level{level}"] = float(np.mean(crps))
                print(f"CRPS for {var}_level{level}: {value:.5f}")
        else:
            truth = targets[var].values
            ens = ensemble_output[var].values
            crps = crps_ensemble(truth, ens, axis=1)
            crps_results[var] = float(np.mean(crps))
            print(f"CRPS for {var}: {crps_results[var]:.5f}")
    
    params_prefix = f"ensemble{args.ensemble_size}_noise{args.noise_strength}_scale{args.noise_scale}_seed{args.noise_seed}"
    os.makedirs(args.output_path, exist_ok=True)
    json.dump(crps_results, open(os.path.join(args.output_path,f"crps_results_{params_prefix}.json"), "w"))


    print("Unwrapping JAX arrays to NumPy...")
    stacked_ensemble_output = model_utils.dataset_to_stacked(ensemble_output)
    preds_np = np.array(stacked_ensemble_output)

    print("Saving ensemble predictions to Zarr...")
    zarr.save(os.path.join(args.output_path, params_prefix + '.zarr'), preds_np)
    print("Done.")


if __name__ == '__main__':
    main()
