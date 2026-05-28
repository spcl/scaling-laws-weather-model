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
import os
import json
import zarr
import pandas as pd

from graphcast import graphcast, autoregressive, casting, normalization, xarray_jax, xarray_tree, data_utils, model_utils
from graphcast_trainer import load_checkpoint, GraphCastTrainingCheckpoint, load_normalization, WeatherBench2Dataset
SURFACE_LEVEL_VARS=["2m_temperature", "mean_sea_level_pressure", "10m_v_component_of_wind", "10m_u_component_of_wind", "total_precipitation_6hr"]
PRESSURE_LEVEL_VARS = ["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind", "vertical_velocity", "specific_humidity"]
PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

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
    parser.add_argument('--zarr-path', default="/ERA5/weatherbench2_original", help='Path to Zarr dataset')
    parser.add_argument('--ckpt-path', default="/scratch/gcjax_checkpoint_n4_mesh5_lat512_gstep16", help='Path to model checkpoint')
    parser.add_argument('--normalization-dir', default="./stats", help='Path to normalization stats')
    parser.add_argument('--output-path', default="/scratch/validation_all/graphcast", help='Where to save model prediction output')
    parser.add_argument('--noise-seed', type=int, default=0, help='Random seed for noise')
    parser.add_argument('--test', action='store_true', help='Test mode')
    args = parser.parse_args()
    test = args.test

    print("Loading checkpoint...")
    
    step_list = [50, 100, 200, 400, 700, 1000, 1500, 2000, 2500, 3000, 4000, 5000,  6000, 8000,  10000]
    if test:
        step_list = [50]

    ckpt = load_checkpoint(Path(args.ckpt_path), global_step=step_list[0])
    ckpt_path = args.ckpt_path
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
    
    @hk.transform_with_state
    def loss_fn(inputs, targets, forcings):
        """Loss function."""
        
        predictor = construct_wrapped_graphcast(
            model_config, task_config)
        loss, diagnostics = predictor.loss(inputs, targets, forcings)
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            (loss, diagnostics))


    print("Loading Zarr dataset...")
    dataset = WeatherBench2Dataset(start_year=2021, end_year=2021, steps=6)
    date_list = ["2021-01-01T00:00", "2021-02-01T06:00", "2021-03-01T12:00", "2021-04-01T18:00", "2021-05-02T00:00", "2021-06-02T06:00", "2021-07-02T12:00", "2021-08-02T18:00", "2021-09-03T00:00", "2021-10-03T06:00", "2021-11-03T12:00", "2021-12-03T18:00"]
    if test:    
        date_list = ["2021-01-01T01:00"]
    batch_list = []
    inputs_list = []
    targets_list = []
    forcings_list = []
    for date in date_list:
        batch = dataset.get_data_by_date(date)
        batch_list.append(batch)

        print("Extracting inputs...")
        
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            batch,
            target_lead_times=slice("6h", "6h"),
            **dataclasses.asdict(task_config)
        )
        print(inputs.dims)
        inputs_list.append(inputs)
        targets_list.append(targets)
        forcings_list.append(forcings)

    print("Generating ensemble predictions with different models...")
    results = {}
    dim = ckpt_path.split("/")[2].split("_")[-2]
    gstep = ckpt_path.split("/")[2].split("_")[-1]
    
    params_prefix = f"validation_{dim}_{gstep}"
    
    for i in step_list:
        
        print("Loading checkpoint...")
        ckpt = load_checkpoint(Path(ckpt_path), global_step=i)
        if ckpt is None:
            print(f"Checkpoint {i} not found")
            continue
        model_config = ckpt.model_config
        task_config = ckpt.task_config
        params = ckpt.params
        state = {}

        print("Constructing model...")
        
        rng = jax.random.PRNGKey(i)
        # pred, state = run_forward.apply(params, state, rng, inputs, targets, forcings)
        
        
        for inputs, targets, forcings, date in zip(inputs_list, targets_list, forcings_list, date_list):
            
            val_corrected_pred, _ = run_forward.apply(params, state, rng, inputs, targets, forcings)
            
            # (loss, diagnostics), next_state
            (val_loss, _), _  = loss_fn.apply(
                params, state, rng,
                inputs, targets, forcings)
            
            val_corrected_pred = jax.device_get(val_corrected_pred)
            val_loss = jax.device_get(val_loss) 
            
            diff = val_corrected_pred - targets
            
            data_dict = {}
            
            for level in PRESSURE_LEVELS:
                diff_level = diff.sel(level=level)
                data_pl_dict = {f"{var}_{level}" : float(np.sqrt((diff_level[var]*diff_level[var]).mean(dim=["lon", "lat"]))) for var in PRESSURE_LEVEL_VARS}
                print("data_pl_dict", data_pl_dict)

                data_dict = {**data_dict, **data_pl_dict}
                
            data_sl_dict = {var : float(np.sqrt((diff[var]*diff[var]).mean(dim=["lon", "lat"]))) for var in SURFACE_LEVEL_VARS}
            print("data_sl_dict", data_sl_dict)
            data_dict = {**data_dict, **data_sl_dict}
            # add val loss
            data_dict["val_loss_full"] = float(val_loss)
            
        
            results[f"GraphCast_rmse_step{i}_date{date}"] = data_dict

    
    
    # get shape from ckpt_path eg. /scratch/gcjax_checkpoint_n4_mesh5_lat512_gstep16/
    os.makedirs(args.output_path, exist_ok=True)
    json.dump(results, open(os.path.join(args.output_path,f"GC_validation_results_{params_prefix}.json"), "w"))

    print("Done.")


if __name__ == '__main__':
    main()
