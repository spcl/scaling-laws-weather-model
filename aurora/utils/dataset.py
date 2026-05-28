## various classes and utilities for dataset and their loaders

import os
import math
import glob
import random
from typing import Optional
from datetime import datetime 

import torch
import cftime
import mmnpz
import numpy as np
import xarray as xr
import pandas as pd
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader 
import pickle


from aurora.normalisation import load_normalization_stats


d_srf_abr2full = dict(zip(("2t", "10u", "10v", "msl"), ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure")))
d_static_abr2full = dict(zip(("lsm", "z", "slt"), ("land_sea_mask", "geopotential_at_surface", "soil_type"))) ## nonexisting variable in wb2: "slt": "soil_type"
d_atmos_abr2full = dict(zip(("z", "u", "v", "t", "q", "w"), ("geopotential", "u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity", "vertical_velocity")))
d_srf_full2abr = {v: k for k, v in d_srf_abr2full.items()}
d_static_full2abr = {v: k for k, v in d_static_abr2full.items()}
d_atmos_full2abr = {v: k for k, v in d_atmos_abr2full.items()}


class WeatherBench2Raw(Dataset):
    def __init__(
        self, 
        path: str = '/data/weatherbench2_original', 
        extended_path: str = None, # path to dataset that contains extra variables
        extended_vars: list = None, # list of the varialbes from extended_dataset to include in original dataset
        stats_path: str = 'aurora/normalization_stats_1979_2021.json',
        inds = None, 
        str_task: str = '6h-forecast', 
        dict_vars: dict = None, 
        atmos_levels = np.asarray([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], dtype=np.int32),
        dict_stats: Optional[dict[str, tuple[float, float]]] = None, 
        slt: np.ndarray = None,
        co2_path: str = None,
        is_global_observation: bool = True,
        grid_resolution: float = 0.25,
        **kwargs,
    ):
        self.path = path
        self.inds = inds
        self.d_ind_pairs = {}
        self.str_task = str_task
        self.dict_vars = dict_vars
        self.atmos_levels = atmos_levels
        self.dict_stats = dict_stats
        self.slt = slt
        self.is_global_observation = is_global_observation
        self.grid_resolution = grid_resolution
        self.ds = xr.open_zarr(path)
        if extended_path:
            self.ds_extended = xr.open_zarr(extended_path)
            self.ds = self.ds.assign({var: self.ds_extended[var] for var in extended_vars})

        if len(self.ds.latitude) == 721:
            self.lat = self.ds.latitude.values[:-1] ## get only 720 out of the 721 latitudes
            self.slt = self.slt[:-1] ## get only 720 out of the 721 latitudes
        else:
            self.lat = self.ds.latitude.values
        self.lon = self.ds.longitude.values

        self.locations, self.scales = load_normalization_stats(stats_path)
        

        
        if self.inds is None: ##assuming training set (<2018)
            self.inds = self.ds.time[self.ds.time.values < np.datetime64(datetime(2018, 1, 1),)]
            # inds_val = self.ds.time[(self.ds.time.values >= np.datetime64(datetime(2018, 1, 1),)) & (self.ds.time.values < np.datetime64(datetime(2019, 1, 1),))]
            # inds_test = self.ds.time[self.ds.time.values >= np.datetime64(datetime(2019, 1, 1),)]
        self.len_dataobj = len(self.inds) ## will be later overwritten.
        if self.dict_vars is None:
            self.dict_vars = {
                'surf_vars': ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"),
                'static_vars': ("land_sea_mask", "geopotential_at_surface", "soil_type"),
                'atmos_vars': ("geopotential", "u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity")
            }
        self.ds = self.ds.sel(level=self.atmos_levels)
        self.ds = self.ds.sel(time=self.inds)
        self.ds = self.ds.sel(latitude=self.lat)
        self.ds = self.ds.sel(longitude=self.lon)
        self.surf_vars = {d_srf_full2abr[k]: self.ds[k] for k in self.dict_vars['surf_vars']} ## respecting the abbreviations from Aurora implementation for dict keys
        self.static_vars = {d_static_full2abr[k]: self.ds[k] for k in self.dict_vars['static_vars']}
        self.atmos_vars = {d_atmos_full2abr[k]: self.ds[k] for k in self.dict_vars['atmos_vars']}
        self._prepare_inds_for_forecast(lead_time_h=6) ## assumes only forecast task for the dataloader (overwrites length of dataset obj.)

        self.co2 = pd.read_csv(co2_path) if co2_path else None
        

    def __len__(self):
        return self.len_dataobj
    
    

    
    def _prepare_inds_for_forecast(self, lead_time_h=6):
        # Determine if this is training or validation data based on time range
        first_time = np.min(self.inds)
        last_time = np.max(self.inds)
        
        # Create a dataset identifier based on time range
        dataset_id = f"{first_time.astype('datetime64[D]')}_{last_time.astype('datetime64[D]')}"
        cache_file = f'utils/forecast_pairs_{lead_time_h}h_{dataset_id}.pkl'

        # Try to load from cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.d_ind_pairs = pickle.load(f)
                    ind_pair_key = f"{lead_time_h}h_forecast"
                    if ind_pair_key in self.d_ind_pairs:
                        self.len_dataobj = len(self.d_ind_pairs[ind_pair_key])
                        return
            except (EOFError, pickle.UnpicklingError):
                # Handle corrupt cache file
                print(f"Warning: Cache file {cache_file} is corrupted. Recreating...")
        
        # If cache doesn't exist or is invalid, compute from scratch
        # Compute the indices
        x_t1 = self.inds
        x_t0 = x_t1 - np.timedelta64(lead_time_h, 'h')
        y_t = x_t1 + np.timedelta64(lead_time_h, 'h')
        l_pairs = []
        for i in range(len(x_t1)):
            if x_t0[i] in self.ds.time and x_t1[i] in self.ds.time and y_t[i] in self.ds.time:
                l_pairs.append((x_t0[i], x_t1[i], y_t[i]))
        pairs = tuple(l_pairs)
        ind_pair_key = f"{lead_time_h}h_forecast"
        self.d_ind_pairs[ind_pair_key] = pairs
        self.len_dataobj = len(pairs)
        
        # Save to cache for future use - only from rank 0
        is_rank_zero = int(os.environ.get("GLOBAL_RANK", "0")) == 0
        
        if is_rank_zero:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.d_ind_pairs, f)

    def __getitem__(self, idx):
        # print('getitem idx=', idx)
        if self.str_task == '6h-forecast':
            return self._get_forecast(idx, lead_time_h=6)
        elif self.str_task == 'unmask':
            return self._get_unmask(idx)
        else:
            raise ValueError(f"Invalid task: {self.str_task}")

    def year_to_co2(self, year):
        # Return co2 value for the given year
        row = self.co2[self.co2['Unnamed: 0'] == year]
        if not row.empty:
            return row['CO2'].values[0]
        else:
            return None

    def _get_forecast(self, idx, lead_time_h=6):
        ## get the forecast data: Returns data of the form (x=(x-6h, x), y=(x+6h), t=x_t)
        if f"{lead_time_h}h_forecast" not in self.d_ind_pairs:
            raise ValueError(f"Invalid lead time: {lead_time_h}.")
        x_ind0, x_ind1, y_ind = self.d_ind_pairs[f"{lead_time_h}h_forecast"][idx]
        x_srf = {d_srf_full2abr[k]: np.stack((self.surf_vars[d_srf_full2abr[k]].sel(time=x_ind0).values, self.surf_vars[d_srf_full2abr[k]].sel(time=x_ind1).values),axis=-3) for k in self.dict_vars['surf_vars']}
        if self.co2 is not None:
            sample_year = int(x_ind0.astype('datetime64[Y]').astype(int) + 1970)
            co2 = self.year_to_co2(sample_year)
            x_srf['co2'] = np.full_like(next(iter(x_srf.values())), co2)

        x_static = {d_static_full2abr[k]: self.static_vars[d_static_full2abr[k]].values for k in self.dict_vars['static_vars']} ## cache this in a next iteration
        x_atmos = {d_atmos_full2abr[k]: np.stack((self.atmos_vars[d_atmos_full2abr[k]].sel(time=x_ind0).values, self.atmos_vars[d_atmos_full2abr[k]].sel(time=x_ind1).values),axis=-4) for k in self.dict_vars['atmos_vars']}
        y_srf = {d_srf_full2abr[k]: self.surf_vars[d_srf_full2abr[k]].sel(time=y_ind).values for k in self.dict_vars['surf_vars']}
        y_static = x_static.copy() #[self.static_vars[k].values for k in self.dict_vars['static_vars']] ## cache this
        y_atmos = {d_atmos_full2abr[k]: self.atmos_vars[d_atmos_full2abr[k]].sel(time=y_ind).values for k in self.dict_vars['atmos_vars']}
        # x_ind0 = str(x_ind0)
        x_time = str(x_ind1)
        y_time = str(y_ind)
        return {
            'x_srf': x_srf,
            'x_static':x_static,
            'x_atmos': x_atmos,
            'y_srf':y_srf,
            'y_static':y_static,
            'y_atmos':y_atmos,
            'x_time':x_time,
            'y_time':y_time,
            'lat': self.lat,
            'lon': self.lon,
            'atmos_levels':self.atmos_levels,
            'locations': self.locations,
            'scales': self.scales,
            'grid_resolution': self.grid_resolution,
            'is_global_observation': self.is_global_observation,
        }
    
    def _get_unmask(self, idx):
        ## get the unmask data
        pass