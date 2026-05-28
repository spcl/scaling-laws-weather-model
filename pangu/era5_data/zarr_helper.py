# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Protocol
import xarray
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torch.utils.data.distributed import DistributedSampler
from typing import Any
import numpy as np
from typing import Mapping, Optional, Tuple



class Params(Protocol):
    """A protocol with the required input parameters

    Useful for typechecking or editor autocompletion.
    """

    in_channels: Any
    out_channels: Any
    batch_size: int
    global_means_path: str
    global_stds_path: str


@dataclass
class Metadata:
    """Image metadata required to initialize the model"""

    img_shape_x: int
    img_shape_y: int
    in_channels: Any
    out_channels: Any

    img_crop_shape_x: int
    img_crop_shape_y: int
    img_crop_offset_x: int
    img_crop_offset_y: int
    img_local_shape_x: int
    img_local_shape_y: int
    img_local_offset_x: int
    img_local_offset_y: int

def variable_to_stacked(
    variable: xarray.Variable,
    sizes: Mapping[str, int],
    preserved_dims: Tuple[str, ...] = ("batch", "lat", "lon"),
) -> xarray.Variable:
  """Converts an xarray.Variable to a stacked format with channels first.

  This takes a variable, converts it into BHWC layout, then puts channels first
  to get BCHW layout.

  Args:
    variable: An xarray.Variable.
    sizes: Mapping including sizes for any dimensions which are not present in
      the `variable` but are needed for the output.
    preserved_dims: dimensions from the variable that should not be folded in
      the predictions channels.

  Returns:
    An xarray.Variable with dimensions ("time", "channels", "lat", "lon").
  """
  # Get the dimensions that will be folded into channels
  stack_to_channels_dims = [d for d in variable.dims if d not in preserved_dims]
  
  # Stack dimensions into channels if there are any to stack
  if stack_to_channels_dims:
    variable = variable.stack(channels=stack_to_channels_dims)
  
  # Ensure all dimensions exist with correct sizes
  dims = {dim: variable.sizes.get(dim) or sizes[dim] for dim in preserved_dims}
  dims["channels"] = variable.sizes.get("channels", 1)
  variable = variable.set_dims(dims)
  
  # Reorder dimensions to match specified order
  if "time" in preserved_dims:
    dim_order = ("time", "channels", "lat", "lon")
  else:
    dim_order = ("channels", "lat", "lon")
    
  # Transpose to get the desired dimension order
  return variable.transpose(*dim_order)


def dataset_to_stacked(
    dataset: xarray.Dataset,
    sizes: Optional[Mapping[str, int]] = None,
    preserved_dims: Tuple[str, ...] = ("batch", "lat", "lon"),
) -> xarray.DataArray:
  """Converts an xarray.Dataset to a stacked format while preserving Dataset structure.

  This takes each constituent data_var, converts it into BHWC layout
  using `variable_to_stacked`, then returns a Dataset with stacked variables.

  Args:
    dataset: An xarray.Dataset.
    sizes: Mapping including sizes for any dimensions which are not present in
      the `dataset` but are needed for the output. See variable_to_stacked.
    preserved_dims: dimensions from the dataset that should not be folded in
      the predictions channels.

  Returns:
    An xarray.Dataset with stacked variables, preserving the original structure.
    Each variable will have dimensions preserved_dims + ("channels",).
  """
  # 4 + 5*13 = 69 channels, missing 100u, 100v
  new_key = ['mean_sea_level_pressure','10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
             'geopotential', 'specific_humidity', 'temperature',
             'u_component_of_wind', 'v_component_of_wind']
  data_vars = [
      variable_to_stacked(dataset.variables[name], sizes or dataset.sizes,
                          preserved_dims)
      for name in new_key
  ]
  coords = {
      dim: coord
      for dim, coord in dataset.coords.items()
      if dim in preserved_dims
  }
  # Create new dataset with processed variables
  return xarray.DataArray(
        data=xarray.Variable.concat(data_vars, dim="channels"), coords=coords)


def get_data_loader(params: Params, zarr_path: str, train: bool):
    """Matches interface used in trainer.py:Trainer"""
    # Note that ds has different attributes than original xarray.Dataset
    ds = xarray.open_zarr(zarr_path)
    # Convert to float32 when loading
    ds = ds.astype(np.float32)
    dataset = _xarray_to_dataset(params, ds, train=train)
    
    # print("dataset=", dataset)

    # shape is (1, channel, 1, 1)
    mean = np.load(params.global_means_path)
    std = np.load(params.global_stds_path)

    # assert mean.shape == (1, len(ds.channel), 1, 1), mean.shape
    # assert not np.any(np.isnan(mean)), np.ravel(std)

    # assert std.shape == (1, len(ds.channel), 1, 1), std.shape
    # assert not np.any(np.isnan(std)), np.ravel(std)

    def reset_pipeline():
        pass

    def get_output_normalization():
        return mean[:, params.out_channels], std[:, params.out_channels]

    def get_input_normalization():
        return mean[:, params.in_channels], std[:, params.in_channels]

    def center(args):
        x, y = args

        xmean = mean[0, params.in_channels]
        xstd = std[0, params.in_channels]

        ymean = mean[0, params.out_channels]
        ystd = std[0, params.out_channels]

        return (x - xmean) / xstd, (y - ymean) / ystd

    dataset = Map(dataset, center)
    
    if train:

        sampler = StatefulDistributedSampler(dataset, shuffle=train, seed=42, num_replicas=params.data_num_shards, rank=params.data_shard_id) if (params.data_num_shards > 1) else None

        dataloader = StatefulDataLoader(
            dataset,
            batch_size=int(params.batch_size),
            num_workers=params.num_data_workers,
            sampler=sampler,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        sampler = DistributedSampler(dataset, shuffle=True, seed=42, num_replicas=params.data_num_shards, rank=params.data_shard_id) if (params.data_num_shards > 1) else None
        dataloader = DataLoader(
            dataset, # 16 is the number of validation steps
            batch_size=int(params.batch_size),
            num_workers=params.num_data_workers,
            shuffle=False,
            sampler=sampler,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )

    dataloader.get_output_normalization = get_output_normalization
    dataloader.get_input_normalization = get_input_normalization
    dataloader.reset_pipeline = reset_pipeline

    shape = ds.sizes
    nlon = shape.get("lon", 1440)
    nlat = shape.get("lat", 721)

    metadata = Metadata(
        img_shape_y=nlon,
        img_shape_x=nlat,
        in_channels=params.in_channels,
        out_channels=params.out_channels,
        img_crop_shape_x=nlat,
        img_crop_shape_y=nlon,
        img_crop_offset_x=0,
        img_crop_offset_y=0,
        img_local_shape_x=nlat,
        img_local_shape_y=nlon,
        img_local_offset_x=0,
        img_local_offset_y=0,
    )

    if train:
        return dataloader, metadata, sampler
    else:
        return dataloader, metadata


def _xarray_to_dataset(params: Params, ds: xarray.Dataset, train: bool):
    year = ds.time.dt.year
    if train:
        mask = (year <= 2020) & (year >= 1979)
        ds = ds.sel(time=mask)
    else:
        mask = (2020 < year) & (year <= 2021)
        ds = ds.sel(time=mask)
        
    mask = ds.level.isin([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
    ds = ds.sel(level=mask)
    hour = ds.time.dt.hour
    ds = ds.sel(time=hour.isin([0, 6, 12, 18]))
    
    # rename dimensions latitude and longitude to lat and lon
    ds = ds.rename_dims({"latitude": "lat", "longitude": "lon"})
    ds = dataset_to_stacked(ds, sizes=None, preserved_dims=("time", "lat", "lon"))
    
    return XarrayDataset(ds, params.in_channels, params.out_channels)


class Map(Dataset):
    def __init__(self, data, func):
        self.data = data
        self.func = func

    def __getitem__(self, i):
        return self.func(self.data[i])

    def __len__(self):
        return len(self.data)


@dataclass
class XarrayDataset(Dataset):
    data: xarray.DataArray
    in_channels: Any = slice(None)
    out_channels: Any = slice(None)

    def _to_array(self, x):
        return x.values.astype(np.float32)

    def __getitem__(self, i):
        input_ = self.data.isel(time=i, channels=self.in_channels)
        target = self.data.isel(time=i + 1, channels=self.out_channels)
        x = torch.tensor(self._to_array(input_), dtype=torch.float32)
        y = torch.tensor(self._to_array(target), dtype=torch.float32)
        return x, y

    def __len__(self):
        times = self.data.time
        if len(times) > 1:
            return len(times) - 1
        else:
            return 0
