import xarray as xr
import numpy as np
from scores.continuous import mae, mse, rmse
from scores.continuous.correlation import pearsonr
from scores.spatial import fss_2d


dict_metrics = {'MAE': mae,
                'RMSE': rmse,
                'R2': pearsonr,
                'FSS': fss_2d
                }


def batch2xr(batch, eval_surf=True, eval_atmos=True):
    ''' metrics from the scores library can only be applied to xarray.DataArray. This function transforms Batch objects to xarray.Dataset '''
    tmp = dict()
    if eval_surf:
        for k in batch.surf_vars:
            tmp[k] = (["latitude", "longitude"], batch.surf_vars[k].detach().squeeze().cpu().numpy())

    if eval_atmos:
        for k in batch.atmos_vars:
            tmp[k] = (["level", "latitude", "longitude"], batch.atmos_vars[k].detach().squeeze().cpu().numpy())

    ds_batch = xr.Dataset(
        coords=dict(
            longitude=batch.metadata.lon.cpu().numpy(),
            latitude=batch.metadata.lat.cpu().numpy(),
            level=np.array(batch.metadata.atmos_levels)
        ),
        data_vars = tmp,
    )

    return ds_batch



class ValidationMetrics():
    def __init__(self, 
                 list_metrics=['MAE', 'RMSE', 'R2', 'FSS'], 
                 eval_surf=True, 
                 eval_atmos=True, 
                 levels=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
                 ):
        self.list_metrics = list_metrics
        self.eval_surf = eval_surf
        self.eval_atmos = eval_atmos
        self.levels = levels


    def get_loss(self, pred, target):
        metrics = {}
        ds_pred = batch2xr(pred, eval_surf=self.eval_surf, eval_atmos=self.eval_atmos)
        ds_target = batch2xr(target, eval_surf=self.eval_surf, eval_atmos=self.eval_atmos)

        metrics = {}
        for metric in self.list_metrics:
            metric_fn = dict_metrics[metric]
            for k in target.surf_vars:
                if metric == 'FSS':
                    q = ds_pred[k].quantile(0.9).data
                    m = metric_fn(ds_pred[k], 
                                    ds_target[k],
                                    event_threshold=q,
                                    window_size=(10, 10), 
                                    spatial_dims=["latitude", "longitude"]
                                    ).data
                    metrics[f"metrics/{metric}_{k}"] = m if isinstance(m, float) else m.item()
                else:
                    m = metric_fn(ds_pred[k], ds_target[k]).data
                    metrics[f"metrics/{metric}_{k}"] = m if isinstance(m, float) else m.item()

            for k in target.atmos_vars:
                metric_mean = 0
                for level in self.levels:
                    if metric == 'FSS':
                        q = ds_pred.sel(level=level)[k].quantile(0.9).data
                        metric_mean += metric_fn(ds_pred.sel(level=level)[k], 
                                                ds_target.sel(level=level)[k],
                                                event_threshold=q,
                                                window_size=(10, 10), 
                                                spatial_dims=["latitude", "longitude"]
                                                ).data
                    else:
                        metric_mean += metric_fn(ds_pred.sel(level=level)[k], 
                                                            ds_target.sel(level=level)[k]).data
                metrics[f"metrics/{metric}_{k}"] = metric_mean/len(self.levels)

        return metrics