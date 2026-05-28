import torch
import logging
import torch.nn.functional as F
from aurora import Batch

epsilon = 1e-6

class MSELoss:
        def __init__(self, d_var_weights=None, d_pres_weights=None):
            '''
            if d_var_weights=None, all vars are weighted equally. Any vars not in dict keys are weighted 1.
            if d_pres_weights=None, all pres levels are weighted equally. Any pres levels not in dict keys are weighted 1. weights in d_pres_weights are multiplied with d_var_weights.
            '''
            self.d_var_weights = d_var_weights if d_var_weights is not None else {}
            self.d_pres_weights = d_pres_weights if d_pres_weights is not None else {}
            self.loss_fn = torch.nn.MSELoss()
            self.loss_no_reduction = torch.nn.MSELoss(reduction='none')
        def get_loss(self, pred_batch, target_batch):
            
            loss_individual_vars = {}
            ## surface vars
            loss_srf = []
            for k in target_batch.surf_vars.keys():
                if k not in self.d_var_weights.keys():
                    self.d_var_weights[k] = 1
                loss_srf_ = self.loss_fn(pred_batch.surf_vars[k], target_batch.surf_vars[k]) * self.d_var_weights[k]
                loss_individual_vars[f'srf_{k}'] = loss_srf_
                loss_srf.append(loss_srf_)
            loss_srf = torch.stack(loss_srf).mean() ## take average over all surf_vars
            ###
            # ## static vars
            ### Static vars do not have gradients.
            # loss_static = []
            # for k in target_batch.static_vars.keys():
            #     if k not in self.d_var_weights.keys():
            #         self.d_var_weights[k] = 1
            #     loss_static_ = self.loss_fn(pred_batch.static_vars[k], target_batch.static_vars[k]) * self.d_var_weights[k]
            #     loss_individual_vars[f'static_{k}'] = loss_static_
            #     loss_static.append(loss_static_)
            # loss_static = torch.stack(loss_static).mean() ## take average over all static_vars
            ###
            ## atmos vars
            loss_atmos = []
            ## check pressure level weights dict
            for c in pred_batch.metadata.atmos_levels:
                if c not in self.d_pres_weights.keys():
                    self.d_pres_weights[c] = 1
            if all(v == 1 for v in self.d_pres_weights.values()):
                scale_pres_lvls = False
            else:
                scale_pres_lvls = True
                weights_pres_lvl = torch.tensor([self.d_pres_weights[c] for c in pred_batch.metadata.atmos_levels], device=pred_batch.atmos_vars[list(pred_batch.atmos_vars.keys())[0]].device)
                pres_lvl_len = torch.tensor(weights_pres_lvl.size(0), dtype=torch.float32, device=weights_pres_lvl.device)
            for k in target_batch.atmos_vars.keys():
                if k not in self.d_var_weights.keys():
                    self.d_var_weights[k] = 1
                if scale_pres_lvls:
                    loss_atmos_ = self.loss_no_reduction(pred_batch.atmos_vars[k], target_batch.atmos_vars[k]) * self.d_var_weights[k] ## shape: (batch, pressure levels, lat, lon)
                    loss_atmos_ = torch.einsum('bchw,c->bchw', loss_atmos_, weights_pres_lvl) / pres_lvl_len ## shape: (batch, pressure levels, lat, lon)
                else:
                    loss_atmos_ = self.loss_fn(pred_batch.atmos_vars[k], target_batch.atmos_vars[k]) * self.d_var_weights[k]
                loss_individual_vars[f'atmos_{k}'] = loss_atmos_
                loss_atmos.append(loss_atmos_)
            loss_atmos = torch.stack(loss_atmos).mean() ## take average over all atmos_vars
            loss_total = loss_atmos + loss_srf # + loss_static
            dict_losses = {**{'loss_srf': loss_srf, 'loss_atmos': loss_atmos}, **loss_individual_vars} #, 'loss_static': loss_static
            # l_nans = []
            # for k,v in dict_losses.items():
            #     if torch.isnan(v):
            #         l_nans.append(k)
            # if len(l_nans) > 0:
            #     logging.info(f'Following losses are NaN: {str(",").join(l_nans)}')
            return loss_total, dict_losses


class WeightedMAELoss:
    def __init__(
        self,
        surf_weight: float = 1/4,
        atmos_weight: float = 1.0,
        surf_var_weights: dict[str, float] = None,
        atmos_var_weights: dict[str, float] = None,
        dataset_weight: int = 2,
        reduction: bool = True,
        latitude_weight: bool = True,
        level_weight: bool = False
    ) -> None:
        if surf_var_weights is None:
            surf_var_weights = {
                'msl': 1.5,
                '10u': 0.77,
                '2t': 3.0,
            }
        if atmos_var_weights is None:
            atmos_var_weights = {
                'z': 2.8,
                'q': 0.78,
                't': 1.7,
                'u': 0.87,
                'v': 0.6
            }

        self.surf_weight = surf_weight
        self.atmos_weight = atmos_weight
        self.dataset_weight = dataset_weight
        self.surf_var_weights = surf_var_weights
        self.atmos_var_weights = atmos_var_weights
        self.reduction = reduction
        self.latitude_weight = latitude_weight
        self.level_weight = level_weight # only in validation

    def __call__(self, pred_batch, target_batch):
        latitudes = torch.deg2rad(pred_batch.metadata.lat)
        latitude_weight = torch.cos(latitudes) / torch.cos(latitudes).mean()
        levels = torch.tensor(pred_batch.metadata.atmos_levels, dtype=torch.float32)
        level_weight = levels / levels.mean() # use mean instead of sum, see https://github.com/google-deepmind/graphcast/issues/156
        num_vars = (len(target_batch.surf_vars) + len(target_batch.atmos_vars))

        groups = [
            (target_batch.surf_vars, pred_batch.surf_vars, self.surf_weight, self.surf_var_weights), 
            (target_batch.atmos_vars, pred_batch.atmos_vars, self.atmos_weight, self.atmos_var_weights)
        ]
        loss_dict = {}
        total_loss = 0
        for target, pred, group_weight, var_weights in groups:
            group_loss = 0
            for var_name in sorted(target.keys()):
                err = abs(pred[var_name] - target[var_name])

                # validation
                if self.level_weight:
                    err = err ** 2

                if self.latitude_weight:
                    err = err * (latitude_weight[:, None].to(err.device))
                
                if self.level_weight and var_name in target_batch.atmos_vars.keys():
                    err = err * (level_weight.view(1, 1, -1, 1, 1).to(err.device))
                # # calculate the average loss over entire loss
                # mean = err.nanmean()
                # mean = 0 if mean.isnan() else mean
                # # calculate the average loss across batches
                # mean0 = err.nanmean(dim=0).nan_to_num()
                # loss_dict[var_name] = mean if self.reduction else mean0

                mean = err.mean()
                loss_dict[var_name] = mean

                group_loss += mean * var_weights.get(var_name, 1.0)

            total_loss += group_weight * group_loss

        #return (self.dataset_weight / num_vars) * total_loss, loss_dict # leave the normalization to MyLoss
        # print("total_loss", total_loss)
        # print("loss_dict", loss_dict)
        return  total_loss, loss_dict

    def get_loss(self, pred_batch, target_batch): # this is here only to support backward compatibility in the code.
        return self.__call__(pred_batch, target_batch)


class MyLoss:
    def __init__(
        self,
        surf_weight: float = 1/4,
        atmos_weight: float = 1.0,
        surf_var_weights: dict[str, float] = None,
        atmos_var_weights: dict[str, float] = None,
        dataset_weight: int = 2,
        reduction: bool = True,
        latitude_weight: bool = True,
        mae_weight: float = 1.0,
        level_weight: bool = False
    ) -> None:
        # Define default weights if None is provided
        if surf_var_weights is None:
            surf_var_weights = {
                'msl': 1.5,
                '10u': 0.77,
                '2t': 3.0,
            }
        if atmos_var_weights is None:
            atmos_var_weights = {
                'z': 2.8,
                'q': 0.78,
                't': 1.7,
                'u': 0.87,
                'v': 0.6
            }

        # Initialize the WeightedMAELoss with the same weights
        self.mae_loss = WeightedMAELoss(
            surf_weight=surf_weight,
            atmos_weight=atmos_weight,
            surf_var_weights=surf_var_weights,
            atmos_var_weights=atmos_var_weights,
            dataset_weight=dataset_weight,
            reduction=reduction,
            latitude_weight=latitude_weight,
            level_weight=level_weight
        )
        
        # Store parameters
        self.surf_weight = surf_weight
        self.atmos_weight = atmos_weight
        self.surf_var_weights = surf_var_weights  # Now guaranteed to not be None
        self.atmos_var_weights = atmos_var_weights  # Now guaranteed to not be None
        self.dataset_weight = dataset_weight
        self.reduction = reduction
        self.latitude_weight = latitude_weight
        self.level_weight = level_weight  # Only used in validation
        
        self.mae_weight = mae_weight


    def get_loss(self, pred_batch, std_batch, ens_batch, target_batch):
        mae_total, mae_dict = self.mae_loss(pred_batch, target_batch)
        total_loss = self.mae_weight * mae_total
        
        latitudes = torch.deg2rad(pred_batch.metadata.lat)
        latitude_weight = torch.cos(latitudes) / torch.cos(latitudes).mean() if self.latitude_weight else 1.0
        latitude_weight = latitude_weight.to(mae_total.device)

        losses = mae_dict  # Start with MAE losses
        num_vars = (len(target_batch.surf_vars) + len(target_batch.atmos_vars))

        # Process both surface and atmospheric variables for other losses
        groups = [
            (target_batch.surf_vars, pred_batch.surf_vars, std_batch.surf_vars, 
             ens_batch.surf_vars, self.surf_weight, self.surf_var_weights),
            (target_batch.atmos_vars, pred_batch.atmos_vars, std_batch.atmos_vars, 
             ens_batch.atmos_vars, self.atmos_weight, self.atmos_var_weights)
        ]

        for target, pred, std, ens, group_weight, var_weights in groups:
            group_loss = 0
            for var_name in sorted(target.keys()):
                weight = var_weights.get(var_name, 1.0)
                
                # Apply latitude weighting if enabled
                if self.latitude_weight:
                    target_var = target[var_name] * latitude_weight[..., None]
                    pred_var = pred[var_name] * latitude_weight[..., None]
                    std_var = std[var_name] * latitude_weight[..., None]
                    ens_var = ens[var_name] * latitude_weight[..., None]
                else:
                    target_var = target[var_name]
                    pred_var = pred[var_name]
                    std_var = std[var_name]
                    ens_var = ens[var_name]

                

                
        # Normalize the total loss by the number of variables, useless for one dataset
        if not self.level_weight:
            total_loss = (self.dataset_weight / num_vars) * total_loss
                
        losses['total'] = total_loss

        return total_loss, losses

