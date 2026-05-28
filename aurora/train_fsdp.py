## import modules

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import os
import glob
import sys
import math
import numpy as np
import wandb
from datetime import datetime, timedelta
import torch
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist

# Import FSDP and Mixed Precision from PyTorch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, CSVLogger

from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import wrap
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

from huggingface_hub import hf_hub_download

from aurora import Aurora, Batch, Metadata
from aurora.model import swin3d as aurora_swin3d_module
from aurora.model.decoder import Perceiver3DDecoder

## import custom modules
import yaml
import pickle
from config import parse_args
from utils import dataset, logging_utils, losses
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torch.utils.data.distributed import DistributedSampler

import psutil
import time

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'

_DEBUG = True
is_rank0 = False
if _DEBUG:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        is_rank0 = True
        logging_utils.log_entire_script(__file__)

args = parse_args()


wandb_key = os.getenv("WANDB_KEY")
if is_rank0:  # Only attempt login on rank 0
    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        if args.wnb_mode != 'disabled':
            print(f'WANDB_KEY is not set. Please set WANDB_KEY to use wandb. not logging to WANDB.')
            args.wnb_mode = 'disabled'

def get_total_gpus():
    # Get number of GPUs on the current node
    gpus_per_node = torch.cuda.device_count()
    # Get total number of nodes (ensure this is set externally)
    num_nodes = int(os.getenv("WORLD_SIZE", "1"))  # Default to 1 node if not set
    total_gpus = gpus_per_node * num_nodes
    return total_gpus

DATA_PATH_PREFIX = '/'
start_time_train = datetime(1979, 1, 1, 0, 0, 0)
#start_time_train = datetime(2020, 11, 1, 0, 0, 0) ## making it 2 months for testing it now
end_time_train = datetime(2020, 12, 31, 23, 0, 0)
start_time_val = datetime(2021, 1, 1, 0, 0, 0)
end_time_val = datetime(2021, 12, 31, 23, 0, 0) ## last date on wb2 is 2021-12-31
#end_time_val = datetime(2021, 1, 2, 23, 0, 0)
path_weatherbench2 = '/data/weatherbench2_original'
path_slt = '/data/ERA5_missing_feats/static/ERA5_Soiltype.npy'
slt = np.load(path_slt)
inds_train = [np.datetime64(start_time_train + timedelta(hours=i)) for i in range(int((end_time_train - start_time_train).total_seconds() // 3600) + 1)]
inds_val = [np.datetime64(start_time_val + timedelta(hours=i)) for i in range(int((end_time_val - start_time_val).total_seconds() // 3600) + 1)]



def get_device(use_gpu):
    return "cuda" if use_gpu and torch.cuda.is_available() else "cpu"


def load_data(yaml_path, datasets_type):
    """Load datasets based on YAML configuration."""
    path_slt = os.path.join(DATA_PATH_PREFIX, 'data/ERA5_missing_feats/static/ERA5_Soiltype.npy')
    slt = np.load(path_slt)
    surf_vars, atmos_vars, static_vars = (), (), ()
    dataset_train, dataset_val = [], []
    batch_sizes = []
    subset_sizes = []

    for dataset_type in datasets_type:
        with open(yaml_path, 'r') as file:
            yml_file = yaml.safe_load(file)
            data_cls = getattr(dataset, yml_file[dataset_type]['class'])
            conf_train = yml_file[dataset_type]['conf']
            conf_train['path'] = os.path.join(DATA_PATH_PREFIX, conf_train['path'])
            
            if dataset_type == 'era5':
                conf_train['inds'] = inds_train
                conf_train['atmos_levels'] = np.asarray(conf_train['atmos_levels'], dtype=np.int32)
                conf_train['slt'] = slt
                conf_val = conf_train.copy()
                conf_val['inds'] = inds_val
            else:
                conf_train['start_idx'] = yml_file[dataset_type]['start_train']
                conf_train['end_idx'] = yml_file[dataset_type]['end_train']
                if 'wb2_path' in conf_train:
                    conf_train['wb2_path'] = os.path.join(DATA_PATH_PREFIX, conf_train['wb2_path'])
                conf_val = conf_train.copy()
                conf_val['start_idx'] = yml_file[dataset_type]['start_val']
                conf_val['end_idx'] = yml_file[dataset_type]['end_val']

            surf_vars += tuple([yml_file[dataset_type]['conf']['variable_name_mapping'].get(var, var) 
                              for var in yml_file[dataset_type]['surf_vars']] 
                              if 'variable_name_mapping' in yml_file[dataset_type]['conf'] 
                              else yml_file[dataset_type]['surf_vars'])
            atmos_vars += tuple([yml_file[dataset_type]['conf']['variable_name_mapping'].get(var, var) 
                                for var in yml_file[dataset_type]['atmos_vars']]
                                if 'variable_name_mapping' in yml_file[dataset_type]['conf']
                                else yml_file[dataset_type]['atmos_vars'])
            static_vars += tuple([yml_file[dataset_type]['conf']['variable_name_mapping'].get(var, var) 
                                for var in yml_file[dataset_type]['static_vars']]
                                if 'variable_name_mapping' in yml_file[dataset_type]['conf']
                                else yml_file[dataset_type]['static_vars'])

            batch_sizes.append(yml_file[dataset_type]['batch_size'])
            subset_sizes.append(yml_file[dataset_type].get('subset_size', 256))

            dataset_train.append(data_cls(**conf_train))
            dataset_val.append(data_cls(**conf_val))
            

    if args.devices > 1:
        dist.init_process_group(backend=args.backend)

    if len(dataset_train) == 1:
        dataloader_train = StatefulDataLoader(
            dataset_train[0], 
            sampler=StatefulDistributedSampler(dataset_train[0]) if args.devices > 1 else None,
            batch_size=batch_sizes[0],
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,  
            persistent_workers=False  
        )
        # do not use StatefulDataLoader for validation, as it is not needed and will cause skipping val
        dataloader_val = DataLoader(
            Subset(dataset_val[0],range(16)), # 16 devices in training
            batch_size=batch_sizes[0],
            num_workers=args.num_workers,
            shuffle=False,  # Ensure deterministic for validation
            drop_last=True,
            pin_memory=True,  
            persistent_workers=False  
        )
    else:
        dataloader_train = dataset.DatasetMixer(
            dataset_train, subset_sizes, batch_sizes, num_workers=args.num_workers,
            drop_last=True, pin_memory=True,  persistent_workers=True 
        )
        dataloader_val = dataset.DatasetMixer(
            dataset_val, subset_sizes, batch_sizes, num_workers=args.num_workers, # all data is used for validation
            drop_last=True, pin_memory=True,  persistent_workers=True,
            shuffle=False
        )

    return dataloader_train, dataloader_val, surf_vars, atmos_vars, static_vars



def main():
    L.seed_everything(args.seed, workers=True)
    if is_rank0: 
        logging_utils.copy_exp_params(log_dir=args.log_dir, config_file=args.config, args=args)
    device = get_device(use_gpu=not args.no_gpu)

    if not args.no_gpu and not torch.cuda.is_available():
        logging.info("GPU training is requested, but no GPU device available.")
        sys.exit(1)


    # Replace the existing dataset setup with:
    dataloader_train, dataloader_val, surf_vars, atmos_vars, static_vars = load_data(
        args.dataset_config_path, args.data_sources
    )
    
    ## Need to check if last.ckpt exists when resume args is passed. If not, one should disable "Resume" status and still consider loading from aurora weights if the other args permit.
    if args.resume: 
        ckpt_fname = os.path.join(args.log_dir, "last.ckpt")
        if os.path.isfile(ckpt_fname):
            trainer_fit_ckpt_path = ckpt_fname
            version = 1
            trainer_fit_last_ckpt_path = ckpt_fname.replace('last',f'last-v{verssion}')
            while os.path.isfile(trainer_fit_last_ckpt_path):
                version += 1
                trainer_fit_ckpt_path = trainer_fit_last_ckpt_path
                trainer_fit_last_ckpt_path = ckpt_fname.replace('last',f'last-v{version}')
            checkpoint = torch.load(trainer_fit_ckpt_path)
            if 'dataloader_state' in checkpoint.keys():
                print(f"Loading dataloader state from {trainer_fit_ckpt_path}")
                dataloader_train.load_state_dict(checkpoint['dataloader_state'])

            logging.info(f'resuming from {trainer_fit_ckpt_path}')
        else:
            logging.info(
                f"\n\n\n--resume is passed, but could not find ckpt at {ckpt_fname}. Starting from scratch.\n\n\n"
            )
            trainer_fit_ckpt_path = None
            args.resume = False
    else:
        trainer_fit_ckpt_path = None

    ## setup model architecture
    assert args.embed_dim % 8 == 0 and args.embed_dim >= 64, "embed_dim must be divisible by 8 and le than 64"
    encoder_num_heads = tuple((x * args.embed_dim) // 512 for x in (8, 16, 32))
    decoder_num_heads = tuple((x * args.embed_dim) // 512 for x in (32, 16, 8))
    encoder_depths = tuple(int(x // (args.depth_scaling_factor)) for x in (6, 10, 8))
    decoder_depths = tuple(int(x // (args.depth_scaling_factor)) for x in (8, 10, 6))

    use_small_model = False
    if use_small_model:
        # Minimal model architecture
        model = Aurora(
            use_lora=False, 
            autocast=False, # Use AMP (mixed precision to fit to GPU)
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            encoder_depths=(3, 5, 4),
            encoder_num_heads=(4, 8, 16),
            decoder_depths=(4, 5, 3),
            decoder_num_heads=(16, 8, 4),
            embed_dim=128,
        )
    else:
        # Default model architecture
        model = Aurora(
            use_lora=False, 
            autocast=True, # Use AMP (mixed precision to fit to GPU)
            embed_dim=args.embed_dim,
            encoder_depths=encoder_depths,
            encoder_num_heads=encoder_num_heads,
            decoder_depths=decoder_depths,
            decoder_num_heads=decoder_num_heads,
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            drop_path=0.2,
        )
        
        # Initially load the pretrained aurora by default.
        if args.load_aurora_pretrain_weights and not args.resume:
            path = hf_hub_download(repo_id="microsoft/aurora", filename="aurora-0.25-pretrained.ckpt") # float32
            model.load_checkpoint_local(path, strict=False) ## we do not yet have slt
        model.configure_activation_checkpointing() ## recalculates backbone activations on the backprop.



    ## setup loss function
    
    # no percipitaion, no vertical windspeed
    # sum of weights=6.3, graphcast sum of weights=7.4
    surf_var_weights={
            # Any variables not specified here are weighted as 1.0.
            # A single-level variable, but an important headline variable
            # and also one which we have struggled to get good performance
            # on at short lead times, so leaving it weighted at 1.0, equal
            # to the multi-level variables:
            "2m": 1.0,
            # New single-level variables, which we don't weight too highly
            # to avoid hurting performance on other variables.
            "10u": 0.1,
            "10v": 0.1,
            "msl": 0.1,
        }
    atmos_var_weights = {
                'z': 1.0,
                'q': 1.0,
                't': 1.0,
                'u': 1.0,
                'v': 1.0,
            }
    
    loss_obj = losses.MyLoss(
                        mae_weight=1.0,          
                    )
    loss_obj_graphcast = losses.MyLoss(
                        mae_weight=1.0,          
                        surf_weight=1.0,
                        surf_var_weights=surf_var_weights, 
                        atmos_var_weights=atmos_var_weights,
                        level_weight=True)
    
    ## setup lightning module & trainer
    class LightningModule(L.LightningModule):
        def __init__(self, net, loss_fn, loss_fn_val, **kwargs):
            super().__init__()
            self.net = net
            self.loss_fn = loss_fn
            self.loss_fn_val = loss_fn_val
            self.rc_quantile = kwargs.pop('rc_quantile', 0.7)  # Add RC loss quantile parameter
            self.example_input_array = kwargs.pop('example_input_array', None)
            self.lr_scheduler_interval = kwargs.pop('lr_scheduler_interval', 'step')
            self.batch_size = kwargs.pop('batch_size', None)
            self.learning_rate = kwargs.pop('learning_rate', 5e-4)  # Changed base learning rate to 5e-4
            self.constant_lr = kwargs.pop('constant_lr', False)  # Use constant learning rate
            self.constant_lr_after_warmup = kwargs.pop('constant_lr_after_warmup', False)  # Use constant learning rate after warmup
            self.warmup_steps = kwargs.pop('warmup_steps', 1000)    # 1k warmup steps
            self.weight_decay = kwargs.pop('weight_decay', 5e-6)    # AdamW weight decay
            for key, val in kwargs.items():
                setattr(self, key, val)
            self.save_hyperparameters(ignore=['net'])
            self.worst_metrics_train, self.worst_metrics_val, self.worst_metrics_test = {}, {}, {}

        def on_train_start(self):
            """Initialize timing variables when training starts"""
            self.last_step_time = time.time()
            print(f"self.net type = {type(self.net)}")

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,  # base_lr = 5e-4
                weight_decay=self.weight_decay,  # weight decay = 5e-6
                # betas=(0.9, 0.98),  # beta1 = 0.9, beta2 = 0.98
            )
            if trainer_fit_ckpt_path is not None:
                # Load optimizer state from checkpoint if available
                checkpoint = torch.load(trainer_fit_ckpt_path, map_location=self.device)
                optimizer.load_state_dict(checkpoint['optimizer_states'][0])
                print("optimizer state loaded from checkpoint with global_step=", checkpoint['global_step'])

            # Calculate total training steps
            total_steps = self.trainer.estimated_stepping_batches 
            print("total_steps (self.trainer.estimated_stepping_batches)", total_steps)

            if self.constant_lr:
                # Use constant learning rate
                print('Using constant learning rate=', self.learning_rate)
                scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=self.warmup_steps)
            else:
                # 1. lienar warmup，from 1e-8 to base_lr，within warmup_steps
                warmup_scheduler = LinearLR(optimizer, start_factor=1e-8/self.learning_rate, end_factor=1.0, total_iters=self.warmup_steps)
                if self.constant_lr_after_warmup:
                    # Use constant learning rate after warmup
                    print('Using constant learning rate after warmup')
                    second_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=total_steps - self.warmup_steps)
                else:
                    # 2. Cosine decay，decrease from base_lr to 0.1x base_lr
                    second_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - self.warmup_steps, eta_min=self.learning_rate * 0.1)

                # combine scheduler
                scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, second_scheduler], milestones=[self.warmup_steps])

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }


        def forward(self, x):
            return self.net.forward(x)

        @staticmethod
        def _make_batch(batch):
            # Create input batch
            x_time = tuple([datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%f') for t in batch['x_time']])
            atmos_levels = tuple(batch['atmos_levels'][0].cpu().numpy().tolist())
            static_vars = {k: batch['x_static'][k][0] for k in batch['x_static'].keys()}

            batch_obj_x = Batch(
                surf_vars={k: batch['x_srf'][k] for k in batch['x_srf'].keys()},
                static_vars=static_vars,
                atmos_vars={k: batch['x_atmos'][k] for k in batch['x_atmos'].keys()},
                metadata=Metadata(
                    lat=batch['lat'][0],
                    lon=batch['lon'][0],
                    time=x_time,
                    atmos_levels=atmos_levels,
                    locations={k: v[0] for k, v in batch['locations'].items()},
                    scales={k: v[0] for k, v in batch['scales'].items()},
                    grid_resolution=batch['grid_resolution'][0],
                    is_global_observation=batch['is_global_observation'][0]
                )
            )

            # Create target batch
            y_time = tuple([datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%f') for t in batch['y_time']])
            batch_obj_y = Batch(
                surf_vars={k: batch['y_srf'][k] for k in batch['y_srf'].keys()},
                static_vars={k: batch['y_static'][k] for k in batch['y_static'].keys()},
                atmos_vars={k: batch['y_atmos'][k] for k in batch['y_atmos'].keys()},
                metadata=Metadata(
                    lat=batch['lat'][0],
                    lon=batch['lon'][0],
                    time=y_time,
                    atmos_levels=atmos_levels,
                    locations={k: v[0] for k, v in batch['locations'].items()},
                    scales={k: v[0] for k, v in batch['scales'].items()},
                    grid_resolution=batch['grid_resolution'][0],
                    is_global_observation=batch['is_global_observation'][0]
                )
            )
            del batch
            return batch_obj_x, batch_obj_y


        
        def _get_system_metrics(self):
            """Gather system metrics including GPU, CPU, memory, and throughput."""
            try:
                # Calculate throughput
                current_time = time.time()
                batch_size = self.trainer.train_dataloader.batch_size
                world_size = self.trainer.world_size
                step_elapsed_time = current_time - self.last_step_time
                step_throughput = (batch_size * world_size) / step_elapsed_time
              
                        
                # Prepare metrics dictionary with proper WandB formatting
                metrics = {
                    'performance/throughput': step_throughput,  # Changed naming for better WandB organization
                    'performance/cpu_usage': psutil.cpu_percent(),
                    'performance/memory_usage_percent': psutil.virtual_memory().percent,
                    'performance/memory_used_gb': psutil.virtual_memory().used / (1024**3),
                    'performance/memory_available_gb': psutil.virtual_memory().available / (1024**3)
                }
                
                # Add GPU metrics
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        metrics.update({
                            f'performance/gpu_{i}/memory_used_mb': torch.cuda.memory_allocated(i) / 1024**2,
                            f'performance/gpu_{i}/memory_reserved_mb': torch.cuda.memory_reserved(i) / 1024**2,
                            f'performance/gpu_{i}/max_memory_mb': torch.cuda.max_memory_allocated(i) / 1024**2
                        })
                
                self.last_step_time = current_time
                return metrics
                
            except Exception as e:
                print(f"Error collecting system metrics: {str(e)}")
                return {}
                

        def training_step(self, batch, batch_idx):
            sync = batch.get('sync', False)
            batch_obj_x, batch_obj_y = self._make_batch(batch)
            batch_obj_y = batch_obj_y.normalise(surf_stats =None) # Normalise target batch in the training step
            del batch
            
            # Forward pass
            batch_pred, batch_std, batch_ens = self.net.forward(batch_obj_x)
            del batch_obj_x

            # Main task loss
            task_loss, loss_dict = self.loss_fn(batch_pred, batch_std, batch_ens, batch_obj_y)
            del batch_std, batch_ens
            
            total_loss = task_loss
            del batch_pred,batch_obj_y
            
            
            # Get system metrics and add to loss_dict 
            loss_dict['loss_train'] = total_loss
            system_metrics = self._get_system_metrics()
            loss_dict.update(system_metrics)

            # Log all losses
            self.log_dict(loss_dict, batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            if sync and torch.distributed.is_initialized():
                torch.distributed.barrier()

            return total_loss

        def validation_step(self, val_batch, batch_idx):
            batch_obj_x, batch_obj_y = self._make_batch(val_batch) # WHY??? Don't normalise target batch in the validation step
            del val_batch
            batch_pred, batch_std, batch_ens = self.net.forward(batch_obj_x)
            
            # normalize
            batch_obj_y = batch_obj_y.normalise(surf_stats =None) # Add normalisation to target batch in the validation step
            batch_pred = batch_pred.normalise(surf_stats =None) # Add normalisation to target batch in the validation step
            
            del batch_obj_x
            
            # Main task loss
            task_loss, loss_dict = self.loss_fn_val(batch_pred, batch_std, batch_ens, batch_obj_y)
            del batch_std, batch_ens
            
            total_loss = task_loss

            del batch_pred,batch_obj_y
            

            # Log metrics
            loss_dict = {f'{key}_val': value for key, value in loss_dict.items()}
            loss_dict['loss_val'] = total_loss
            self.log_dict(loss_dict, batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        
        def optimizer_step(
            self,
            epoch=None,
            batch_idx=None,
            optimizer=None,
            optimizer_closure=None,
            optimizer_idx=None,
            on_tpu=None,
            using_native_amp=None,
            using_lbfgs=None,
        ):  
            # Run the closure to get the loss and compute gradients
            if optimizer_closure is not None:
                optimizer_closure()
            
            # Clip gradients using FSDP's clip_grad_norm_: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_
            if hasattr(self, 'net'):
                # Check if we're using FSDP strategy
                using_fsdp = isinstance(self.trainer.strategy, FSDPStrategy)
                try:
                    if using_fsdp:
                        # First check the gradient norm
                        pre_clip_norm = self.trainer.strategy.model.clip_grad_norm_(max_norm=float('inf'))
                        self.log('grad_norm_pre_clip', pre_clip_norm)
                        
                        # Track skipped steps
                        if pre_clip_norm > 30.0:
                            self.skipped_steps = getattr(self, 'skipped_steps', 0) + 1
                            skip_rate = self.skipped_steps / (self.global_step + 1)
                            self.log('skip_rate', skip_rate)
                            
                            print(f"Warning: High skip rate ({skip_rate:.2%}). Consider adjusting learning rate or model architecture.")
                            
                            print(f"Skipping step {self.global_step} due to large gradient norm: {pre_clip_norm:.3f}")
                            optimizer.zero_grad()
                            return
                        
                        # More moderate clipping threshold
                        post_clip_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=args.max_norm)
                        self.log('grad_norm_post_clip', post_clip_norm)
                        
                        if self.global_step % 100 == 0:
                            print(f"Grad norms - Pre: {pre_clip_norm:.3f}, Post: {post_clip_norm:.3f}")
                
                except RuntimeError as e:
                    print(f"Gradient clipping failed: {e}")
                    optimizer.zero_grad()
                    return
            # Update parameters
            optimizer.step()
            
            # Zero gradients
            optimizer.zero_grad()

    # Initialize LightningModule
    model_lightning = LightningModule(
        net=model, 
        loss_fn=loss_obj.get_loss,  
        loss_fn_val=loss_obj_graphcast.get_loss, 
        example_input_array=None, 
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        constant_lr=args.constant_lr,
        constant_lr_after_warmup=args.constant_lr_after_warmup,
        weight_decay=5e-6,
        warmup_steps=200,
        trainer_fit_ckpt_path=trainer_fit_ckpt_path
    )
    
    # Setup WandB
    # wnb_id = f'{args.wnb_project}_{str(args.log_dir).split("/")[-1]}' #_{time.strftime("%Y%m%d-%H%M%S")}'
    # lightning_logs_path = os.path.join(args.log_dir, 'lightning_logs', 'version_1')
    # wandb.tensorboard.patch(root_logdir=lightning_logs_path, pytorch=True)
    # wandb.init(
    #     project=args.wnb_project, 
    #     name=args.wnb_name, 
    #     id=wnb_id, 
    #     config=args, 
    #     save_code=True, 
    #     resume='allow', 
    #     mode=args.wnb_mode
    # )

    ### setup, callbacks,  logging and checkpoints
    class PushProfilingTraceToWB(Callback):
        def __init__(self, lightning_logs_path):
            super().__init__()
            self.lightning_logs_path = lightning_logs_path
            self.profiling_files = glob.glob(os.path.join(lightning_logs_path, f'.*.pt.trace.json'))
        def on_train_epoch_end(self, *args, **kwargs):
            profiling_files = glob.glob(os.path.join(self.lightning_logs_path, f'.*.pt.trace.json'))
            new_files = []
            for file in profiling_files:
                if file not in self.profiling_files:
                    new_files.append(file)
                    self.profiling_files.append(file)
            for f in new_files:
                wandb.save(os.path.basename(f), base_path=self.lightning_logs_path, policy='now')
    
    class StatefulDataLoaderCallback(Callback):
        """
        PyTorch Lightning callback to save and restore dataloader state during training.

        Args:
            dataloader (DataLoader): The dataloader instance to manage
            checkpoint_dir (str): Directory to save dataloader state checkpoints
        """
        def __init__(self, dataloader: DataLoader, checkpoint_dir: str = './dataloader_checkpoints'):
            super().__init__()
            self.dataloader = dataloader
            self.checkpoint_dir = checkpoint_dir
            os.makedirs(checkpoint_dir, exist_ok=True)

        def _save_dataloader_state(self, trainer, checkpoint_path):
            """Save dataloader state to a file"""
            state_path = os.path.join(self.checkpoint_dir, f"dataloader_state_{trainer.global_step}.pkl")
            with open(state_path, 'wb') as f:
                pickle.dump(self.dataloader.state_dict(), f)
            return state_path

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            """Save dataloader state when a checkpoint is saved"""
            # Only save on rank 0 to prevent file conflicts
            print("saving checkpoint")
            if trainer.is_global_zero:
                # Check if the checkpoint is triggered by modelcheckpoint_callback_regular_step_save
                for callback in trainer.callbacks:
                    if isinstance(callback, ModelCheckpoint) and callback == modelcheckpoint_callback_regular_step_save:
                        # Add the dataloader state in the checkpoint
                        checkpoint['dataloader_state'] = self.dataloader.state_dict()
                        break
        
        # not used actually, restored in main
        def on_load_checkpoint(self, trainer, pl_module, checkpoint):
            """Restore dataloader state from the checkpoint"""
            if trainer.is_global_zero and 'dataloader_state' in checkpoint:
                self.dataloader.load_state_dict(checkpoint['dataloader_state'])

    ## Define Callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True, log_weight_decay=True)
    modelcheckpoint_callback_last_step_save = ModelCheckpoint(
        dirpath=args.log_dir, 
        filename="model_ckpt-last-{step}-{loss_train:.5f}",
        save_last=True
    )
    # os.makedirs(os.path.join(args.log_dir,'regular_save'), exist_ok=True)
    modelcheckpoint_callback_regular_step_save = ModelCheckpoint(
        dirpath=args.log_dir, 
        filename="model_ckpt-{step}-{loss_train:.5f}", 
        every_n_train_steps=args.val_every_n_steps,
        monitor="loss_val", 
        save_top_k=-1, 
        mode='min', 
    )
    modelcheckpoint_callback_regular_epoch_save = ModelCheckpoint(
        dirpath=args.log_dir, 
        filename="model_ckpt-{epoch}-{step}-{loss_train:.5f}", 
        save_on_train_epoch_end=True, 
        every_n_epochs=1,
        save_last=False
    )
    modelcheckpoint_callback_best_val_save = ModelCheckpoint(
        dirpath=args.log_dir, 
        filename="model_best_val_ckpt-{epoch}-{step}-{loss_val:.2f}", 
        monitor="loss_val", 
        save_top_k=3, 
        mode='min', 
        save_on_train_epoch_end=False
    )
    callbacks = [
        modelcheckpoint_callback_regular_step_save,
        modelcheckpoint_callback_last_step_save,
        # modelcheckpoint_callback_regular_epoch_save, 
        modelcheckpoint_callback_best_val_save,
        # PushProfilingTraceToWB(lightning_logs_path=lightning_logs_path),
        StatefulDataLoaderCallback(dataloader=dataloader_train, checkpoint_dir=args.log_dir),
        lr_monitor,
        # DeviceStatsMonitor(),  # Logs CPU stats
    ]

    # CSV logger
    # no version, automatically creates a new folder
    logger = CSVLogger(save_dir=args.log_dir, name=f"train_dim{args.embed_dim}_depth{args.depth_scaling_factor}")

    # Alternatively, you can use WandbLogger if preferred
    # logger = WandbLogger(
    #     save_dir=args.log_dir,
    #     entity=args.wnb_entity,
    #     name=args.wnb_name,
    #     project=args.wnb_project,
    #     id=args.wnb_id,
    #     log_model=False,
    #     save_code=True,
    #     resume='allow',
    #     mode=args.wnb_mode,
    #     config=args
    # )

    ## Setup and Launch Lightning Trainer
    deterministic_trainer = False  # Might make training slower
    check_val_every_n_epoch = 1  # Use val_check_interval if you want to run val every N steps
    total_train_minibatches = int(len(dataloader_train)//get_total_gpus())
    print(f'get_total_gpus returns {get_total_gpus()}')
    if total_train_minibatches >= 900:
        val_check_interval = int(total_train_minibatches)
    else:
        val_check_interval = total_train_minibatches
    val_check_interval = args.val_every_n_steps # val is run on all data
    log_every_n_steps = args.log_every_n_steps  # How often to add logging rows
    max_epochs = args.epochs
    accelerator = 'gpu' if not args.no_gpu else 'cpu'
    devices = args.devices  # Number of GPUs on each node
    num_nodes = args.num_nodes  # Number of nodes. Total GPUs = num_nodes x devices

    ### Strategy
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 1:
        # Define FSDPStrategy with Mixed Precision
        fsdp_strategy = FSDPStrategy(
            activation_checkpointing_policy={
                Perceiver3DDecoder,   
            },
            cpu_offload=True,
            sharding_strategy="FULL_SHARD", 
            #sharding_strategy="HYBRID_SHARD", 
            backward_prefetch="SHARD",
            use_orig_params=True,
            timeout = timedelta(seconds=6000), # set NCCL timeout to 100 mins
            process_group_backend=args.backend,
        )
        strategy = fsdp_strategy
    else:
        strategy = 'auto'
    if is_rank0:
        logging.info(f"Strategy: {strategy}, num_gpus: {num_gpus}.")

    # Keep precision as float32 in trainer
    trainer = L.Trainer(
        accelerator=accelerator, 
        devices=devices, 
        num_nodes=num_nodes, 
        strategy=strategy, 
        precision='32-true',  # Keep this as 32-bit
        deterministic=deterministic_trainer, 
        callbacks=callbacks, 
        # check_val_every_n_epoch=check_val_every_n_epoch, 
        val_check_interval = val_check_interval,
        log_every_n_steps=log_every_n_steps, 
        logger=logger, 
        min_epochs=10, 
        max_epochs=max_epochs, 
        profiler="pytorch", 
        enable_progress_bar=True, 
        num_sanity_val_steps=0, 
        gradient_clip_val=1.0, 
        gradient_clip_algorithm='value',
    )
    logging.info(f"trainer state fn: {trainer.state.fn}, status: {trainer.state.status}.")
        
    trainer.fit(model_lightning, dataloader_train, dataloader_val, ckpt_path=trainer_fit_ckpt_path)

    wandb.finish()
    
    logging.info("Training is completed.")

if __name__ == "__main__":
    main()
