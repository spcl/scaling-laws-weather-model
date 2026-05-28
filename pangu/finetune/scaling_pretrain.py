import sys

sys.path.append("/data")
from era5_data import utils, utils_data
from era5_data.utils_dist import get_dist_info, init_dist
from era5_data.config import cfg
from models.pangu_model import PanguModel
import torch
import os
from torch.utils import data
from models.pangu_sample import test, train
import argparse
import time
import logging
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from era5_data.zarr_helper import get_data_loader
"""
Fully finetune the pretrained model
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_net', type=str, default="scaling_pretrain")
    parser.add_argument('--load_my_best', type=bool, default=False)
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--dist', default=True)
    parser.add_argument('--resume', type=bool, default=False)

    args = parser.parse_args()
    starts = time.time()

    PATH = cfg.PG_INPUT_PATH


    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if args.dist:
        init_dist('pytorch')
    rank, world_size = get_dist_info()
    print("The rank and world size is", rank, world_size)
        
    device = torch.device(f"cuda:{rank}" if world_size > 1 else 'cuda')

    print(f"Predicting on {device}")

    output_path = os.path.join(cfg.PG_OUT_PATH, args.type_net, str(cfg.PG.HORIZON))
    os.makedirs(output_path, exist_ok=True)
    
    checkpoint_path = os.path.join(output_path, "ckpt.tar")

    writer_path = os.path.join(output_path, "writer")
    os.makedirs(writer_path, exist_ok=True)

    writer = SummaryWriter(writer_path)

    logger_name = args.type_net + str(cfg.PG.HORIZON)
    utils.logger_info(logger_name, os.path.join(output_path, logger_name + '.log'))

    logger = logging.getLogger(logger_name)

    # Zarr-based dataloader pipeline
    class Params:
        in_channels = slice(None)
        out_channels = slice(None)
        # batch_size = cfg.PG.TRAIN.BATCH_SIZE // len(opt['gpu_ids'])
        batch_size = 1
        global_means_path = "/data/era5_stats_69channels/global_mean.npy"
        global_stds_path = "/data/era5_stats_69channels/global_std.npy"
        data_num_shards = world_size
        data_shard_id = rank
        num_data_workers = 0

    train_dataloader, train_metadata, train_sampler = get_data_loader(Params, zarr_path=PATH, train=True)
    val_dataloader, val_metadata = get_data_loader(Params, zarr_path=PATH, train=False)

    dataset_length = len(train_dataloader)
    if rank == 0:
        print("dataset_length", dataset_length)

    if args.type_net == "scaling_pretrain":
        model = PanguModel(device=device).to(device)
    
    elif args.type_net == "deep_scaling_pretrain":
        model = PanguModel(device=device, depths = [3,9,9,3], num_heads = [6, 12, 12, 6], dims = [192, 384, 384, 192]).to(device)
    
    elif args.type_net == "wide_scaling_pretrain":
        model = PanguModel(device=device, depths = [2,6,6,2], num_heads = [6, 12, 12, 6], dims = [288, 576, 576, 288]).to(device)
    
    elif args.type_net == "large_scaling_pretrain":
        model = PanguModel(device=device, depths = [3,9,9,3], num_heads = [6, 12, 12, 6], dims = [288, 576, 576, 288]).to(device)
    
    
    elif args.type_net == "shallow_scaling_pretrain":
        model = PanguModel(device=device, depths = [1,3,3,1], num_heads = [6, 12, 12, 6], dims = [192, 384, 384, 192]).to(device)
        
    elif args.type_net == "narrow_scaling_pretrain":
        model = PanguModel(device=device, depths = [2,4,4,2], num_heads = [6, 12, 12, 6], dims = [96, 192, 192, 96]).to(device)
    
    elif args.type_net == "small_scaling_pretrain":
        model = PanguModel(device=device, depths = [1,3,3,1], num_heads = [6, 12, 12, 6], dims = [96, 192, 192, 96]).to(device)
    
    else:
        raise ValueError(f"Invalid model type: {args.type_net}")
    
    # deep
    # model = PanguModel(device=device, depths = [3,9,9,3], num_heads = [6, 12, 12, 6], dims = [192, 384, 384, 192]).to(device)
    # print("model.device", model.device)

    # checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch)
    # model.load_state_dict(checkpoint['model'])
    #Fully finetune
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = cfg.PG.TRAIN.LR , weight_decay= cfg.PG.TRAIN.WEIGHT_DECAY)

    if rank == 0:
        msg = '\n'
        msg += utils.torch_summarize(model, show_weights=False)
        logger.info(msg)

    #weather_statistics = utils.LoadStatic_pretrain()
    if rank == 0:
        print("weather statistics are loaded!")
    torch.set_num_threads(cfg.GLOBAL.NUM_STREADS)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50], gamma=0.5)
    start_epoch = 1

    model = train(model, train_loader=train_dataloader,
                     val_loader=val_dataloader,
                     optimizer=optimizer,
                     lr_scheduler=lr_scheduler,
                     res_path = output_path,
                     device=device,
                     rank=rank,
                     writer=writer, logger = logger, 
                     start_epoch=start_epoch, 
                     checkpoint_path=checkpoint_path,
                     resume=args.resume)

    if args.load_my_best:
        best_model = torch.load(os.path.join(output_path,"models/best_model.pth"),map_location='cuda:0')

    logger.info("Begin testing...")

    # TODO: add testing in scaling pretrain
    # test(test_loader=test_dataloader,
    #      model=best_model,
    #      device=device,
    #      res_path=output_path)
    #CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 finetune_lastLayer_ddp.py --dist True
