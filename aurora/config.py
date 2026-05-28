import argparse

import yaml


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.safe_load(f)
        yaml_args = argparse.Namespace()
        yaml_args.__dict__.update(config)
    return yaml_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--net",
        type=str,
        default="aurora",
        help="Network architecture to use. [default: aurora] Options: [aurora]",
    )
    parser.add_argument("--config", help="Load settings from yaml.")
    parser.add_argument(
        "--dataset_config_path", 
        default='dataset_config.yaml',
        help="Load dataset configs from yaml.")
    parser.add_argument("--no_gpu", action="store_true", default=False, help="Explicitly use CPU [default: uses gpu]")
    parser.add_argument("--num_nodes", type=int, default=1, help="num nodes to train on")
    parser.add_argument("--devices", type=int, default=1, help="num GPU devices on each node to train on")
    parser.add_argument("--fix_seedcudnn", action="store_false", default=True, help="true if fixing cudnn")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="#threads to run for dataloaders") # big value oom
    parser.add_argument("--backend", type=str, default='nccl', help="Backend for distributed trianing ")

    # parser.add_argument('--category', default=None, help='Which single class to train on [default: None]')
    parser.add_argument("--log_dir", default="logs", help="Log dir [default: log]")
    parser.add_argument("--n_pts", type=int, default=5000, help="Point Number [default: 2048]")
    parser.add_argument("--n_verts", type=int, default=2463, help="Vertex Number [default: 2463]")

    parser.add_argument("--data", type=str, default="./data", help="dataset path")
    
    parser.add_argument("--stats_trainingset_name",
        type=str,
        default=None,
        help="Filename of npz file which stores training set statistics.",
    )
    
    parser.add_argument("--epochs", type=int, default=100, help="Epoch to run [default: 200]")
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Batch Size during training [default: 100]"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Initial learning rate [default: 1e-4]"
    )
    parser.add_argument(
        "--max_norm", type=float, default=1.0, help="Max norm for gradient clipping [default: 1.0]"
    )
    parser.add_argument(
        "--constant_lr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Constant learning rate [default: False]",
    )
    parser.add_argument(
        "--constant_lr_after_warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Constant learning rate after warmup [default: False]",
    )
    parser.add_argument("--optimizer", default="adamW", help="adam or sgd [default: adamW]")


    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume training using {log_dir}/last.ckpt.",
    )

    # load checkpoint for inference
    parser.add_argument('--ckpt_name', type=str, 
                    default="last.ckpt",
                    help='Name of the checkpoint file to load')
    
    parser.add_argument(
        "--load_aurora_pretrain_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize model weights to Aurora pretrained weights, where applicable. [default: True]",
    )
    

    parser.add_argument("--restore_checkpoint", default=None, help="restore_model")

    parser.add_argument("--valid_checkpoint", default=None, help="restore_model")

    parser.add_argument("--report_freq", type=int, default=10, help="report batch frequency")
    parser.add_argument("--wnb_entity", type=str, default="aurora", help="W&B project name")
    parser.add_argument("--wnb_project", type=str, default="aurora_era5", help="W&B project name")
    parser.add_argument("--wnb_name", type=str, default="", help="W&B run name")
    parser.add_argument("--wnb_id", type=str, default=None, help="W&B project id")
    parser.add_argument(
        "--wnb_mode", type=str, default="online", help="W&B mode. use online or disabled"
    )
    parser.add_argument("--log_every_n_steps", type=int, default=1, help="log freq for wandb")
    parser.add_argument("--val_every_n_steps", type=float, default=5, help="validation")

    parser.add_argument("--embed_dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--depth_scaling_factor", type=float, default=1, help="encode / decoder depth scaling factor")


    parser.add_argument("--note", type=str, default="", help="extra note about the run")
    
    

    parser.add_argument(
        "--data_sources",
        nargs='+',
        default=['era5'],
        help="List of data sources to use.",
    )
    args = parser.parse_args()

    if args.config:
        print(f"Loading config from {args.config}. Will overwrite any command line arguments with yaml content.")
        yaml_args = parse_config(args.config)
        args.__dict__.update({**args.__dict__, **yaml_args.__dict__})

    return args
