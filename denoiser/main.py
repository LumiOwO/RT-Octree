import torch

from pathlib import Path

from denoiser.utils import seed_everything
from denoiser.runner import Runner
from denoiser.dataset import DenoiserDataset
from denoiser.logger.base_logger import BaseLogger
from denoiser.logger.wandb_logger import WandbLogger

import configargparse

def main(args):
    # Init
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # test
    from denoiser.network import _denoiser
    x = torch.ones((1, 800, 800, 3), dtype=torch.float32).to(device)
    print("compile succeed")
    imgs_in = torch.ones((1, 800, 800, 3), dtype=torch.float32).to(device)
    imgs_out = torch.ones((1, 800, 800, 3), dtype=torch.float32).to(device)
    _denoiser.filtering(x, imgs_in, imgs_out)
    return

    # Logger
    if args.use_wandb:
        logger = WandbLogger(args)
    else:
        logger = BaseLogger()
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    dataset = DenoiserDataset(args, device=device)
    logger.print("Dataset loaded.")

    # Create model
    model = DenoiserNetwork()

    # Create runner
    runner = Runner(args, dataset, logger, device=device)
    if args.task == "train":
        runner.train(model)
    elif args.task == "test":
        runner.test(model)
    elif args.task == "compact":
        runner.compact(model)
    else:
        raise NotImplementedError("Invalid task type.")


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, 
                        help="config file path")
    parser.add_argument("--task", type=str, choices=["train", "test", "compact"],
                        help="task type")
    parser.add_argument("--exp_name", type=str, 
                        help="experiment name")
    parser.add_argument("--logs_dir", type=str, default="../logs/", 
                        help="where to store ckpts and logs")
    parser.add_argument("--data_dir", type=str, default="../data/nerf_synthetic/lego", 
                        help="input data directory")

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help="layers in network")
    parser.add_argument("--netwidth", type=int, default=256, 
                        help="channels per layer")
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help="layers in fine network")
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help="channels per layer in fine network")
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help="batch size (number of random rays per gradient step)")
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help="learning rate")
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help="exponential learning rate decay (in 1000 steps)")
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help="number of rays processed in parallel, decrease if running out of memory")
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help="number of pts sent through network in parallel, decrease if running out of memory")
    parser.add_argument("--no_batching", action="store_true", 
                        help="only take random rays from 1 image at a time")
    parser.add_argument("--no_reload", action="store_true", 
                        help="do not reload weights from saved ckpt")
    parser.add_argument("--ft_path", type=str, default=None, 
                        help="specific weights npy file to reload for coarse network")
    parser.add_argument("--batch_size", type=str, default=16, 
                        help="batch_size for dataloader")
    parser.add_argument("--num_workers", type=int, default=12, 
                        help="num_workers for dataloader")

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help="number of coarse samples per ray")
    parser.add_argument("--N_importance", type=int, default=0,
                        help="number of additional fine samples per ray")
    parser.add_argument("--perturb", type=float, default=1.,
                        help="set to 0. for no jitter, 1. for jitter")
    parser.add_argument("--use_viewdirs", action="store_true", 
                        help="use full 5D input instead of 3D")
    parser.add_argument("--i_embed", type=int, default=0, 
                        help="set 0 for default positional encoding, -1 for none")
    parser.add_argument("--multires", type=int, default=10, 
                        help="log2 of max freq for positional encoding (3D location)")
    parser.add_argument("--multires_views", type=int, default=4, 
                        help="log2 of max freq for positional encoding (2D direction)")
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help="std dev of noise added to regularize sigma_a output, 1e0 recommended")

    parser.add_argument("--render_only", action="store_true", 
                        help="do not optimize, reload weights and render out render_poses path")
    parser.add_argument("--render_test", action="store_true", 
                        help="render the test set instead of render_poses path")
    parser.add_argument("--render_factor", type=int, default=0, 
                        help="downsampling factor to speed up rendering, set 4 or 8 for fast preview")

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help="number of steps to train on central crops")
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help="fraction of img taken for central crops") 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default="llff", 
                        help="options: llff / blender / deepvoxels")
    parser.add_argument("--testskip", type=int, default=8, 
                        help="will load 1/N images from test/val sets, useful for large datasets like deepvoxels")

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default="greek", 
                        help="options : armchair / cube / greek / vase")

    ## blender flags
    parser.add_argument("--white_bkgd", action="store_true", 
                        help="set to render synthetic data on a white bkgd (always use for dvoxels)")
    parser.add_argument("--half_res", action="store_true", 
                        help="load blender synthetic data at 400x400 instead of 800x800")

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help="downsample factor for LLFF images")
    parser.add_argument("--no_ndc", action="store_true", 
                        help="do not use normalized device coordinates (set for non-forward facing scenes)")
    parser.add_argument("--lindisp", action="store_true", 
                        help="sampling linearly in disparity rather than depth")
    parser.add_argument("--spherify", action="store_true", 
                        help="set for spherical 360 scenes")
    parser.add_argument("--llffhold", type=int, default=8, 
                        help="will take every 1/N images as LLFF test set, paper uses 8")

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help="frequency of console printout and metric loggin")
    parser.add_argument("--i_img",     type=int, default=500, 
                        help="frequency of tensorboard image logging")
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help="frequency of weight ckpt saving")
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help="frequency of testset saving")
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help="frequency of render_poses video saving")

    # args check
    args = parser.parse_args()
    if args.task != "train":
        args.use_wandb = False

    main(args)