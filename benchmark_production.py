#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_combined
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from PIL import Image  
import numpy as np
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import torchvision
from utils.sh_utils import SH2RGB, RGB2SH
from torch.nn.functional import normalize

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    benchmark_log_folder = 'benchmark_logs'
    os.makedirs(benchmark_log_folder, exist_ok=True)

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    scene.gaussians.load_ply("/home/koca/projects/gaussian-splatting/output/7b6477cb95_try_regularization0_80_inverse_try_stdloss0_02_3by3_dilated_mask80/point_cloud/iteration_30000/point_cloud.ply")
    gaussians = scene.gaussians
    gaussians.training_setup(opt)
    gaussians.optimizer.param_groups[3]['lr'] = 0.0025 # 0.0025
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    distance_gaussians = torch.ones(gaussians.get_xyz.shape[0], dtype=torch.float32, device="cuda")*-1

    iteration_per_view = 200

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    epsilon = 0.001

    view_num = 0

    gaussians.get_mro.grad = None

    for iteration in range(first_iter, opt.iterations + 1):

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        view_num += 1

        N = torch.clip(SH2RGB(gaussians.get_normal), 0, 1)
        N = N*2-1
        N = normalize(N, dim=2)


        # dir_pp = (viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1) - gaussians.get_xyz).unsqueeze(1)

        dir_pp  = viewpoint_cam.world_view_transform[:3, 2]*-1

        dir_pp = normalize(dir_pp, dim=0)

        

        cos_sim = torch.sum(N*dir_pp, axis=2).squeeze()

        dir_pp_norm = cos_sim

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # render
        # met_rough_occ
        met_rough_occ_render_pkg = render_combined(viewpoint_cam, gaussians, pipe, bg, data_type = 'met_rough_occ')
        met_rough_occ_image = met_rough_occ_render_pkg["render"]

        if iteration %100 == 0:
            #print(iteration)
            #print(viewpoint_cam.world_view_transform[:3, 2]*-1)
            gt = viewpoint_cam.mro_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(benchmark_log_folder, '{0:03d}'.format(view_num) + "_gt.png"))
            torchvision.utils.save_image(met_rough_occ_image, os.path.join(benchmark_log_folder, '{0:03d}'.format(view_num) + ".png"))

        visibility_filter = met_rough_occ_render_pkg["visibility_filter"]

        # Loss
        met_rough_occ_gt_image = viewpoint_cam.mro_image.cuda()
        Ll1 = l1_loss(met_rough_occ_image, met_rough_occ_gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(met_rough_occ_image, met_rough_occ_gt_image))


        loss.backward()

        iter_end.record()

        with torch.no_grad():

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            
            # Optimizer step
            
            #gaussians.get_xyz.grad = None
            #gaussians.get_opacity_parameter.grad = None
            #gaussians.get_scaling_parameter.grad = None
            #gaussians.get_rotation_parameter.grad = None
            gaussians.get_features.grad = None
            gaussians.get_bc.grad = None
            gaussians.get_normal.grad = None
            gaussians.get_depth_global_scale.grad = None

            #gaussians.get_mro.grad[(~visibility_filter)|(distance_gaussians<dir_pp_norm)] = 0


            gaussians.get_mro.grad[(~visibility_filter)|((distance_gaussians-epsilon)>dir_pp_norm)] = 0



            #print(gaussians.get_mro[(~visibility_filter)|(distance_gaussians<dir_pp_norm)].mean())
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            #print(gaussians.get_mro[(~visibility_filter)|(distance_gaussians<dir_pp_norm)].mean())

            distance_gaussians[visibility_filter] = torch.maximum(distance_gaussians[visibility_filter], dir_pp_norm[visibility_filter])
        
        
        #torchvision.utils.save_image(met_rough_occ_image, os.path.join(benchmark_log_folder, '{0:03d}'.format(view_num) + "_after.png"))
        

                
    iteration = iteration_per_view
    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    scene.save(iteration)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
