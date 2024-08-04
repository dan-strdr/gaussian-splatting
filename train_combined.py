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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    image_log_folder = 'image_logs'
    os.makedirs(image_log_folder, exist_ok=True)

    met_rough_occ_folder = 'met_rough_occ'
    os.makedirs(os.path.join(image_log_folder, met_rough_occ_folder), exist_ok=True)

    base_color_folder = 'base_color'
    os.makedirs(os.path.join(image_log_folder, base_color_folder), exist_ok=True)

    normal_folder = 'normal'
    os.makedirs(os.path.join(image_log_folder, normal_folder), exist_ok=True)

    mask_coef = 60/255
    regularization_start_iteration = 30100 # 1500

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    #scene.gaussians.load_ply_original("/home/meric/umut/gaussian-splatting/data/sugarfine_3Dgs7000_sdfestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.ply")
    #gaussians = scene.gaussians
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):     
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # render
        render_pkg = render_combined(viewpoint_cam, gaussians, pipe, bg, data_type = 'render')
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        if iteration >= regularization_start_iteration:

            # met_rough_occ
            met_rough_occ_render_pkg = render_combined(viewpoint_cam, gaussians, pipe, bg, data_type = 'met_rough_occ')
            met_rough_occ_image = met_rough_occ_render_pkg["render"]

            # Loss
            met_rough_occ_gt_image = viewpoint_cam.mro_image.cuda()
            met_rough_occ_gt_image_mask = viewpoint_cam.mro_image_mask.cuda()
            met_rough_occ_gt_image_mask = torch.clip(met_rough_occ_gt_image_mask, 0, 1)
            Ll1 = l1_loss(met_rough_occ_image*met_rough_occ_gt_image_mask, met_rough_occ_gt_image*met_rough_occ_gt_image_mask)
            Lmask = ((met_rough_occ_gt_image_mask*-1)+1).mean()
            loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(met_rough_occ_image, met_rough_occ_gt_image)) + (1.0 - opt.lambda_dssim) * Lmask * mask_coef

            #print('Lmask metroughocc', Lmask.item())
            #print('met_rough_occ_gt_image_mask.mean()', met_rough_occ_gt_image_mask.mean().item())

            # base_color
            base_color_render_pkg = render_combined(viewpoint_cam, gaussians, pipe, bg, data_type = 'base_color')
            base_color_image = base_color_render_pkg["render"]

            # Loss
            base_color_gt_image = viewpoint_cam.bc_image.cuda()
            base_color_gt_image_mask = viewpoint_cam.bc_image_mask.cuda()
            base_color_gt_image_mask = torch.clip(base_color_gt_image_mask, 0, 1)
            Ll1 = l1_loss(base_color_image*base_color_gt_image_mask, base_color_gt_image*base_color_gt_image_mask)
            Lmask = ((base_color_gt_image_mask*-1)+1).mean()
            loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(base_color_image, base_color_gt_image)) + (1.0 - opt.lambda_dssim) * Lmask * mask_coef

            # normal
            normal_render_pkg = render_combined(viewpoint_cam, gaussians, pipe, bg, data_type = 'normal')
            normal_image = normal_render_pkg["render"]

            # Loss
            normal_gt_image = viewpoint_cam.normal_image.cuda()
            normal_gt_image_mask = viewpoint_cam.normal_image_mask.cuda()
            normal_gt_image_mask = torch.clip(normal_gt_image_mask, 0, 1)
            Ll1 = l1_loss(normal_image*normal_gt_image_mask, normal_gt_image*normal_gt_image_mask)
            Lmask = ((normal_gt_image_mask*-1)+1).mean()
            loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(normal_image, normal_gt_image)) + (1.0 - opt.lambda_dssim) * Lmask * mask_coef


            if iteration%100==0:
                saved_img = Image.fromarray((torch.transpose(torch.transpose(met_rough_occ_gt_image_mask, 0, 2), 0, 1).detach().cpu().numpy()*255).astype(np.uint8))
                saved_img.save(os.path.join(image_log_folder, met_rough_occ_folder, f'met_rough_occ_mask_{iteration:05d}_{viewpoint_cam.image_name}.png'))

                saved_img = Image.fromarray((torch.transpose(torch.transpose(base_color_gt_image_mask, 0, 2), 0, 1).detach().cpu().numpy()*255).astype(np.uint8))
                saved_img.save(os.path.join(image_log_folder, base_color_folder, f'base_color_mask_{iteration:05d}_{viewpoint_cam.image_name}.png'))

                saved_img = Image.fromarray((torch.transpose(torch.transpose(normal_gt_image_mask, 0, 2), 0, 1).detach().cpu().numpy()*255).astype(np.uint8))
                saved_img.save(os.path.join(image_log_folder, normal_folder, f'normal_mask_{iteration:05d}_{viewpoint_cam.image_name}.png'))
        else:
            
            # met_rough_occ
            met_rough_occ_render_pkg = render_combined(viewpoint_cam, gaussians, pipe, bg, data_type = 'met_rough_occ')
            met_rough_occ_image = met_rough_occ_render_pkg["render"]

            # Loss
            met_rough_occ_gt_image = viewpoint_cam.mro_image.cuda()
            Ll1 = l1_loss(met_rough_occ_image, met_rough_occ_gt_image)
            loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(met_rough_occ_image, met_rough_occ_gt_image))

            # base_color
            base_color_render_pkg = render_combined(viewpoint_cam, gaussians, pipe, bg, data_type = 'base_color')
            base_color_image = base_color_render_pkg["render"]

            # Loss
            base_color_gt_image = viewpoint_cam.bc_image.cuda()
            Ll1 = l1_loss(base_color_image, base_color_gt_image)
            loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(base_color_image, base_color_gt_image))

            # normal
            normal_render_pkg = render_combined(viewpoint_cam, gaussians, pipe, bg, data_type = 'normal')
            normal_image = normal_render_pkg["render"]

            # Loss
            normal_gt_image = viewpoint_cam.normal_image.cuda()
            Ll1 = l1_loss(normal_image, normal_gt_image)
            loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(normal_image, normal_gt_image))

            if iteration >= 3000:

                # position
                position_render_pkg = render_combined(viewpoint_cam, gaussians, pipe, bg, data_type = 'position')
                position_image = position_render_pkg["render"]

                # Loss
                depth_gt_image = viewpoint_cam.depth_image.cuda()
                #print("gaussians.get_depth_global_scale", gaussians.get_depth_global_scale)
                #Ll1 = l1_loss(position_image, depth_gt_image*gaussians.get_depth_global_scale)
                #print("inf sum", torch.isfinite(depth_gt_image).sum())
                Ll1 = l1_loss(position_image[torch.isfinite(depth_gt_image)], depth_gt_image[torch.isfinite(depth_gt_image)]*gaussians.get_depth_global_scale)
                loss += (1.0 - opt.lambda_dssim) * Ll1 * 0.5

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_combined, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            
            # Optimizer step
            if iteration < opt.iterations:
                #gaussians.get_xyz.grad = None
                #gaussians.get_opacity.grad = None
                #gaussians.get_scaling.grad = None
                #gaussians.get_rotation.grad = None

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                if iteration >= regularization_start_iteration:
                    scene.optimizer.step()
                    scene.optimizer.zero_grad(set_to_none = True)


            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {} render: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                
                met_rough_occ_l1_test = 0.0
                met_rough_occ_psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    met_rough_occ_image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, data_type = 'met_rough_occ')["render"], 0.0, 1.0)
                    met_rough_occ_gt_image = torch.clamp(viewpoint.mro_image.to("cuda"), 0.0, 1.0)
                    
                    met_rough_occ_l1_test += l1_loss(met_rough_occ_image, met_rough_occ_gt_image).mean().double()
                    met_rough_occ_psnr_test += psnr(met_rough_occ_image, met_rough_occ_gt_image).mean().double()
                met_rough_occ_psnr_test /= len(config['cameras'])
                met_rough_occ_l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {} met_rough_occ: L1 {} PSNR {}".format(iteration, config['name'], met_rough_occ_l1_test, met_rough_occ_psnr_test))


                base_color_l1_test = 0.0
                base_color_psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    base_color_image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, data_type = 'base_color')["render"], 0.0, 1.0)
                    base_color_gt_image = torch.clamp(viewpoint.bc_image.to("cuda"), 0.0, 1.0)
                    
                    base_color_l1_test += l1_loss(base_color_image, base_color_gt_image).mean().double()
                    base_color_psnr_test += psnr(base_color_image, base_color_gt_image).mean().double()
                base_color_psnr_test /= len(config['cameras'])
                base_color_l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {} base_color: L1 {} PSNR {}".format(iteration, config['name'], base_color_l1_test, base_color_psnr_test))


                normal_l1_test = 0.0
                normal_psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    normal_image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, data_type = 'normal')["render"], 0.0, 1.0)
                    normal_gt_image = torch.clamp(viewpoint.normal_image.to("cuda"), 0.0, 1.0)
                    
                    normal_l1_test += l1_loss(normal_image, normal_gt_image).mean().double()
                    normal_psnr_test += psnr(normal_image, normal_gt_image).mean().double()
                normal_psnr_test /= len(config['cameras'])
                normal_l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {} normal: L1 {} PSNR {}".format(iteration, config['name'], normal_l1_test, normal_psnr_test))

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

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
