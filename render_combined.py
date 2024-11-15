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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_combined
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from time import time

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):

    shading_folder_name = 'shading_light_200_radiance_30_radius_2_try'

    render_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders", "render")
    render_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt", "render")

    met_rough_occ_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders", "met_rough_occ")
    met_rough_occ_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt", "met_rough_occ")

    base_color_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders", "base_color")
    base_color_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt", "base_color")

    normal_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders", "normal")
    normal_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt", "normal")

    position_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders", "position")

    shading_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders", shading_folder_name)
    shading_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt", "shading")

    makedirs(render_render_path, exist_ok=True)
    makedirs(render_gts_path, exist_ok=True)

    makedirs(met_rough_occ_render_path, exist_ok=True)
    makedirs(met_rough_occ_gts_path, exist_ok=True)

    makedirs(base_color_render_path, exist_ok=True)
    makedirs(base_color_gts_path, exist_ok=True)

    makedirs(normal_render_path, exist_ok=True)
    makedirs(normal_gts_path, exist_ok=True)

    makedirs(position_render_path, exist_ok=True)

    makedirs(shading_render_path, exist_ok=True)
    makedirs(shading_gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        t1 = time()

        rendering = render_combined(view, gaussians, pipeline, background, data_type = 'render')["render"]
        gt = view.original_image[0:3, :, :].cuda().float()
        torchvision.utils.save_image(rendering, os.path.join(render_render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(render_gts_path, '{0:05d}'.format(idx) + ".png"))

        t2 = time()

        rendering = render_combined(view, gaussians, pipeline, background, data_type = 'met_rough_occ')["render"]
        gt = view.mro_image[0:3, :, :].cuda().float()
        torchvision.utils.save_image(rendering, os.path.join(met_rough_occ_render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(met_rough_occ_gts_path, '{0:05d}'.format(idx) + ".png"))

        t3 = time()

        rendering = render_combined(view, gaussians, pipeline, background, data_type = 'base_color')["render"]
        gt = view.bc_image[0:3, :, :].cuda().float()
        torchvision.utils.save_image(rendering, os.path.join(base_color_render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(base_color_gts_path, '{0:05d}'.format(idx) + ".png"))

        t4 = time()

        rendering = render_combined(view, gaussians, pipeline, background, data_type = 'normal')["render"]
        gt = view.normal_image[0:3, :, :].cuda().float()
        torchvision.utils.save_image(rendering, os.path.join(normal_render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(normal_gts_path, '{0:05d}'.format(idx) + ".png"))

        t4 = time()

        rendering = render_combined(view, gaussians, pipeline, background, data_type = 'depth')["render"]
        rendering = (rendering-rendering.min())/(rendering.max()-rendering.min())
        #print(rendering.shape)
        #print(rendering[0].min(), rendering[0].max(), rendering[0].mean())
        #print(rendering[1].min(), rendering[1].max(), rendering[1].mean())
        #print(rendering[2].min(), rendering[2].max(), rendering[2].mean())
        torchvision.utils.save_image(rendering, os.path.join(position_render_path, '{0:05d}'.format(idx) + ".png"))

        t5 = time()

        rendering = render_combined(view, gaussians, pipeline, background, data_type = 'shading')["render"]
        gt = view.original_image[0:3, :, :].cuda().float()
        torchvision.utils.save_image(rendering, os.path.join(shading_render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(shading_gts_path, '{0:05d}'.format(idx) + ".png"))

        t6 = time()

        #print('render time:', t2-t1)
        #print('met_rough_occ time:', t3-t2)
        #print('base_color time:', t4-t3)
        #print('normal time:', t5-t4)
        #print('shading time:', t6-t5)

        #break

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)