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
from random import randint
from utils.loss_utils import l1_loss, ssim

#torch.autograd.set_detect_anomaly(True)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, scene):

    output_path = 'lighting_results'
    os.makedirs(output_path, exist_ok=True)

    viewpoint_stack = None
    nof_lights = 50

    #light_pos_tensor = torch.tensor([1, 0.8, 0.6], dtype=torch.float32).to('cuda')
    light_pos_tensor = (torch.rand(nof_lights, 3, dtype=torch.float32)*torch.tensor([20.0, 8.0, 18.0])+torch.tensor([-12.0, -2.0, -1.0])).to('cuda')

    light_pos = torch.nn.Parameter(light_pos_tensor.requires_grad_(True))

    #light_color_tensor = torch.tensor([1, 0.8, 0.6], dtype=torch.float32).to('cuda')
    light_color_tensor = torch.rand(nof_lights, 3, dtype=torch.float32).to('cuda')

    light_color = torch.nn.Parameter(light_color_tensor.requires_grad_(True))

    optimizer = torch.optim.Adam([light_pos, light_color], lr=0.01, eps=1e-15)

    for iteration in range(1000): 
        if iteration == 500:
            optimizer.param_groups[0]['lr'] = 0.002

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        render = render_combined(viewpoint_cam, gaussians, pipeline, background, data_type = 'render')["render"]

        shading = render_combined(viewpoint_cam, gaussians, pipeline, background, data_type = 'shading', light_pos = light_pos, light_color = light_color, lighting_optimization = True)["render"]

        #if iteration == 0: # iteration%10==0:
        #    torchvision.utils.save_image(shading, os.path.join(output_path, '{0:05d}'.format(iteration) + ".png"))

        Ll1 = l1_loss(shading, render)
        loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim(shading, render))

        loss *= 100

        print('loss:', loss.item())

        optimizer.zero_grad()

        loss.backward()

        #print('light_pos.grad', light_pos.grad)

        optimizer.step()

        if viewpoint_cam.image_name== 'sample_0': # iteration%10==0:
            torchvision.utils.save_image(shading, os.path.join(output_path, '{0:05d}'.format(iteration) + ".png"))


        #print('light_pos:', light_pos.data)
        #print('light_color:', light_color.data)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, scene)


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