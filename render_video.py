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
from scene.cameras import Camera
import numpy as np


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):

    data_type = 'render'

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders", f"{data_type}_video")

    makedirs(render_path, exist_ok=True)

    idx = 0
    for i in tqdm(range(len(views)-1), desc='Rendering progress'):
        view1 = views[i]
        view2 = views[i+1]
        for alpha in np.arange(0, 1.01, 0.03):

            T = (1-alpha)*view1.T + alpha*view2.T
            R = (1-alpha)*view1.R + alpha*view2.R

            view3 = Camera(0, R, T, view1.FoVx, view1.FoVy, view1.original_image, gt_alpha_mask=None,
                        image_name=view1.image_name, uid=view1.uid,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda")

            rendering = render_combined(view3, gaussians, pipeline, background, data_type = data_type)["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            idx += 1


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

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