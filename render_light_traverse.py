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
import numpy as np
import cv2

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):

    makedirs('videos', exist_ok=True)

    light_traverse_folder_name = 'light_traverse_white'
    suffix = 'diffusion_regularized'

    frame_number = 20

    """
    coordinates = np.array([[-5.5, 5.7, 3.5],
                        [-5.5, -1.0, 3.5],
                        [-8, -1.0, 12],
                        [-8, 5.7, 12],
                        [-4.5, 5.0, 9.5],
                        [-4.5, -0.5, 9.5],
                        [2, -0.5, 14],
                        [2, 3.0, 14],
                        [1.0, 4.0, 11],
                        [2.6, 0.0, 10.5],
                        [2.6, 2.0, 10.5]], dtype=np.float32)
    """

    """
    coordinates = np.array([[0, 1.25, 0],
                        [0, 1.25, 0.9],
                        [0, 0.4, 0.9],
                        [0.8, 0.4, 0.9],
                        [0.8, 0.4, 0],
                        [0, 0.4, 0],
                        [0, 0.4, 0.9]], dtype=np.float32)
    """
    coordinates = np.array([[2.5, 0.6, 1.6],
                        [2.5, 0.6, 0.4],
                        [2.5, 1.2, 0.4],
                        [3.6, 1.2, 0.4],
                        [3.6, 1.2, 1.6],
                        [3.6, 0.6, 1.6]], dtype=np.float32)

    light_coordinates = []

    for coordinate_id in range(len(coordinates)-1):
        for alpha in np.linspace(0, 1, frame_number):
            light_coordinates.append((1-alpha)*coordinates[coordinate_id]+ (alpha)*coordinates[coordinate_id+1])

    light_color = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32).to('cuda')

    light_traverse_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders", light_traverse_folder_name)

    makedirs(light_traverse_render_path, exist_ok=True)

    view = views[26]

    #for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

    for light_id, light_coordinate in tqdm(enumerate(light_coordinates), desc="Rendering progress"):
        light_pos = torch.from_numpy(light_coordinates[light_id]).unsqueeze(0).to('cuda')

        rendering = render_combined(view, gaussians, pipeline, background, data_type = 'shading', light_pos = light_pos, light_color = light_color)["render"]
        torchvision.utils.save_image(rendering, os.path.join(light_traverse_render_path, '{0:05d}'.format(light_id) + ".png"))

        #break
    video_name = os.path.join('videos', f'{light_traverse_folder_name}_{suffix}_video.avi')

    images = [img for img in sorted(os.listdir(light_traverse_render_path)) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(light_traverse_render_path, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), 10, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(light_traverse_render_path, image)))

    cv2.destroyAllWindows()
    video.release()

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