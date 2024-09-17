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
from utils.shading_utils import deferred_shade
import numpy as np
import cv2
import torchvision.transforms as T

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):

    shading_folder_name = 'deferred_shading'

    shading_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders", shading_folder_name)

    makedirs(shading_render_path, exist_ok=True)

    makedirs('videos', exist_ok=True)

    light_traverse_folder_name = 'light_traverse_white_deferred_shading'
    suffix = 'diffusion_regularized'

    frame_number = 20
    """
    coordinates = np.array([[2.5, 0.6, 1.6],
                        [2.5, 0.6, 0.4],
                        [2.5, 1.2, 0.4],
                        [3.6, 1.2, 0.4],
                        [3.6, 1.2, 1.6],
                        [3.6, 0.6, 1.6]], dtype=np.float32)
    """
    """
    coordinates = np.array([[2.5, 0.7, 1.6],
                        [2.5, 0.7, 0.5],
                        [2.5, 1.2, 0.5],
                        [3.6, 1.2, 0.5],
                        [3.6, 1.2, 1.6],
                        [3.6, 0.7, 1.6]], dtype=np.float32)
    """
    
    coordinates = np.array([[-1, 3.5, 8.5],
                        [-4.5, 2.0, 3.5],
                        [-7.5, 2.0, 12],
                        [-7.5, 5.7, 12],
                        [-4.5, 5.0, 9.5],
                        [-4.5, -5.5, 9.5],
                        [2, 1.5, 14],
                        [2, 3.0, 14],
                        [1.0, 4.0, 11],
                        [2.6, 1.0, 10.5],
                        [2.6, 2.0, 10.5]], dtype=np.float32)

    coordinates = np.array([[-6, 3, 12],
                        [-4.0, 3, 3.5],
                        [-4.0, 5.0, 3.5],
                        [-2, 3.7, 4],
                        [-2.5, 3.7, 9.5],
                        [1, 3.7, 10.3],
                        [2.5, 2.7, 6.5],
                        [3, 2.0, 8],], dtype=np.float32)
    
    """
    coordinates = np.array([[-9, 1, 12],
                        [5, 1, 8],
                        [-4.0, 1, 2],
                        [-2, 1, 4],], dtype=np.float32)
    """

    coordinates = np.array([[2.5, 0.6, 1.6],
                        [2.5, 0.6, 0.4],
                        [2.5, 1.2, 0.4],
                        [3.6, 1.2, 0.4],
                        [3.6, 1.2, 1.6],
                        [3.6, 0.6, 1.6]], dtype=np.float32)
    
    """
    coordinates = np.array([[-6, 3, 12],
                        [-4.0, 3, 3.5],
                        [-4.0, 5.0, 3.5],
                        [-2, 3.7, 4],
                        [-2.5, 3.7, 9.5],
                        [1, 3.7, 10.3],
                        [2.5, 2.7, 6.5],
                        [3, 2.0, 8],], dtype=np.float32)
    """
    light_coordinates = []

    for coordinate_id in range(len(coordinates)-1):
        for alpha in np.linspace(0, 1, frame_number):
            light_coordinates.append((1-alpha)*coordinates[coordinate_id]+ (alpha)*coordinates[coordinate_id+1])

    light_color = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32).to('cuda')

    #view = views[99]
    #view = views[18]
    view = views[18]

    met_rough_occ_image = render_combined(view, gaussians, pipeline, background, data_type = 'met_rough_occ')["render"]
    #met_rough_occ_image = view.mro_image.cuda()

    t3 = time()

    base_color_image = render_combined(view, gaussians, pipeline, background, data_type = 'base_color')["render"]

    t4 = time()

    normal_image = render_combined(view, gaussians, pipeline, background, data_type = 'normal')["render"]

    #transform = T.GaussianBlur(kernel_size=(15, 15), sigma=(50, 50))

    #std_filter = torch.ones(3, 3)/9
    #std_filter = std_filter.float()
    
    #std_filter = std_filter.repeat(3, 3, 1, 1).cuda()
    #print(normal_image.shape)

    #normal_image = torch.nn.functional.conv2d(normal_image.unsqueeze(0), std_filter, padding=1)
    #normal_image = torch.nn.functional.conv2d(normal_image, std_filter, padding=1)
    #normal_image = torch.nn.functional.conv2d(normal_image, std_filter, padding=1)
    #normal_image = torch.nn.functional.conv2d(normal_image, std_filter, padding=1)

    #normal_image = normal_image.squeeze(0)

    #print(normal_image.squeeze(0).shape)

    #normal_image = transform(normal_image)

    t4 = time()

    position_image = render_combined(view, gaussians, pipeline, background, data_type = 'position')["render"]
    #position_image = transform(position_image)
    position_image = position_image*(gaussians.get_xyz.max()-gaussians.get_xyz.min()) + gaussians.get_xyz.min()
    #position_image = (position_image-position_image.min())/(position_image.max()-position_image.min())
    #torchvision.utils.save_image(position_image, "try.png")

    for light_id, light_coordinate in tqdm(enumerate(light_coordinates), desc="Rendering progress"):
        light_pos = torch.from_numpy(light_coordinates[light_id]).unsqueeze(0).to('cuda')

        light_rendering = deferred_shade(view, gaussians, met_rough_occ_image, base_color_image, normal_image, position_image, light_pos = light_pos, light_color = light_color)
        #print(light_rendering.shape)

        #light_rendering = transform(light_rendering)

        #light_rendering = torch.nn.functional.conv2d(light_rendering.unsqueeze(0), std_filter, padding=1)
        #light_rendering = light_rendering.squeeze(0)
        #print(light_rendering.shape)
        #print("light_rendering.min(), light_rendering.max()", light_rendering.min(), light_rendering.max())

        torchvision.utils.save_image(light_rendering, os.path.join(shading_render_path, '{0:05d}'.format(light_id) + ".png"))

        #break
    video_name = os.path.join('videos', f'{light_traverse_folder_name}_{suffix}_video.avi')

    images = [img for img in sorted(os.listdir(shading_render_path)) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(shading_render_path, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), 10, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(shading_render_path, image)))

    cv2.destroyAllWindows()
    video.release()

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

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