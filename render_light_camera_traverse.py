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
from scene.colmap_loader import qvec2rotmat
from scene.cameras import Camera

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):

    makedirs('videos', exist_ok=True)

    light_traverse_folder_name = 'camera_traverse_white'
    suffix = 'diffusion_regularized'

    frame_number = 20

    
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
    coordinates = np.array([[-6, 3, 12],
                        [-4.0, 3, 3.5],
                        [-4.0, 5.0, 3.5],
                        [-2, 3.7, 4],
                        [-2.5, 3.7, 9.5],
                        [1, 3.7, 10.3],
                        [2.5, 2.7, 6.5],
                        [3, 2.0, 8],], dtype=np.float32)
    """
    """
    coordinates = np.array([[-9, 1, 12],
                        [5, 1, 8],
                        [-4.0, 1, 2],
                        [-2, 1, 4],], dtype=np.float32)
    """

    """
    coordinates = np.array([[2.5, 0.7, 1.6],
                        [2.5, 0.7, 0.5],
                        [2.5, 1.2, 0.5],
                        [3.6, 1.2, 0.5],
                        [3.6, 1.2, 1.6],
                        [3.6, 0.7, 1.6]], dtype=np.float32)
    
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

    light_traverse_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders", light_traverse_folder_name)

    makedirs(light_traverse_render_path, exist_ok=True)

    #view = views[18] #99

    light_pos = torch.from_numpy(light_coordinates[72]).unsqueeze(0).to('cuda')

    #for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

    idx = 0
    for i in tqdm(range(15, 35)):
        view1 = views[i]
        view2 = views[i+1]
        for alpha in np.arange(0, 1.01, 0.03):

            T = (1-alpha)*view1.T + alpha*view2.T
            #print(os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            #print("T", T)
            #R = (1-alpha)*view1.R + alpha*view2.R

            #print("view1.R", view1.R)
            #print("view2.R", view2.R)

            q1 = rotmat2qvec(view1.R)
            q2 = rotmat2qvec(view2.R)

            if np.abs(q1-q2).sum()>np.abs(-q1-q2).sum():
                q1 *= -1

            #print("q1", q1)
            #print("q2", q2)

            theta = np.arccos(np.dot(q1, q2))
            #print("theta", theta)

            q3 = (np.sin((1-alpha)*theta)/np.sin(theta))*q1 + (np.sin(alpha*theta)/np.sin(theta))*q2
            R = qvec2rotmat(q3)
            #print("R", R)

            view3 = Camera(0, R, T, view1.FoVx, view1.FoVy, view1.original_image, gt_alpha_mask=None,
                        image_name=view1.image_name, uid=view1.uid,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda")
            
            view3.projection_matrix = view1.projection_matrix.detach().clone()
            view3.full_proj_transform = (view3.world_view_transform.unsqueeze(0).bmm(view3.projection_matrix.unsqueeze(0))).squeeze(0)

            rendering = render_combined(view3, gaussians, pipeline, background, data_type = 'shading', light_pos = light_pos, light_color = light_color)["render"]
            torchvision.utils.save_image(rendering, os.path.join(light_traverse_render_path, '{0:05d}'.format(idx) + ".png"))
            idx += 1

        #break
    video_name = os.path.join('videos', f'{light_traverse_folder_name}_{suffix}_video.avi')

    images = [img for img in sorted(os.listdir(light_traverse_render_path)) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(light_traverse_render_path, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), 25, (width,height))

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