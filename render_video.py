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
from scene.colmap_loader import qvec2rotmat
import cv2


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

    #data_types = ['render', 'base_color', 'met_rough_occ', 'normal', 'shading']
    #data_type_folder_names = ['render', 'base_color', 'met_rough_occ', 'normal', 'shading_light_200_radiance_30_radius_2']
    data_types = ['render', 'base_color', 'met_rough_occ', 'normal']
    data_type_folder_names = ['render', 'base_color', 'met_rough_occ', 'normal']
    suffix = 'samples_new_120'

    makedirs('videos', exist_ok=True)
    
    for data_type_idx, data_type in enumerate(data_types):
        #data_type = 'shading'
        #data_type_folder_name = 'shading_light_1_radiance_500_radius_inf_white'
        data_type_folder_name = data_type_folder_names[data_type_idx]

        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders", f"{data_type_folder_name}_video")

        makedirs(render_path, exist_ok=True)

        idx = 0
        for i in tqdm(range(len(views)-1), desc=f'Rendering progress for {data_type}'):
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

                rendering = render_combined(view3, gaussians, pipeline, background, data_type = data_type)["render"]
                torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                idx += 1

                #break
                
            #break
        
        video_name = os.path.join('videos', f'{data_type_folder_name}_{suffix}_video.avi')

        images = [img for img in sorted(os.listdir(render_path)) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(render_path, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), 10, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(render_path, image)))

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