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


data_types = ['render', 'base_color', 'met_rough_occ', 'normal']
data_type_folder_names = ['render', 'base_color', 'met_rough_occ', 'normal']
suffix = 'samples_new_120'

makedirs('videos', exist_ok=True)
    
for data_type_idx, data_type in enumerate(data_types):
    #data_type = 'shading'
    #data_type_folder_name = 'shading_light_1_radiance_500_radius_inf_white'
    data_type_folder_name = data_type_folder_names[data_type_idx]

    render_path = os.path.join("output/7b6477cb95_colmap_try", "train", "ours_{}".format(30000), "renders", f"{data_type_folder_name}_video")

    makedirs(render_path, exist_ok=True)

    video_name = os.path.join('videos', f'{data_type_folder_name}_{suffix}_video.avi')

    images = [img for img in sorted(os.listdir(render_path)) if img.endswith(".png")][:1000]
    frame = cv2.imread(os.path.join(render_path, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), 25, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(render_path, image)))

    cv2.destroyAllWindows()
    video.release()