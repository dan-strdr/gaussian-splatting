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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from math import sin, cos, tan, pi
import torch
import cv2

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    resized_bc_image_rgb = PILtoTorch(cam_info.bc_image, resolution)
    bc_image = resized_bc_image_rgb[:3, ...]

    resized_mro_image_rgb = PILtoTorch(cam_info.mro_image, resolution)
    mro_image = resized_mro_image_rgb[:3, ...]

    resized_normal_image_rgb = PILtoTorch(cam_info.normal_image, resolution)
    normal_image = resized_normal_image_rgb[:3, ...]


    resized_bc_image_rgb = PILtoTorch(cam_info.bc_image_gt, resolution)
    bc_image_gt = resized_bc_image_rgb[:3, ...]

    resized_mro_image_rgb = PILtoTorch(cam_info.mro_image_gt, resolution)
    mro_image_gt = resized_mro_image_rgb[:3, ...]

    resized_normal_image_rgb = PILtoTorch(cam_info.normal_image_gt, resolution)
    normal_image_gt = resized_normal_image_rgb[:3, ...]

    if cam_info.depth_image is not None:
        resized_depth_image_rgb = PILtoTorch(cam_info.depth_image, resolution)
        depth_image = resized_depth_image_rgb[:3, ...]
    else:
        depth_image = None
        
    """
    if cam_info.depth_image is not None:
        depth_image = torch.from_numpy(cv2.resize(cam_info.depth_image, resolution, interpolation = cv2.INTER_AREA).transpose(2, 0, 1))
    else:
        depth_image = None
    """
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                image=gt_image, gt_alpha_mask=loaded_mask,
                image_name=cam_info.image_name, uid=id, data_device="cpu",# data_device=args.data_device,
                projection_matrix=cam_info.projection_matrix, bc_image=bc_image, 
                mro_image=mro_image, normal_image=normal_image, camera_position=cam_info.camera_position,
                depth_image=depth_image, K=cam_info.K, bc_image_gt=bc_image_gt, 
                mro_image_gt=mro_image_gt, normal_image_gt=normal_image_gt)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        print(id)
        camera_list.append(loadCam(args, id, c, resolution_scale))
        if id==20:
            break

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
