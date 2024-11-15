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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 projection_matrix = None, bc_image = None, mro_image = None, normal_image=None,
                 camera_position=None, depth_image = None, K=None, bc_image_gt = None, mro_image_gt = None, normal_image_gt=None,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.K = K
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        if camera_position is not None:
            self.camera_position = camera_position

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        
        self.original_image = image.to(self.data_device) # there were .clamp(0.0, 1.0) for each image
        if bc_image is not None:
            self.bc_image = bc_image.to(self.data_device)
            self.bc_image_gt = bc_image_gt.to(self.data_device)
            #self.bc_image_mask = nn.Parameter(torch.ones_like(bc_image, dtype=torch.float32).to(self.data_device).requires_grad_(True))
            self.bc_image_mask = nn.Parameter(torch.zeros_like(bc_image, dtype=torch.float32).to(self.data_device).requires_grad_(True))
        if mro_image is not None:
            self.mro_image = mro_image.to(self.data_device)
            self.mro_image_gt = mro_image_gt.to(self.data_device)
            #self.mro_image_mask = nn.Parameter(torch.ones_like(mro_image, dtype=torch.float32).to(self.data_device).requires_grad_(True))
            self.mro_image_mask = nn.Parameter(torch.zeros_like(mro_image, dtype=torch.float32).to(self.data_device).requires_grad_(True))
        if normal_image is not None:
            self.normal_image = normal_image.to(self.data_device)
            self.normal_image_gt = normal_image_gt.to(self.data_device)
            #self.normal_image_mask = nn.Parameter(torch.ones_like(normal_image, dtype=torch.float32).to(self.data_device).requires_grad_(True))
            self.normal_image_mask = nn.Parameter(torch.zeros_like(normal_image, dtype=torch.float32).to(self.data_device).requires_grad_(True))
        if depth_image is not None:
            self.depth_image = depth_image.to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()

        if projection_matrix is not None:
            # Correct z axis orientation from OpenGL to COLMAP
            projection_matrix[3, 2] = 1
            self.projection_matrix = torch.from_numpy(projection_matrix).transpose(0,1).cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()

        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

