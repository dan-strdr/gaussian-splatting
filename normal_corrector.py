from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
import os
import numpy as np
from PIL import Image

path = '/home/meric/umut/gaussian-splatting/data/scannet++_colmap'
cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")

cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)

output_path = 'normal_corrected'

os.makedirs(output_path, exist_ok=True)

for idx, key in enumerate(cam_extrinsics):
    extr = cam_extrinsics[key]

    R = np.transpose(qvec2rotmat(extr.qvec))
    T = np.array(extr.tvec)

    image_path = os.path.join(path, 'images', os.path.basename(extr.name))
    image_name = os.path.basename(image_path).split(".")[0]
    image = Image.open(image_path)

    normal_image_path = os.path.join(path, 'normal', os.path.basename(extr.name))
    normal_image = Image.open(normal_image_path)
    image_shape = np.array(normal_image).shape
    
    normal_image = np.array(normal_image, dtype=np.float32)/255
    normal_image = normal_image*2-1

    normal_image[:,:,1:3] *= -1

    normal_image = np.transpose(np.reshape(normal_image, (-1 ,3)))

    normal_image = np.matmul(R, normal_image)

    normal_image = np.transpose(normal_image)

    normal_image = np.reshape(normal_image, image_shape)

    normal_image = (normal_image+1)/2
    normal_image = np.clip(normal_image, 0, 1)

    normal_image = Image.fromarray(np.uint8(normal_image*255))

    normal_image.save(os.path.join(output_path, extr.name))

