from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.image

path = '/home/koca/projects/gaussian-splatting/data/L3D124S21ENDIDR4BOIUI5NYALUF3P3XA888_samples'
#cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")

#cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)

cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")

cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)

output_path = 'normal_corrected6'

os.makedirs(output_path, exist_ok=True)

for idx, key in enumerate(cam_extrinsics):
    extr = cam_extrinsics[key]

    R = np.transpose(qvec2rotmat(extr.qvec))
    T = np.array(extr.tvec)

    image_path = os.path.join(path, 'images', os.path.basename(extr.name))
    image_name = os.path.basename(image_path).split(".")[0]
    image = Image.open(image_path)

    normal_image_path = os.path.join(path, 'normal_wrong_gt', os.path.basename(extr.name))
    normal_image = Image.open(normal_image_path)
    image_shape = np.array(normal_image).shape
    
    normal_image = np.array(normal_image, dtype=np.float64)/255
    normal_image = normal_image*2-1

    normal_image[:,:,1:3] *= -1

    normal_image = np.transpose(np.reshape(normal_image, (-1 ,3)))

    normal_image = normal_image / np.linalg.norm(normal_image, axis=0, keepdims=True)

    #R = np.array([[1, 0, 0], [0, 1, 0],[0, 0, 1]], dtype = np.float64)

    normal_image = np.matmul(R, normal_image)

    normal_image = np.transpose(normal_image)

    normal_image = normal_image / np.linalg.norm(normal_image, axis=1, keepdims=True)

    normal_image = np.reshape(normal_image, image_shape)

    normal_image = (normal_image+1)/2
    normal_image = np.clip(normal_image, 0, 1)

    normal_image = Image.fromarray(np.uint8(np.round(normal_image*255)))

    normal_image.save(os.path.join(output_path, extr.name))

    #print(extr.name)
    #normal_image.save("try.jpg", quality=95)
    #normal_image.save("try.png", "PNG", compress_level=0)

    #open_cv_image = np.array(normal_image)
    # Convert RGB to BGR
    #open_cv_image = open_cv_image[:, :, ::-1].copy()
    #cv2.imwrite("try.jpg", open_cv_image, [cv2.IMWRITE_JPEG_QUALITY, 100])

    #break

