python train_combined.py -s data/scannet++_colmap -m output/scannet++_colmap_try_regularization0_20_inverse_try
python render_combined.py -m output/scannet++_colmap_try_regularization0_20_inverse_try
python render_video.py -m output/scannet++_colmap_try_regularization0_20_inverse_try
python render_light_traverse.py -m output/scannet++_colmap_try_regularization0_20_inverse_try