python train_combined.py -s data/scannet++_colmap -m output/scannet++_colmap_try
python render_combined.py -m output/L3D124S21ENDIDR4BOIUI5NYALUF3P3XA888_samples_depthfeedback_05_activate3k
python render_video.py -m output/scannet++_colmap_try
python render_light_traverse.py -m output/samples_new_120