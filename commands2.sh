python train_combined.py -s data/7b6477cb95 -m output/7b6477cb95_try_regularization0_60_inverse_try_stdloss2_02_3by3_dilated
python render_combined.py -m output/7b6477cb95_try_regularization0_60_inverse_try_stdloss2_02_3by3_dilated
python render_video.py -m output/7b6477cb95_try_regularization0_60_inverse_try
python render_light_traverse.py -m output/7b6477cb95_try_regularization0_60_inverse_try