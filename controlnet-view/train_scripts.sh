rm -rf ./models/control_sd21_view_ini.ckpt
python tool_add_control_sd21_view.py ./models/v2-1_512-ema-pruned.ckpt ./models/control_sd21_view_ini.ckpt
python train_view.py
