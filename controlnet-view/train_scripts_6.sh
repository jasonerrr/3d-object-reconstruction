CUDA_VISIBLE_DEVICES=6 rm -rf ./models/control_sd21_view_ini.ckpt
CUDA_VISIBLE_DEVICES=6 python tool_add_control_sd21_view.py ./models/v2-1_512-ema-pruned.ckpt ./models/control_sd21_view_ini.ckpt
CUDA_VISIBLE_DEVICES=6 python train_view.py --checkdir='model_checkpoint_6' --split='train_6'
