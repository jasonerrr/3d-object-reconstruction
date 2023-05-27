rm -rf image_log/train_7
mkdir image_log/train_7
CUDA_VISIBLE_DEVICES=7 rm -rf ./models/control_sd21_view_ini.ckpt
CUDA_VISIBLE_DEVICES=7 python tool_add_control_sd21_view.py ./models/v2-1_512-ema-pruned.ckpt ./models/control_sd21_view_ini.ckpt
CUDA_VISIBLE_DEVICES=7 python train_view.py --checkdir='model_checkpoint_7' --split='train_7'
