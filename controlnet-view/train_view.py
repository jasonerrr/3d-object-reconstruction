from share import *
import sys
sys.path.append(r'/DATA/disk1/cihai/lrz/3d-object-reconstruction/controlnet-view')
import shutil
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from dataset_view import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

parser = argparse.ArgumentParser(description='train_view')
parser.add_argument('--resume_path', default='./models/control_sd21_view_ini.ckpt', type=str, help='init parameter')
parser.add_argument('--bs', default=4, type=int, help='batch size')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--checkdir', default='model_checkpoint_7', type=str, help='checkpoint save dir')
parser.add_argument('--split', default='train', type=str, help='train sample save dir')
args = parser.parse_args()

# Configs
resume_path = args.resume_path  # './models/control_sd21_view_ini.ckpt'
batch_size = args.bs  # 4
logger_freq = 1000
learning_rate = args.lr  # 1e-4
sd_locked = False
only_mid_control = False

checkpoint_callback = ModelCheckpoint(
    dirpath=args.checkdir,  # 'model_checkpoint_7',
    save_last=True,
    every_n_train_steps=1000,
)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21_view.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset(
    path="../../../yxd/dataset/co3d",
    split="train",
    resolution=512,
    pairs=6 * 4 * 1,
    full_dataset=False,
    transform="center_crop",
    kind="car",
    dropout=0.1
)
dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq, split=args.split)
trainer = pl.Trainer(gpus=1, precision=32, accumulate_grad_batches=6, callbacks=[logger, checkpoint_callback], max_epochs=10000)


# Train!
trainer.fit(model, dataloader)