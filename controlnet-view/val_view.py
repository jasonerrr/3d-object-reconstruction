from share import *

import os
import numpy as np
import torch
import torchvision
from PIL import Image

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset_view import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import io

import lpips

# Configs
resume_path = './models/control_sd21_view_ini.ckpt'
batch_size = 6
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

save_dir = ''
split = 'val'


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21_view.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
print('preparing dataset')
dataset = MyDataset("../../../yxd/dataset/co3d", split="train", resolution=512, pairs=12)
print('preparing dataset done')
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=False)

# validation

model.to('cuda:0')
model.eval()

for batch_idx, batch in enumerate(dataloader):
    # print('the batch is:', batch)
    with torch.no_grad():
        images = model.log_images(batch)

    # PSNR, SSIM, LPIPS, FID

    for k in images:
        if isinstance(images[k], torch.Tensor):
            images[k] = images[k].detach().cpu()
            images[k] = torch.clamp(images[k], -1., 1.)

    root = os.path.join(save_dir, "image_log", split)

    for k in images:
        o_grid = torchvision.utils.make_grid(images[k], nrow=2)
        o_grid = (o_grid + 1.0) / 2.0
        o_grid = o_grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        o_grid = o_grid.numpy()
        o_grid = (o_grid * 255).astype(np.uint8)
        filefolder = "gs-{:08}".format(batch_idx)
        filename = "{}_gs-{:08}.png".format(k, batch_idx)
        path = os.path.join(root, filefolder, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        Image.fromarray(o_grid).save(path)


















