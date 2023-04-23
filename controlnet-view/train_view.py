from share import *
import shutil

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset_view import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

shutil.rmtree('image_log')
# shutil.rmtree('lightning_logs')

# Configs
resume_path = '/data2/yanxudong/liuruizhe/controlnet-test/ControlNet-main0/models/control_sd21_view_ini.ckpt'
batch_size = 4
logger_freq = 1000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


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
    pairs=100,
    full_dataset=False,
    transform="center_crop",
    kind="car"
)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], max_epochs=10000)


# Train!
trainer.fit(model, dataloader)
