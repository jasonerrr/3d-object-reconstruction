from share import *
import shutil

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from dataset_view import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

shutil.rmtree('image_log')
# shutil.rmtree('lightning_logs')

# Configs
resume_path = './models/control_sd21_view_ini.ckpt'
batch_size = 6
logger_freq = 1000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

checkpoint_callback = ModelCheckpoint(
    dirpath='model_checkpoint',
    filename='cldm-view-{epoch}',
    save_last=True,
    every_n_epochs=40
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
    pairs=100,
    full_dataset=False,
    transform="center_crop",
    kind="car",
    dropout=0.1
)
dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, checkpoint_callback], max_epochs=10000)


# Train!
trainer.fit(model, dataloader)