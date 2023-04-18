import math
import json
import torchvision
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
import random
from einops import rearrange
import webdataset as wds
import pytorch_lightning as pl
from torch.utils.data import Dataset,DataLoader
from omegaconf import DictConfig, ListConfig
from pathlib import Path
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt



class ObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view, train=None, validation=None,
                 test=None, num_workers=4, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

        if 'image_transforms' in dataset_config:
            image_transforms = [torchvision.transforms.Resize(dataset_config.image_transforms.size)]
        else:
            image_transforms = []
        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)


    def train_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=False, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=True, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=self.validation),\
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='.objaverse/hf-objaverse-v1/views',
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=4,
        validation=False
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_trans = default_trans
        self.return_paths = return_paths
        '''
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        '''
        self.postprocess = postprocess
        self.total_view = total_view

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]
        '''
        with open(os.path.join(root_dir, 'valid_paths.json')) as f:
            self.paths = json.load(f)
        '''
        self.paths=[root_dir]
        total_objects = len(self.paths)
        '''
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):] # used last 1% as validation
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)] # used first 99% as training
        '''
        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths)
        
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        
        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img

    def __getitem__(self, index):

        data = {}
        if self.paths[index][-2:] == '_1': # dirty fix for rendering dataset twice
            total_view = 8
        else:
            total_view = 4
        #index_target, index_cond = random.sample(range(total_view), 2) # without replacement
        index_target,index_cond=0,1
        print(index_target,index_cond)
        filename = self.paths[0]
        #print(filename)
        # print(self.paths[index])

        if self.return_paths:
            data["path"] = str(filename)
        
        color = [1., 1., 1., 1.]

        
        target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
        cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
        target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
        cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
        #print(target_im)

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["T"] = self.get_T(target_RT, cond_RT)

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)


image_transforms = []
image_transforms.extend([transforms.ToTensor(),transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
image_transforms = torchvision.transforms.Compose(image_transforms)
data=ObjaverseData("./8476c4170df24cf5bbe6967222d1a42d",image_transforms=image_transforms)
loader=DataLoader(data,batch_size=1,shuffle=True)
for result in loader:
    #print(result)
    print(result["image_target"].shape)
    #print(result["image_target"])
    print(result["image_target"][0][256][256])
    #print(result["image_target"][0][256][256]/255.0)
    print(result["image_cond"].shape)
    print(result["image_cond"][0][256][256])
    #print(result["image_cond"][0][256][256]/255.0)
    #print(result["image_cond"])
    print(result["T"])
    torchvision.utils.save_image(result["image_target"].squeeze().permute(2,0,1),"./zero/0.png")
    torchvision.utils.save_image(result["image_cond"].squeeze().permute(2,0,1),"./zero/1.png")
    #print(result["view_linear"])
