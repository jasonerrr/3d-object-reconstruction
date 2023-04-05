from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision
import random
import torch
import gzip
import matplotlib.pyplot as plt
import os
import demjson
import numpy as np
import math

import cv2

import matplotlib


class MyDataset(Dataset):
    def __init__(self, path, crop, resolution):
        self.path = path
        self.crop = crop
        self.resolution = resolution
        self.paths = []
        self.dirs = []
        self.files = []
        self.dir_file = {}
        self.file_pose = {}
        self.img_mask = {}
        g = os.listdir(path)

        # tmp_cnt0 = 0

        for dir in g:
            # tmp_cnt0 += 1
            # print(tmp_cnt0)
            if dir[0] != '_' and '.' not in dir:
                obj_path = path + '/' + dir
                self.paths.append(obj_path)
                folders = os.listdir(obj_path)
                for folder in folders:
                    if folder[0].isdigit():
                        self.dirs.append(obj_path + '/' + folder + "/images")
                        self.dir_file[obj_path + '/' + folder + "/images"] = []
                name = obj_path + "/frame_annotations.jgz"
                g_file = gzip.GzipFile(name)
                content = g_file.read()
                text = demjson.decode(content)

                # tmp_cnt1 = 0

                for frame in text:
                    # tmp_cnt1 += 1
                    # print(tmp_cnt1)
                    if frame["meta"]["frame_type"][-6:] != "unseen":
                        img_path = self.path + "/" + frame["image"]["path"]
                        self.files.append(img_path)
                        if self.crop:
                            self.img_mask[img_path] = self.path + '/' + frame["mask"]["path"]
                        folder = os.path.dirname(img_path)
                        self.dir_file[folder].append(img_path)
                        R = torch.Tensor(frame["viewpoint"]["R"])
                        T = torch.Tensor(frame["viewpoint"]["T"])
                        self.file_pose[img_path] = [R, T]

    def __getitem__(self, index):
        """
        img1_path,pose=self.data[index]
        folder=os.path.dirname(img1_path)
        print(img1_path)
        print(folder)
        img2_file=random.choice(os.listdir(folder))
        img2_path=os.path.join(folder,img2_file)
        #pose="apple"
        pose=str(pose)
        img1=read_image(img1_path)
        img2=read_image(img2_path)
        #img=self.transform(img)
        return img1,img2,pose
        """

        img1_path = self.files[index]
        img1 = read_image(img1_path)
        pose1 = self.file_pose[img1_path]
        folder = os.path.dirname(img1_path)
        img2_path = random.choice(self.dir_file[folder])
        img2 = read_image(img2_path)
        pose2 = self.file_pose[img2_path]
        if self.crop:
            mask1_path = self.img_mask[img1_path]
            mask1 = read_image(mask1_path)
            mask1 = torch.squeeze(mask1)
            mask2_path = self.img_mask[img2_path]
            mask2 = read_image(mask2_path)
            mask2 = torch.squeeze(mask2)
            img1 = self.cropWithMask(img1, mask1, self.resolution)
            img2 = self.cropWithMask(img2, mask2, self.resolution)
        relative = self.relative_pose(pose1[0].numpy(), pose1[1].numpy(), pose2[0].numpy(), pose2[1].numpy())
        relative = relative.view(-1)
        return dict(
            jpg=img1.permute(1, 2, 0),
            txt="An image",
            hint=img2.permute(1, 2, 0),
            view_linear=relative
        )

    def __len__(self):
        return len(self.files)

    def cropWithMask(self, img, mask, resolution):
        mask = mask.numpy()
        img = img.permute(1, 2, 0).numpy()
        # print(img)
        # print(img.shape)
        x, y = (mask > 0).nonzero()  # bug
        up = min(x)
        down = max(x)
        left = min(y)
        right = max(y)
        height = down - up
        width = right - left
        center_y = (up + down) // 2
        center_x = (left + right) // 2
        radius = max(height, width) // 2
        # print(up,down,left,right)
        crop = img[max(0, center_y - radius):min(mask.shape[0], center_y + radius),
               max(0, center_x - radius):min(mask.shape[1], center_x + radius)]
        crop = torch.from_numpy(crop)
        crop = crop.permute(2, 0, 1)
        # crop=torchvision.transforms.Resize(crop,(resolution,resolution))
        crop = torchvision.transforms.functional.resize(crop, [resolution, resolution], interpolation=2)
        return crop

    # pose2-pose1
    def relative_pose(self, R1, T1, R2, T2):
        pw1 = np.dot(-(R1.T), T1)
        pw2 = np.dot(-(R2.T), T2)
        r1, theta1, phi1 = self.cart2sph(pw1[0], pw1[1], pw1[2])
        r2, theta2, phi2 = self.cart2sph(pw2[0], pw2[1], pw2[2])
        theta = theta2 - theta1
        phi = phi2 - phi1
        r = r2 - r1
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        return torch.tensor([theta, sin_phi, cos_phi, r])

    def cart2sph(self, x, y, z):
        r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = math.acos(z / r)
        phi = math.atan(y / x)
        return r, theta, phi


"""
train_data = MyDataset("../co3d-main/dataset", True, 512)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
for img1, pose1, img2, pose2, relative in train_loader:
    print(img1.shape)
    print(pose1)
    print(img2.shape)
    print(pose2)
    print(relative)
    print(relative.shape)
"""
# print(train_data[0])
# print(train_data[0][0])
# plt.imshow(train_data[0][0].permute(1,2,0))
# plt.show()
# train_loader=DataLoader(train_data,batch_size=1,shuffle=False)
# img,pose=next(iter(train_loader))
# print(img)
# print(pose)
# print(train_loader)
# for img,pose in train_loader:
# print(img)
# print(pose)

'''
dataset = MyDataset()
print(len(dataset))

item = dataset[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)
'''

'''
print('test dataset preparing')
dataset = MyDataset("/data2/yanxudong/yxd/co3d-main/dataset", True, 512)
print('test dataset prepared')
print(len(dataset))

item = dataset[1234]
jpg = item['jpg']
jpg_np = jpg.cpu().detach().numpy()

matplotlib.image.imsave('fxxk_off/fxxking_test.png', jpg_np)
'''
