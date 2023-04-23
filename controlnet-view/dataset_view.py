import sys
sys.path.append(r'/DATA/disk1/cihai/lrz/3d-object-reconstruction')

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision
import random
import torch
import torch.nn as nn
import gzip
import matplotlib.pyplot as plt
import os
import json
import demjson
import numpy as np
import math
import cv2
import albumentations as A
import sys
from plyfile import PlyData
import pandas as pd
# print(sys.path)
#from blip.demo import blip_run


# from models.blip import blip_decoder
class MyDataset(Dataset):
    # split in ["train","val","test"]
    # transform in ["add_zero","center_crop"]
    def __init__(self, path, split, resolution, pairs, full_dataset, transform, kind=""):
        self.path = path
        self.resolution = resolution
        self.sequence = []
        self.kind = kind
        # split in ["train","val","test"]
        self.split = split
        self.transform = transform
        self.full_dataset = full_dataset
        self.pairs = pairs
        # key:sequence_path
        # value:[[img1_path,img2_path,...],text]
        self.sequence_imgs = {}
        # key:img_path
        # value:camera pose
        self.img_camera = {}
        # if(self.split=="train"):
        g = os.listdir(self.path)
        for dir in g:
            if (dir[0] != '_' and '.' not in dir):
                if (self.kind != "" and self.kind != dir):
                    continue
                # self.sequence.append(self.path+'/'+dir)
                seq_path = self.path + '/' + dir + '/' + "set_lists"
                seqs = os.listdir(seq_path)
                for seq in seqs:
                    with open(seq_path + '/' + seq, 'r') as file:
                        seq_info = json.load(file)
                        train_data = seq_info[self.split]
                        sequence_path = self.path + '/' + dir + '/' + train_data[0][0] + '/images'
                        self.sequence.append(sequence_path)
                        # self.sequence_imgs[sequence_path]=[[]]
                        imgs = []
                        for data in train_data:
                            imgs.append(self.path + '/' + data[2])
                        self.sequence_imgs[sequence_path] = [imgs]
                        ply_path = self.path + '/' + dir + '/' + train_data[0][0] + '/pointcloud.ply'
                        origin = self.get_origin(ply_path)
                        self.sequence_imgs[sequence_path].append(origin)
                anno_path = self.path + '/' + dir + "/frame_annotations.jgz"
                g_file = gzip.GzipFile(anno_path)
                content = g_file.read()
                text = demjson.decode(content)
                for frame in text:
                    img_path = self.path + '/' + frame["image"]["path"]
                    R = torch.Tensor(frame["viewpoint"]["R"])
                    T = torch.Tensor(frame["viewpoint"]["T"])
                    self.img_camera[img_path] = [R, T]
        if (self.full_dataset == False and os.path.exists("text.txt") == False):
            self.save_text("text.txt")
        if (self.full_dataset == True and os.path.exists("full_text.txt") == False):
            self.save_text("full_text.txt")
        # print(self.sequence)
        if (self.full_dataset == False):
            f = open("text.txt")
        else:
            f = open("full_text.txt")
        while (1):
            dir = f.readline()
            text = f.readline()
            if (len(dir) == 0):
                break
            dir = self.path + '/' + dir
            if (dir.replace('\n', '') in self.sequence_imgs):
                self.sequence_imgs[dir.replace('\n', '')].append(text.replace('\n', ''))
        f.close()

    def __getitem__(self, index):
        sequence_index = index // self.pairs
        imgs = random.sample(self.sequence_imgs[self.sequence[sequence_index]][0], 2)
        # img_pair=random.sample(self.sequence_imgs[self.sequence[sequence_index]][0],2)
        # print(img_pair)
        img1_path = imgs[0]
        img2_path = imgs[1]
        # print(img1_path)
        # print(img2_path)
        img1 = read_image(img1_path)
        img2 = read_image(img2_path)
        pose1 = self.img_camera[img1_path]
        pose2 = self.img_camera[img2_path]
        origin = self.sequence_imgs[self.sequence[sequence_index]][1]
        # print(origin)
        text = self.sequence_imgs[self.sequence[sequence_index]][2]
        if (self.transform == "add_zero"):
            img1, mask1 = self.image_transform(img1)
            img2, mask2 = self.image_transform(img2)
            # torchvision.utils.save_image(img1/255.0,"./1.png")
            # torchvision.utils.save_image(img2/255.0,"./2.png")
            mask1 = torch.tensor(mask1)
            mask2 = torch.tensor(mask2)
        elif (self.transform == "center_crop"):
            img1 = self.image_transform(img1)
            img2 = self.image_transform(img2)
            # print(img1)
            # print(img2)
            # torchvision.utils.save_image(img1/255.0,"./1.png")
            # torchvision.utils.save_image(img2/255.0,"./2.png")
        relative = self.relative_pose(pose1[0].numpy(), pose1[1].numpy(), pose2[0].numpy(), pose2[1].numpy(), origin)
        relative = torch.tensor(relative)
        relative = relative.view(-1)
        # print(text)
        # print(type(text))
        # return img1,mask1,img2,mask2,relative,text
        return dict(
            jpg=img1.permute(1, 2, 0),
            txt="An image",
            hint=img2.permute(1, 2, 0),
            view_linear=relative
        )

    def __len__(self):
        return self.pairs * len(self.sequence)

    # add zero to make the img to square and resize, also return mask
    def image_transform(self, img):
        if (self.transform == "add_zero"):
            C, H, W = img.shape
            pad_1 = int(abs(H - W) // 2)  # length to add zero on one side
            pad_2 = int(abs(H - W) - pad_1)  # length to add zero on the other side
            img = img.unsqueeze(0)  # add axis
            if H > W:
                img = nn.ZeroPad2d((pad_1, pad_2, 0, 0))(img)  # add zero to left and right
                x1 = (pad_1 * self.resolution) // H
                y1 = 0
                x2 = self.resolution - ((pad_2 * self.resolution) // H)
                y2 = self.resolution
            else:
                img = nn.ZeroPad2d((0, 0, pad_1, pad_2))(img)  # add zero to up and down
                x1 = 0
                y1 = (pad_1 * self.resolution) // W
                x2 = self.resolution
                y2 = self.resolution - ((pad_2 * self.resolution) // W)
            img = img.squeeze(0)
            img = torchvision.transforms.functional.resize(img, [self.resolution, self.resolution], interpolation=2)
            return img, [x1, y1, x2, y2]
        elif (self.transform == "center_crop"):
            trans = A.Compose([
                A.SmallestMaxSize(max_size=self.resolution, interpolation=cv2.INTER_AREA),
                A.CenterCrop(height=self.resolution, width=self.resolution)
            ])
            img = img.permute(1, 2, 0).numpy()
            img = trans(image=img)
            img = torch.from_numpy(img["image"])
            img = img.permute(2, 0, 1)
            return img

    # save the text of each sequence in text.txt using blip
    def save_text(self, path):
        file = open(path, 'w')
        for seq in self.sequence:
            img_path = random.choice(self.sequence_imgs[seq][0])
            text = blip_run(img_path)
            seq.replace(self.path + '/', '')
            file.write(seq)
            file.write('\n')
            file.write(text)
            file.write('\n')
            # print(seq)
            # print(text)
        '''
        text=blip_run(self.files[0])
        file.write(self.files[0])
        file.write('\n')
        file.write(text)
        file.write('\n')
        '''
        file.close()

    # pose2-pose1
    def relative_pose(self, R1, T1, R2, T2, origin):
        pw1 = np.dot(-R1.T, T1)
        pw2 = np.dot(-R2.T, T2)
        r1, theta1, phi1 = self.cart2sph(pw1[0] - origin[0], pw1[1] - origin[1], pw1[2] - origin[2])
        r2, theta2, phi2 = self.cart2sph(pw2[0] - origin[0], pw2[1] - origin[1], pw2[2] - origin[2])
        theta = theta2 - theta1
        phi = phi2 - phi1
        r = r2 - r1
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        return theta, sin_phi, cos_phi, r

    def cart2sph(self, x, y, z):
        r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = math.acos(z / r)
        phi = math.atan(y / x)
        return r, theta, phi

    def get_origin(self, ply_path):
        if (os.path.exists(ply_path)):
            plydata = PlyData.read(ply_path)  # 读取文件
            data = plydata.elements[0].data  # 读取数据
            data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
            data_np = np.zeros(data_pd.shape, dtype=np.float64)  # 初始化储存数据的array
            property_names = data[0].dtype.names  # 读取property的名字
            for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
                data_np[:, i] = data_pd[name]
            # print(data_np)
            # print(data_np.shape)
            origin = np.average(data_np[:, :3], axis=0)  # 按列求均值
            # print(result)
        else:
            origin = np.zeros(3)
        return origin


# train_data=MyDataset("../co3d-main/dataset","train",512,12,False)
if __name__ == '__main__':
    train_data = MyDataset("../../../dataset/co3d", "train", 512, 120, False, "add_zero", "car")
    train_loader = DataLoader(train_data, batch_size=6, shuffle=True)
    for result in train_loader:
    # print(result)
        print(result["jpg"].shape)
        print(result["txt"])
        print(result["hint"].shape)
        print(result["view_linear"])
    # sys.exit(0)
    # continue
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

