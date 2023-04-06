from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision
import random
import torch
import torch.nn as nn
import gzip
import matplotlib.pyplot as plt
import os
import demjson
import numpy as np
import math
#import cv2
import sys
#print(sys.path)
from blip.demo import blip_run
#from models.blip import blip_decoder
class MyDataset(Dataset):
    def __init__(self,path,resolution):
        self.path=path
        #self.crop=crop
        self.resolution=resolution
        #get_data=GetDataset(path=self.path)
        #self.data=get_data.getImageCamera()
        #paths of all object
        #e.g. ../co3d-main/dataset/plant
        self.paths=[]
        #paths of all folders that contain training images
        #e.g. ../co3d-main/dataset/plant/247_26441_50907/images
        self.dirs=[]
        #paths of all images
        #e.g. ../co3d-main/dataset/plant/461_65179_127609/images/frame000001.jpg
        self.files=[]
        #key : father folder of images
        #value : list of paths of images
        self.dir_file={}
        #key : path of images
        #value : camera pose in the form of [R,T] of the image
        self.file_pose={}
        self.img_mask={}
        self.dir_text={}
        g=os.listdir(path)
        for dir in g:
            if(dir[0]!='_' and '.' not in dir):
                #print(dir)
                #self.objects.append(dir)
                obj_path=path+'/'+dir
                self.paths.append(obj_path)
                folders=os.listdir(obj_path)
                for folder in folders:
                    if(folder[0].isdigit()):
                        self.dirs.append(obj_path+'/'+folder+"/images")
                        self.dir_file[obj_path+'/'+folder+"/images"]=[]
                name=obj_path+"/frame_annotations.jgz"
                g_file=gzip.GzipFile(name)
                content=g_file.read()
                text=demjson.decode(content)
                

                for frame in text:
                    if(frame["meta"]["frame_type"][-6:]!="unseen"):
                        img_path=self.path+"/"+frame["image"]["path"]
                        self.files.append(img_path)
                        #if(self.crop):
                            #self.img_mask[img_path]=self.path+'/'+frame["mask"]["path"]
                        folder=os.path.dirname(img_path)
                        self.dir_file[folder].append(img_path)
                        R=torch.Tensor(frame["viewpoint"]["R"])
                        #print(R.shape)
                        T=torch.Tensor(frame["viewpoint"]["T"])
                        #print(T.shape)
                        self.file_pose[img_path]=[R,T]
        #print(self.dirs)
        #add text to each sequence
        if(os.path.exists("text.txt")==False):
            self.save_text("text.txt")
        f=open("text.txt")
        while(1):
            dir=f.readline()
            text=f.readline()
            if(len(dir)==0):
                break
            self.dir_text[dir.replace('\n','')]=text.replace('\n','')
        f.close()
        print(self.dir_text)
        #print(self.path)
        #print(self.paths[0])
        #print(self.dirs[0])
        #print(self.files[0])
        #print(self.dir_file[self.dirs[0]])
        #print(self.file_pose[self.files[0]])
    def __getitem__(self,index):
        '''
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
        '''
        img1_path=self.files[index]
        img1=read_image(img1_path)
        pose1=self.file_pose[img1_path]
        folder=os.path.dirname(img1_path)
        img2_path=random.choice(self.dir_file[folder])
        img2=read_image(img2_path)
        pose2=self.file_pose[img2_path]
        text=self.dir_text[folder]
        '''
        if(self.crop):
            mask1_path=self.img_mask[img1_path]
            mask1=read_image(mask1_path)
            mask1=torch.squeeze(mask1)
            mask2_path=self.img_mask[img2_path]
            mask2=read_image(mask2_path)
            mask2=torch.squeeze(mask2)
            #img1=self.cropWithMask(img1,mask1,self.resolution)
            #img2=self.cropWithMask(img2,mask2,self.resolution)
            img1=self.cropWithMask(img1,mask1,512)
            img2=self.cropWithMask(img2,mask2,512)
        '''
        img1,mask1=self.image_transform(img1)
        img2,mask2=self.image_transform(img2)
        relative=self.relative_pose(pose1[0].numpy(),pose1[1].numpy(),pose2[0].numpy(),pose2[1].numpy())
        relative=torch.tensor(relative)
        mask1=torch.tensor(mask1)
        mask2=torch.tensor(mask2)

        return img1,mask1,pose1,img2,mask2,pose2,relative,text
    def __len__(self):
        return len(self.files)
    def cropWithMask(self,img,mask,resolution):
        mask=mask.numpy()
        img=img.permute(1,2,0).numpy()
        #print(img)
        #print(img.shape)
        x,y=(mask>0).nonzero() #bug
        up=min(x)
        down=max(x)
        left=min(y)
        right=max(y)
        height=down-up
        width=right-left
        center_y=(up+down)//2
        center_x=(left+right)//2
        radius=max(height,width)//2
        #print(up,down,left,right)
        crop=img[max(0,center_y-radius):min(mask.shape[0],center_y+radius),max(0,center_x-radius):min(mask.shape[1],center_x+radius)]
        crop=torch.from_numpy(crop)
        crop=crop.permute(2,0,1)
        #crop=torchvision.transforms.Resize(crop,(resolution,resolution))
        crop=torchvision.transforms.functional.resize(crop, [resolution,resolution], interpolation=2)
        return crop
    #add zero to make the img to square and resize, also return mask
    def image_transform(self,img):
        C, H, W = img.shape
        pad_1 = int(abs(H - W) // 2)  # 一侧填充长度
        pad_2 = int(abs(H - W) - pad_1)  # 另一侧填充长度
        img = img.unsqueeze(0)  # 加轴
        if H > W:
            img = nn.ZeroPad2d((pad_1, pad_2, 0, 0))(img)  # 左右填充，填充值是0
            x1=(pad_1*self.resolution)//H
            y1=0
            x2=self.resolution-((pad_2*self.resolution)//H)
            y2=self.resolution
        elif H < W:
            img = nn.ZeroPad2d((0, 0, pad_1, pad_2))(img)  # 上下填充，填充值是0
            x1=0
            y1=(pad_1*self.resolution)//W
            x2=self.resolution
            y2=self.resolution-((pad_2*self.resolution)//W)
        img = img.squeeze(0)
        img=torchvision.transforms.functional.resize(img, [self.resolution,self.resolution], interpolation=2)
        return img,[x1,y1,x2,y2]

    #save the text of each sequence in text.txt using blip
    def save_text(self,path):
        file=open(path,'w')
        for seq in self.dirs:
            img_path=random.choice(self.dir_file[seq])
            text=blip_run(img_path)
            file.write(seq)
            file.write('\n')
            file.write(text)
            file.write('\n')
            print(seq)
            print(text)
        '''
        text=blip_run(self.files[0])
        file.write(self.files[0])
        file.write('\n')
        file.write(text)
        file.write('\n')
        '''
        file.close()

    #pose2-pose1
    def relative_pose(self,R1,T1,R2,T2):
        pw1=np.dot(-(R1.T),T1)
        pw2=np.dot(-(R2.T),T2)
        r1,theta1,phi1=self.cart2sph(pw1[0],pw1[1],pw1[2])
        r2,theta2,phi2=self.cart2sph(pw2[0],pw2[1],pw2[2])
        theta=theta2-theta1
        phi=phi2-phi1
        r=r2-r1
        sin_phi=math.sin(phi)
        cos_phi=math.cos(phi)
        return theta,sin_phi,cos_phi,r
    
    def cart2sph(self,x,y,z):
        r=math.sqrt(x**2+y**2+z**2)
        theta=math.acos(z/r)
        phi=math.atan(y/x)
        return r,theta,phi
    
train_data=MyDataset("../co3d-main/dataset",512)
train_loader=DataLoader(train_data,batch_size=1,shuffle=False)
for img1,mask1,pose1,img2,mask2,pose2,relative,text in train_loader:
    print(img1.shape)
    print(mask1)
    #print(img1)
    print(pose1)
    print(img2.shape)
    print(mask2)
    #print(img2)
    print(pose2)
    print(relative)
    print(relative.shape)
    torchvision.utils.save_image(img1/255.0,"./1.png")
    torchvision.utils.save_image(img2/255.0,"./2.png")
    print(text)
    sys.exit(0)
#print(train_data[0])
#print(train_data[0][0])
#plt.imshow(train_data[0][0].permute(1,2,0))
#plt.show()
#train_loader=DataLoader(train_data,batch_size=1,shuffle=False)
#img,pose=next(iter(train_loader))
#print(img)
#print(pose)
#print(train_loader)
#for img,pose in train_loader:
    #print(img)
    #print(pose)

