from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision
import random
import torch
import gzip
import matplotlib.pyplot as plt
import os
import demjson
class MyDataset(Dataset):
    def __init__(self,path,crop,resolution):
        self.path=path
        self.crop=crop
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
                        if(self.crop):
                            self.img_mask[img_path]=self.path+'/'+frame["mask"]["path"]
                        folder=os.path.dirname(img_path)
                        self.dir_file[folder].append(img_path)
                        R=torch.Tensor(frame["viewpoint"]["R"])
                        #print(R.shape)
                        T=torch.Tensor(frame["viewpoint"]["T"])
                        #print(T.shape)
                        self.file_pose[img_path]=[R,T]
    
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
        if(self.crop):
            mask1_path=self.img_mask[img1_path]
            mask1=read_image(mask1_path)
            mask1=torch.squeeze(mask1)
            mask2_path=self.img_mask[img2_path]
            mask2=read_image(mask2_path)
            mask2=torch.squeeze(mask2)
            img1=self.cropWithMask(img1,mask1,self.resolution)
            img2=self.cropWithMask(img2,mask2,self.resolution)
        return img1,pose1,img2,pose2
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
train_data=MyDataset("../co3d-main/dataset",True,512)
train_loader=DataLoader(train_data,batch_size=1,shuffle=False)
for img1,pose1,img2,pose2 in train_loader:
    print(img1.shape)
    print(pose1)
    print(img2.shape)
    print(pose2)
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

