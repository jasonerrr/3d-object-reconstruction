import demjson
import gzip
import matplotlib.pyplot as plt
import json
from plyfile import PlyData
import pandas as pd
import os
from mpl_toolkits import mplot3d
import numpy as np
path="../../data/CO3D/co3d-main/dataset"
sequence=""
kind="toytrain"
#split in ["train","val","test"]

#key:sequence_path
#value:[[img1_path,img2_path,...],text]

#key:img_path
#value:camera pose

#if(self.split=="train"):
g=os.listdir(path)
d={}
for dir in g:
    if(dir[0]!='_' and '.' not in dir):
        if(kind!="" and kind!=dir):
            continue
        #self.sequence.append(self.path+'/'+dir)
        seq_path=path+'/'+dir+'/'+"set_lists"
        seqs=os.listdir(seq_path)
        for seq in seqs:
            with open(seq_path+'/'+seq,'r') as file:
                seq_info=json.load(file)
                seq=seq_info["train"][0][0]
                sequence=path+"/"+kind+"/"+seq
        anno_path=path+'/'+dir+"/frame_annotations.jgz"
        g_file=gzip.GzipFile(anno_path)
        content=g_file.read()
        text=demjson.decode(content)
        for frame in text:
            d[path+"/"+frame["image"]["path"]]=frame["viewpoint"]

#print(d)

#x1=[0]
#y1=[0]
#z1=[0]
x2=[0]
y2=[0]
z2=[0]
print(sequence)
ply_path=sequence+"/pointcloud.ply"
for i in range(1,203):
    name=sequence+"/images/frame"+str(i).zfill(6)+".jpg"
    if(name in d):
    #print(d[path1+str(i).zfill(6)+".jpg"]["R"])
        R=np.array(d[name]["R"],dtype=float)
        T=np.array(d[name]["T"],dtype=float)
        #dot1=T
        dot2=np.dot(-(R.T),T)
        #dot2=T
        #x1.append(float(dot1[0]))
        #y1.append(float(dot1[1]))
        #z1.append(float(dot1[2]))
        x2.append(float(dot2[0]))
        y2.append(float(dot2[1]))
        z2.append(float(dot2[2]))
#fig = plt.figure()
#ax1 = plt.axes(projection='3d')
#ax1.scatter3D(x1, y1, z1)
#plt.show()
if(os.path.exists(ply_path)):
    plydata = PlyData.read(ply_path)  # 读取文件
    data = plydata.elements[0].data  # 读取数据
    data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
    data_np = np.zeros(data_pd.shape, dtype=np.float64)  # 初始化储存数据的array
    property_names = data[0].dtype.names  # 读取property的名字
    for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
        data_np[:, i] = data_pd[name]
    #print(data_np)
    #print(data_np.shape)
    #origin=np.average(data_np[:,:3], axis=0)  # 按列求均值
    j=50
    for temp in data_np:
        if(j==0):
            x2.append(temp[0])
            y2.append(temp[1])
            z2.append(temp[2])
            j=50
        else:
            j-=1
    #print(result)

for i in range(20):
    x2.append(i)
    y2.append(0)
    z2.append(0)
for i in range(20):
    x2.append(0)
    y2.append(i)
    z2.append(0)
for i in range(20):
    x2.append(0)
    y2.append(0)
    z2.append(i)

#plt.zlabel('z')
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x2, y2, z2,s=0.1)
'''
x=sum(x2)
y=sum(y2)
z=sum(z2)
x/=202
y/=202
z/=202
xx=np.array([-x,-y,-z])
for i in range(1,203):
    yy=np.array([x2[i]-x,y2[i]-y,z2[i]-z])
    #print(xx)
    #print(yy)
    print(np.dot(xx,yy))
'''


plt.show()


