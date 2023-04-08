# 3d-object-reconstruction
## This is a dataLoader
### Introduction

There are five parameters in class MyDataset  
path: the path of the co3d dataset  
split: choose from "train","val","test", define which split of the dataset you want to use
resolution: an integer, the image transformer will return an image of size resolution*resolution  
pairs: an integer, howmany pairs of images you want in an sequence  
full_dataset: boolean, True means you want to use the full co3d dataset of size 5.5T, False means you want to use single-sequence dataset subset of size 8.9G  
transform: choose from "add_zero", "center_crop", define which transform method you want to use  
