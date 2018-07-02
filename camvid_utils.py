
# coding: utf-8

# In[1]:


import torch 
import torch.nn as nn
import os 
import cv2
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


root_dir='camvid-master'
data_dir=os.path.join(root_dir,'701_StillsRaw_full')
label_dir=os.path.join(root_dir,'LabeledApproved_full')
label_colors_file=os.path.join(root_dir,'label_colors.txt')
#create dir for labeled idx
label_idx_dir=os.path.join(root_dir,'Labeled_idx')
if not os.path.exists(label_idx_dir):
    os.mkdir(label_idx_dir)


# In[4]:


all_images=os.listdir(data_dir)
all_labels=os.listdir(label_dir)
label2color={}
color2label={}
label2index={}
index2label={}
def parse_label():
    f=open(label_colors_file,'r').read().split("\n")[:-1] #ignoring the last empty line
    for idx,line in enumerate(f):
        label=line.split()[-1]
        color=tuple([int(x) for x in line.split()[:-1]])
    
        color2label[color]=label
        label2color[label]=color
        label2index[label]=idx
        index2label[idx]=label
    
    for idx,name in enumerate(all_labels):
        filename=os.path.join(label_idx_dir,name)
        image=cv2.imread(os.path.join(label_dir,name))
        height,width,_=image.shape
        idx_mat=np.zeros((height,width))
        for h in range(height):
            for w in range(width):
                color=tuple(image[h,w])
                try:
                    color2label[color]=label
                    label2index[label]=index
                    idx_mat[h,w]=index
                except:
                    pass
                
        idx_mat.astype(np.uint8)
        np.save(filename,idx_mat)
    
    


# In[5]:


parse_label()

