
# coding: utf-8

# In[7]:


from __future__ import print_function

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# In[2]:


import numpy as np
import os
from torchvision.models.vgg import VGG
from torchvision import models



# In[3]:


class FCN32s(nn.Module):
    def __init__(self,pretrained_net,n_classes):
        
        super(FCN32s,self).__init__()
        self.pretrained_net=pretrained_net
        self.n_classes=n_classes
        self.relu=nn.ReLU(inplace=True)
        self.deconv1=nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn1=nn.BatchNorm2d(512)
        self.deconv2=nn.ConvTranspose2d(512,256,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn2=nn.BatchNorm2d(256)
        self.deconv3=nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn3=nn.BatchNorm2d(128)
        self.deconv4=nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn4=nn.BatchNorm2d(64)
        self.deconv5=nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn5=nn.BatchNorm2d(32)
        
        self.classifier=nn.Conv2d(32,n_class,kernel_size=1)
        
        
    def forward(self,x):
        output=self.pretrained_net(x)
        x5=output['x5']
        
        score=self.bn1(self.relu(self.deconv1(x5)))
        score=self.bn2(self.relu(self.deconv2(score)))
        score=self.bn3(self.relu(self.deconv3(score)))
        score=self.bn4(self.relu(self.deconv4(score)))
        score=self.bn5(self.relu(self.deconv5(score)))
        
        score=self.classifier(score)
        
        return score  


# In[8]:


class FCN16s(nn.Module):
    def __init__(self,n_classes,pretrained_net):
        super(FCN16s,self).__init__()
        self.pretrained_net=pretrained_net
        self.n_classes=n_classes
        self.relu=nn.ReLU(inplace=True)
        self.deconv1=nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn1=nn.BatchNorm2d(512)
        self.deconv2=nn.ConvTranspose2d(512,256,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn2=nn.BatchNorm2d(256)
        self.deconv3=nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn3=nn.BatchNorm2d(128)
        self.deconv4=nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn4=nn.BatchNorm2d(64)
        self.deconv5=nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn5=nn.BatchNorm2d(32)
        
        self.classifier=nn.Conv2d(32,n_classes,kernel_size=1)
        
    def forward(self,x):
        output=self.pretrained_net(x)
        x5=output['x5']
        x4=output['x4']
        score=self.relu(self.deconv1(x5))
        score=self.bn1(score+x4)
        score=self.bn2(self.relu(self.deconv2(score)))
        score=self.bn3(self.relu(self.deconv3(score)))
        score=self.bn4(self.relu(self.deconv4(score)))
        score=self.bn5(self.relu(self.deconv4(score)))
        
        score=self.classifier(score)
        
        return score #size (N,n_classes,X.H, X.W)
        


# In[5]:


class FCN8s(nn.Module):
    def __init__(self,n_classes,pretrained_net):
        super(FCN8s,self).__init__()
        self.pretrained_net=pretrained_net
        self.n_classes=n_classes
        self.relu=nn.ReLU(inplace=True)
        self.deconv1=nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn1=nn.BatchNorm2d(512)
        self.deconv2=nn.ConvTranspose2d(512,256,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn2=nn.BatchNorm2d(256)
        self.deconv3=nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn3=nn.BatchNorm2d(128)
        self.deconv4=nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn4=nn.BatchNorm2d(64)
        self.deconv5=nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn5=nn.BatchNorm2d(32)
        
        self.classifier=nn.Conv2d(32,n_classes,kernel_size=1)
    
    def forward(self,x):
        output=pretrained_net(x)
        x5=output['x5']
        x4=output['x4']
        x3=output['x3']
        score=self.relu(self.deconv1(x5))
        score=self.bn1(x4+score)
        score=self.relu(self.deconv2(score))
        score=self.bn2(x3+score)
        score=self.bn3(self.relu(self.deconv3(score)))
        score=self.bn4(self.relu(self.deconv4(score)))
        score=self.bn5(self.relu(self.deconv5(score)))
        
        score=self.classifier(score)
        return score


# In[6]:


class FCNs(nn.Module):
    def __init__(self,n_classes,pretrained_net):
        super(FCNs,self).__init__()
        self.pretrained_net=pretrained_net
        self.n_classes=n_classes
        self.relu=nn.ReLU(inplace=True)
        self.deconv1=nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn1=nn.BatchNorm2d(512)
        self.deconv2=nn.ConvTranspose2d(512,256,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn2=nn.BatchNorm2d(256)
        self.deconv3=nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn3=nn.BatchNorm2d(128)
        self.deconv4=nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn4=nn.BatchNorm2d(64)
        self.deconv5=nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn5=nn.BatchNorm2d(32)
        
        self.classifier=nn.Conv2d(32,n_classes,kernel_size=1)
    
    def forward(self,x):
        output=pretrained_net(x)
        x5=output['x5']
        x4=output['x4']
        x3=output['x3']
        x2=output['x2']
        x1=output['x1']
        
        score=self.relu(self.deconv1(x5))
        score=self.bn1(x4+score)
        score=self.relu(self.deconv2(score))
        score=self.bn2(x3+score)
        score=self.relu(self.deconv3(score))
        score=self.bn3(x2+score)
        score=self.relu(self.deconv4(score))
        score=self.bn4(x1+score)
        score=self.bn5(self.relu(self.deconv5(score)))
        return score


# In[ ]:


class VGGNet(VGG):
    def __init__(self,pretrained=True,model='vgg16',requires_grad=True,remove_fc=True,show_params=False):
        super(VGGNet,self).__init__(make_layers(cfg[model])) #making layers module of vgg 16 
        self.ranges=ranges[model]  #super arguement specific to python 2
                                   #in python 3 super() works
        
        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model) #loads the model
            
        if not requires_grad:
            for param in super().parameters():
                param.requires_grad=True
        if remove_fc:
            del self.classifier #removing fully connected layers
        if show_params:
            for name,param in self.named_parameters():
                
                print (name,param.size())
    
    def forward(self,x):
        output={}
        
        #getting the output of the max pool layers(5 maxpools in vgg net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0],self.ranges[idx][1]):
                x=self.features[layer](x)
            output["x%d"%(idx+1)]=x
            
        
        return output

ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg,batch_norm=False):
    layers=[]  #making a list of layers in the vgg network
    in_channels=3
    for v in cfg: #cfg gives a order of conv and maxpool layers
        if v=='M':
            layers+=[nn.MaxPool2d(kernel_size=2,stride=2)] #adding a maxpool layer
        else:
            conv2d=nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
            if batch_norm:
                layers+=[conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
            else:
                layers+=[conv2d,nn.ReLU(inplace=True)]
            in_channels=v        
    
    return nn.Sequential(*layers) #making a nn module for these layers


        


# In[ ]:


if __name__=='__main__':
    pass

