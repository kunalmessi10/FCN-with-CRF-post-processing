
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader


# In[3]:


import torchvision.transforms as transform
import numpy as np
from torch.autograd import Variable
from crf import dense_crf
from fcn import FCNs,FCN8s,FCN32s,VGGNet


# In[4]:


from skimage import io
from skimage import transform as trans
import cv2
import os
from sklearn.model_selection import train_test_split


# In[8]:


root_dir='camvid-master'
data_dir=os.path.join(root_dir,'701_StillsRaw_full')
label_dir=os.path.join(root_dir,'LabeledApproved_full')
label_colors_file=os.path.join(root_dir,'label_colors.txt')
label_idx_dir=os.path.join(root_dir,'Labeled_idx')
n_classes=32
h,w=720,960
train_h,train_w=int(h*2/3),int(w*2/3)
val_h,val_w=int(h/32)*32,int(w)
momentum=0


# In[9]:


configs= "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, weight_decay)
print("Configs:", configs)


# In[10]:


model_dir=os.path.join(root_dir,'model')
model_path = os.path.join(model_dir, configs)


# In[11]:


epochs=500
batch_size=6
lr=1e-4
weight_decay=1e-5
step_size=30
gamma=0.5


# In[13]:


score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
IU_scores    = np.zeros((epochs, n_classes))
pixel_scores = np.zeros(epochs)


# In[14]:


all_images=os.listdir(data_dir)
all_labels=os.listdir(label_dir)


# In[15]:


cam_train,cam_val=train_test_split(all_images,train_size=0.9,test_size=0.1)


# In[16]:


class camvid(Dataset):
    def __init__(self,data_dir,label_idx_dir,cam_train,cam_val,train=True,transform=None):
        self.data_dir=data_dir
        self.label_idx_dir=label_idx_dir
        self.cam_train=cam_train
        self.cam_val=cam_val
        self.train=train
        self.transform=transform
    def __len__(self):
        if self.train:
            return len(self.cam_train)
        else:
            return len(self.cam_eval)
    def __getitem__(self,idx):
        if self.train:
            im=io.imread(os.path.join(self.data_dir,self.cam_train[idx]))
            lb=np.load(os.path.join(self.label_idx_dir,self.cam_train[idx].split(".")[0]+'_L.png.npy'))
            sample={'image':im,'label':lb}
        else:
            im=io.imread(os.path.join(self.data_dir,self.cam_val[idx]))
            lb=np.load(os.path.join(self.label_idx_dir,self.cam_val[idx].split(".")[0]+'_L.png.npy'))
            sample={'image':im,'label':lb}
        if self.transform:
            sample=self.transform(sample)
        label=sample['label']    
        h_,w_=label.size()
        target=torch.zeros(n_classes,h_,w_)
        for c in range(n_classes):
            target[c][label==c]=1
        sample['target']=target
        
        return sample
    
    
        


# In[17]:


class resize(object):
    def __init__(self,train=True):
        self.train=train
    def __call__(self,sample):
        image=sample['image']
        label=sample['label']
        
        if self.train:
            image=trans.resize(image,(train_h,train_w))
            label=trans.resize(label,(train_h,train_w))
        else:
            image=trans.resize(image,(val_h,val_w))
            label=trans.resize(label,(val_h,val_w))
        sample={'image':image,'label':label}
        return sample    

class ToTensor(object):
    def __call__(self,sample):
        image,label=sample['image'],sample['label']
        t=transform.ToTensor()
        image=t(image)
        label=torch.from_numpy(label.copy()).long()
        return {'image':image,'label':label}

class Norm(object):
    def __init__(self,mean,std):
        self.mean=mean
        self.std=std
    def __call__(self,sample):
        t=transform.Normalize(self.mean,self.std)
        image,label=sample['image'],sample['label']
        im=t(image)
        return {'image':im,'label':label}
        
        


# In[18]:


mean=np.array([123.68,103.939,116.779])/255
std=[1,1,1]
tsfm1=transform.Compose([resize(train=True),ToTensor(),Norm(mean,std)])
tsfm2=transform.Compose([resize(train=False),ToTensor(),Norm(mean,std)])


# In[19]:


CamVid_train=camvid(data_dir,label_idx_dir,cam_train,cam_val,train=True,transform=tsfm1)
CamVid_val=camvid(data_dir,label_idx_dir,cam_train,cam_val,train=False,transform=tsfm2)


# In[20]:


train_loader=DataLoader(CamVid_train,batch_size=batch_size,shuffle=True)
eval_loader=DataLoader(CamVid_val,batch_size=batch_size,shuffle=True)


# In[21]:


vgg_model=VGGNet(pretrained=True,model='vgg16',requires_grad=True,remove_fc=True)
fcn_model=FCNs(n_classes=n_classes,pretrained_net=vgg_model)


# In[22]:


criterion=nn.BCEWithLogitsLoss()
optimizer=optim.RMSprop(fcn_model.parameters(),lr=lr,momentum=0,weight_decay=weight_decay)
scheduler=lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)


# In[23]:


def train():
    for epoch in range(epochs):
        scheduler.step()
        for iter,batch in enumerate(train_loader):
            optimizer.zero_grad()
            inputs=Variable(batch['image'])
            labels=Variable(batch['target'])
            outputs=fcn_model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            if i%10==0:
                print ("epoch{},iter{},loss:{}".format(epoch,iter,loss.data[0]))
        print ("Finish epoch:{}".format(epoch))
        torch.save(fcn_model,model_path)
        val(epoch)


# In[24]:


def val(epoch,crf=True):
    fcn_model.eval()
    total_ious=[]
    pixel_accs=[]
    for iter,batch in enumerate(eval_loader):
        inputs=Variable(batch['image'])
        output=fcn_model(inputs)
        output.data.cpu().numpy()
        if crf:
            crf_output = np.zeros(output.shape)
            images = inputs.data.cpu().numpy().astype(np.uint8)
            for i, (image, prob_map) in enumerate(zip(images, output)):
                image = image.transpose(1, 2, 0)
                crf_output[i] = dense_crf(image, prob_map)
            output = crf_output
            
        N,_,h,w=output.shape
        pred=output.transpose(0,2,3,1).reshape(-1,n_class).argmax(axis=1).reshape(N,h,w)
        target=batch['label'].cpu().numpy().reshape(N,h,w)
        
        for p,t in zip(pred,target):
            total_ious.append(iou(p,t))
            pixel_accs.append(pixel_acc(p,t))
        #calculate average IOUs
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    IU_scores[epoch] = ious
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    pixel_scores[epoch] = pixel_accs
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)


# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total    


# In[ ]:


val(0)
train()

