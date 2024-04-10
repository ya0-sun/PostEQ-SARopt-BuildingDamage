import torch
import numpy as np
import h5py
import os
import cv2
import kornia


class EarthquakeDataset(torch.utils.data.Dataset):
    '''
    Dataset class for earthquake building dataset.
    '''
    def __init__(self, root, splits=['fold-2.txt','fold-3.txt','fold-4.txt','fold-5.txt']):
        self.root = root
        self.splits = splits
        
        self.ids = []
        self.labels = []
        for split in splits:
            with open(os.path.join(root,split),'r') as f:
                for line in f.readlines():
                    fid = line.split(',')[0]
                    label = line.split(',')[1]
                    self.ids.append(fid)
                    self.labels.append(int(label))
                
        self.length = len(self.ids)
        
    def __getitem__(self,index):
        osmid = self.ids[index]
        label = np.float32(self.labels[index])
        
        sar_path = os.path.join(self.root,osmid+'_SAR.mat')
        sarftp_path = os.path.join(self.root,osmid+'_SARftp.mat')
        opt_path = os.path.join(self.root,osmid+'_opt.mat')
        optftp_path = os.path.join(self.root,osmid+'_optftp.mat')
        
        with h5py.File(sar_path, 'r') as f1:
            x1 = np.float32(f1['x1'])
            x1[x1<-100] = x1[x1>-100].min() # fill missing values with min of non-missing values
            x1 = cv2.resize(x1, dsize=(224, 224), interpolation=cv2.INTER_LINEAR_EXACT)
            p1,p99 = np.percentile(x1,(1,99))
            x1 = np.clip(x1,p1,p99)
            x1 = (x1-p1)/(p99-p1)
            x1 = np.stack((x1,x1,x1),0) # 3,224,224
        with h5py.File(sarftp_path, 'r') as f2:
            x2 = np.float32(f2['x2'])
            x2 = cv2.resize(x2, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)
            x2 = np.stack((x2,x2,x2),0) # 3,224,224
        with h5py.File(opt_path, 'r') as f3:
            x3 = np.float32(f3['x3'])
            x3 = cv2.resize(np.transpose(x3,(1,2,0)), dsize=(224, 224), interpolation=cv2.INTER_LINEAR_EXACT)
            x3 = np.transpose(x3,(2,0,1))
            p1,p99 = np.percentile(x3,(1,99))
            x3 = np.clip(x3,p1,p99)
            x3 = (x3-p1)/(p99-p1)
        with h5py.File(optftp_path, 'r') as f4:
            x4 = np.float32(f4['x4'])
            x4 = cv2.resize(x4, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)
            x4 = np.stack((x4,x4,x4),0) # 3,224,224
    
        images = {'sar':x1,'sarftp':x2,'opt':x3,'optftp':x4}        

        return images, label#, osmid
        
    def __len__(self):
        return self.length
    

class MyAug(torch.nn.Module):
    '''
    A custom augmentation module using Kornia.
    '''
    def __init__(self):
        super(MyAug,self).__init__()
        self.k1 = kornia.augmentation.RandomResizedCrop((224,224),scale=(0.8,1.0))
        self.k2 = kornia.augmentation.RandomHorizontalFlip(p=0.5)
        self.k3 = kornia.augmentation.RandomVerticalFlip(p=0.5)
        #self.k4 = kornia.augmentation.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5)

    def forward(self,sar,sarftp,opt,optftp):

        sar = self.k1(sar)
        sarftp = self.k1(sar,self.k1._params)
        opt = self.k1(opt,self.k1._params)
        optftp = self.k1(optftp,self.k1._params)

        sar = self.k2(sar)
        sarftp = self.k2(sar,self.k2._params)
        opt = self.k2(opt,self.k2._params)
        optftp = self.k2(optftp,self.k2._params)
        
        sar = self.k3(sar)
        sarftp = self.k3(sar,self.k3._params)
        opt = self.k3(opt,self.k3._params)
        optftp = self.k3(optftp,self.k3._params)
        
        #sar = self.k4(sar)
        #opt = self.k4(opt)

        return sar,sarftp,opt,optftp