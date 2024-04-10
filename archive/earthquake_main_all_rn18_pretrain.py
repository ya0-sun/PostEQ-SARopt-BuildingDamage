import argparse
import random
import os
import time
import torch
from torchvision import models
import numpy as np
import h5py
import cv2
import kornia
from metrics import compute_imagewise_retrieval_metrics,compute_imagewise_f1_metrics

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


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

    
class MODEL_MM(torch.nn.Module):
    '''
    A late fusion model that combines SAR, SARftp, OPT, OPTftp.
    '''
    def __init__(self,sar_pretrain=None,opt_pretrain=None):
        super(MODEL_MM, self).__init__()
        
        # sar model
        self.model_sar = models.resnet18()
        if sar_pretrain is not None:
            ckpt = torch.load(sar_pretrain)
            del ckpt['fc.weight']
            del ckpt['fc.bias']
            msg = self.model_sar.load_state_dict(ckpt, strict=False)
            #print(msg)
        self.model_sar.fc = torch.nn.Linear(512,128)
        
        # sarftp model
        self.model_sarftp = models.resnet18(pretrained=True)
        self.model_sarftp.fc = torch.nn.Linear(512,128)
        
        # opt model
        if opt_pretrain is not None:
            opt_pretrain = True
        self.model_opt = models.resnet18(pretrained=opt_pretrain)
        self.model_opt.fc = torch.nn.Linear(512,128)
        
        # optftp model
        self.model_optftp = models.resnet18(pretrained=True)
        self.model_optftp.fc = torch.nn.Linear(512,128)
        
        
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512,128),
            torch.nn.Dropout(),
            torch.nn.Linear(128,1)
        )
        
    def forward(self,sar,sarftp,opt,optftp):
        x_sar = self.model_sar(sar)
        x_sarftp = self.model_sarftp(sarftp)
        x_opt = self.model_opt(opt)
        x_optftp = self.model_optftp(optftp)
        x = torch.cat((x_sar,x_sarftp,x_opt,x_optftp),1)
        x = self.fc(x)
        return x
    

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--root", default='/data/earthquake_building_dataset/', type=str)
parser.add_argument("--val_split", default='fold-1.txt', type=str)
parser.add_argument("--checkpoints", default='checkpoints_all/fold1', type=str)
parser.add_argument("--sar_pretrain", default=None, type=str)
parser.add_argument("--opt_pretrain", default=None, type=str)
args = parser.parse_args()

## create dataset and dataloader
if 'fold-1.txt' == args.val_split:
    train_splits = ['fold-2.txt','fold-3.txt','fold-4.txt','fold-5.txt']
    val_split = ['fold-1.txt']
elif 'fold-2.txt' == args.val_split:
    train_splits = ['fold-1.txt','fold-3.txt','fold-4.txt','fold-5.txt']
    val_split = ['fold-2.txt']
elif 'fold-3.txt' == args.val_split:
    train_splits = ['fold-1.txt','fold-2.txt','fold-4.txt','fold-5.txt']
    val_split = ['fold-3.txt']
elif 'fold-4.txt' == args.val_split:
    train_splits = ['fold-1.txt','fold-2.txt','fold-3.txt','fold-5.txt']
    val_split = ['fold-4.txt']
elif 'fold-5.txt' == args.val_split:
    train_splits = ['fold-1.txt','fold-2.txt','fold-3.txt','fold-4.txt']
    val_split = ['fold-5.txt']

train_dataset = EarthquakeDataset(args.root,train_splits)
val_dataset = EarthquakeDataset(args.root,val_split)

print(len(train_dataset),len(val_dataset))

# class weighted data sampler
y_train = train_dataset.labels
class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in y_train])
sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler = sampler, pin_memory=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=False)


## create model, criterion, optimizer, scheduler
model = MODEL_MM(args.sar_pretrain, args.opt_pretrain)
model.cuda()
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)

## define augmentation
common_aug = MyAug().cuda()

## training loop
os.makedirs(args.checkpoints,exist_ok=True)
start_time = time.time()
best_f1 = 0
best_auroc = 0
best_epoch = 0
for epoch in range(args.epochs):
    # training
    model.train()
    train_loss = 0
    count = 0
    for i, data in enumerate(train_loader, 0):
        images = data[0]
        labels = data[1].cuda()
        sar,sarftp,opt,optftp = common_aug(images['sar'].cuda(),images['sarftp'].cuda(),images['opt'].cuda(),images['optftp'].cuda())
        optimizer.zero_grad()
        outputs = model(sar,sarftp,opt,optftp)
        loss = criterion(outputs.squeeze(1), labels.float())
        loss.backward()
        optimizer.step()
        count += 1
        train_loss += loss.item()
        if i%10==0:
            print(args.val_split, epoch, i, loss.item())
    
    # validation
    model.eval()
    out_all = []
    gt_all = []
    for i, data in enumerate(val_loader, 0):
        images = data[0]
        labels = data[1].cuda()
        sar,sarftp,opt,optftp = images['sar'].cuda(),images['sarftp'].cuda(),images['opt'].cuda(),images['optftp'].cuda()
        outputs = model(sar,sarftp,opt,optftp)
        out_all.append(outputs.detach().cpu())
        gt_all.append(labels.detach().cpu())
    out_all = torch.cat(out_all,0).squeeze(1)
    gt_all = torch.cat(gt_all,0)
    loss = criterion(out_all,gt_all.float()).item()
    out_all = torch.nn.Sigmoid()(out_all)

    f1_metrics = compute_imagewise_f1_metrics(out_all.numpy(),gt_all.numpy())
    auroc = compute_imagewise_retrieval_metrics(out_all.numpy(),gt_all.numpy())
    
    print(args.val_split, 'Epoch',epoch,'train loss {:.4f}'.format(train_loss/count),'val loss {:.4f}'.format(loss), 'auroc', auroc['auroc'],
          'f1',f1_metrics['f1'], 'precision',f1_metrics['precision'],'recall',f1_metrics['recall'], 
          'best_f1',f1_metrics['best_f1'], 'precision',f1_metrics['best_f1_precision'],'recall',f1_metrics['best_f1_recall'], 'threshold',f1_metrics['best_threshold']
          )
    
    if auroc['auroc']>best_auroc:
        best_f1 = f1_metrics['best_f1']
        best_auroc = auroc['auroc']
        best_epoch = epoch
        torch.save(model.state_dict(),os.path.join(args.checkpoints,"checkpoint_ep{:02d}.pth".format(epoch)))
        
print(args.val_split, 'best epoch',best_epoch, 'best_f1', best_f1, 'best_auroc', best_auroc, 'time', time.time()-start_time)
