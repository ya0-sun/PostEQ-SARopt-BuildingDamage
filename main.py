import argparse
import random
import os
import time
import torch
import numpy as np
from dataset import EarthquakeDataset, MyAug
from model import MODEL_MM, MODEL_SAR, MODEL_OPT
from metrics import compute_imagewise_retrieval_metrics,compute_imagewise_f1_metrics

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

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
parser.add_argument("--mode", default='all', type=str, choices=['all','sar','opt'])
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
if args.mode == 'sar':
    model = MODEL_SAR(args.sar_pretrain)
elif args.mode == 'opt':
    model = MODEL_OPT(args.opt_pretrain)
elif args.mode == 'all':
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
        if args.mode == 'sar':
            outputs = model(sar,sarftp)
        elif args.mode == 'opt':
            outputs = model(opt,optftp)
        elif args.mode == 'all':
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
