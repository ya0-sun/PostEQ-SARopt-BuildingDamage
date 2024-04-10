import torch
from torchvision import models

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
    

class MODEL_SAR(torch.nn.Module):
    '''
    late fusion model class for SAR and SARftp.
    '''
    def __init__(self,pretrain=None):
        super(MODEL_SAR, self).__init__()
        self.model_sar = models.resnet18()
        self.model_sar.fc = torch.nn.Linear(512,128)
        self.model_ftp = models.resnet18(pretrained=True)
        self.model_ftp.fc = torch.nn.Linear(512,128)
        
        if pretrain is not None:
            ckpt = torch.load(pretrain)
            del ckpt['fc.weight']
            del ckpt['fc.bias']
            msg = self.model_sar.load_state_dict(ckpt, strict=False)
            print(msg)
        
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256,128),
            torch.nn.Dropout(),
            torch.nn.Linear(128,1)
        )
        
    def forward(self,sar,ftp):
        x_sar = self.model_sar(sar)
        x_ftp = self.model_ftp(ftp)
        x = torch.cat((x_sar,x_ftp),1)
        x = self.fc(x)
        return x
    

class MODEL_OPT(torch.nn.Module):
    '''
    A late fusion model that combines OPT, OPTftp.
    '''
    def __init__(self,pretrain=None):
        super(MODEL_OPT, self).__init__()
        
        if pretrain is not None:
            self.pretrained = True 
        else:
            self.pretrained = False
        self.model_opt = models.resnet18(pretrained=self.pretrained)
        self.model_opt.fc = torch.nn.Linear(512,128)
        self.model_ftp = models.resnet18(pretrained=self.pretrained)
        self.model_ftp.fc = torch.nn.Linear(512,128)
        
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256,128),
            torch.nn.Dropout(),
            torch.nn.Linear(128,1)
        )
        
    def forward(self,opt,ftp):
        x_opt = self.model_opt(opt)
        x_ftp = self.model_ftp(ftp)
        x = torch.cat((x_opt,x_ftp),1)
        x = self.fc(x)
        return x