from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
import os
from loss import *
from model import *
from skimage.feature.tests.test_orb import img
import torchinfo
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        
        self.cal_loss = SoftIoULoss()
        if model_name == 'DNANet':
            if mode == 'train':
                self.model = DNANet(mode='train')
            else:
                self.model = DNANet(mode='test')
        # elif model_name == 'ACM':
        #     self.model = ACM()
        # elif model_name == 'ALCNet':
        #     self.model = ALCNet()
        # elif model_name == 'ISNet':
        #     if mode == 'train':
        #         self.model = ISNet(mode='train')
        #     else:
        #         self.model = ISNet(mode='test')
        #     self.cal_loss = ISNetLoss()
        # elif model_name == 'RISTDnet':
        #     self.model = RISTDnet()
        # elif model_name == 'UIUNet':
        #     if mode == 'train':
        #         self.model = UIUNet(mode='train')
        #     else:
        #         self.model = UIUNet(mode='test')
        # elif model_name == 'U-Net':
        #     self.model = Unet()
        # elif model_name == 'ISTDU-Net':
        #     self.model = ISTDU_Net()
        # elif model_name == 'RDIAN':
        #     self.model = RDIAN()
        # elif model_name == 'NestedSwinUnet':
        #     self.model = NestedSwinUnet()
        #     self.cal_loss = SwinUnetLoss()
        # elif model_name == 'SearchNet':
        #     self.model = SearchNet()
        #     self.cal_loss = SearchNetLoss()
        # elif model_name == 'IsDet':
        #     self.model = IsDet()
        # elif model_name == 'RepirDet':
        #     if mode == 'train':
        #         self.model = RepirDet(deploy=False, mode='train')
        #     else:
        #         self.model = RepirDet(deploy=False, mode='test')
        #     self.cal_loss = RepirLoss()
        # elif model_name == 'LineNet':
        #     self.model = LineNet()
        # elif model_name == 'ExtractOne':
        #     self.model = ExtractOne()
        #     self.cal_loss = EOLoss()
        # elif model_name == 'BNN':
        #     self.model = BNN()
        #     self.cal_loss = BiisLoss()
        # elif model_name == 'ABC':
        #     self.model = ABCNet()
        if model_name == 'LKUnet':
            self.model = LKUnet()
            # torchinfo.summary(self.model)
        elif model_name == "WTNet":
            self.model = WTNet()
            torchinfo.summary(self.model)
        elif model_name == "RepirDet":
            if mode == 'train':
                self.model = RepirDet(deploy=False, mode='train')
            else:
                self.model = RepirDet(deploy=False, mode='test')
            self.cal_loss = RepirLoss()
        elif model_name == "NoiseNet":
            self.model = NoiseNet()
            # torchinfo.summary(self.model)
        elif model_name == "LinearVIT":
            self.model = LinearVIT()
        elif model_name == "NoiseAttNet":
            self.model = NoiseAttNet()
        
    def forward(self, img,):
        return self.model(img)

    def loss(self, pred, gt_mask,idx_iter=None,idx_epoch=None,img=None):
        if self.model_name == 'RepirDet' or self.model_name == 'SearchNet':
            loss = self.cal_loss(pred, gt_mask , idx_iter,idx_epoch,img)
        else:
            loss = self.cal_loss(pred, gt_mask)
        return loss
