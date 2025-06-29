import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from torch.autograd import Variable
import torchvision.utils as vutils
#from point_matcher import HungarianMatcher
# ----------------------------------------------------------------
#from LibMTL.weighting import Aligned_MTL as Aligned_MTL
# ----------------------------------------------------------------
import logging

def img2windowsCHW(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)
    return img_perm
def windows2imgCHW(img_splits_hw,B, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' C H W
    """

    img = img_splits_hw.reshape(B, -1, H // H_sp, W // W_sp, H_sp, W_sp)
    img = img.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H, W)
    return img

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=True)

        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=True)

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        functions = [self._ce_forward] * \
                    (len(weights) - 1) + [self._ohem_forward]
        return sum([
            w * func(x, target)
            for (w, x, func) in zip(weights, score, functions)
        ])

class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            # 若列表或者元组，每一个计算loss并且取平均
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                new = torch.zeros_like(pred)
                mask = pred > 0.5
                new[mask] = pred[mask]
                pred = new
                smooth = 1
                intersection = pred * gt_masks
                loss = (intersection.sum()+smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            new = torch.zeros_like(pred)
            mask = pred > 0.5
            new[mask] = pred[mask]
            pred = new
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            # 若列表或者元组，每一个计算loss并且取平均
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                new = torch.zeros_like(pred)
                mask = pred > 0
                new[mask] = pred[mask]
                pred = new
                smooth = 1
                intersection = pred * gt_masks
                loss = (intersection.sum()+smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            # new = torch.zeros_like(pred)
            # mask = pred > 0.5
            # new[mask] = 1.0
            # pred = new
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss


class nIoULoss(nn.Module):
    def __init__(self):
        super(nIoULoss, self).__init__()
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            # 若列表或者元组，每一个计算loss并且取平均
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                new = torch.zeros_like(pred)
                mask = pred > 0
                new[mask] = pred[mask]
                pred = new
                smooth = 1e-7

                intersection = pred * (mask == gt_masks) # TP
                union = pred + gt_masks - intersection
                loss = (intersection+smooth) / (union + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            new = torch.zeros_like(pred)
            mask = pred > 0.5
            new[mask] = 1.0
            pred = new
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss

class nSoftIoULoss(nn.Module):
    def __init__(self):
        super(nSoftIoULoss, self).__init__()
    def get_weight(self,x):
        # return (0.5+(x*(1-x)))*x
        return x

    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            # 若列表或者元组，每一个计算loss并且取平均
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]

                smooth = 1e-3
                intersection = pred * gt_masks

                A = self.get_weight(pred - intersection)
                B = self.get_weight(gt_masks -intersection)

                loss = (A.sum()+B.sum()+smooth) / (A.sum() + B.sum() + intersection.sum() + smooth)
                loss = loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds

            smooth = 1e-11
            intersection = pred * gt_masks
            batchsz = pred.shape[0]
            A =(pred - intersection).reshape([batchsz,-1])
            B =(gt_masks - intersection).reshape([batchsz,-1])
            intersection = intersection.reshape([batchsz,-1])
            loss = (A + B).sum(dim = -1) / (A + B + intersection).sum(dim = -1) + smooth

            loss = (loss * loss).sum() / preds.shape[0]
            return loss


class FocalSoftIoULoss(nn.Module):
    def __init__(self):
        super(FocalSoftIoULoss, self).__init__()
    def get_weight(self,x):
        # return (0.5+(x*(1-x)))*x
        return x

    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            # 若列表或者元组，每一个计算loss并且取平均
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]

                smooth = 1e-3
                intersection = pred * gt_masks

                A = self.get_weight(pred - intersection)
                B = self.get_weight(gt_masks -intersection)

                loss = (A.sum()+B.sum()+smooth) / (A.sum() + B.sum() + intersection.sum() + smooth)
                loss = loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds

            smooth = 1e-3
            intersection = pred * gt_masks
            batchsz = pred.shape[0]
            A = self.get_weight(pred - intersection).reshape([batchsz,-1])
            B = self.get_weight(gt_masks - intersection).reshape([batchsz,-1])
            intersection = intersection.reshape([batchsz,-1])
            loss = (A.sum(dim = -1) + B.sum(dim = -1) + smooth) / (A.sum(dim = -1) + B.sum(dim = -1) + intersection.sum(dim = -1) + smooth)
            T=5.6
            loss = loss * (loss/T).softmax(dim=-1)
            loss = loss.sum()
            return loss

class FocalSoftIoULoss1(nn.Module):
    def __init__(self):
        super(FocalSoftIoULoss1, self).__init__()
    def get_weight(self,x):
        # return (0.5+(x*(1-x)))*x
        return x

    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            # 若列表或者元组，每一个计算loss并且取平均
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                new = torch.zeros_like(pred)
                mask = pred > 0.5
                new[mask] = pred[mask]
                pred = new
                smooth = 1e-3
                intersection = pred * gt_masks

                A = self.get_weight(pred - intersection)
                B = self.get_weight(gt_masks -intersection)

                loss = (A.sum()+B.sum()+smooth) / (A.sum() + B.sum() + intersection.sum() + smooth)
                loss = loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            new = torch.zeros_like(pred)
            mask = pred > 0.5
            new[mask] = pred[mask]
            pred = new
            smooth = 1e-3
            intersection = pred * gt_masks
            batchsz = pred.shape[0]
            A = self.get_weight(pred - intersection).reshape([batchsz,-1])
            B = self.get_weight(gt_masks - intersection).reshape([batchsz,-1])
            intersection = intersection.reshape([batchsz,-1])
            loss = (A.sum(dim = -1) + B.sum(dim = -1) + smooth) / (A.sum(dim = -1) + B.sum(dim = -1) + intersection.sum(dim = -1) + smooth)
            loss = loss * (loss / loss.sum())
            loss = loss.sum()
            return loss

class FocalHardIoULoss1(nn.Module):
    def __init__(self):
        super(FocalHardIoULoss1, self).__init__()
    def get_weight(self,x):
        # return (0.5+(x*(1-x)))*x
        return x

    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            # 若列表或者元组，每一个计算loss并且取平均
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                new = torch.zeros_like(pred)
                mask = pred > 0.5
                new[mask] = pred[mask]
                pred = new
                smooth = 1e-5
                intersection = pred * gt_masks

                A = self.get_weight(pred - intersection)
                B = self.get_weight(gt_masks -intersection)

                loss = (A.sum()+B.sum()+smooth) / (A.sum() + B.sum() + intersection.sum() + smooth)
                loss = loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            new = torch.zeros_like(pred)
            mask = pred > 0.5
            new[mask] = 1.0
            pred = new
            smooth = 1e-7
            intersection = pred * gt_masks
            batchsz = pred.shape[0]
            A = self.get_weight(pred - intersection).reshape([batchsz, -1])
            B = self.get_weight(gt_masks - intersection).reshape([batchsz, -1])
            intersection = intersection.reshape([batchsz, -1])
            loss = (A.sum(dim=-1) + B.sum(dim=-1) + smooth) / (
                        A.sum(dim=-1) + B.sum(dim=-1) + intersection.sum(dim=-1) + smooth)
            T = 9.7
            loss = loss * (loss / T).softmax(dim=-1)
            loss = loss.sum()
            return loss

#(pred-intersection).sum() + (gt_masks -intersection).sum() / (pred-intersection).sum() + (gt_masks.sum() -intersection.sum())+intersection.sum()

class AdditionMSELoss(nn.Module):
    def __init__(self):
        super(AdditionMSELoss, self).__init__()
        self.MSEloss = nn.MSELoss(reduction='sum')
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            # 若列表或者元组，每一个计算loss并且取平均
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                AmB = (pred - gt_masks).clamp(0,1)
                BmA = (gt_masks - pred).clamp(0,1)

                loss = (self.MSEloss(AmB,BmA)) / (AmB.sum()+BmA.sum()+1)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            AmB = (pred - gt_masks).clamp(0, 1)
            BmA = (gt_masks - pred).clamp(0, 1)
            loss = (self.MSEloss(AmB, BmA))/ (AmB.sum()+BmA.sum()+1)
            return loss


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


class ISNetLoss(nn.Module):
    def __init__(self):
        super(ISNetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss()
        self.grad = Get_gradient_nopadding()
        
    def forward(self, preds, gt_masks):
        edge_gt = self.grad(gt_masks.clone())
        
        ### img loss
        loss_img = self.softiou(preds[0], gt_masks)
        
        ### edge loss
        loss_edge = 10 * self.bce(preds[1], edge_gt)+ self.softiou(preds[1].sigmoid(), edge_gt)
        
        return loss_img + loss_edge

# def dice_loss(y_true, y_pred, smooth=1.0):
#     intersection = torch.sum(y_true * y_pred)
#     dice = (2.0 * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
#     loss = 1.0 - dice
#     return loss.mean()



class BCEFocalLoss(nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2.3, alpha=0.00001, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class pixelsumLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.bceloss = nn.BCELoss(reduction='mean')

    def forward(self, input, target):
        b,_ , h, w = input.shape
        input = input.reshape(b,-1)
        target = target.reshape(b,-1)
        p_in = torch.sum(input,dim=1).cuda() / (torch.ones([b,1]).cuda()*(h*w))
        p_ta = torch.sum(target, dim=1).cuda() / (torch.ones([b, 1]).cuda() * (h * w))
        loss = self.bceloss(p_in,p_ta)
        return loss

class SwinUnetLoss(nn.Module):
    def __init__(self):
        super(SwinUnetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        #self.bce = nn.BCELoss(reduction='mean')
        #self.psloss = pixelsumLoss()
    def forward(self, pred, gt_masks):
        b,_,_,_ = pred.shape
        ### img loss
        loss_img = self.softiou(pred, gt_masks)
        #loss_ps = self.psloss(pred, gt_masks)
        ### edge loss
        #print(loss_focal,loss_img + 3*loss_focal)
        #loss_bce = dice_loss(pred, gt_masks)
        #loss_bce = self.bce(pred.reshape(b,-1), gt_masks.reshape(b,-1))
        #print(loss_img, loss_bce)
        loss_dice = dice_loss(pred.reshape(b, -1), gt_masks.reshape(b, -1))
        return loss_img

from torch.utils.tensorboard import SummaryWriter
from mmcv.ops.point_sample import point_sample
from mmdet.models.utils.point_sample import get_uncertain_point_coords_with_randomness
class SearchNetLoss1(nn.Module):
    def __init__(self):
        super(SearchNetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss(reduction='mean')
        #self.psloss = pixelsumLoss()
        self.bce_sum = nn.BCELoss(reduction='sum')
        # 512*512 --> 2*2
        self.pool2 = nn.MaxPool2d(kernel_size=256, stride=256)
        # 512*512 --> 4*4
        self.pool4 = nn.MaxPool2d(kernel_size=128, stride=128)
        # 512*512 --> 8*8   256*256 --> 4*4
        self.pool8 = nn.MaxPool2d(kernel_size=64, stride=64)
        # 512*512 --> 16*16 256*256 --> 8*8
        self.pool16 = nn.MaxPool2d(kernel_size=32, stride=32)
        #                   256*256 --> 16*16
        self.pool32 = nn.MaxPool2d(kernel_size=16, stride=16)
        #                   256*256 --> 32*32
        self.pool64 = nn.MaxPool2d(kernel_size=8, stride=8)

        self.pool128 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.layer_lvl_num = 3

        self.threshold = 0.45

        #self.weighting = Aligned_MTL()
        self.grad = Get_gradient_nopadding()

        self.writer = SummaryWriter('K:\\BasicIRSTD-main\\tensorboard')



    def forward(self, pred, gt_masks, idx_iter, idx_epoch):
        pred_,weighted_mask_1, weighted_mask_2, weighted_mask_3 = pred
        b,_,_,_ = pred_.shape

        loss_img = self.softiou(pred_, gt_masks)
        #print(pred_)
        loss_search = 0
        for i in range(self.layer_lvl_num-1):
            loss_search+=self.bce(self.pool4(weighted_mask_1[i]).reshape(b, -1),self.pool2(gt_masks).reshape(b, -1))
            loss_search += self.bce(self.pool8(weighted_mask_2[i]).reshape(b, -1), self.pool4(gt_masks).reshape(b, -1))
            loss_search += self.bce(self.pool16(weighted_mask_3[i]).reshape(b, -1), self.pool8(gt_masks).reshape(b, -1))

        vutils.save_image(self.pool64(weighted_mask_3[-1]), f"output/z_output_pool_{idx_iter}.png")

        #local = torch.tensor(self.pool64(weighted_mask_3[-1])>self.threshold,dtype=float)
        local = self.pool16(gt_masks)
        mask = F.interpolate(
            local,
            pred_.shape[-2:],
            mode='bicubic',
            align_corners=True)

        masked_pred = torch.masked_select(pred_, mask>self.threshold)
        masked_gt = torch.masked_select(gt_masks, mask>self.threshold)

        #loss_local = self.bce(masked_pred, masked_gt)
        loss_local = self.bce(masked_pred, masked_gt)
        #vutils.save_image(torch.tensor(mask>self.threshold,dtype=float), f"output/z_output_mask_{idx_iter}.png")

        #edge
        edge_gt = self.grad(gt_masks.clone())

        ### edge loss
        loss_edge =self.softiou(self.grad(pred_.clone()), edge_gt)


        # weight = self.weighting(loss_img,loss_search,loss_local)
        # print(weight)

        if idx_epoch<0:
            loss_total = loss_img + loss_search
        else:
            #print(loss_local)
            loss_total = loss_img + 0.5 * loss_search +  loss_local + loss_edge
            # loss_total = loss_img + loss_search
            # loss_total = loss_img + 0.5 * loss_search + 5 * loss_local + loss_edge

        iter_ = idx_epoch * 80 + idx_iter + 1
        self.writer.add_scalar("SoftIoULoss", loss_img,iter_)
        self.writer.add_scalar("SearchLoss", loss_search, iter_)
        self.writer.add_scalar("LocalLoss", loss_local, iter_)
        self.writer.add_scalar("EdgeLoss", loss_edge, iter_)
        self.writer.add_scalar("TotalLoss", loss_total, iter_)


        if idx_epoch == 400:
            self.writer.close()
        return loss_total


class FocalLoss(nn.Module):

    def __init__(self, gamma=5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        # input:size is M*2. M　is the batch　number
        # target:size is M.
        pt = torch.softmax(input, dim=1)
        p = pt[:, 1]
        loss = -self.alpha * (1 - p) ** self.gamma * (target * torch.log(p)) - \
               (1 - self.alpha) * p ** self.gamma * ((1 - target) * torch.log(1 - p))
        return loss.mean()


class VFLoss(nn.Module):
    def __init__(self, loss_fcn=nn.BCEWithLogitsLoss(reduction='mean',pos_weight=torch.tensor(10)), gamma=1.5, alpha=0.25):
        super(VFLoss, self).__init__()
        # 传递 nn.BCEWithLogitsLoss() 损失函数  must be nn.BCEWithLogitsLoss()
        self.loss_fcn = loss_fcn  #
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply VFL to each element

    def forward(self, pred, true):

        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits

        focal_weight = true * (true > 0.0).float() + self.alpha * (pred_prob - true).abs().pow(self.gamma) * (
                    true <= 0.0).float()
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean() / num_masks



class SplitIoULoss(nn.Module):
    def __init__(self,size):
        super(SplitIoULoss, self).__init__()
        self.size = size
    def forward(self, pred, gt_masks):
        pred = img2windowsCHW(pred,self.size,self.size)
        gt_masks = img2windowsCHW(gt_masks,self.size,self.size)
        smooth = 1e-7
        intersection = pred * gt_masks
        intersection = intersection.sum(dim=0)
        pred = pred.sum(dim=0)
        gt_masks = gt_masks.sum(dim=0)
        loss = intersection / (pred+gt_masks-intersection+smooth) #b维度上有一部分为iou为0，不计算这部分的loss
        loss = loss[loss>0].mean()
        loss = 1 - loss
        return loss


class SearchNetLoss(nn.Module):
    def __init__(self):
        super(SearchNetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        # self.softiou = FocalSoftIoULoss()
        # self.softiou32 = SplitIoULoss(32)
        self.softiou64 = SplitIoULoss(64)
        self.iou = IoULoss()
        #self.iou = FocalHardIoULoss1()
        self.amse = AdditionMSELoss()

        self.bce = nn.BCEWithLogitsLoss(reduction='mean',pos_weight=torch.tensor(10))
        #self.psloss = pixelsumLoss()
        self.bce_sum = nn.BCELoss(reduction='sum')
        # 512*512 --> 2*2
        self.pool2 = nn.MaxPool2d(kernel_size=256, stride=256)
        # 512*512 --> 4*4
        self.pool4 = nn.MaxPool2d(kernel_size=128, stride=128)
        # 512*512 --> 8*8   256*256 --> 4*4
        self.pool8 = nn.MaxPool2d(kernel_size=64, stride=64)
        # 512*512 --> 16*16 256*256 --> 8*8
        self.pool16 = nn.MaxPool2d(kernel_size=32, stride=32)
        #                   256*256 --> 16*16
        self.pool32 = nn.MaxPool2d(kernel_size=16, stride=16)
        #                   256*256 --> 32*32
        self.pool64 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.pool128 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool256 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer_lvl_num = 2

        self.threshold = 0.45

        #self.weighting = Aligned_MTL()
        self.grad = Get_gradient_nopadding()
        #self.vfocal = VFLoss()
        self.writer = SummaryWriter('K:\\BasicIRSTD-main\\tensorboard')

        self.up2x = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
        self.up4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16x = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        #self.point_loss = HungarianMatcher(num_points=200)
        # self.lovaze = LovaszSoftmax()


    def forward(self, pred, gt_masks, idx_iter, idx_epoch,img):
        pred_, thresh, weighted_mask_1, weighted_mask_2, weighted_mask_3,x_4_3,edge= pred
        b,_,_,_ = pred_.shape
        # with torch.no_grad():
        #     bipred = torch.where(pred_ > thresh,torch.ones_like(pred_),torch.zeros_like(pred_))
        # pred_ = pred_ * bipred
        # loss_search = 0
        # for i in range(self.layer_lvl_num-1):
        #     loss_search += self.bce(self.pool4(weighted_mask_1[i]).reshape(b, -1),self.pool2(gt_masks).reshape(b, -1))
        #     loss_search += self.bce(self.pool8(weighted_mask_2[i]).reshape(b, -1), self.pool4(gt_masks).reshape(b, -1))
        #     loss_search += self.bce(self.pool16(weighted_mask_3[i]).reshape(b, -1), self.pool8(gt_masks).reshape(b, -1))
        # loss_search /= 3

        loss_mse = self.amse(pred_, gt_masks)
        loss_point = self.point_loss(pred_, gt_masks)
        # loss_img = self.iou(pred_, gt_masks)
        loss_dice = dice_loss(pred_, gt_masks.flatten(1),1)
        # loss_lovaze = self.lovaze(pred_ > 0.5, gt_masks)
        #loss_img = structure_loss(pred_,gt_masks)
        # edge_gt =  self.grad(gt_masks.clone())
        loss_edge = 0
        # for i in [0,1]:
        #     loss_edge += self.softiou(self.up2x(weighted_mask_1[i]), edge_gt)
        #     loss_edge += self.softiou(self.up2x(weighted_mask_2[i]), edge_gt)
        #     loss_edge += self.softiou(self.up2x(weighted_mask_3[i]), edge_gt)
        # loss_edge =  self.softiou(edge, edge_gt)

        loss_iou = self.iou(pred_, gt_masks)
        loss_features = 0
        # loss_features +=self.softiou(self.up8x(features[0]), gt_masks)
        # loss_features += self.softiou(self.up4x(features[1]), gt_masks)
        # loss_features += self.softiou(self.up2x(features[2]), gt_masks)
        # loss_features += self.bce(self.up8x(features[0]).sigmoid(), gt_masks)
        # loss_features += self.bce(self.up4x(features[1]).sigmoid(), gt_masks)
        # loss_features += self.bce(self.up2x(features[2]).sigmoid(), gt_masks)
        #
        # loss_features /= 3

        #
        # loss_search = self.bce(x_4_3.sigmoid(),self.pool16(gt_masks))

        # local = self.pool16(gt_masks)
        # mask = F.interpolate(
        #     local,
        #     pred_.shape[-2:],
        #     mode='bicubic',
        #     align_corners=True)
        #
        # masked_pred = torch.masked_select(pred_, mask>self.threshold)
        # masked_gt = torch.masked_select(gt_masks, mask>self.threshold)
        # loss_local = self.bce(masked_pred, masked_gt)

        #loss_local = self.bce(masked_pred, masked_gt)
        #vutils.save_image(torch.tensor(mask>self.threshold,dtype=float), f"output/z_output_mask_{idx_iter}.png")
        ### edge loss
        #loss_edge =self.softiou(self.grad(pred_.clone()), edge_gt)


        # weight = self.weighting(loss_img,loss_search,loss_local)
        # print(weight)

        #loss_total = loss_img + loss_search + loss_edge + loss_features
        # loss_total = 15*loss_mse + 1.5*loss_img
        # loss_total = 10*loss_mse+loss_point + loss_iou
        loss_total = loss_iou
        # print(loss_total)
        # loss_total = loss_point  + loss_iou + 5*loss_mse
        # loss_total = loss_img + loss_search
        # loss_total = 10 * loss_mse + loss_iou + loss_img +loss_point
        # loss_total = loss_img + 0.5 * loss_search + 5 * loss_local + loss_edge


        # vutils.save_image(gt_masks, f"output/gtmask_{idx_iter}.png")
        # vutils.save_image(pred_, f"output/output_{idx_iter}.png")
        # if idx_epoch > 1498:
        #     vutils.save_image(edge, f"output/z_output_edge_{idx_iter}_{loss_img}.png")
        #     vutils.save_image(pred_, f"output/output_{idx_iter}_{loss_img}.png")
        #     vutils.save_image(img, f"output/img_{idx_iter}_{loss_img}.png")
        iter_ = idx_epoch * 80 + idx_iter + 1
        self.writer.add_scalar("SoftIoULoss", loss_iou,iter_)
        # self.writer.add_scalar("SearchLoss", loss_search, iter_)
        self.writer.add_scalar("FeatureLoss", loss_dice, iter_)
        # self.writer.add_scalar("PointLoss", loss_point, iter_)
        self.writer.add_scalar("TotalLoss", loss_total, iter_)


        if idx_epoch == 400:
            self.writer.close()
        return loss_total


class RepirLoss(nn.Module):
    def __init__(self):
        super(RepirLoss, self).__init__()
        self.softiou = SoftIoULoss()
        # self.softiou = FocalSoftIoULoss()
        # self.softiou32 = SplitIoULoss(32)
        self.softiou64 = SplitIoULoss(64)
        self.iou = IoULoss()
        #self.iou = FocalHardIoULoss1()
        self.amse = AdditionMSELoss()
        self.l1 = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean',pos_weight=torch.tensor(10))
        #self.psloss = pixelsumLoss()
        self.bce_sum = nn.BCELoss(reduction='sum')
        # 512*512 --> 2*2
        self.pool2 = nn.MaxPool2d(kernel_size=256, stride=256)
        # 512*512 --> 4*4
        self.pool4 = nn.MaxPool2d(kernel_size=128, stride=128)
        # 512*512 --> 8*8   256*256 --> 4*4
        self.pool8 = nn.MaxPool2d(kernel_size=64, stride=64)
        # 512*512 --> 16*16 256*256 --> 8*8
        self.pool16 = nn.MaxPool2d(kernel_size=32, stride=32)
        self.down16 = nn.Upsample(scale_factor=1/32,mode='bilinear')
        #                   256*256 --> 16*16
        self.pool32 = nn.MaxPool2d(kernel_size=16, stride=16)
        #                   256*256 --> 32*32
        self.pool64 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.pool128 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool256 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer_lvl_num = 2

        self.threshold = 0.45

        #self.weighting = Aligned_MTL()
        self.grad = Get_gradient_nopadding()
        #self.vfocal = VFLoss()
        self.writer = SummaryWriter('tensorboard')

        self.up2x = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
        self.up4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16x = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        # self.point_loss = HungarianMatcher(num_points=200)
        # self.lovaze = LovaszSoftmax()
        self.focal = nn.BCELoss()
        self.nsoft = nSoftIoULoss()

    def forward(self, pred, gt_masks, idx_iter, idx_epoch,img):
        pred_, classify_result= pred
        b,_,_,_ = pred_.shape

        with torch.no_grad():
            classify_gt = self.pool256(gt_masks)
        loss_class = self.iou(classify_result,classify_gt)
        # loss_class = self.softiou(classify_result,gt_masks)
        # loss_mse = self.amse(pred_, gt_masks)
        # loss_point = self.point_loss(pred_, gt_masks)
        # loss_dice = dice_loss(pred_, gt_masks.flatten(1),1)


        loss_iou = self.iou(pred_, gt_masks)

        # loss_total = idx_epoch/400 * loss_iou + (400 - idx_epoch)/400 * loss_class
        loss_total =  loss_iou + 0.8 * loss_class


        return loss_total



class EOLoss(nn.Module):
    def __init__(self):
        super(EOLoss, self).__init__()
        self.softiou = SoftIoULoss()
        # self.softiou = FocalSoftIoULoss()
        # self.softiou32 = SplitIoULoss(32)
        self.softiou64 = SplitIoULoss(64)
        self.iou = IoULoss()
        #self.iou = FocalHardIoULoss1()
        self.amse = AdditionMSELoss()

        self.bce = nn.BCELoss(reduction='mean')
        #self.psloss = pixelsumLoss()
        self.bce_sum = nn.BCELoss(reduction='sum')
        # 512*512 --> 2*2
        self.pool2 = nn.MaxPool2d(kernel_size=256, stride=256)
        # 512*512 --> 4*4
        self.pool4 = nn.MaxPool2d(kernel_size=128, stride=128)
        # 512*512 --> 8*8   256*256 --> 4*4
        self.pool8 = nn.MaxPool2d(kernel_size=64, stride=64)
        # 512*512 --> 16*16 256*256 --> 8*8
        self.pool16 = nn.MaxPool2d(kernel_size=32, stride=32)
        #                   256*256 --> 16*16
        self.pool32 = nn.MaxPool2d(kernel_size=16, stride=16)
        #                   256*256 --> 32*32
        self.pool64 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.pool128 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool256 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer_lvl_num = 2

        self.threshold = 0.45

        #self.weighting = Aligned_MTL()
        self.grad = Get_gradient_nopadding()
        #self.vfocal = VFLoss()
        self.writer = SummaryWriter('K:\\BasicIRSTD-main\\tensorboard')

        self.up2x = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
        self.up4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16x = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        # self.point_loss = HungarianMatcher(num_points=200)
        # self.lovaze = LovaszSoftmax()
        self.focal = nn.BCELoss()

    def forward(self, pred, gt_masks):
        pred_= pred
        b,_,_,_ = pred_.shape


        loss_class = self.bce(pred_,gt_masks)
        loss_iou = self.iou(pred_, gt_masks)

        loss_total = loss_iou

        return loss_total

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def __init__(self, reduction='mean'):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (target_c + (~input_c)).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        inputs, targets = self.prob_flatten(inputs, targets)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses


class net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(net, self).__init__()
        self.conv = nn.Conv3d(in_channels, num_classes, (1, 3, 3), padding=(0, 1, 1))

    def forward(self, input):
        out = self.conv(input)
        return out


def test():
    from torch.optim import Adam
    BS = 2
    num_classes = 8
    dim, hei, wid = 8, 64, 64
    data = torch.rand(BS, num_classes, dim, hei, wid)
    model = net(num_classes, num_classes)
    target = torch.zeros(BS, dim, hei, wid).random_(num_classes)
    Loss = LovaszSoftmax()
    optim = Adam(model.parameters(), lr=0.01,betas=(0.99,0.999))
    for step in range(1000):
        out = model(data)
        loss = Loss(out, target)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(loss)


class BiisLoss(nn.Module):
    def __init__(self):
        super(BiisLoss, self).__init__()
        self.softiou = SoftIoULoss()
        # self.softiou = FocalSoftIoULoss()
        # self.softiou32 = SplitIoULoss(32)
        self.softiou64 = SplitIoULoss(64)
        self.iou = IoULoss()
        #self.iou = FocalHardIoULoss1()
        self.amse = AdditionMSELoss()
        self.l1 = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean',pos_weight=torch.tensor(10))
        #self.psloss = pixelsumLoss()
        self.bce_sum = nn.BCELoss(reduction='sum')
        # 512*512 --> 2*2
        self.pool2 = nn.MaxPool2d(kernel_size=256, stride=256)
        # 512*512 --> 4*4
        self.pool4 = nn.MaxPool2d(kernel_size=128, stride=128)
        # 512*512 --> 8*8   256*256 --> 4*4
        self.pool8 = nn.MaxPool2d(kernel_size=64, stride=64)
        # 512*512 --> 16*16 256*256 --> 8*8
        self.pool16 = nn.MaxPool2d(kernel_size=32, stride=32)
        self.down16 = nn.Upsample(scale_factor=1/32,mode='bilinear')
        #                   256*256 --> 16*16
        self.pool32 = nn.MaxPool2d(kernel_size=16, stride=16)
        #                   256*256 --> 32*32
        self.pool64 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.pool128 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool256 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer_lvl_num = 2

        self.threshold = 0.45

        #self.weighting = Aligned_MTL()
        self.grad = Get_gradient_nopadding()
        #self.vfocal = VFLoss()
        self.writer = SummaryWriter('K:\\BasicIRSTD-main\\tensorboard')

        self.up2x = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
        self.up4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16x = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        # self.point_loss = HungarianMatcher(num_points=200)
        # self.lovaze = LovaszSoftmax()
        self.focal = nn.BCELoss()
        self.nsoft = nSoftIoULoss()
        self.cos = nn.PairwiseDistance()

    def forward(self, pred, gt_masks,):
        pred_,vq_loss= pred
        b,_,_,_ = pred_.shape


        # with torch.no_grad():
        #     mask = self.pool256(self.pool256(self.pool256(gt_masks)))
        # c = origin.shape[1]
        # dis = 0
        # for i in range(b):
        #     dis_t = 0
        #     for j in range(proto.shape[0]):
        #         target_feat = F.normalize(origin[i,:,:,:][(mask[i]>0).repeat([c,1,1])].reshape([c,-1]),dim=0)
        #         dis_t += torch.sum(self.cos(target_feat,proto[j].unsqueeze(-1).repeat([1,target_feat.shape[-1]])))
        #     dis_t /= proto.shape[0]
        #     dis += dis_t
        # dis /= b
        # # dis = self.cos(,)


        loss_iou = self.iou(pred_, gt_masks)

        # loss_total = idx_epoch/400 * loss_iou + (400 - idx_epoch)/400 * loss_class
        loss_total =  loss_iou + 0.5*vq_loss


        return loss_total

if __name__ == '__main__':
    test()