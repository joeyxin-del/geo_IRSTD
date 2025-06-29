import argparse
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import numpy as np
import os
from tqdm import tqdm
import torchvision.utils as vutils
from CalculateFPS import *

from torch.cuda.amp import autocast
use_amp = False

import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2


print("PYTHONPATH:", os.environ.get('PYTHONPATH'))
print("PATH:", os.environ.get('PATH'))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")

# =======================================model=========================================
# parser.add_argument("--model_names", default=['WTNet'], type=list,
# parser.add_argument("--model_names", default=['RepirDet'], type=list,
parser.add_argument("--model_names", default=['DNANet'], type=list,
                    help="model_name: Dataset-mask  'ACM', 'ALCNet', 'DNANet', 'ISNet', "
                         "'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet', 'NestedSwinUnet',"
                         "'SearchNet','RepirDet','ExtractOne'"
                         "'LKUnet', 'WTNet',")

# =======================================datasets============================
parser.add_argument("--dataset_names", default=['IRSTD-1K'], type=list,
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--dataset_dir", default='./datasets', type=str, help="train_dataset_dir")

# =====================================batchSize===========================================
parser.add_argument("--batchSize", type=int, default=2, help="Training batch sizse")
parser.add_argument("--patchSize", type=int, default=512, help="Training patch size")
parser.add_argument("--save", default='./log', type=str, help="Save path of checkpoints")
parser.add_argument("--resume", default=None, type=list, help="Resume from exisiting checkpoints (default: None) use ['path']")

# ======================================Epoch==============================================
parser.add_argument("--nEpochs", type=int, default=400, help="Number of epochs")
#=======================================optimizer==========================================
parser.add_argument("--optimizer_name", default='Adamw', type=str, help="optimizer name: Adam, Adagrad, SGD,Adamw")
#==========================================lr=============================================
parser.add_argument("--optimizer_settings", default={'lr': 5e-4}, type=dict, help="optimizer settings")
parser.add_argument("--scheduler_name", default='MultiStepLR', type=str, help="scheduler name: MultiStepLR")
parser.add_argument("--scheduler_settings", default={'step': [200, 300], 'gamma': 0.1}, type=dict,help="scheduler settings")

# =======================================
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--transform", type=bool, default=True, help="Augmentation for training")
parser.add_argument("--seed", type=int, default=42, help="Random Seed for test (i.e.44)")
parser.add_argument("--orth", type=bool, default=False, help="Orthogonal Regularization")
parser.add_argument("--inferenceFPS", type=bool, default=False, help="claculate FPS for inference")

global opt
opt = parser.parse_args()
seed_pytorch(opt.seed)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
bestiou=-1.0

def train():
    train_set = TrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize,
                               img_norm_cfg=opt.img_norm_cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    net = Net(model_name=opt.model_name, mode='train').cuda()

    net.train()

    #AMP
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    #EMA
    ema = EMA(net, 0.0)
    ema.register()

    #satis = MaskSatistical(opt.patchSize)

    epoch_state = 0
    total_loss_list = [0]
    total_loss_epoch = []

    if opt.resume:
        for resume_pth in opt.resume:
            print(resume_pth, ' is loaded')
            if opt.model_name in resume_pth:
                print(resume_pth,' is loaded')
                ckpt = torch.load(resume_pth)
                net.load_state_dict(ckpt['state_dict'])
                epoch_state = ckpt['epoch']
                total_loss_list = ckpt['total_loss']
                # for i in range(len(opt.step)):
                #     opt.step[i] = opt.step[i] - ckpt['epoch']
    resume = False


    resume_temp = opt.save + '/' + opt.dataset_name + '/' + opt.model_name+ '_best_m' + '.pth'
    # resume_temp = 'K:\BasicIRSTD-main\log\IRSTD-1K\LinearVitR_1_69.pth'
    if resume:
        resume_pth=resume_temp
        print(resume_pth, ' is loaded')
        if opt.model_name in resume_pth:
            print(resume_pth,' is loaded')
            ckpt = torch.load(resume_pth)
            net.load_state_dict(ckpt['state_dict'],strict=True)
            # epoch_state = ckpt['epoch']
            total_loss_list = ckpt['total_loss']


    ### Default settings
    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': 5e-4 * 4}
        opt.scheduler_name = 'MultiStepLR'
        opt.scheduler_settings = {'epochs':400, 'step': [200, 300], 'gamma': 0.1}

    if opt.optimizer_name == 'Adamw':
        #opt.optimizer_settings = {'lr': 0.0007}
        opt.optimizer_settings['lr'] = 0.0015
        # opt.scheduler_name = 'MultiStepLR'
        # opt.scheduler_settings = {'epochs': 400, 'step': [80, 280], 'gamma': 0.3333}
        opt.scheduler_name = 'CosineAnnealingLR'
        # # opt.scheduler_name = 'CyclicLR'
        opt.scheduler_settings['epochs'] = 800
        opt.scheduler_settings['min_lr'] = 0.0005

    if opt.optimizer_name == 'Lion':
        #opt.optimizer_settings = {'lr': 0.0007}
        opt.optimizer_settings['lr'] = 0.00015
        # opt.scheduler_name = 'MultiStepLR'
        # opt.scheduler_settings = {'epochs': 400, 'step': [80, 280], 'gamma': 0.3333}
        opt.scheduler_name = 'CosineAnnealingLR'
        # # opt.scheduler_name = 'CyclicLR'
        opt.scheduler_settings['epochs'] = 400
        opt.scheduler_settings['min_lr'] = 0.00005/5

    ### Default settings of DNANet
    if opt.optimizer_name == 'Adagrad':
        opt.optimizer_settings['lr'] = 0.05
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings['epochs'] = 1500
        opt.scheduler_settings['min_lr'] = 1e-3
        opt.scheduler_settings['epochs'] = 400

    if opt.model_name == 'ExtractOne':
        opt.optimizer_name = 'Adamw'
        opt.optimizer_settings['lr'] = 0.015
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings['epochs'] = 400
        opt.scheduler_settings['min_lr'] = 0.00005

    opt.nEpochs = opt.scheduler_settings['epochs']

    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings,
                                         opt.scheduler_settings)

    # bestiou = 0

    for idx_epoch in range(epoch_state, opt.nEpochs):
        tbar = tqdm(train_loader)
        for idx_iter, (img, gt_mask) in enumerate(tbar):
            img, gt_mask = img.cuda(), gt_mask.cuda()


            # satis.update(gt_mask)
            # satis.choose()
            # vutils.save_image(img, f"output/img_{idx_iter}.png")
            # vutils.save_image(gt_mask, f"output/gtmask_{idx_iter}.png")
            if img.shape[0] == 1:
                continue

            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = net.forward(img)
                    loss = net.loss(pred,gt_mask,idx_iter,idx_epoch,img)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = net.forward(img)
                # vutils.save_image(pred[0], f"output/pred_{idx_iter}.png")
                loss = net.loss(pred,gt_mask,idx_iter,idx_epoch,img)
                # if opt.orth == True:
                #     with torch.enable_grad():
                #         reg = 1e-6
                #         orth_loss = torch.zeros(1,device=loss.device)
                #         for name, param in net.named_parameters():
                #             if ('stage1' in name or 'stage2' in name or 'stage3' in name or 'stage4' in name) and 'bias' not in name:
                #                 param_flat = param.view(param.shape[0], -1)
                #                 sym = torch.mm(param_flat, torch.t(param_flat))
                #                 sym -= torch.eye(param_flat.shape[0],device=sym.device)
                #                 orth_loss = orth_loss + (reg * sym.abs().sum())
                #     loss += orth_loss.item() / 5

                if opt.orth == True:
                    with torch.enable_grad():
                        reg = 1e-6
                        orth_loss = torch.zeros(1,device=loss.device)
                        param_flat_1 = None
                        param_flat_2 = None
                        param_flat_3 = None
                        param_flat_4 = None
                        for name, param in net.named_parameters():
                            if 'bias' and 'norm' not in name:
                                if 'stage1.0.branch_conv_list' in name :
                                    param_flat = param.view(param.shape[0], -1)
                                    if param_flat_1 is not None:
                                        param_flat_1 = torch.cat([param_flat_1,param_flat],dim=0)
                                    else:
                                        param_flat_1 = param_flat
                                elif 'stage2.0.branch_conv_list' in name:
                                    param_flat = param.view(param.shape[0], -1)
                                    if param_flat_2 is not None:
                                        param_flat_2 = torch.cat([param_flat_2,param_flat],dim=0)
                                    else:
                                        param_flat_2 = param_flat
                                elif 'stage3.0.branch_conv_list' in name:
                                    param_flat = param.view(param.shape[0], -1)
                                    if param_flat_3 is not None:
                                        param_flat_3 = torch.cat([param_flat_3,param_flat],dim=0)
                                    else:
                                        param_flat_3 = param_flat
                                elif'stage4.0.branch_conv_list' in name:
                                    param_flat = param.view(param.shape[0], -1)
                                    if param_flat_4 is not None:
                                        param_flat_4 = torch.cat([param_flat_4,param_flat],dim=0)
                                    else:
                                        param_flat_4 = param_flat

                        sym1 = torch.mm(param_flat_1, torch.t(param_flat_1))
                        sym1 -= torch.eye(param_flat_1.shape[0],device=sym1.device)
                        sym2 = torch.mm(param_flat_2, torch.t(param_flat_2))
                        sym2 -= torch.eye(param_flat_2.shape[0],device=sym2.device)
                        sym3 = torch.mm(param_flat_3, torch.t(param_flat_3))
                        sym3 -= torch.eye(param_flat_3.shape[0],device=sym3.device)
                        sym4 = torch.mm(param_flat_4, torch.t(param_flat_4))
                        sym4 -= torch.eye(param_flat_4.shape[0],device=sym4.device)
                        orth_loss =  (reg * sym1.abs().sum())+(reg * sym2.abs().sum())+(reg * sym3.abs().sum())+(reg * sym4.abs().sum())
                    loss += orth_loss.item() / 4

                loss.backward()
                optimizer.step()
            # for name, parms in net.named_parameters():
            #     if parms.grad is not None:
            #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
            #     else:
            #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', 0.)
            total_loss_epoch.append(loss.detach().cpu())
            #ema
            ema.update()
            #total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            # if idx_epoch == 0:
            #     total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            tbar.set_description("Epoch:%3d, lr:%f, total_loss:%f" %(idx_epoch+1, optimizer.param_groups[0]['lr'], total_loss_list[-1]))


        if (idx_epoch + 1) % 1 == 0:
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            # print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,'
            #       % (idx_epoch + 1, total_loss_list[-1]))
            opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n'
                        % (idx_epoch + 1, total_loss_list[-1]))
            total_loss_epoch = []

        if (idx_epoch + 1) % 2 == 0:
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name+ '_temp ' + '.pth'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
            }, save_pth)
            # ema.apply_shadow()

            iou = test(save_pth,idx_epoch)
            global bestiou
            if iou>bestiou:
                save_checkpoint({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                    'total_loss': total_loss_list,
                }, opt.save + '/' + opt.dataset_name + '/' + opt.model_name+ '_best' + '.pth')
                bestiou=iou
            print("best_epoch: "+str(idx_epoch+1))


            # ema.restore()

        if (idx_epoch + 1) == opt.nEpochs and (idx_epoch + 1) % 50 != 0:
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
            }, save_pth)
            test(save_pth,idx_epoch)
        scheduler.step()


def test(save_pth,idx_epoch):
    test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, patch_size=opt.patchSize,img_norm_cfg=opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)
    net = Net(model_name=opt.model_name, mode='test').cuda()
    ckpt = torch.load(save_pth)
    net.load_state_dict(ckpt['state_dict'])

    if opt.model_name == 'RepirDet' or opt.model_name == 'ExtractOne':
        net.model.switch_to_deploy()

    net.eval()
    eval_mIoU = mIoU()
    eval_PD_FA = PD_FA()
    eval_iou = SigmoidMetric()
    eval_nIoU = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    eval_iou.reset()
    eval_nIoU.reset()
    time_ = 0

    for idx_iter, (img, gt_mask, size, _) in enumerate(test_loader):
        img = img.cuda()

        start_time = time.time()
        pred = net.forward(img)
        end_time = time.time()
        time_ = time_ +(end_time - start_time)


        isthresh = True
        if  isinstance(pred,list):
            if opt.model_name == 'LKUnet':
                pred = pred[-1]
            else:
                pred = pred[0]

        # pred = pred[:, :, :size[0], :size[1]]
        # gt_mask = gt_mask[:, :, :size[0], :size[1]]

        if isthresh:
            # threshold_choose = eval_mIoU.calc(pred.cpu(),gt_mask)
            threshold_choose = 0.5
            threshold_choose = torch.tensor(threshold_choose, device=pred.device)
            t1 = pred > threshold_choose
            t2 = pred[0, 0, :, :] > threshold_choose
            # pred_ = torch.where(pred > threshold_choose,torch.tensor(1.),torch.zeros_like(pred))
            pred_ = torch.where(pred > threshold_choose, torch.tensor(1., device=pred.device), torch.zeros_like(pred))
        else:
            t1 = pred > opt.threshold
            t2 = pred[0, 0, :, :] > opt.threshold

        # vutils.save_image(img, f"output_test/img_{idx_iter}.png")
        # vutils.save_image(pred_, f"output_test/output_{idx_iter}.png")
        # vutils.save_image(gt_mask, f"output_test/gt_{idx_iter}.png")
        eval_mIoU.update(t1.cpu(), gt_mask)
        eval_PD_FA.update(t2.cpu(), gt_mask[0, 0, :, :], size)
        eval_nIoU.update(t1.cpu(), gt_mask)
        eval_iou.update(t1.cpu(), gt_mask)
    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    results3 = eval_nIoU.get()
    results4 = eval_iou.get()
    print("pixAcc, mIoU:\t" + str(results1))
    print("PD, FA:\t" + str(results2))
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')
    print("nIoU:\t" + str(results3[-1]))
    print("IoU:\t" + str(results4))
    opt.f.write("nIoU:\t" + str(results3[-1]) + '\n')
    opt.f.write("IoU:\t" + str(results4) + '\n')
    print(f"推理时间为：{time_:.4f}秒")

    if opt.inferenceFPS and (idx_epoch + 1) % 1 == 0:
        # 新增(start)添加FPS
        FPSBenchmark(
            model=net,
            device="cuda:0",
            datasets=test_loader,
            iterations=test_loader.__len__(),
            log_interval=10
        ).measure_inference_speed()

    return results1[-1]


def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path


if __name__ == '__main__':

    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            print(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ', '_').replace( ':', '_') + '.txt')
            opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ', '_').replace( ':', '_') + '.txt', 'w')
            print(opt.dataset_name + '\t' + opt.model_name)
            train()
            print('\n')
            opt.f.close()
