import argparse
import time
import matplotlib.pyplot as plt
import tarfile
import shutil
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

import torchinfo
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
# parser.add_argument("--model_names", default=['NoiseNet'], type=list,
# parser.add_argument("--model_names", default=['LinearVIT'], type=list,
parser.add_argument("--model_names", default=['NoiseAttNet'], type=list,
# parser.add_argument("--model_names", default=['RepirDet'], type=list,
                    # parser.add_argument("--model_names", default=['DNANet'], type=list,
                    help="model_name: Dataset-mask  'ACM', 'ALCNet', 'DNANet', 'ISNet', "
                         "'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet', 'NestedSwinUnet',"
                         "'SearchNet','RepirDet','ExtractOne'"
                         "'LKUnet', 'WTNet',")

# =======================================datasets============================
parser.add_argument("--dataset_names", default=['IRSTD-1K'], type=list,
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--experiment_name", default='basicBlock2222', type=str,
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")

parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--dataset_dir", default='./datasets', type=str, help="train_dataset_dir")

# =====================================batchSize===========================================
parser.add_argument("--batchSize", type=int, default=8 * 2, help="Training batch sizse")
parser.add_argument("--patchSize", type=int, default=512, help="Training patch size")
parser.add_argument("--save", default='./log', type=str, help="Save path of checkpoints")
parser.add_argument("--resume", default=None, type=list,
                    help="Resume from exisiting checkpoints (default: None) use ['path']")

# ======================================Epoch==============================================
parser.add_argument("--nEpochs", type=int, default=400, help="Number of epochs")
# =======================================optimizer==========================================
parser.add_argument("--optimizer_name", default='Adamw', type=str, help="optimizer name: Adam, Adagrad, SGD,Adamw")
# ==========================================lr=============================================
parser.add_argument("--optimizer_settings", default={'lr': 5e-4}, type=dict, help="optimizer settings")
parser.add_argument("--scheduler_name", default='MultiStepLR', type=str, help="scheduler name: MultiStepLR")
parser.add_argument("--scheduler_settings", default={'step': [200, 300], 'gamma': 0.1}, type=dict,
                    help="scheduler settings")

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


def plot_loss_curve(total_loss_list, experiment_dir):
    plt.figure()
    plt.plot(total_loss_list)
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(experiment_dir, 'loss.png'))
    plt.close()


def plot_mIoU_curve(mIoU_list, experiment_dir):
    plt.figure()
    plt.plot(mIoU_list)
    plt.title('mIoU Curve')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.savefig(os.path.join(experiment_dir, 'mIoU.png'))
    plt.close()


# 定义一个函数来创建 tar.gz 压缩包
def create_tar_gz(source_dir, output_dir, archive_name):
    # 构造完整的输出文件路径
    output_filename = os.path.join(output_dir, f'{archive_name}.tar.gz')
    with tarfile.open(output_filename, "w:gz") as tar:
        # 遍历源目录并添加到压缩包
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # 计算文件在压缩包中的相对路径
                tar.add(file_path, arcname=os.path.relpath(file_path, start=source_dir))


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


bestiou = -1.0


def train():
    # 记录程序开始的时间
    start_time = time.time()

    # 创建路径：./log/opt.dataset_name/opt.model_name/日期＋特性/
    experiment_dir_name = f'{opt.model_name}_{opt.experiment_name}_{time.strftime("%Y-%m-%d_%H-%M-%S")}'
    experiment_dir = os.path.join('./log', opt.dataset_name, opt.model_name, experiment_dir_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # 假设模型目录是 'model/模型名称'
    model_dir_path = os.path.join('model', opt.model_name)  
    
    # 检查路径是否存在
    if os.path.exists(model_dir_path):
        # 如果是目录，使用 shutil.copytree
        if os.path.isdir(model_dir_path):
            dest_dir = os.path.join(experiment_dir, opt.model_name)
            shutil.copytree(model_dir_path, dest_dir)
            print(f"Model directory {model_dir_path} has been copied to {dest_dir}")
        else:
            # 如果是文件，使用 shutil.copy
            shutil.copy(model_dir_path, experiment_dir)
            print(f"Model file {model_dir_path} has been copied to {experiment_dir}")
    else:
        print(f"Model path {model_dir_path} does not exist and cannot be copied.")

    # 训练日志文件
    log_file = os.path.join(experiment_dir, 'training_log.txt')
    opt.f = open(log_file, 'w')

    # 保存训练超参数信息
    hyperparameters = f"""
    Dataset: {opt.dataset_name}
    Model: {opt.model_name}
    Optimizer: {opt.optimizer_name}
    Learning Rate: {opt.optimizer_settings['lr']}
    Epochs: {opt.nEpochs}
    Batch Size: {opt.batchSize}
    Patch Size: {opt.patchSize}
    """
    with open(os.path.join(experiment_dir, 'hyperparameters.txt'), 'w') as f:
        f.write(hyperparameters)

    train_set = TrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize,
                               img_norm_cfg=opt.img_norm_cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    net = Net(model_name=opt.model_name, mode='train').cuda()
    torchinfo.summary(net)

    net.train()

    # AMP
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # EMA
    ema = EMA(net, 0.0)
    ema.register()

    # satis = MaskSatistical(opt.patchSize)

    epoch_state = 0
    total_loss_list = [0]
    total_loss_epoch = []

    if opt.resume:
        for resume_pth in opt.resume:
            print(resume_pth, ' is loaded')
            if opt.model_name in resume_pth:
                print(resume_pth, ' is loaded')
                ckpt = torch.load(resume_pth)
                net.load_state_dict(ckpt['state_dict'])
                epoch_state = ckpt['epoch']
                total_loss_list = ckpt['total_loss']
                # for i in range(len(opt.step)):
                #     opt.step[i] = opt.step[i] - ckpt['epoch']
    resume = False

    resume_temp = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_best_m' + '.pth'
    # resume_temp = 'K:\BasicIRSTD-main\log\IRSTD-1K\LinearVitR_1_69.pth'
    if resume:
        resume_pth = resume_temp
        print(resume_pth, ' is loaded')
        if opt.model_name in resume_pth:
            print(resume_pth, ' is loaded')
            ckpt = torch.load(resume_pth)
            net.load_state_dict(ckpt['state_dict'], strict=True)
            # epoch_state = ckpt['epoch']
            total_loss_list = ckpt['total_loss']

    ### Default settings
    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': 5e-4 * 4}
        opt.scheduler_name = 'MultiStepLR'
        opt.scheduler_settings = {'epochs': 400, 'step': [200, 300], 'gamma': 0.1}

    if opt.optimizer_name == 'Adamw':
        # opt.optimizer_settings = {'lr': 0.0007}
        opt.optimizer_settings['lr'] = 0.0015
        # opt.scheduler_name = 'MultiStepLR'
        # opt.scheduler_settings = {'epochs': 400, 'step': [80, 280], 'gamma': 0.3333}
        opt.scheduler_name = 'CosineAnnealingLR'
        # # opt.scheduler_name = 'CyclicLR'
        opt.scheduler_settings['epochs'] = 800
        opt.scheduler_settings['min_lr'] = 0.0005

    if opt.optimizer_name == 'Lion':
        # opt.optimizer_settings = {'lr': 0.0007}
        opt.optimizer_settings['lr'] = 0.00015
        # opt.scheduler_name = 'MultiStepLR'
        # opt.scheduler_settings = {'epochs': 400, 'step': [80, 280], 'gamma': 0.3333}
        opt.scheduler_name = 'CosineAnnealingLR'
        # # opt.scheduler_name = 'CyclicLR'
        opt.scheduler_settings['epochs'] = 400
        opt.scheduler_settings['min_lr'] = 0.00005 / 5

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

    # 保存目录创建
    save_dir = os.path.join(experiment_dir, 'models')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    bestiou = -1.0
    mIoU_list = []  # 用于保存每个epoch的mIoU

    for idx_epoch in range(epoch_state, opt.nEpochs):
        tbar = tqdm(train_loader)
        for idx_iter, (img, gt_mask) in enumerate(tbar):
            img, gt_mask = img.cuda(), gt_mask.cuda()

            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = net.forward(img)
                    loss = net.loss(pred, gt_mask, idx_iter, idx_epoch, img)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = net.forward(img)
                loss = net.loss(pred, gt_mask, idx_iter, idx_epoch, img)
                loss.backward()
                optimizer.step()

            total_loss_epoch.append(loss.detach().cpu())
            ema.update()

            tbar.set_description(f"Epoch: {idx_epoch + 1}, Loss: {loss.item():.4f}")

        total_loss_list.append(float(np.array(total_loss_epoch).mean()))
        opt.f.write(f"Epoch {idx_epoch + 1}: Loss = {total_loss_list[-1]}\n")

        # 每10个epoch画一次损失图
        # if (idx_epoch + 1) % 10 == 0:
        #     plot_loss_curve(total_loss_list, experiment_dir)

        # 保存模型和进行测试
        if (idx_epoch + 1) % 2 == 0:
            save_pth = os.path.join(save_dir, f'{opt.model_name}_temp.pth')
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
            }, save_pth)

            # 计算当前的mIoU
            iou = test(save_pth, idx_epoch)
            mIoU_list.append(iou)  # 保存每个epoch的mIoU
            if iou > bestiou:
                save_checkpoint({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                    'total_loss': total_loss_list,
                }, os.path.join(save_dir, f'{opt.model_name}_best.pth'))
                bestiou = iou

        scheduler.step()

    # 绘制mIoU图
    plot_mIoU_curve(mIoU_list, experiment_dir)

    # 绘制最终损失图
    plot_loss_curve(total_loss_list, experiment_dir)

    # 训练结束后，记录程序结束的时间
    end_time = time.time()

    # 计算程序运行的总时长（秒）
    total_time = end_time - start_time

    # 获取当前时间并格式化
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    if opt.f.closed:
        opt.f = open(log_file, 'a')  # 重新打开文件，追加模式
    # 将当前时间和总时长写入日志文件
    opt.f.write(f"\n\nTraining completed at {current_time}\n")
    opt.f.write(f"Total training time: {total_time / 3600:.2f} hours\n")  # 转换为小时，保留两位小数

    # 调用函数，创建压缩包
    create_tar_gz(experiment_dir, os.path.join('./log', opt.dataset_name, opt.model_name), experiment_dir_name)
    # 关闭文件
    opt.f.close()


def test(save_pth, idx_epoch):
    test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, patch_size=opt.patchSize,
                             img_norm_cfg=opt.img_norm_cfg)
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
        time_ = time_ + (end_time - start_time)

        isthresh = True
        if isinstance(pred, list):
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
            print(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ',
                                                                                                          '_').replace(
                ':', '_') + '.txt')
            opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ',
                                                                                                                 '_').replace(
                ':', '_') + '.txt', 'w')
            print(opt.dataset_name + '\t' + opt.model_name)
            train()
            print('\n')
            opt.f.close()
