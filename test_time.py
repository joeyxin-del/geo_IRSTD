import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
import matplotlib.pyplot as plt
import torchinfo
from metrics import *
import os
import time
import cv2
from CalculateFPS import FPSBenchmark  # 确保导入你的FPS计算类

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument("--model_names", default=['LKUnet'], type=list,
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet','LKUnetK'")
# parser.add_argument("--pth_dirs", default=['checkpoint/1k/LKUnet_best.pth'], type=list,
# parser.add_argument("--pth_dirs", default=['checkpoint/NUDT-SIRST/LKUnetlr_best.pth'], type=list,
#parser.add_argument("--pth_dirs", default=['checkpoint/NUDT-SIRST/LKUnet_best.pth'], type=list,
parser.add_argument("--pth_dirs", default=['log/IRSTD-1K/LKUnet_best.pth'], type=list,
                    help="checkpoint dir, default=None or ['NUDT-SIRST/ACM_400.pth.tar','NUAA-SIRST/ACM_400.pth.tar']")
parser.add_argument("--dataset_dir", default='./datasets', type=str, help="train_dataset_dir")
#parser.add_argument("--dataset_names", default=['NUDT-SIRST'], type=list,
parser.add_argument("--dataset_names", default=['IRSTD-1K'], type=list,
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--save_img", default=False, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='./log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)

global opt
opt = parser.parse_args()


def Save_Img(img, pred, gt_mask, img_dir, img_IoU):
    img = img.cpu().detach().numpy()
    img_pred = transforms.ToPILImage()((pred[0, 0, :, :]).cpu())
    gt_mask = transforms.ToPILImage()((gt_mask[0, 0, :, :]).cpu())

    img_size = 512
    img_src = np.reshape(img, (img_size, img_size))

    image_pred = np.array(img_pred)
    image_mask = np.array(gt_mask)

    # Check if image_pred is grayscale and convert it to three channels if necessary
    if len(img_src.shape) == 2:  # Grayscale image
        img_src = cv2.cvtColor(img_src, cv2.COLOR_GRAY2RGB)

    if len(image_pred.shape) == 2:  # Grayscale image
        image_pred = cv2.cvtColor(image_pred, cv2.COLOR_GRAY2RGB)

    if len(image_mask.shape) == 2:  # Grayscale image
        image_mask = cv2.cvtColor(image_mask, cv2.COLOR_GRAY2RGB)

    # 创建 subplot 图表
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    # # 添加主标题
    fig.suptitle(img_dir[0] + ' IoU: ' + str(format(img_IoU * 100, '.2f')) + "%")

    # 显示原图
    axes[0].imshow(img_src)
    axes[0].set_title('Original')

    # 显示差异图像
    axes[1].imshow(image_pred)
    axes[1].set_title('prediction')

    # 显示标签图像
    axes[2].imshow(image_mask)
    axes[2].set_title('Ground Truth')

    # 调整子图布局
    plt.tight_layout()

    # 保存结果图像
    if img_IoU > 0.716:
        plt.savefig(
            opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name + '/' + "Good/" + img_dir[0] + '.png')
    elif img_IoU <= 0.716 and img_IoU > 0.25:
        plt.savefig(
            opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name + '/' + "Medium/" + img_dir[0] + '.png')
    elif img_IoU <= 0.25:
        plt.savefig(
            opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name + '/' + "Bad/" + img_dir[0] + '.png')


def test():
    print(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    # test_set = TestSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, patch_size= 512,opt.img_norm_cfg)
    test_set = TestSetLoader(dataset_dir=opt.dataset_dir,  train_dataset_name=opt.train_dataset_name,test_dataset_name = opt.test_dataset_name, patch_size=512,
                             img_norm_cfg = opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)

    net = Net(model_name=opt.model_name, mode='test').cuda()
    net.load_state_dict(torch.load(opt.pth_dir)['state_dict'])
    torchinfo.summary(net)
    net.eval()

    eval_mIoU = mIoU()
    eval_IoU = SigmoidMetric()
    eval_PD_FA = PD_FA()
    eval_nIoU = SamplewiseSigmoidMetric(1, score_thresh=0.5)

    # FPSBenchmark setup
    benchmark = FPSBenchmark(
        datasets=test_loader,
        iterations=100,  # Adjust this as necessary
        model=net,
        device="cuda:0",
        warmup_num=5,
        log_interval=10,
        repeat_num=1
    )
    fps_calculateFPS = benchmark.measure_inference_speed()

    total_time = 0
    num_frames = 0

    for idx_iter, (img, gt_mask, size, img_dir) in enumerate(test_loader):
        img = img.cuda()  # 将输入数据移到GPU上

        # Start timing
        start_time = time.time()
        pred = net.forward(img)
        # End timing
        end_time = time.time()

        if isinstance(pred, list):
            pred = pred[0]  # 如果预测结果为列表，则取第一个
        pred = pred[:, :, :size[0], :size[1]]
        gt_mask = gt_mask[:, :, :size[0], :size[1]]

        eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask)
        eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], size)
        eval_IoU.update((pred > opt.threshold).cpu(), gt_mask)
        eval_nIoU.update((pred > opt.threshold).cpu(), gt_mask)

        # Calculate time taken for this iteration
        inference_time = end_time - start_time
        total_time += inference_time
        num_frames += 1

        # Optional: print the time for each iteration
        # print(f"Time for frame {idx_iter}: {inference_time:.6f} seconds")

        ### save img
        if opt.save_img:
            inter, Union, img_IoU = eval_mIoU.IoU((pred > opt.threshold).cpu(), gt_mask)
            print(img_dir[0], inter, Union, img_IoU)
            Save_Img(img, pred, gt_mask, img_dir, img_IoU)

    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    results3 = eval_nIoU.get()
    results4 = eval_IoU.get()

    print("pixAcc, mIoU:\t" + str(results1))
    print("PD, FA:\t" + str(results2))
    print("nIoU:\t" + str(results3))
    print("IoU:\t" + str(results4))

    # Calculate and print FPS using time method
    fps_time_method = num_frames / total_time
    print(f"FPS (time method): {fps_time_method:.2f}")
    print(f"FPS (calculateFPS method): {fps_calculateFPS:.2f}")

    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')
    opt.f.write("nIoU:\t" + str(results3) + '\n')
    opt.f.write("IoU:\t" + str(results4) + '\n')
    opt.f.write("FPS (time method): " + str(fps_time_method) + '\n')
    opt.f.write("FPS (calculateFPS method): " + str(fps_calculateFPS) + '\n')


if __name__ == '__main__':
    opt.f = open(opt.save_log + 'test_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
    if opt.pth_dirs == None:
        for i in range(len(opt.model_names)):
            opt.model_name = opt.model_names[i]
            print(opt.model_name)
            opt.f.write(opt.model_name + '_400.pth.tar' + '\n')
            for dataset_name in opt.dataset_names:
                print(dataset_name, "11111111111111111")
                opt.dataset_name = dataset_name
                opt.train_dataset_name = opt.dataset_name
                opt.test_dataset_name = opt.dataset_name
                print(dataset_name)
                opt.f.write(opt.dataset_name + '\n')
                opt.pth_dir = opt.save_log + opt.dataset_name + '/' + opt.model_name + '_400.pth.tar'
                test()
            print('\n')
            opt.f.write('\n')
        opt.f.close()
    else:
        for model_name in opt.model_names:
            for dataset_name in opt.dataset_names:
                for pth_dir in opt.pth_dirs:
                    print(dataset_name, "11111111111111111")
                    opt.test_dataset_name = dataset_name
                    opt.train_dataset_name = dataset_name
                    opt.model_name = model_name
                    print(pth_dir, "222222222222222")
                    opt.f.write(pth_dir)
                    print(opt.test_dataset_name)
                    opt.f.write(opt.test_dataset_name + '\n')
                    opt.pth_dir = pth_dir
                    test()
                    print('\n')
                    opt.f.write('\n')
        opt.f.close()
