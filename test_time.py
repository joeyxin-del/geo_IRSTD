import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
# from dataset import *
from dataset_spotgeo import *
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端
import matplotlib.pyplot as plt
import torchinfo
from metrics import *
from metrics import F1Metric, MSEMetric
import os
import time
import cv2
from CalculateFPS import FPSBenchmark  # 确保导入你的FPS计算类
import torch
import numpy as np
from skimage import measure
from typing import List, Dict, Any
from tqdm import tqdm
import json

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
# parser.add_argument("--model_names", default=['LKUnet'], type=list,
parser.add_argument("--model_names", default=['TADNet'], type=list,
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet','LKUnetK'")
# parser.add_argument("--pth_dirs", default=['checkpoint/1k/LKUnet_best.pth'], type=list,
# parser.add_argument("--pth_dirs", default=['checkpoint/NUDT-SIRST/LKUnetlr_best.pth'], type=list,
#parser.add_argument("--pth_dirs", default=['checkpoint/NUDT-SIRST/LKUnet_best.pth'], type=list,
# parser.add_argument("--pth_dirs", default=['log/IRSTD-1K/LKUnet_best.pth'], type=list,
parser.add_argument("--pth_dirs", default=['checkpoints/TADNet_780.pth'], type=list,
                    help="checkpoint dir, default=None or ['NUDT-SIRST/ACM_400.pth.tar','NUAA-SIRST/ACM_400.pth.tar']")
parser.add_argument("--dataset_dir", default='./datasets', type=str, help="train_dataset_dir")
#parser.add_argument("--dataset_names", default=['NUDT-SIRST'], type=list,
# parser.add_argument("--dataset_names", default=['IRSTD-1K'], type=list,
parser.add_argument("--dataset_names", default=['spotgeov2'], type=list,
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--save_img", default=True, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='./log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)

global opt
opt = parser.parse_args()


# def Save_Img(img, pred, gt_mask, img_dir, img_IoU):
#     img = img.cpu().detach().numpy()
#     img_pred = transforms.ToPILImage()((pred[0, 0, :, :]).cpu())
#     gt_mask = transforms.ToPILImage()((gt_mask[0, 0, :, :]).cpu())

#     # 获取实际图像尺寸
#     img_shape = img.shape
#     if len(img_shape) == 4:  # (batch, channel, height, width)
#         img_src = img[0, 0, :, :]  # 取第一个batch的第一个通道
#     elif len(img_shape) == 3:  # (channel, height, width)
#         img_src = img[0, :, :]  # 取第一个通道
#     else:
#         img_src = img

#     # 确保所有图像都是3通道RGB格式
#     if len(img_src.shape) == 2:  # 如果是灰度图
#         img_src = cv2.cvtColor((img_src * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
#     image_pred = np.array(img_pred)
#     image_mask = np.array(gt_mask)

#     if len(image_pred.shape) == 2:  # 灰度图
#         image_pred = cv2.cvtColor(image_pred, cv2.COLOR_GRAY2RGB)

#     if len(image_mask.shape) == 2:  # 灰度图
#         image_mask = cv2.cvtColor(image_mask, cv2.COLOR_GRAY2RGB)

#     # 创建 subplot 图表
#     fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

#     # 添加主标题
#     fig.suptitle(img_dir[0] + ' IoU: ' + str(format(img_IoU * 100, '.2f')) + "%")

#     # 显示原图
#     axes[0].imshow(img_src)
#     axes[0].set_title('Original')

#     # 显示差异图像
#     axes[1].imshow(image_pred)
#     axes[1].set_title('prediction')

#     # 显示标签图像
#     axes[2].imshow(image_mask)
#     axes[2].set_title('Ground Truth')

#     # 调整子图布局
#     plt.tight_layout()

#     # 创建保存目录（如果不存在）
#     save_dir = os.path.join(opt.save_img_dir, opt.test_dataset_name, opt.model_name)
#     for subdir in ['Good', 'Medium', 'Bad']:
#         os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)

#     # 保存结果图像
#     if img_IoU > 0.716:
#         plt.savefig(os.path.join(save_dir, "Good", img_dir[0] + '.png'))
#     elif img_IoU <= 0.716 and img_IoU > 0.25:
#         plt.savefig(os.path.join(save_dir, "Medium", img_dir[0] + '.png'))
#     elif img_IoU <= 0.25:
#         plt.savefig(os.path.join(save_dir, "Bad", img_dir[0] + '.png'))
    
#     plt.close()


def mask_to_coords(pred_mask: torch.Tensor, conf_thresh: float = 0.5) -> List[Dict[str, Any]]:
    """
    将分割掩码转换为目标中心坐标列表
    Args:
        pred_mask: [B, 1, H, W] 预测的分割掩码
        conf_thresh: 二值化阈值
    Returns:
        List[Dict]，每个元素格式：
        {
            'frame': int,
            'num_objects': int,
            'object_coords': List[[x, y], ...]
        }
    """
    results = []
    # 先detach掉梯度再转换为numpy
    pred_mask = pred_mask.detach().cpu().numpy()
    B = pred_mask.shape[0]
    
    for b in range(B):
        # 取出当前帧的掩码并二值化
        mask = pred_mask[b, 0] > conf_thresh
        
        # 使用连通区域分析找到所有目标
        labels = measure.label(mask, connectivity=2)
        regions = measure.regionprops(labels)
        
        # 提取每个目标的中心坐标
        coords = []
        for region in regions:
            # regionprops返回的centroid是(y, x)顺序，需要转换为(x, y)
            y, x = region.centroid
            coords.append([float(x), float(y)])  # 转换为Python float
            
        results.append({
            'frame': int(b),
            'num_objects': len(coords),
            'object_coords': coords
        })
    
    return results


def extract_points_from_mask(mask):
    """
    从mask中提取坐标点
    Args:
        mask: 二值化掩码图像
    Returns:
        points: 坐标点列表 [[x1,y1], [x2,y2], ...]
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # 确保mask是2D的
    if mask.ndim == 3:
        mask = mask.squeeze()
    
    # 找到所有非零点的坐标
    points = np.where(mask > 0)
    points = np.array(list(zip(points[1], points[0])))  # 转换为[[x,y], [x,y], ...]格式
    
    # 如果没有点，返回空数组
    if len(points) == 0:
        return np.array([])
        
    return points


def save_visualization_results(img, pred_points, gt_points, save_dir, img_dir, img_size):
    """
    保存可视化结果
    Args:
        img: 输入图像
        pred_points: 预测点坐标
        gt_points: 真实点坐标
        save_dir: 保存目录
        img_dir: 图像路径
        img_size: 图像尺寸
    Returns:
        seq_id: 序列ID
        frame_id: 帧ID
        img_size: 图像尺寸
        pred_points: 预测点坐标
        gt_points: 真实点坐标
    """
    # 获取序列ID和帧ID
    img_name = os.path.basename(img_dir)
    seq_id = int(img_name.split('_')[0])
    frame_id = int(img_name.split('_')[1].split('.')[0])
    
    # 创建序列目录
    seq_dir = os.path.join(save_dir, f'sequence_{seq_id}')
    os.makedirs(seq_dir, exist_ok=True)
    
    # 设置matplotlib后端为Agg
    plt.switch_backend('Agg')
    
    # 创建图像
    plt.figure(figsize=(8, 8), dpi=100)
    plt.axis('off')
    
    # 保存预测结果
    plt.clf()
    plt.imshow(img.squeeze(), cmap='gray')
    if len(pred_points) > 0:
        plt.scatter(pred_points[:, 0], pred_points[:, 1], c='r', s=40, marker='x', linewidth=2)
    plt.savefig(os.path.join(seq_dir, f'{frame_id}_pred.png'), bbox_inches='tight', pad_inches=0)
    
    # 保存真实标注
    plt.clf()
    plt.imshow(img.squeeze(), cmap='gray')
    if len(gt_points) > 0:
        plt.scatter(gt_points[:, 0], gt_points[:, 1], c='g', s=40, marker='x', linewidth=2)
    plt.savefig(os.path.join(seq_dir, f'{frame_id}_gt.png'), bbox_inches='tight', pad_inches=0)
    
    plt.close()
    
    return seq_id, frame_id, img_size, pred_points, gt_points


def create_sequence_visualization(save_dir, sequence_data, annotation_file=None):
    """
    为每个序列创建轨迹可视化
    Args:
        save_dir: 保存目录
        sequence_data: 序列数据字典
        annotation_file: 标注文件路径
    """
    # 读取json标注文件
    json_annotations = {}
    if annotation_file and os.path.exists(annotation_file):
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
            # 按序列和帧组织标注数据
            for anno in annotations:
                seq_id = anno['sequence_id']
                frame_id = anno['frame']
                if seq_id not in json_annotations:
                    json_annotations[seq_id] = {}
                json_annotations[seq_id][frame_id] = anno['object_coords']

    # 为每个序列创建可视化
    for seq_id, seq_info in sequence_data.items():
        print(f"正在处理序列 {seq_id} 的可视化...")
        
        img_size = seq_info['img_size']
        
        # 创建黑色背景图像
        track_img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        
        # 创建图像，设置大小完全匹配原始图像尺寸
        plt.figure(figsize=(img_size[1]/100, img_size[0]/100))  # 转换为英寸
        plt.imshow(track_img)
        
        # 用于跟踪是否已添加图例
        legend_added = {'pred': False, 'gt': False, 'json': False}
        
        # 绘制每一帧的点
        for frame_id, frame_data in seq_info['frames'].items():
            # 绘制预测点（红色叉）
            pred_points = frame_data['pred_points']
            if len(pred_points) > 0:
                plt.scatter(pred_points[:, 0], pred_points[:, 1], 
                          c='red', s=150, marker='x', linewidth=2,
                          label='Prediction' if not legend_added['pred'] else "")
                legend_added['pred'] = True
            
            # 绘制从mask提取的真实标注点（蓝色点）
            gt_points = frame_data['gt_points']
            if len(gt_points) > 0:
                plt.scatter(gt_points[:, 0], gt_points[:, 1], 
                          c='blue', s=75, marker='o', linewidth=2,
                          label='Mask GT' if not legend_added['gt'] else "")
                legend_added['gt'] = True
            
            # 绘制从json文件读取的标注点（黄色点）
            if seq_id in json_annotations and frame_id in json_annotations[seq_id]:
                json_points = np.array(json_annotations[seq_id][frame_id])
                if len(json_points) > 0:
                    plt.scatter(json_points[:, 0], json_points[:, 1], 
                              c='yellow', s=40, marker='o', linewidth=2,
                              label='Anno GT' if not legend_added['json'] else "")
                    legend_added['json'] = True
        
        # 设置坐标轴范围并隐藏所有元素
        plt.xlim(0, img_size[1])
        plt.ylim(img_size[0], 0)
        plt.axis('off')
        
        # 添加图例到图像内部右上角
        legend = plt.legend(loc='upper right', frameon=True, 
                          facecolor='black', edgecolor='white',
                          labelcolor='white', fontsize=8)
        
        # 移除所有边距
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        
        # 保存图像，确保完全没有边距
        plt.savefig(os.path.join(save_dir, f'sequence_{seq_id}', f'track_sequence_{seq_id}_contrast.png'),
                   bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()


def test():
    print(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    test_set = TestSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, patch_size= 512, img_norm_cfg = opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)

    net = Net(model_name=opt.model_name, mode='test').cuda()
    state_dict = torch.load(opt.pth_dir)['state_dict'] 
    net.load_state_dict(state_dict)
    torchinfo.summary(net)
    net.eval()

    # FPSBenchmark setup
    benchmark = FPSBenchmark(
        datasets=test_loader,
        iterations=100,
        model=net,
        device="cuda:0",
        warmup_num=5,
        log_interval=10,
        repeat_num=1
    )
    fps_calculateFPS = benchmark.measure_inference_speed()

    # 初始化指标
    f1_metric = F1Metric(threshold=opt.threshold)
    mse_metric = MSEMetric()
    
    total_time = 0
    num_frames = 0
    all_predictions = {}  # 存储所有预测结果

    # 用于存储序列数据的字典
    sequence_data = {}
    
    # 遍历数据集
    for idx_iter, (img, gt_mask, size, img_dir) in enumerate(tqdm(test_loader)):
        try:
            img = img.cuda()
            torch.cuda.empty_cache()  # 清理GPU缓存

            start_time = time.time()
            with torch.cuda.amp.autocast():  # 使用混合精度
                pred = net.forward(img)
            end_time = time.time()

            if isinstance(pred, list):
                pred = pred[0]
            pred = pred[:, :, :size[0], :size[1]]
            
            # 更新F1和MSE指标
            f1_metric.update(pred, gt_mask)
            mse_metric.update(pred, gt_mask)
            
            # 将预测掩码转换为坐标形式
            coords = mask_to_coords(pred, conf_thresh=opt.threshold)
            
            # 保存预测结果
            for coord_info in coords:
                img_name = img_dir[coord_info['frame']]
                all_predictions[img_name] = {
                    'coords': coord_info['object_coords'],
                    'num_objects': coord_info['num_objects']
                }

            inference_time = end_time - start_time
            total_time += inference_time
            num_frames += 1

            # if opt.save_img:
            #     # 可视化预测结果
            #     Save_Img(img, pred, gt_mask, img_dir, 0)  # IoU暂时设为0

            # 主动释放内存
            del pred
            torch.cuda.empty_cache()

            # if idx_iter % 10 == 0:  # 每处理10张图片打印一次进度
            #     print(f"已处理: {idx_iter}/{len(test_loader)} 图片")

            # 对每张图片进行处理
            for batch_idx in range(len(img)):
                try:
                    # 打印调试信息
                    print(f"\n处理批次 {idx_iter} 中的第 {batch_idx} 张图片")
                    print(f"图片信息: {img_dir[batch_idx]}")
                    
                    # 从gt_mask中提取真实坐标点
                    current_gt_mask = gt_mask[batch_idx]
                    gt_points = extract_points_from_mask(current_gt_mask)
                    
                    # 获取预测点坐标
                    pred_points = np.array(coord_info['object_coords'])
                    if len(pred_points.shape) > 2:  # 如果维度过高，需要降维
                        pred_points = pred_points.reshape(-1, 2)
                    
                    # 保存可视化结果并获取序列信息
                    seq_id, frame_id, img_size, pred_points, _ = save_visualization_results(
                        img[batch_idx].cpu().numpy(), 
                        pred_points,  # 使用处理后的预测点
                        gt_points,  # 使用提取出的真实坐标点
                        opt.save_img_dir, 
                        img_dir[batch_idx], 
                        img[batch_idx].shape[1:]
                    )
                    
                    # 收集序列数据
                    if seq_id not in sequence_data:
                        sequence_data[seq_id] = {
                            'img_size': img_size,
                            'frames': {}
                        }
                    sequence_data[seq_id]['frames'][frame_id] = {
                        'pred_points': pred_points,
                        'gt_points': gt_points  # 使用提取出的真实坐标点
                    }
                except Exception as e:
                    print(f"警告：处理单张图片时出错: {str(e)}")
                    continue
        
        except Exception as e:
            print(f"警告：处理批次时出错: {str(e)}")
            continue
    
    # 创建序列可视化
    create_sequence_visualization(opt.save_img_dir, sequence_data, 
                               annotation_file='datasets/spotgeov2/test_anno.json')

    # 计算FPS
    fps_time_method = num_frames / total_time
    
    # 获取F1和MSE指标结果
    f1_score = f1_metric.get()
    mse_score = mse_metric.get()

    # 保存预测结果到json文件
    save_path = os.path.join(opt.save_img_dir, opt.test_dataset_name, opt.model_name, 'predictions.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(all_predictions, f, indent=2)

    print("\n=== 测试结果 ===")
    print(f"FPS (time method): {fps_time_method:.2f}")
    print(f"FPS (calculateFPS method): {fps_calculateFPS:.2f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"MSE: {mse_score:.4f}")
    print(f"Successfully processed {num_frames} images")
    print(f"预测结果已保存到: {save_path}")
    print("================\n")

    # 写入结果到文件
    opt.f.write("\n=== 测试结果 ===\n")
    opt.f.write(f"FPS (time method): {fps_time_method:.2f}\n")
    opt.f.write(f"FPS (calculateFPS method): {fps_calculateFPS:.2f}\n")
    opt.f.write(f"F1 Score: {f1_score:.4f}\n")
    opt.f.write(f"MSE: {mse_score:.4f}\n")
    opt.f.write(f"Successfully processed {num_frames} images\n")
    opt.f.write(f"预测结果已保存到: {save_path}\n")
    opt.f.write("================\n")


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
