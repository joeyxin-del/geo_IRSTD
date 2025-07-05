import numpy as np
import torch
from skimage import measure
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist  # 从scipy导入cdist用于计算距离矩阵
from typing import List, Tuple
import multiprocessing as mp
from functools import partial
import warnings
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def calculate_distance(point1: List[float], point2: List[float]) -> float:
    """计算两点之间的欧氏距离"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def match_points_optimized(pred_points: List[List[float]], gt_points: List[List[float]], 
                          distance_threshold: float = 5.0) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    使用优化的匈牙利算法匹配预测点和真实点
    参考score_frame函数的实现，使用cdist高效计算距离矩阵
    """
    if len(pred_points) == 0 or len(gt_points) == 0:
        return [], list(range(len(pred_points))), list(range(len(gt_points)))

    # 转换为numpy数组以便使用cdist
    X = np.array(pred_points)
    Y = np.array(gt_points)
    
    # 使用cdist高效计算欧几里得距离矩阵
    D = cdist(X, Y)
    
    # 截断超过阈值的距离
    D[D > distance_threshold] = 1000
    
    # 使用匈牙利算法解决线性分配问题
    row_ind, col_ind = linear_sum_assignment(D)
    matching = D[row_ind, col_ind]
    
    # 筛选有效匹配（距离在阈值内）
    valid_matches = []
    for i, (pred_idx, gt_idx) in enumerate(zip(row_ind, col_ind)):
        if matching[i] <= distance_threshold:
            valid_matches.append((pred_idx, gt_idx))
    
    # 计算未匹配的点
    matched_preds = set(match[0] for match in valid_matches)
    matched_gts = set(match[1] for match in valid_matches)
    
    unmatched_preds = [i for i in range(len(pred_points)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(gt_points)) if i not in matched_gts]
    
    return valid_matches, unmatched_preds, unmatched_gts

def match_points(pred_points: List[List[float]], gt_points: List[List[float]], 
                distance_threshold: float = 5.0) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    使用匈牙利算法匹配预测点和真实点（优化版本）
    Args:
        pred_points: 预测的点坐标列表 [[x1,y1], [x2,y2], ...]
        gt_points: 真实的点坐标列表 [[x1,y1], [x2,y2], ...]
        distance_threshold: 匹配的距离阈值
    Returns:
        matches: 匹配的点对索引
        unmatched_preds: 未匹配的预测点索引
        unmatched_gts: 未匹配的真实点索引
    """
    return match_points_optimized(pred_points, gt_points, distance_threshold)

def extract_points_from_mask(mask):
    """
    从mask中提取质心坐标点
    Args:
        mask: 二值化掩码图像
    Returns:
        points: 质心坐标点列表 [[x1,y1], [x2,y2], ...]
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # 确保mask是2D的
    if mask.ndim == 3:
        mask = mask.squeeze()
    
    # 使用skimage的label和regionprops来找到连通区域并计算质心
    labeled_mask = measure.label(mask, connectivity=2)
    regions = measure.regionprops(labeled_mask)
    
    # 提取每个区域的质心坐标
    points = []
    for region in regions:
        # region.centroid返回(y, x)格式，需要转换为(x, y)
        centroid_y, centroid_x = region.centroid
        points.append([centroid_x, centroid_y])
    
    return np.array(points) if points else np.array([])

class mIoU():
    
    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union
        # print(self.total_correct,self.total_label,self.total_inter,self.total_union)
    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return float(pixAcc), mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

    def IoU(self, preds, labels):
        inter, union = batch_intersection_union(preds, labels)
        # print("inter: ",inter)
        # print("union: ", union)
        IoU = 1.0 * inter / (np.spacing(1) + union)
        return inter[0], union[0],IoU[0]

class PD_FA():
    def __init__(self,):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0
        # 添加缺失的bins属性
        self.bins = 1
        
    def update(self, preds, labels, size=None):
        # 支持批量输入
        if preds.ndim == 4:  # [B, 1, H, W]
            batch_size = preds.shape[0]
            for i in range(batch_size):
                self._update_single_sample(preds[i, 0], labels[i, 0], size)
        else:  # [1, H, W] or [H, W]
            self._update_single_sample(preds.squeeze(), labels.squeeze(), size)
    
    def _update_single_sample(self, preds, labels, size):
        """更新单个样本的PD_FA指标"""
        # 确保张量在CPU上再转换为numpy
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu()
            
        predits  = np.array(preds).astype('int64')
        labelss = np.array(labels).astype('int64') 

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss , connectivity=2)
        coord_label = measure.regionprops(label)

        self.target    += len(coord_label)
        self.image_area_total = []
        self.distance_match   = []
        self.dismatch         = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        true_img = np.zeros(predits.shape)
        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    true_img[coord_image[m].coords[:,0], coord_image[m].coords[:,1]] = 1
                    del coord_image[m]
                    break

        self.dismatch_pixel += (predits - true_img).sum()
        if size is not None:
            self.all_pixel += size[0] * size[1]
        else:
            # 如果没有提供size，使用图像的实际尺寸
            self.all_pixel += predits.shape[0] * predits.shape[1]
        self.PD += len(self.distance_match)

    def get(self):
        Final_FA =  self.dismatch_pixel / self.all_pixel
        Final_PD =  self.PD / self.target
        return Final_PD, float(Final_FA)

    def reset(self):
        # 修复reset方法，重置所有相关属性
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0

def batch_pix_accuracy(output, target):   
    if len(target.shape) == 3:
        target = np.expand_dims(target.astype('float64'), axis=1)
    elif len(target.shape) == 4:
        target = target.astype('float64')
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).astype('float64')
    pixel_labeled = (target > 0).astype('float64').sum()
    pixel_correct = (((predict == target).astype('float64'))*((target > 0).astype('float64'))).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

def batch_intersection_union(output, target):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).astype('float64')
    if len(target.shape) == 3:
        target = np.expand_dims(target.astype('float64'), axis=1)
    elif len(target.shape) == 4:
        target = target.astype('float64')
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).astype('float64'))

    # 优化：减少CPU-GPU数据传输，在GPU上计算histogram
    if hasattr(torch, 'histc') and isinstance(intersection, torch.Tensor):  # 使用torch的histc函数
        area_inter = torch.histc(intersection, bins=nbins, min=mini, max=maxi)
        area_pred = torch.histc(predict, bins=nbins, min=mini, max=maxi)
        area_lab = torch.histc(target, bins=nbins, min=mini, max=maxi)
        area_union = area_pred + area_lab - area_inter
        return area_inter, area_union
    else:
        # 回退到原来的numpy方法
        if isinstance(intersection, torch.Tensor):
            intersection = intersection.cpu()
            predict = predict.cpu()
            target = target.cpu()
        area_inter, _  = np.histogram(intersection, bins=nbins, range=(mini, maxi))
        area_pred,  _  = np.histogram(predict, bins=nbins, range=(mini, maxi))
        area_lab,   _  = np.histogram(target, bins=nbins, range=(mini, maxi))
        area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union

class SigmoidMetric():
    def __init__(self):
        self.reset()

    def update(self, pred, labels):
        correct, labeled = self.batch_pix_accuracy(pred, labels)
        inter, union = self.batch_intersection_union(pred, labels)

        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        """Gets the current evaluation result."""
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

    def batch_pix_accuracy(self, output, target):
        assert output.shape == target.shape
        # 优化：如果输入是tensor，在GPU上计算，最后再转到numpy
        if isinstance(output, torch.Tensor):
            output = output.detach()
            target = target.detach()
            
            predict = (output > 0).float()
            pixel_labeled = (target > 0).float().sum()
            pixel_correct = ((predict == target).float() * (target > 0).float()).sum()
            
            return pixel_correct.item(), pixel_labeled.item()
        else:
            # 原来的numpy方法
            output = output.detach().numpy()
            target = target.detach().numpy()

            predict = (output > 0).astype('int64') # P
            pixel_labeled = np.sum(target > 0) # T
            pixel_correct = np.sum((predict == target)*(target > 0)) # TP
            assert pixel_correct <= pixel_labeled
            return pixel_correct, pixel_labeled

    def batch_intersection_union(self, output, target):
        mini = 1
        maxi = 1 # nclass
        nbins = 1 # nclass
        
        # 优化：如果输入是tensor，在GPU上计算
        if isinstance(output, torch.Tensor):
            output = output.detach()
            target = target.detach()
            
            predict = (output > 0).float()
            intersection = predict * (predict == target).float()
            
            # 使用torch的histc函数在GPU上计算
            if hasattr(torch, 'histc'):
                area_inter = torch.histc(intersection, bins=nbins, min=mini, max=maxi)
                area_pred = torch.histc(predict, bins=nbins, min=mini, max=maxi)
                area_lab = torch.histc(target, bins=nbins, min=mini, max=maxi)
                area_union = area_pred + area_lab - area_inter
                return area_inter, area_union
        
        # 原来的numpy方法
        predict = (output.detach().numpy()).astype('int64') # P
        target = target.numpy().astype('int64') # T
        intersection = predict * (predict == target) # TP

        # areas of intersection and union
        area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
        area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
        area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
        area_union = area_pred + area_lab - area_inter
        assert (area_inter <= area_union).all()
        return area_inter, area_union


class SamplewiseSigmoidMetric():
    def __init__(self, nclass, score_thresh=0.5):
        # 设置分类数量
        self.nclass = nclass
        # 设置分数阈值
        self.score_thresh = score_thresh
        # 重置模型状态
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result."""
        # 计算每个类别的交集和并集
        inter_arr, union_arr = self.batch_intersection_union(preds, labels,
                                                             self.nclass, self.score_thresh)

        # 将当前批次的交集结果追加到总交集数组中
        self.total_inter = np.append(self.total_inter, inter_arr)

        # 将当前批次的并集结果追加到总并集数组中
        self.total_union = np.append(self.total_union, union_arr)

    def get(self):
        """Gets the current evaluation result."""
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return IoU, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = np.array([])
        self.total_union = np.array([])
        self.total_correct = np.array([])
        self.total_label = np.array([])

    def batch_intersection_union(self, output, target, nclass, score_thresh):
        """mIoU"""
        # 输入是tensor
        # 类别0是忽略的类别，通常用于背景/边界
        # inputs are tensor
        # the category 0 is ignored class, typically for background / boundary
        mini = 1
        maxi = 1  # nclass
        nbins = 1  # nclass
        # 大于score_thresh
        #> score_thresh
        
        # 确保张量在CPU上再转换为numpy
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
            
        predict = output.astype('int64') # P
        target = target.astype('int64') # T
        # 计算交集（真正例）
        intersection = predict * (predict == target) # TP

        num_sample = intersection.shape[0]
        area_inter_arr = np.zeros(num_sample)
        area_pred_arr = np.zeros(num_sample)
        area_lab_arr = np.zeros(num_sample)
        area_union_arr = np.zeros(num_sample)

        for b in range(num_sample):
            # 计算交集和并集的面积
            # areas of intersection and union
            area_inter, _ = np.histogram(intersection[b], bins=nbins, range=(mini, maxi))
            area_inter_arr[b] = area_inter

            area_pred, _ = np.histogram(predict[b], bins=nbins, range=(mini, maxi))
            area_pred_arr[b] = area_pred

            area_lab, _ = np.histogram(target[b], bins=nbins, range=(mini, maxi))
            area_lab_arr[b] = area_lab

            # 并集面积 = 预测面积 + 真实标签面积 - 交集面积
            area_union = area_pred + area_lab - area_inter
            area_union_arr[b] = area_union

            assert (area_inter <= area_union).all()

        return area_inter_arr, area_union_arr

def _process_single_sample_worker(args):
    """
    工作函数：处理单个样本的F1和MSE计算
    这个函数需要在模块级别定义以便多进程使用
    """
    pred_mask, gt_mask, threshold, distance_threshold = args
    
    try:
        # 应用阈值获取二值化预测
        pred_binary = (pred_mask > threshold).astype(np.int64)
        gt_binary = (gt_mask > 0).astype(np.int64)
        
        # 从掩码中提取点坐标
        pred_points = extract_points_from_mask(pred_binary)
        gt_points = extract_points_from_mask(gt_binary)
        
        # 使用优化的score_frame函数计算指标
        # print(f"pred_points: {pred_points[:10]},{len(pred_points)}, gt_points: {gt_points},{len(gt_points)}" )    
        # score_time_start = time.time()
        TP, FN, FP, sse = _score_frame_optimized_worker(
            pred_points, 
            gt_points, 
            tau=distance_threshold, 
            eps=3
        )
        # score_time_end = time.time()
        # score_time = score_time_end - score_time_start
        # print(f"Score time: {score_time:.2f}s")
        
        # 计算当前样本的平均MSE
        total_points = max(len(gt_points), len(pred_points))
        if total_points > 0:
            current_mse = sse / total_points
        else:
            current_mse = 0
        
        return {
            'tp': TP,
            'fp': FP,
            'fn': FN,
            'mse': current_mse,
            'success': True
        }
        
    except Exception as e:
        # 返回错误信息
        return {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'mse': 0,
            'success': False,
            'error': str(e)
        }

def _score_frame_optimized_worker(X, Y, tau=1000.0, eps=3.0):
    """ 
    优化的评分函数，专门用于多进程工作函数
    Scoring Prediction X on ground-truth Y by linear assignment.
    """
    if len(X) == 0 and len(Y) == 0:
        # no objects, no predictions means perfect score
        TP, FN, FP, sse = 0, 0, 0, 0
    elif len(X) == 0 and len(Y) > 0:
        # no predictions but objects means false negatives
        TP, FN, FP, sse = 0, len(Y), 0, len(Y) * tau**2
    elif len(X) > 0 and len(Y) == 0:
        # predictions but no objects means false positives
        TP, FN, FP, sse = 0, 0, len(X), len(X) * tau**2
    else:
        # compute Euclidean distances between prediction and ground truth
        D = cdist(X, Y)
        
        # truncate distances that violate the threshold
        D[D > tau] = 1000
        
        # compute matching by solving linear assignment problem
        row_ind, col_ind = linear_sum_assignment(D)
        matching = D[row_ind, col_ind]
        
        # true positives are matches within the threshold
        TP = sum(matching <= tau)
        
        # false negatives are missed ground truth points or matchings that violate the threshold
        FN = len(Y) - len(row_ind) + sum(matching > tau)
        
        # false positives are missing predictions or matchings that violate the threshold
        FP = len(X) - len(row_ind) + sum(matching > tau)
        
        # compute truncated regression error
        tp_distances = matching[matching < tau]
        # truncation
        tp_distances[tp_distances < eps] = 0
        # squared error with constant punishment for false negatives and true positives
        sse = sum(tp_distances) + (FN + FP) * tau**2
    
    return TP, FN, FP, sse

class F1MSEMetric():
    """Combined F1 Score and MSE metric using Hungarian algorithm for point matching"""
    
    def __init__(self, threshold=0.5, distance_threshold=5.0, use_multiprocessing=False, num_workers=None, chunk_size=None):
        self.threshold = threshold
        self.distance_threshold = distance_threshold
        self.use_multiprocessing = use_multiprocessing
        self.num_workers = num_workers if num_workers is not None else min(mp.cpu_count(), 8)
        # 设置chunk_size以优化多进程性能
        self.chunk_size = chunk_size if chunk_size is not None else max(1, 32 // self.num_workers)
        self.reset()
    
    def score_frame_optimized(self, X, Y, tau=10.0, eps=3.0):
        """ 
        优化的评分函数，参考用户提供的score_frame实现
        Scoring Prediction X on ground-truth Y by linear assignment.
        """
        if len(X) == 0 and len(Y) == 0:
            # no objects, no predictions means perfect score
            TP, FN, FP, sse = 0, 0, 0, 0
        elif len(X) == 0 and len(Y) > 0:
            # no predictions but objects means false negatives
            TP, FN, FP, sse = 0, len(Y), 0, len(Y) * tau**2
        elif len(X) > 0 and len(Y) == 0:
            # predictions but no objects means false positives
            TP, FN, FP, sse = 0, 0, len(X), len(X) * tau**2
        else:
            # compute Euclidean distances between prediction and ground truth
            D = cdist(X, Y)
            
            # truncate distances that violate the threshold
            D[D > tau] = 1000
            
            # compute matching by solving linear assignment problem
            row_ind, col_ind = linear_sum_assignment(D)
            matching = D[row_ind, col_ind]
            
            # true positives are matches within the threshold
            TP = sum(matching <= tau)
            
            # false negatives are missed ground truth points or matchings that violate the threshold
            FN = len(Y) - len(row_ind) + sum(matching > tau)
            
            # false positives are missing predictions or matchings that violate the threshold
            FP = len(X) - len(row_ind) + sum(matching > tau)
            
            # compute truncated regression error
            tp_distances = matching[matching < tau]
            # truncation
            tp_distances[tp_distances < eps] = 0
            # squared error with constant punishment for false negatives and true positives
            sse = sum(tp_distances) + (FN + FP) * tau**2
        
        return TP, FN, FP, sse
    
    def update(self, preds, labels):
        """Update both F1 and MSE with new predictions and labels"""
        # 确保张量在CPU上再转换为numpy
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # 处理批次维度
        if preds.ndim == 4:  # [B, 1, H, W]
            batch_size = preds.shape[0]
            
            if self.use_multiprocessing and batch_size > 1:
                self._update_batch_multiprocessing(preds, labels, batch_size)
            else:
                # 单进程处理
                for b in range(batch_size):
                    self._update_single_sample(preds[b, 0], labels[b, 0])
        else:  # [1, H, W] or [H, W]
            self._update_single_sample(preds.squeeze(), labels.squeeze())
    
    def _update_batch_multiprocessing(self, preds, labels, batch_size):
        """使用优化的多进程处理批次数据"""
        try:
            # 准备参数
            args_list = []
            for b in range(batch_size):
                args = (
                    preds[b, 0].copy(),  # 使用copy避免共享内存问题
                    labels[b, 0].copy(), 
                    self.threshold, 
                    self.distance_threshold
                )
                args_list.append(args)
            
            # 使用进程池处理，设置chunk_size优化性能
            with mp.Pool(processes=self.num_workers, 
                        initializer=_init_worker, 
                        initargs=(self.threshold, self.distance_threshold)) as pool:
                results = pool.map(_process_single_sample_worker, args_list, 
                                 chunksize=self.chunk_size)
            
            # 收集结果
            successful_results = 0
            for result in results:
                if result['success']:
                    self.total_tp += result['tp']
                    self.total_fp += result['fp']
                    self.total_fn += result['fn']
                    self.total_mse += result['mse']
                    successful_results += 1
                else:
                    # 记录错误但继续处理
                    warnings.warn(f"Sample processing failed: {result.get('error', 'Unknown error')}")
            
            self.count += successful_results
                    
        except Exception as e:
            # 如果多进程失败，回退到单进程
            warnings.warn(f"Multiprocessing failed, falling back to single process: {e}")
            for b in range(batch_size):
                self._update_single_sample(preds[b, 0], labels[b, 0])
    
    def _update_single_sample(self, pred_mask, gt_mask):
        """更新单个样本的F1和MSE分数（优化版本）"""
        # 应用阈值获取二值化预测
        pred_binary = (pred_mask > self.threshold).astype(np.int64)
        gt_binary = (gt_mask > 0).astype(np.int64)
        
        # 从掩码中提取点坐标
        pred_points = extract_points_from_mask(pred_binary)
        gt_points = extract_points_from_mask(gt_binary)
        
        # 使用优化的score_frame函数计算指标
        TP, FN, FP, sse = self.score_frame_optimized(
            pred_points, 
            gt_points, 
            tau=self.distance_threshold, 
            eps=3
        )
        
        # 更新累计指标
        self.total_tp += TP
        self.total_fp += FP
        self.total_fn += FN
        
        # 计算当前样本的平均MSE
        total_points = max(len(gt_points), len(pred_points))
        if total_points > 0:
            current_mse = sse / total_points
        else:
            current_mse = 0
        
        self.total_mse += current_mse
        self.count += 1
    
    def get_f1(self):
        """Get current F1 score"""
        precision = self.total_tp / (self.total_tp + self.total_fp + np.spacing(1))
        recall = self.total_tp / (self.total_tp + self.total_fn + np.spacing(1))
        f1_score = 2 * precision * recall / (precision + recall + np.spacing(1))
        return float(f1_score)
    
    def get_mse(self):
        """Get current average MSE"""
        return float(self.total_mse / (self.count + np.spacing(1)))
    
    def get(self):
        """Get both F1 and MSE scores"""
        return self.get_f1(), self.get_mse()
    
    def reset(self):
        """Reset metric state"""
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_mse = 0.0
        self.count = 0

def _init_worker(threshold, distance_threshold):
    """初始化工作进程，设置全局变量"""
    global _worker_threshold, _worker_distance_threshold
    _worker_threshold = threshold
    _worker_distance_threshold = distance_threshold
