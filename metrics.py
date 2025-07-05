import numpy as np
import torch
from skimage import measure
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple
       
def calculate_distance(point1: List[float], point2: List[float]) -> float:
    """计算两点之间的欧氏距离"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def match_points(pred_points: List[List[float]], gt_points: List[List[float]], 
                distance_threshold: float = 5.0) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    使用匈牙利算法匹配预测点和真实点
    Args:
        pred_points: 预测的点坐标列表 [[x1,y1], [x2,y2], ...]
        gt_points: 真实的点坐标列表 [[x1,y1], [x2,y2], ...]
        distance_threshold: 匹配的距离阈值
    Returns:
        matches: 匹配的点对索引
        unmatched_preds: 未匹配的预测点索引
        unmatched_gts: 未匹配的真实点索引
    """
    if len(pred_points) == 0 or len(gt_points) == 0:
        return [], list(range(len(pred_points))), list(range(len(gt_points)))

    # 构建代价矩阵
    cost_matrix = np.zeros((len(pred_points), len(gt_points)))
    for i, pred in enumerate(pred_points):
        for j, gt in enumerate(gt_points):
            distance = calculate_distance(pred, gt)
            cost_matrix[i, j] = distance

    # 使用匈牙利算法进行匹配
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    # 根据距离阈值筛选有效匹配
    matches = []
    unmatched_preds = list(range(len(pred_points)))
    unmatched_gts = list(range(len(gt_points)))

    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if cost_matrix[pred_idx, gt_idx] <= distance_threshold:
            matches.append((pred_idx, gt_idx))
            if pred_idx in unmatched_preds:
                unmatched_preds.remove(pred_idx)
            if gt_idx in unmatched_gts:
                unmatched_gts.remove(gt_idx)

    return matches, unmatched_preds, unmatched_gts

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
        self.target= 0
    def update(self, preds, labels, size):
        predits  = np.array((preds).cpu()).astype('int64')
        labelss = np.array((labels).cpu()).astype('int64') 

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
        self.all_pixel +=size[0]*size[1]
        self.PD +=len(self.distance_match)

    def get(self):
        Final_FA =  self.dismatch_pixel / self.all_pixel
        Final_PD =  self.PD /self.target
        return Final_PD, float(Final_FA.cpu().detach().numpy())

    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])

def batch_pix_accuracy(output, target):   
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

def batch_intersection_union(output, target):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
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
        predict = output.detach().numpy().astype('int64') # P
        target = target.detach().numpy().astype('int64') # T
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

class F1Metric():
    """F1 Score metric using Hungarian algorithm for point matching"""
    
    def __init__(self, threshold=0.5, distance_threshold=5.0):
        self.threshold = threshold
        self.distance_threshold = distance_threshold
        self.reset()
    
    def update(self, preds, labels):
        """Update F1 score with new predictions and labels"""
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # 处理批次维度
        if preds.ndim == 4:  # [B, 1, H, W]
            batch_size = preds.shape[0]
            for b in range(batch_size):
                self._update_single_sample(preds[b, 0], labels[b, 0])
        else:  # [1, H, W] or [H, W]
            self._update_single_sample(preds.squeeze(), labels.squeeze())
    
    def _update_single_sample(self, pred_mask, gt_mask):
        """更新单个样本的F1分数"""
        # 应用阈值获取二值化预测
        pred_binary = (pred_mask > self.threshold).astype(np.int64)
        gt_binary = (gt_mask > 0).astype(np.int64)
        
        # 从掩码中提取点坐标
        pred_points = extract_points_from_mask(pred_binary)
        gt_points = extract_points_from_mask(gt_binary)
        
        # 使用匈牙利算法匹配点
        matches, unmatched_preds, unmatched_gts = match_points(
            pred_points.tolist() if len(pred_points) > 0 else [],
            gt_points.tolist() if len(gt_points) > 0 else [],
            self.distance_threshold
        )
        
        # 计算TP, FP, FN
        tp = len(matches)
        fp = len(unmatched_preds)
        fn = len(unmatched_gts)
        
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
    
    def get(self):
        """Get current F1 score"""
        precision = self.total_tp / (self.total_tp + self.total_fp + np.spacing(1))
        recall = self.total_tp / (self.total_tp + self.total_fn + np.spacing(1))
        f1_score = 2 * precision * recall / (precision + recall + np.spacing(1))
        return float(f1_score)
    
    def reset(self):
        """Reset metric state"""
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0


class MSEMetric():
    """Mean Squared Error metric using Hungarian algorithm for point matching"""
    
    def __init__(self, threshold=0.5, distance_threshold=5.0):
        self.threshold = threshold
        self.distance_threshold = distance_threshold
        self.reset()
    
    def update(self, preds, labels):
        """Update MSE with new predictions and labels"""
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # 处理批次维度
        if preds.ndim == 4:  # [B, 1, H, W]
            batch_size = preds.shape[0]
            for b in range(batch_size):
                self._update_single_sample(preds[b, 0], labels[b, 0])
        else:  # [1, H, W] or [H, W]
            self._update_single_sample(preds.squeeze(), labels.squeeze())
    
    def _update_single_sample(self, pred_mask, gt_mask):
        """更新单个样本的MSE"""
        # 应用阈值获取二值化预测
        pred_binary = (pred_mask > self.threshold).astype(np.int64)
        gt_binary = (gt_mask > 0).astype(np.int64)
        
        # 从掩码中提取点坐标
        pred_points = extract_points_from_mask(pred_binary)
        gt_points = extract_points_from_mask(gt_binary)
        
        # 使用匈牙利算法匹配点
        matches, unmatched_preds, unmatched_gts = match_points(
            pred_points.tolist() if len(pred_points) > 0 else [],
            gt_points.tolist() if len(gt_points) > 0 else [],
            self.distance_threshold
        )
        
        # 计算当前样本的MSE
        current_mse = 0
        tau_square = self.distance_threshold * self.distance_threshold
        
        # 对匹配点计算实际距离的平方
        for pred_idx, gt_idx in matches:
            distance = calculate_distance(pred_points[pred_idx], gt_points[gt_idx])
            if distance <= self.distance_threshold:
                current_mse += 0  # 距离小于阈值时为0
            else:
                current_mse += distance * distance
        
        # 对未匹配点添加惩罚项
        current_mse += (len(unmatched_preds) + len(unmatched_gts)) * tau_square
        
        # 计算当前样本的平均MSE
        total_points = max(len(gt_points), len(pred_points))
        if total_points > 0:
            current_mse = current_mse / total_points
        else:
            current_mse = 0
        
        self.total_mse += current_mse
        self.count += 1
    
    def get(self):
        """Get current average MSE"""
        return float(self.total_mse / (self.count + np.spacing(1)))
    
    def reset(self):
        """Reset metric state"""
        self.total_mse = 0.0
        self.count = 0
