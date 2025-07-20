import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import copy
import math

class StatisticalOutlierProcessor:
    """基于统计方法的异常点处理器"""
    
    def __init__(self, 
                 base_distance_threshold: float = 200.0,
                 mahalanobis_threshold: float = 2.0,
                 z_score_threshold: float = 3.0,
                 iqr_multiplier: float = 1.5,
                 min_points_for_stats: int = 3):
        """
        初始化统计异常点处理器
        
        Args:
            base_distance_threshold: 基础距离阈值
            mahalanobis_threshold: 马氏距离阈值
            z_score_threshold: Z-score阈值
            iqr_multiplier: IQR倍数阈值
            min_points_for_stats: 进行统计计算的最小点数
        """
        self.base_distance_threshold = base_distance_threshold
        self.mahalanobis_threshold = mahalanobis_threshold
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier
        self.min_points_for_stats = min_points_for_stats
    
    def calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """计算两点之间的欧氏距离"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def extract_sequence_info(self, predictions: Dict) -> Dict[int, Dict[int, Dict]]:
        """从预测结果中提取序列信息"""
        sequence_data = defaultdict(dict)
        
        for img_name, pred_info in predictions.items():
            parts = img_name.split('_')
            if len(parts) >= 2:
                try:
                    sequence_id = int(parts[0])
                    frame = int(parts[1])
                    sequence_data[sequence_id][frame] = {
                        'coords': pred_info['coords'],
                        'num_objects': pred_info['num_objects'],
                        'image_name': img_name
                    }
                except ValueError:
                    continue
        
        return sequence_data
    
    def mahalanobis_outlier_detection(self, points: List[List[float]]) -> Tuple[List[bool], Dict]:
        """
        使用马氏距离进行异常点检测
        
        Args:
            points: 点列表
            
        Returns:
            is_outlier: 是否为异常点的布尔列表
            stats: 统计信息
        """
        if len(points) < self.min_points_for_stats:
            return [False] * len(points), {}
        
        points_array = np.array(points)
        
        # 计算统计特征
        x_coords = points_array[:, 0]
        y_coords = points_array[:, 1]
        
        x_mean, x_std = np.mean(x_coords), np.std(x_coords)
        y_mean, y_std = np.mean(y_coords), np.std(y_coords)
        
        # 计算马氏距离
        is_outlier = []
        mahalanobis_distances = []
        
        for point in points:
            x_score = abs(point[0] - x_mean) / (x_std + 1e-6)
            y_score = abs(point[1] - y_mean) / (y_std + 1e-6)
            mahalanobis_dist = np.sqrt(x_score**2 + y_score**2)
            mahalanobis_distances.append(mahalanobis_dist)
            
            # 使用马氏距离阈值判断异常点
            is_outlier.append(mahalanobis_dist > self.mahalanobis_threshold)
        
        stats = {
            'x_mean': x_mean,
            'x_std': x_std,
            'y_mean': y_mean,
            'y_std': y_std,
            'mahalanobis_distances': mahalanobis_distances,
            'mahalanobis_threshold': self.mahalanobis_threshold
        }
        
        return is_outlier, stats
    
    def z_score_outlier_detection(self, points: List[List[float]]) -> Tuple[List[bool], Dict]:
        """
        使用Z-score进行异常点检测
        
        Args:
            points: 点列表
            
        Returns:
            is_outlier: 是否为异常点的布尔列表
            stats: 统计信息
        """
        if len(points) < self.min_points_for_stats:
            return [False] * len(points), {}
        
        points_array = np.array(points)
        
        # 计算统计特征
        x_coords = points_array[:, 0]
        y_coords = points_array[:, 1]
        
        x_mean, x_std = np.mean(x_coords), np.std(x_coords)
        y_mean, y_std = np.mean(y_coords), np.std(y_coords)
        
        # 计算Z-score
        is_outlier = []
        z_scores = []
        
        for point in points:
            x_z_score = abs(point[0] - x_mean) / (x_std + 1e-6)
            y_z_score = abs(point[1] - y_mean) / (y_std + 1e-6)
            max_z_score = max(x_z_score, y_z_score)
            z_scores.append(max_z_score)
            
            # 使用Z-score阈值判断异常点
            is_outlier.append(max_z_score > self.z_score_threshold)
        
        stats = {
            'x_mean': x_mean,
            'x_std': x_std,
            'y_mean': y_mean,
            'y_std': y_std,
            'z_scores': z_scores,
            'z_score_threshold': self.z_score_threshold
        }
        
        return is_outlier, stats
    
    def iqr_outlier_detection(self, points: List[List[float]]) -> Tuple[List[bool], Dict]:
        """
        使用IQR（四分位距）进行异常点检测
        
        Args:
            points: 点列表
            
        Returns:
            is_outlier: 是否为异常点的布尔列表
            stats: 统计信息
        """
        if len(points) < self.min_points_for_stats:
            return [False] * len(points), {}
        
        points_array = np.array(points)
        
        # 分别计算x和y坐标的IQR
        x_coords = points_array[:, 0]
        y_coords = points_array[:, 1]
        
        x_q1, x_q3 = np.percentile(x_coords, [25, 75])
        y_q1, y_q3 = np.percentile(y_coords, [25, 75])
        
        x_iqr = x_q3 - x_q1
        y_iqr = y_q3 - y_q1
        
        # 计算边界
        x_lower_bound = x_q1 - self.iqr_multiplier * x_iqr
        x_upper_bound = x_q3 + self.iqr_multiplier * x_iqr
        y_lower_bound = y_q1 - self.iqr_multiplier * y_iqr
        y_upper_bound = y_q3 + self.iqr_multiplier * y_iqr
        
        # 判断异常点
        is_outlier = []
        for point in points:
            x_outlier = point[0] < x_lower_bound or point[0] > x_upper_bound
            y_outlier = point[1] < y_lower_bound or point[1] > y_upper_bound
            is_outlier.append(x_outlier or y_outlier)
        
        stats = {
            'x_q1': x_q1, 'x_q3': x_q3, 'x_iqr': x_iqr,
            'y_q1': y_q1, 'y_q3': y_q3, 'y_iqr': y_iqr,
            'x_bounds': [x_lower_bound, x_upper_bound],
            'y_bounds': [y_lower_bound, y_upper_bound],
            'iqr_multiplier': self.iqr_multiplier
        }
        
        return is_outlier, stats
    
    def distance_based_outlier_detection(self, points: List[List[float]]) -> Tuple[List[bool], Dict]:
        """
        基于距离的异常点检测
        
        Args:
            points: 点列表
            
        Returns:
            is_outlier: 是否为异常点的布尔列表
            stats: 统计信息
        """
        if len(points) < self.min_points_for_stats:
            return [False] * len(points), {}
        
        # 计算所有点对之间的距离
        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = self.calculate_distance(points[i], points[j])
                distances.append(dist)
        
        if not distances:
            return [False] * len(points), {}
        
        # 计算距离统计
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        distance_threshold = mean_distance + self.z_score_threshold * std_distance
        
        # 计算每个点到其他点的平均距离
        is_outlier = []
        avg_distances = []
        
        for i, point in enumerate(points):
            point_distances = []
            for j, other_point in enumerate(points):
                if i != j:
                    dist = self.calculate_distance(point, other_point)
                    point_distances.append(dist)
            
            if point_distances:
                avg_dist = np.mean(point_distances)
                avg_distances.append(avg_dist)
                is_outlier.append(avg_dist > distance_threshold)
            else:
                avg_distances.append(0)
                is_outlier.append(False)
        
        stats = {
            'mean_distance': mean_distance,
            'std_distance': std_distance,
            'distance_threshold': distance_threshold,
            'avg_distances': avg_distances
        }
        
        return is_outlier, stats
    
    def simple_statistical_detection(self, points: List[List[float]]) -> Tuple[List[bool], Dict]:
        """
        简单的统计异常点检测（只使用马氏距离）
        
        Args:
            points: 点列表
            
        Returns:
            is_outlier: 是否为异常点的布尔列表
            stats: 统计信息
        """
        if len(points) < self.min_points_for_stats:
            return [False] * len(points), {}
        
        points_array = np.array(points)
        
        # 计算统计特征
        x_coords = points_array[:, 0]
        y_coords = points_array[:, 1]
        
        x_mean, x_std = np.mean(x_coords), np.std(x_coords)
        y_mean, y_std = np.mean(y_coords), np.std(y_coords)
        
        # 计算马氏距离
        is_outlier = []
        mahalanobis_distances = []
        
        for point in points:
            x_score = abs(point[0] - x_mean) / (x_std + 1e-6)
            y_score = abs(point[1] - y_mean) / (y_std + 1e-6)
            mahalanobis_dist = np.sqrt(x_score**2 + y_score**2)
            mahalanobis_distances.append(mahalanobis_dist)
            
            # 使用马氏距离阈值判断异常点
            is_outlier.append(mahalanobis_dist > self.mahalanobis_threshold)
        
        stats = {
            'x_mean': x_mean,
            'x_std': x_std,
            'y_mean': y_mean,
            'y_std': y_std,
            'mahalanobis_distances': mahalanobis_distances,
            'mahalanobis_threshold': self.mahalanobis_threshold,
            'outlier_count': sum(is_outlier)
        }
        
        return is_outlier, stats
    
    def calculate_statistical_consensus(self, methods_results: Dict) -> List[bool]:
        """
        计算统计方法的共识异常点
        
        Args:
            methods_results: 各种统计方法的检测结果
            
        Returns:
            consensus_outliers: 共识异常点列表
        """
        if not methods_results:
            return []
        
        # 统计每个点被多少种方法标记为异常点
        point_votes = defaultdict(int)
        total_methods = len(methods_results)
        
        for method_name, method_result in methods_results.items():
            if 'is_outlier' in method_result:
                for i, is_outlier in enumerate(method_result['is_outlier']):
                    if is_outlier:
                        point_votes[i] += 1
        
        # 需要至少2种方法认为异常才标记为异常点
        min_consensus = max(2, total_methods // 2)
        consensus_outliers = [point_votes[i] >= min_consensus for i in range(len(next(iter(methods_results.values()))['is_outlier']))]
        
        return consensus_outliers
    
    def filter_outliers_from_predictions(self, predictions: Dict) -> Tuple[Dict, Dict]:
        """
        从预测结果中过滤异常点（使用简单的马氏距离方法）
        
        Args:
            predictions: 原始预测结果
            
        Returns:
            filtered_predictions: 过滤后的预测结果
            outlier_info: 异常点信息
        """
        sequence_data = self.extract_sequence_info(predictions)
        filtered_predictions = copy.deepcopy(predictions)
        outlier_info = {}
        
        for sequence_id, frames_data in sequence_data.items():
            print(f"处理序列 {sequence_id}，包含 {len(frames_data)} 帧")
            
            # 收集序列中的所有点
            all_points = []
            point_to_image = []
            
            for frame, frame_data in frames_data.items():
                for coord in frame_data['coords']:
                    all_points.append(coord)
                    point_to_image.append(frame_data['image_name'])
            
            if len(all_points) < self.min_points_for_stats:
                continue
            
            # 进行简单的统计异常点检测
            is_outlier, detection_stats = self.simple_statistical_detection(all_points)
            
            # 记录异常点信息
            outlier_points = []
            for i, (point, image_name) in enumerate(zip(all_points, point_to_image)):
                if is_outlier[i]:
                    outlier_points.append({
                        'point': point,
                        'image_name': image_name,
                        'sequence_id': sequence_id
                    })
            
            outlier_info[sequence_id] = {
                'outlier_points': outlier_points,
                'total_points': len(all_points),
                'outlier_count': len(outlier_points),
                'detection_stats': detection_stats
            }
            
            # 过滤异常点
            for frame, frame_data in frames_data.items():
                image_name = frame_data['image_name']
                if image_name in filtered_predictions:
                    # 找到该图像中的异常点索引
                    outlier_indices = []
                    for i, (point, img_name) in enumerate(zip(all_points, point_to_image)):
                        if img_name == image_name and is_outlier[i]:
                            # 找到在原始coords中的索引
                            for j, coord in enumerate(filtered_predictions[image_name]['coords']):
                                if self.calculate_distance(point, coord) < 1e-6:  # 浮点数比较
                                    outlier_indices.append(j)
                                    break
                    
                    # 移除异常点
                    for idx in sorted(outlier_indices, reverse=True):
                        del filtered_predictions[image_name]['coords'][idx]
                    
                    # 更新num_objects
                    filtered_predictions[image_name]['num_objects'] = len(filtered_predictions[image_name]['coords'])
            
            print(f"  序列 {sequence_id}: 总点数 {len(all_points)}, 异常点数 {len(outlier_points)}")
        
        return filtered_predictions, outlier_info
    
    def process_sequence(self, predictions: Dict) -> Dict:
        """
        处理整个预测结果，进行简单统计异常点筛选
        
        Args:
            predictions: 原始预测结果
            
        Returns:
            filtered_predictions: 过滤后的预测结果
        """
        print("开始简单统计异常点筛选处理...")
        
        # 使用马氏距离方法进行异常点检测
        filtered_predictions, outlier_info = self.filter_outliers_from_predictions(predictions)
        
        # 统计结果
        total_outliers = sum(info['outlier_count'] for info in outlier_info.values())
        total_points = sum(info['total_points'] for info in outlier_info.values())
        
        print(f"简单统计异常点筛选完成！")
        print(f"总检测点数: {total_points}")
        print(f"异常点数: {total_outliers}")
        print(f"异常点比例: {total_outliers/total_points*100:.2f}%")
        
        return filtered_predictions
    
    def evaluate_improvement(self, original_predictions: Dict, filtered_predictions: Dict, ground_truth: List[Dict]) -> Dict:
        """评估统计异常点筛选的效果改善"""
        import sys
        import os
        # 添加父目录到路径，以便导入eval_predictions
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from eval_predictions import calculate_metrics
        
        # 计算原始预测的指标
        original_metrics = calculate_metrics(original_predictions, ground_truth, self.base_distance_threshold)
        
        # 计算过滤后预测的指标
        filtered_metrics = calculate_metrics(filtered_predictions, ground_truth, self.base_distance_threshold)
        
        # 计算改善程度
        improvement = {
            'precision_improvement': filtered_metrics['precision'] - original_metrics['precision'],
            'recall_improvement': filtered_metrics['recall'] - original_metrics['recall'],
            'f1_improvement': filtered_metrics['f1'] - original_metrics['f1'],
            'mse_improvement': original_metrics['mse'] - filtered_metrics['mse'],
            'original_metrics': original_metrics,
            'filtered_metrics': filtered_metrics,
            'total_tp_improvement': filtered_metrics['total_tp'] - original_metrics['total_tp'],
            'total_fp_improvement': filtered_metrics['total_fp'] - original_metrics['total_fp'],
            'total_fn_improvement': filtered_metrics['total_fn'] - original_metrics['total_fn'],
        }
        
        return improvement

def main():
    """主函数，演示简单统计异常点筛选和评测"""
    # 加载预测结果和真实标注
    pred_path = 'results/spotgeov2/WTNet/sequence_slope_processed_predictions.json'
    gt_path = 'datasets/spotgeov2-IRSTD/test_anno.json'
    
    print("正在加载预测结果和真实标注...")
    with open(pred_path, 'r') as f:
        original_predictions = json.load(f)
    
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    # 创建简单统计异常点处理器
    processor = StatisticalOutlierProcessor(
        base_distance_threshold=500.0,
        mahalanobis_threshold=3.0,
        z_score_threshold=3.0,
        iqr_multiplier=1.5,
        min_points_for_stats=3
    )
    
    # 进行简单统计异常点筛选
    filtered_predictions = processor.process_sequence(original_predictions)
    
    # 评估改善效果
    improvement = processor.evaluate_improvement(original_predictions, filtered_predictions, ground_truth)
    
    # 打印结果
    print("\n=== 简单统计异常点筛选效果评估 ===")
    print(f"Precision 改善: {improvement['precision_improvement']:.4f}")
    print(f"Recall 改善: {improvement['recall_improvement']:.4f}")
    print(f"F1 Score 改善: {improvement['f1_improvement']:.4f}")
    print(f"MSE 改善: {improvement['mse_improvement']:.4f}")
    print(f"TP 改善: {improvement['total_tp_improvement']}")
    print(f"FP 改善: {improvement['total_fp_improvement']}")
    print(f"FN 改善: {improvement['total_fn_improvement']}")
    
    print("\n=== 原始指标 ===")
    print(f"Precision: {improvement['original_metrics']['precision']:.4f}")
    print(f"Recall: {improvement['original_metrics']['recall']:.4f}")
    print(f"F1 Score: {improvement['original_metrics']['f1']:.4f}")
    print(f"MSE: {improvement['original_metrics']['mse']:.4f}")
    
    print("\n=== 筛选后指标 ===")
    print(f"Precision: {improvement['filtered_metrics']['precision']:.4f}")
    print(f"Recall: {improvement['filtered_metrics']['recall']:.4f}")
    print(f"F1 Score: {improvement['filtered_metrics']['f1']:.4f}")
    print(f"MSE: {improvement['filtered_metrics']['mse']:.4f}")
    
    # 保存筛选后的结果
    output_path = 'results/spotgeov2/WTNet/simple_statistical_outlier_filtered_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(filtered_predictions, f, indent=2)
    print(f"\n简单统计异常点筛选结果已保存到: {output_path}")
    
    # 保存评测结果
    evaluation_path = 'results/spotgeov2/WTNet/simple_statistical_outlier_evaluation.json'
    with open(evaluation_path, 'w') as f:
        json.dump(improvement, f, indent=2)
    print(f"评测结果已保存到: {evaluation_path}")

if __name__ == '__main__':
    main() 