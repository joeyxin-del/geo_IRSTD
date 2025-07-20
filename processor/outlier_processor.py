import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, Counter
import copy
import math
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

class OutlierProcessor:
    """异常点处理器，专门用于筛选分布外的检测点"""
    
    def __init__(self, 
                 base_distance_threshold: float = 80.0,
                 outlier_threshold: float = 3.0,
                 eps_factor: float = 3.0,
                 min_samples: int = 2,
                 mahalanobis_threshold: float = 2.0,
                 contamination: float = 0.1,
                 consensus_threshold: float = 0.5):
        """
        初始化异常点处理器
        
        Args:
            base_distance_threshold: 基础距离阈值
            outlier_threshold: 异常检测阈值（σ倍数）
            eps_factor: DBSCAN的eps参数倍数
            min_samples: DBSCAN的最小样本数
            mahalanobis_threshold: 马氏距离阈值
            contamination: 预期的异常点比例
            consensus_threshold: 共识阈值（多少比例的方法认为某点是异常点）
        """
        self.base_distance_threshold = base_distance_threshold
        self.outlier_threshold = outlier_threshold
        self.eps_factor = eps_factor
        self.min_samples = min_samples
        self.mahalanobis_threshold = mahalanobis_threshold
        self.contamination = contamination
        self.consensus_threshold = consensus_threshold
    
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
    
    def dbscan_outlier_detection(self, points: List[List[float]], eps: Optional[float] = None) -> Tuple[List[bool], List[int]]:
        """
        使用DBSCAN进行异常点检测
        
        Args:
            points: 点列表
            eps: DBSCAN的eps参数，如果为None则自动计算
            
        Returns:
            is_outlier: 是否为异常点的布尔列表
            labels: DBSCAN的聚类标签
        """
        if len(points) < 2:
            return [False] * len(points), [0] * len(points)
        
        points_array = np.array(points)
        
        # 自动计算eps参数
        if eps is None:
            # 计算所有点对之间的距离
            distances = []
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    dist = self.calculate_distance(points[i], points[j])
                    distances.append(dist)
            
            if distances:
                eps = np.percentile(distances, 75) * self.eps_factor
            else:
                eps = self.base_distance_threshold
        
        # 使用DBSCAN聚类
        clustering = DBSCAN(eps=eps, min_samples=self.min_samples).fit(points_array)
        labels = clustering.labels_
        
        # 标签为-1的点是异常点
        is_outlier = [label == -1 for label in labels]
        
        return is_outlier, labels
    
    def statistical_outlier_detection(self, points: List[List[float]]) -> Tuple[List[bool], Dict]:
        """
        使用统计方法进行异常点检测
        
        Args:
            points: 点列表
            
        Returns:
            is_outlier: 是否为异常点的布尔列表
            stats: 统计信息
        """
        if len(points) < 3:
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
    
    def isolation_forest_detection(self, points: List[List[float]]) -> List[bool]:
        """
        使用Isolation Forest进行异常点检测
        
        Args:
            points: 点列表
            
        Returns:
            is_outlier: 是否为异常点的布尔列表
        """
        if len(points) < 3:
            return [False] * len(points)
        
        points_array = np.array(points)
        
        # 使用Isolation Forest
        iso_forest = IsolationForest(contamination=self.contamination, random_state=42)
        labels = iso_forest.fit_predict(points_array)
        
        # 标签为-1的点是异常点
        is_outlier = [label == -1 for label in labels]
        
        return is_outlier
    
    def local_outlier_factor_detection(self, points: List[List[float]]) -> List[bool]:
        """
        使用Local Outlier Factor进行异常点检测
        
        Args:
            points: 点列表
            
        Returns:
            is_outlier: 是否为异常点的布尔列表
        """
        if len(points) < 3:
            return [False] * len(points)
        
        points_array = np.array(points)
        
        # 使用Local Outlier Factor
        lof = LocalOutlierFactor(contamination=self.contamination)
        labels = lof.fit_predict(points_array)
        
        # 标签为-1的点是异常点
        is_outlier = [label == -1 for label in labels]
        
        return is_outlier
    
    def elliptic_envelope_detection(self, points: List[List[float]]) -> List[bool]:
        """
        使用Elliptic Envelope进行异常点检测
        
        Args:
            points: 点列表
            
        Returns:
            is_outlier: 是否为异常点的布尔列表
        """
        if len(points) < 3:
            return [False] * len(points)
        
        points_array = np.array(points)
        
        # 使用Elliptic Envelope
        envelope = EllipticEnvelope(contamination=self.contamination, random_state=42)
        labels = envelope.fit_predict(points_array)
        
        # 标签为-1的点是异常点
        is_outlier = [label == -1 for label in labels]
        
        return is_outlier
    
    def comprehensive_outlier_detection(self, points: List[List[float]]) -> Dict:
        """
        综合多种方法进行异常点检测
        
        Args:
            points: 点列表
            
        Returns:
            results: 包含各种方法检测结果的字典
        """
        results = {
            'points': points,
            'total_points': len(points),
            'methods': {}
        }
        
        # 1. DBSCAN方法
        dbscan_outliers, dbscan_labels = self.dbscan_outlier_detection(points)
        results['methods']['dbscan'] = {
            'is_outlier': dbscan_outliers,
            'labels': dbscan_labels,
            'outlier_count': sum(dbscan_outliers)
        }
        
        # 2. 统计方法
        if len(points) >= 3:
            stat_outliers, stat_stats = self.statistical_outlier_detection(points)
            results['methods']['statistical'] = {
                'is_outlier': stat_outliers,
                'stats': stat_stats,
                'outlier_count': sum(stat_outliers)
            }
        
        # 3. Isolation Forest方法
        if len(points) >= 3:
            iso_outliers = self.isolation_forest_detection(points)
            results['methods']['isolation_forest'] = {
                'is_outlier': iso_outliers,
                'outlier_count': sum(iso_outliers)
            }
        
        # 4. Local Outlier Factor方法
        if len(points) >= 3:
            lof_outliers = self.local_outlier_factor_detection(points)
            results['methods']['local_outlier_factor'] = {
                'is_outlier': lof_outliers,
                'outlier_count': sum(lof_outliers)
            }
        
        # 5. Elliptic Envelope方法
        if len(points) >= 3:
            env_outliers = self.elliptic_envelope_detection(points)
            results['methods']['elliptic_envelope'] = {
                'is_outlier': env_outliers,
                'outlier_count': sum(env_outliers)
            }
        
        # 计算综合结果
        results['consensus_outliers'] = self.calculate_consensus_outliers(results['methods'])
        
        return results
    
    def calculate_consensus_outliers(self, methods_results: Dict) -> List[bool]:
        """
        计算多种方法的共识异常点
        
        Args:
            methods_results: 各种方法的检测结果
            
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
        
        # 如果超过共识阈值比例的方法认为某点是异常点，则认为是共识异常点
        threshold = total_methods * self.consensus_threshold
        consensus_outliers = [point_votes[i] > threshold for i in range(len(next(iter(methods_results.values()))['is_outlier']))]
        
        return consensus_outliers
    
    def filter_outliers_from_predictions(self, predictions: Dict, method: str = 'consensus') -> Tuple[Dict, Dict]:
        """
        从预测结果中过滤异常点
        
        Args:
            predictions: 原始预测结果
            method: 使用的异常检测方法
            
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
            
            if len(all_points) < 2:
                continue
            
            # 进行异常点检测
            detection_results = self.comprehensive_outlier_detection(all_points)
            
            # 根据选择的方法确定异常点
            if method == 'consensus':
                is_outlier = detection_results['consensus_outliers']
            elif method in detection_results['methods']:
                is_outlier = detection_results['methods'][method]['is_outlier']
            else:
                print(f"未知的方法: {method}，使用共识方法")
                is_outlier = detection_results['consensus_outliers']
            
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
                'detection_results': detection_results
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
        处理整个预测结果，进行异常点筛选
        
        Args:
            predictions: 原始预测结果
            
        Returns:
            filtered_predictions: 过滤后的预测结果
        """
        print("开始异常点筛选处理...")
        
        # 使用共识方法进行异常点检测
        filtered_predictions, outlier_info = self.filter_outliers_from_predictions(
            predictions, method='consensus'
        )
        
        # 统计结果
        total_outliers = sum(info['outlier_count'] for info in outlier_info.values())
        total_points = sum(info['total_points'] for info in outlier_info.values())
        
        print(f"异常点筛选完成！")
        print(f"总检测点数: {total_points}")
        print(f"异常点数: {total_outliers}")
        print(f"异常点比例: {total_outliers/total_points*100:.2f}%")
        
        return filtered_predictions
    
    def evaluate_improvement(self, original_predictions: Dict, filtered_predictions: Dict, ground_truth: List[Dict]) -> Dict:
        """评估异常点筛选的效果改善"""
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
    """主函数，演示异常点筛选和评测"""
    # 加载预测结果和真实标注
    pred_path = 'results/spotgeov2/WTNet/sequence_slope_processed_predictions.json'
    gt_path = 'datasets/spotgeov2-IRSTD/test_anno.json'
    
    print("正在加载预测结果和真实标注...")
    with open(pred_path, 'r') as f:
        original_predictions = json.load(f)
    
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    # 创建异常点处理器
    processor = OutlierProcessor(
        base_distance_threshold=500.0,
        outlier_threshold=5.0,
        eps_factor=4.0,
        min_samples=2,
        mahalanobis_threshold=3.0,
        contamination=0.05,
        consensus_threshold=0.5
    )
    
    # 进行异常点筛选
    filtered_predictions = processor.process_sequence(original_predictions)
    
    # 评估改善效果
    improvement = processor.evaluate_improvement(original_predictions, filtered_predictions, ground_truth)
    
    # 打印结果
    print("\n=== 异常点筛选效果评估 ===")
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
    output_path = 'results/spotgeov2/WTNet/outlier_filtered_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(filtered_predictions, f, indent=2)
    print(f"\n异常点筛选结果已保存到: {output_path}")
    
    # 保存异常点信息
    outlier_info_path = 'results/spotgeov2/WTNet/outlier_info.json'
    with open(outlier_info_path, 'w') as f:
        json.dump(improvement, f, indent=2)
    print(f"评测结果已保存到: {outlier_info_path}")

if __name__ == '__main__':
    main() 