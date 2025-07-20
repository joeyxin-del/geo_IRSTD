import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import copy
from sklearn.cluster import DBSCAN

class DBSCANOutlierProcessor:
    """基于DBSCAN的异常点处理器"""
    
    def __init__(self, 
                 base_distance_threshold: float = 200.0,
                 eps_factor: float = 3.0,
                 min_samples: int = 2,
                 min_points_for_clustering: int = 3):
        """
        初始化DBSCAN异常点处理器
        
        Args:
            base_distance_threshold: 基础距离阈值
            eps_factor: DBSCAN的eps参数倍数
            min_samples: DBSCAN的最小样本数
            min_points_for_clustering: 进行聚类的最小点数
        """
        self.base_distance_threshold = base_distance_threshold
        self.eps_factor = eps_factor
        self.min_samples = min_samples
        self.min_points_for_clustering = min_points_for_clustering
    
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
    
    def adaptive_eps_parameter(self, points: List[List[float]]) -> float:
        """
        自适应计算DBSCAN的eps参数
        
        Args:
            points: 点列表
            
        Returns:
            eps: 计算得到的eps参数
        """
        if len(points) < 2:
            return self.base_distance_threshold
        
        # 计算所有点对之间的距离
        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = self.calculate_distance(points[i], points[j])
                distances.append(dist)
        
        if not distances:
            return self.base_distance_threshold
        
        # 使用距离的75%分位数作为eps的基础
        base_eps = np.percentile(distances, 75)
        
        # 应用倍数因子
        eps = base_eps * self.eps_factor
        
        # 限制在合理范围内
        min_eps = 5.0
        max_eps = self.base_distance_threshold * 2
        
        return max(min_eps, min(eps, max_eps))
    
    def dbscan_outlier_detection(self, points: List[List[float]], eps: Optional[float] = None) -> Tuple[List[bool], Dict]:
        """
        使用DBSCAN进行异常点检测
        
        Args:
            points: 点列表
            eps: DBSCAN的eps参数，如果为None则自动计算
            
        Returns:
            is_outlier: 是否为异常点的布尔列表
            stats: 统计信息
        """
        if len(points) < self.min_points_for_clustering:
            return [False] * len(points), {}
        
        points_array = np.array(points)
        
        # 自动计算eps参数
        if eps is None:
            eps = self.adaptive_eps_parameter(points)
        
        # 使用DBSCAN聚类
        clustering = DBSCAN(eps=eps, min_samples=self.min_samples).fit(points_array)
        labels = clustering.labels_
        
        # 标签为-1的点是异常点（噪声点）
        is_outlier = [label == -1 for label in labels]
        
        # 统计聚类信息
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = sum(is_outlier)
        
        stats = {
            'eps': eps,
            'min_samples': self.min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'labels': labels.tolist(),
            'outlier_count': n_noise
        }
        
        return is_outlier, stats
    
    def filter_outliers_from_predictions(self, predictions: Dict) -> Tuple[Dict, Dict]:
        """
        从预测结果中过滤异常点（使用DBSCAN方法）
        
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
            
            if len(all_points) < self.min_points_for_clustering:
                continue
            
            # 进行DBSCAN异常点检测
            is_outlier, detection_stats = self.dbscan_outlier_detection(all_points)
            
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
            
            print(f"  序列 {sequence_id}: 总点数 {len(all_points)}, 异常点数 {len(outlier_points)}, 聚类数 {detection_stats.get('n_clusters', 0)}")
        
        return filtered_predictions, outlier_info
    
    def process_sequence(self, predictions: Dict) -> Dict:
        """
        处理整个预测结果，进行DBSCAN异常点筛选
        
        Args:
            predictions: 原始预测结果
            
        Returns:
            filtered_predictions: 过滤后的预测结果
        """
        print("开始DBSCAN异常点筛选处理...")
        
        # 使用DBSCAN方法进行异常点检测
        filtered_predictions, outlier_info = self.filter_outliers_from_predictions(predictions)
        
        # 统计结果
        total_outliers = sum(info['outlier_count'] for info in outlier_info.values())
        total_points = sum(info['total_points'] for info in outlier_info.values())
        
        print(f"DBSCAN异常点筛选完成！")
        print(f"总检测点数: {total_points}")
        print(f"异常点数: {total_outliers}")
        print(f"异常点比例: {total_outliers/total_points*100:.2f}%")
        
        return filtered_predictions
    
    def evaluate_improvement(self, original_predictions: Dict, filtered_predictions: Dict, ground_truth: List[Dict]) -> Dict:
        """评估DBSCAN异常点筛选的效果改善"""
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
    """主函数，演示DBSCAN异常点筛选和评测"""
    # 加载预测结果和真实标注
    pred_path = 'results/spotgeov2/WTNet/sequence_slope_processed_predictions.json'
    gt_path = 'datasets/spotgeov2-IRSTD/test_anno.json'
    
    print("正在加载预测结果和真实标注...")
    with open(pred_path, 'r') as f:
        original_predictions = json.load(f)
    
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    # 创建DBSCAN异常点处理器
    processor = DBSCANOutlierProcessor(
        base_distance_threshold=500.0,
        eps_factor=3.0,
        min_samples=2,
        min_points_for_clustering=5
    )
    
    # 进行DBSCAN异常点筛选
    filtered_predictions = processor.process_sequence(original_predictions)
    
    # 评估改善效果
    improvement = processor.evaluate_improvement(original_predictions, filtered_predictions, ground_truth)
    
    # 打印结果
    print("\n=== DBSCAN异常点筛选效果评估 ===")
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
    output_path = 'results/spotgeov2/WTNet/dbscan_outlier_filtered_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(filtered_predictions, f, indent=2)
    print(f"\nDBSCAN异常点筛选结果已保存到: {output_path}")
    
    # 保存评测结果
    evaluation_path = 'results/spotgeov2/WTNet/dbscan_outlier_evaluation.json'
    with open(evaluation_path, 'w') as f:
        json.dump(improvement, f, indent=2)
    print(f"评测结果已保存到: {evaluation_path}")

if __name__ == '__main__':
    main() 