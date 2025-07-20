import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, Counter
import copy
import math

class DominantSlopeFilter:
    """基于主导斜率的异常点过滤处理器"""
    
    def __init__(self, 
                 base_distance_threshold: float = 200.0,
                 slope_tolerance: float = 0.1,  # 斜率容差
                 min_slope_count: int = 2,      # 最小斜率出现次数
                 slope_match_threshold: float = 0.2):  # 斜率匹配阈值
        """
        初始化基于主导斜率的过滤处理器
        
        Args:
            base_distance_threshold: 基础距离阈值
            slope_tolerance: 斜率容差，用于聚类相似斜率
            min_slope_count: 最小斜率出现次数，用于确定主导斜率
            slope_match_threshold: 斜率匹配阈值，用于判断点对是否属于主导斜率
        """
        self.base_distance_threshold = base_distance_threshold
        self.slope_tolerance = slope_tolerance
        self.min_slope_count = min_slope_count
        self.slope_match_threshold = slope_match_threshold
    
    def calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """计算两点之间的欧氏距离"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_slope(self, point1: List[float], point2: List[float]) -> float:
        """计算两点之间的斜率"""
        if abs(point2[0] - point1[0]) < 1e-6:  # 避免除零
            return float('inf') if point2[1] > point1[1] else float('-inf')
        return (point2[1] - point1[1]) / (point2[0] - point1[0])
    
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
    
    def collect_dominant_slopes_and_points(self, frames_data: Dict[int, Dict]) -> Dict:
        """收集主导斜率和对应的点对信息"""
        frames = sorted(frames_data.keys())
        
        if len(frames) < 2:
            return {'dominant_slopes': [], 'valid_points': set(), 'slope_pairs': []}
        
        all_slopes = []
        slope_pairs = []  # 存储斜率对应的点对信息
        
        # 计算所有可能的帧间配对
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                frame1 = frames[i]
                frame2 = frames[j]
                
                coords1 = frames_data[frame1]['coords']
                coords2 = frames_data[frame2]['coords']
                
                if coords1 and coords2:
                    # 遍历所有可能的点配对组合
                    for pos1_idx, pos1 in enumerate(coords1):
                        for pos2_idx, pos2 in enumerate(coords2):
                            # 计算两点之间的距离
                            distance = self.calculate_distance(pos1, pos2)
                            
                            # 只考虑距离合理的配对
                            if distance <= self.base_distance_threshold * 2:
                                slope = self.calculate_slope(pos1, pos2)
                                all_slopes.append(slope)
                                slope_pairs.append({
                                    'frame1': frame1,
                                    'frame2': frame2,
                                    'pos1': pos1,
                                    'pos2': pos2,
                                    'slope': slope,
                                    'distance': distance,
                                    'pos1_idx': pos1_idx,
                                    'pos2_idx': pos2_idx,
                                    'image1': frames_data[frame1]['image_name'],
                                    'image2': frames_data[frame2]['image_name']
                                })
        
        # 统计斜率分布
        slope_counter = Counter()
        for slope in all_slopes:
            if not math.isinf(slope):
                # 将斜率聚类到相近的区间
                slope_key = round(slope / self.slope_tolerance) * self.slope_tolerance
                slope_counter[slope_key] += 1
        
        # 找到主导斜率
        dominant_slopes = []
        for slope, count in slope_counter.most_common():
            if count >= self.min_slope_count:
                dominant_slopes.append((slope, count))
        
        # 记录属于主导斜率的点对
        valid_points = set()
        dominant_slope_pairs = []
        
        for pair in slope_pairs:
            pair_slope = pair['slope']
            if not math.isinf(pair_slope):
                # 检查是否属于主导斜率
                for dominant_slope, count in dominant_slopes:
                    if abs(pair_slope - dominant_slope) <= self.slope_match_threshold:
                        # 这个点对属于主导斜率
                        valid_points.add((pair['image1'], tuple(pair['pos1'])))
                        valid_points.add((pair['image2'], tuple(pair['pos2'])))
                        dominant_slope_pairs.append(pair)
                        break
        
        return {
            'dominant_slopes': dominant_slopes,
            'valid_points': valid_points,
            'slope_pairs': slope_pairs,
            'dominant_slope_pairs': dominant_slope_pairs,
            'total_slopes': len(all_slopes),
            'valid_point_count': len(valid_points),
            'dominant_pair_count': len(dominant_slope_pairs)
        }
    
    def filter_points_by_dominant_slopes(self, predictions: Dict) -> Tuple[Dict, Dict]:
        """
        基于主导斜率过滤异常点
        
        Args:
            predictions: 原始预测结果
            
        Returns:
            filtered_predictions: 过滤后的预测结果
            filter_info: 过滤信息
        """
        sequence_data = self.extract_sequence_info(predictions)
        filtered_predictions = copy.deepcopy(predictions)
        filter_info = {}
        
        total_points_before = 0
        total_points_after = 0
        total_outliers = 0
        
        for sequence_id, frames_data in sequence_data.items():
            print(f"处理序列 {sequence_id}，包含 {len(frames_data)} 帧")
            
            # 计算序列总目标数
            sequence_total_points = 0
            for frame, frame_data in frames_data.items():
                sequence_total_points += len(frame_data['coords'])
            
            # 如果5帧预测总目标小于5，跳过筛选
            if sequence_total_points < 5:
                print(f"  序列 {sequence_id}: 总目标数 {sequence_total_points} < 5，跳过主导斜率筛选")
                # 记录跳过信息
                filter_info[sequence_id] = {
                    'dominant_slopes': [],
                    'valid_point_count': 0,
                    'dominant_pair_count': 0,
                    'points_before': sequence_total_points,
                    'points_after': sequence_total_points,
                    'outliers_removed': 0,
                    'outlier_ratio': 0.0,
                    'skipped': True,
                    'reason': 'total_points_less_than_5'
                }
                total_points_before += sequence_total_points
                total_points_after += sequence_total_points
                continue
            
            # 收集主导斜率和有效点
            slope_info = self.collect_dominant_slopes_and_points(frames_data)
            valid_points = slope_info['valid_points']
            dominant_slopes = slope_info['dominant_slopes']
            
            # 统计当前序列的过滤情况
            sequence_outliers = 0
            sequence_points_before = 0
            sequence_points_after = 0
            
            # 过滤每个图像中的点
            for frame, frame_data in frames_data.items():
                image_name = frame_data['image_name']
                if image_name in filtered_predictions:
                    original_coords = filtered_predictions[image_name]['coords']
                    sequence_points_before += len(original_coords)
                    
                    # 过滤不属于主导斜率的点
                    filtered_coords = []
                    for coord in original_coords:
                        point_key = (image_name, tuple(coord))
                        if point_key in valid_points:
                            filtered_coords.append(coord)
                        else:
                            sequence_outliers += 1
                    
                    # 更新预测结果
                    filtered_predictions[image_name]['coords'] = filtered_coords
                    filtered_predictions[image_name]['num_objects'] = len(filtered_coords)
                    sequence_points_after += len(filtered_coords)
            
            # 记录序列过滤信息
            filter_info[sequence_id] = {
                'dominant_slopes': dominant_slopes,
                'valid_point_count': len(valid_points),
                'dominant_pair_count': slope_info['dominant_pair_count'],
                'points_before': sequence_points_before,
                'points_after': sequence_points_after,
                'outliers_removed': sequence_outliers,
                'outlier_ratio': sequence_outliers / max(sequence_points_before, 1) * 100,
                'skipped': False
            }
            
            total_points_before += sequence_points_before
            total_points_after += sequence_points_after
            total_outliers += sequence_outliers
            
            print(f"  序列 {sequence_id}: 主导斜率 {len(dominant_slopes)} 个, 有效点 {len(valid_points)} 个")
            print(f"  过滤前: {sequence_points_before} 点, 过滤后: {sequence_points_after} 点, 移除: {sequence_outliers} 点")
        
        # 总体统计
        overall_stats = {
            'total_points_before': total_points_before,
            'total_points_after': total_points_after,
            'total_outliers_removed': total_outliers,
            'overall_outlier_ratio': total_outliers / max(total_points_before, 1) * 100
        }
        
        return filtered_predictions, filter_info, overall_stats
    
    def process_sequence(self, predictions: Dict) -> Dict:
        """
        处理整个预测结果，进行基于主导斜率的异常点筛选
        
        Args:
            predictions: 原始预测结果
            
        Returns:
            filtered_predictions: 过滤后的预测结果
        """
        print("开始基于主导斜率的异常点筛选处理...")
        
        # 基于主导斜率进行异常点检测
        filtered_predictions, filter_info, overall_stats = self.filter_points_by_dominant_slopes(predictions)
        
        print(f"主导斜率异常点筛选完成！")
        print(f"总检测点数: {overall_stats['total_points_before']}")
        print(f"异常点数: {overall_stats['total_outliers_removed']}")
        print(f"异常点比例: {overall_stats['overall_outlier_ratio']:.2f}%")
        
        return filtered_predictions
    
    def evaluate_improvement(self, original_predictions: Dict, filtered_predictions: Dict, ground_truth: List[Dict]) -> Dict:
        """评估基于主导斜率的异常点筛选的效果改善"""
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
    """主函数，演示基于主导斜率的异常点筛选和评测"""
    # 加载预测结果和真实标注
    pred_path = 'results/spotgeov2/WTNet/sequence_slope_processed_predictions.json'
    gt_path = 'datasets/spotgeov2-IRSTD/test_anno.json'
    
    print("正在加载预测结果和真实标注...")
    with open(pred_path, 'r') as f:
        original_predictions = json.load(f)
    
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    # 创建基于主导斜率的过滤处理器
    processor = DominantSlopeFilter(
        base_distance_threshold=500.0,
        slope_tolerance=0.05,
        min_slope_count=2,
        slope_match_threshold=0.05
    )
    
    # 进行基于主导斜率的异常点筛选
    filtered_predictions = processor.process_sequence(original_predictions)
    
    # 评估改善效果
    improvement = processor.evaluate_improvement(original_predictions, filtered_predictions, ground_truth)
    
    # 打印结果
    print("\n=== 基于主导斜率的异常点筛选效果评估 ===")
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
    output_path = 'results/spotgeov2/WTNet/dominant_slope_filtered_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(filtered_predictions, f, indent=2)
    print(f"\n基于主导斜率的异常点筛选结果已保存到: {output_path}")
    
    # 保存评测结果
    evaluation_path = 'results/spotgeov2/WTNet/dominant_slope_evaluation.json'
    with open(evaluation_path, 'w') as f:
        json.dump(improvement, f, indent=2)
    print(f"评测结果已保存到: {evaluation_path}")

if __name__ == '__main__':
    main() 