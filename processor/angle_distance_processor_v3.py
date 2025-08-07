import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import copy
import math
import os
from pattern_analyzer import PatternAnalyzer, ClusteringPatternAnalyzer, GMMPatternAnalyzer, SpectralPatternAnalyzer
from utils import SequenceExtractor, TrajectoryGenerator, TrajectoryCompleter, OutlierFilter, GeometryUtils


class AngleDistanceProcessorV3:
    """基于角度和步长的轨迹补全处理器 - 重构后的核心版本"""
    
    def __init__(self, 
                 angle_tolerance: float = 10.0,
                 min_angle_count: int = 2,
                 step_tolerance: float = 0.2,
                 min_step_count: int = 1,
                 max_step_size: float = 40.0,
                 point_distance_threshold: float = 5.0,
                 use_clustering: bool = True):
        """
        初始化处理器
        
        Args:
            angle_tolerance: 角度容差（度）
            min_angle_count: 最小角度出现次数
            step_tolerance: 步长容差（比例）
            min_step_count: 最小步长出现次数
            max_step_size: 最大步长限制
            point_distance_threshold: 重合点过滤阈值
        """
        self.angle_tolerance = angle_tolerance
        self.min_angle_count = min_angle_count
        self.step_tolerance = step_tolerance
        self.min_step_count = min_step_count
        self.max_step_size = max_step_size
        self.point_distance_threshold = point_distance_threshold
        self.use_clustering = use_clustering
        
        # 初始化组件
        if use_clustering:
            self.pattern_analyzer = ClusteringPatternAnalyzer(angle_tolerance, min_angle_count, min_step_count)
            # self.pattern_analyzer = GMMPatternAnalyzer(angle_tolerance, min_angle_count, min_step_count)
            # self.pattern_analyzer = SpectralPatternAnalyzer(angle_tolerance, min_angle_count, min_step_count)
        else:
            self.pattern_analyzer = PatternAnalyzer(angle_tolerance, min_angle_count, min_step_count)
        self.trajectory_completer = TrajectoryCompleter(point_distance_threshold)
    
    def process_sequence(self, sequence_id: int, frames_data: Dict[int, Dict]) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
        """处理单个序列，返回轨迹补全和异常点筛选的结果"""
        # 收集序列内的角度和步长统计
        pattern_stats = self.pattern_analyzer.collect_sequence_angles_and_steps(frames_data)
        
        if not pattern_stats['dominant_angles'] and not pattern_stats['dominant_steps']:
            return frames_data, frames_data
        
        # 步骤1: 轨迹补全
        completed_sequence, generated_points_from_pairs = self.trajectory_completer.complete_trajectory_with_patterns(
            frames_data, pattern_stats['dominant_angles'], pattern_stats['dominant_steps'], self.pattern_analyzer
        )
        
        # 步骤2: 异常点筛选
        filtered_sequence = OutlierFilter.filter_outliers_by_dominant_patterns(
            completed_sequence, pattern_stats['point_participation']
        )

        return completed_sequence, filtered_sequence
    
    def process_dataset(self, predictions: Dict) -> Dict:
        """处理整个数据集，使用角度和步长补全策略"""
        print("开始基于角度和步长的轨迹补全处理...")
        
        # 提取序列信息
        sequence_data = SequenceExtractor.extract_sequence_info(predictions)
        print(f"提取了 {len(sequence_data)} 个序列的信息")
        
        # 处理每个序列
        processed_predictions = {}
        
        for sequence_id, frames_data in sequence_data.items():
            print(f"处理序列 {sequence_id}...")
            # 处理单个序列
            completed_sequence, filtered_sequence = self.process_sequence(sequence_id, frames_data)
            
            # 使用过滤后的结果
            processed_sequence = filtered_sequence
            
            # 转换回原始格式
            for frame, frame_data in processed_sequence.items():
                image_name = frame_data['image_name']
                processed_predictions[image_name] = {
                    'coords': frame_data['coords'],
                    'num_objects': frame_data['num_objects']
                }
        
        print(f"基于角度和步长的轨迹补全完成，处理了 {len(processed_predictions)} 张图像")
        return processed_predictions


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='角度距离轨迹补全处理器 V3')
    parser.add_argument('--pred_path', type=str, 
                       default='results/spotgeov2/WTNet/predictions_8807.json',
                       help='预测结果文件路径')
    parser.add_argument('--gt_path', type=str, 
                       default='datasets/spotgeov2-IRSTD/test_anno.json',
                       help='真实标注文件路径')
    parser.add_argument('--output_path', type=str, 
                       default='results/spotgeov2/WTNet/angle_distance_processed_predictions_v3.json',
                       help='输出文件路径')
    parser.add_argument('--angle_tolerance', type=float, default=12,
                       help='角度容差（度）')
    parser.add_argument('--use_clustering', type=bool, default=True,
                       help='是否使用聚类')
    parser.add_argument('--min_angle_count', type=int, default=1,
                       help='最小角度出现次数')
    parser.add_argument('--step_tolerance', type=float, default=0.5,
                       help='步长容差（比例）')
    parser.add_argument('--min_step_count', type=int, default=1,
                       help='最小步长出现次数')
    parser.add_argument('--point_distance_threshold', type=float, default=65,
                       help='重合点过滤阈值')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.pred_path):
        print(f"错误：预测结果文件不存在: {args.pred_path}")
        return
    
    if not os.path.exists(args.gt_path):
        print(f"错误：真实标注文件不存在: {args.gt_path}")
        return
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载数据
    print("正在加载预测结果和真实标注...")
    try:
        with open(args.pred_path, 'r') as f:
            original_predictions = json.load(f)
        print(f"成功加载预测结果，包含 {len(original_predictions)} 张图像")
        
        with open(args.gt_path, 'r') as f:
            ground_truth = json.load(f)
        print(f"成功加载真实标注，包含 {len(ground_truth)} 个标注")
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return
    
    # 创建处理器
    processor = AngleDistanceProcessorV3(
        angle_tolerance=args.angle_tolerance,
        min_angle_count=args.min_angle_count,
        step_tolerance=args.step_tolerance,
        min_step_count=args.min_step_count,
        point_distance_threshold=args.point_distance_threshold,
        use_clustering=args.use_clustering
    )
    
    # 处理数据
    processed_predictions = processor.process_dataset(original_predictions)
    
    # 评估效果
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from eval_predictions import calculate_metrics
        
        original_metrics = calculate_metrics(original_predictions, ground_truth, 1000)
        processed_metrics = calculate_metrics(processed_predictions, ground_truth, 1000)
        
        print("\n=== 处理效果评估 ===")
        print(f"Precision: {original_metrics['precision']:.4f} -> {processed_metrics['precision']:.4f}")
        print(f"Recall: {original_metrics['recall']:.4f} -> {processed_metrics['recall']:.4f}")
        print(f"F1 Score: {original_metrics['f1']:.4f} -> {processed_metrics['f1']:.4f}")
        print(f"MSE: {original_metrics['mse']:.4f} -> {processed_metrics['mse']:.4f}")
        
    except Exception as e:
        print(f"评估过程中出错: {e}")
    
    # 保存结果
    try:
        with open(args.output_path, 'w') as f:
            json.dump(processed_predictions, f, indent=2)
        print(f"\n处理结果已保存到: {args.output_path}")
    except Exception as e:
        print(f"保存结果时出错: {e}")


if __name__ == '__main__':
    main() 