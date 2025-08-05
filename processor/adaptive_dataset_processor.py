import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Set
import sys
import os
from PIL import Image
import cv2
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import math
import time
from tqdm import tqdm
import argparse

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class AdaptiveDatasetProcessor:
    """对整个数据集进行自适应角度距离处理的处理器"""
    
    def __init__(self, 
                 min_cluster_size: int = 3,
                 angle_cluster_eps: float = 5.0,
                 step_cluster_eps: float = 0.3,
                 confidence_threshold: float = 0.6,
                 angle_tolerance: float = 15.0,
                 step_tolerance: float = 0.5):
        """
        初始化自适应数据集处理器
        
        Args:
            min_cluster_size: 最小聚类大小
            angle_cluster_eps: 角度聚类半径（度）
            step_cluster_eps: 步长聚类半径（比例）
            confidence_threshold: 主导模式置信度阈值
            angle_tolerance: 角度容差（度）
            step_tolerance: 步长容差（比例）
        """
        self.min_cluster_size = min_cluster_size
        self.angle_cluster_eps = angle_cluster_eps
        self.step_cluster_eps = step_cluster_eps
        self.confidence_threshold = confidence_threshold
        self.angle_tolerance = angle_tolerance
        self.step_tolerance = step_tolerance
        
        # 统计信息
        self.dataset_stats = {
            'total_sequences': 0,
            'processed_sequences': 0,
            'successful_sequences': 0,
            'failed_sequences': 0,
            'total_original_points': 0,
            'total_filtered_points': 0,
            'total_removed_points': 0,
            'sequences_with_patterns': 0,
            'sequences_without_patterns': 0
        }
    
    def calculate_angles_and_steps(self, frames_data: Dict[int, Dict]) -> Tuple[List[float], List[float]]:
        """计算所有点对的角度和步长"""
        angles = []
        steps = []
        
        frames = sorted(frames_data.keys())
        
        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            
            coords1 = frames_data[frame1]['coords']
            coords2 = frames_data[frame2]['coords']
            
            if not coords1 or not coords2:
                continue
            
            # 计算所有点对之间的角度和距离
            for coord1 in coords1:
                for coord2 in coords2:
                    # 计算角度（弧度）
                    dx = coord2[0] - coord1[0]
                    dy = coord2[1] - coord1[1]
                    
                    if dx == 0 and dy == 0:
                        continue
                    
                    angle_rad = math.atan2(dy, dx)
                    angle_deg = math.degrees(angle_rad)
                    
                    # 标准化角度到 [0, 360)
                    if angle_deg < 0:
                        angle_deg += 360
                    
                    # 计算步长（欧几里得距离）
                    step = math.sqrt(dx*dx + dy*dy)
                    
                    angles.append(angle_deg)
                    steps.append(step)
        
        return angles, steps
    
    def cluster_angles(self, angles: List[float]) -> Tuple[Optional[float], float]:
        """对角度进行聚类，找出主导角度"""
        if len(angles) < self.min_cluster_size:
            return None, 0.0
        
        # 将角度转换为二维坐标（考虑角度的周期性）
        angles_rad = np.array(angles) * np.pi / 180.0
        X = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 使用DBSCAN聚类
        clustering = DBSCAN(eps=self.angle_cluster_eps/180.0*np.pi, min_samples=self.min_cluster_size).fit(X_scaled)
        
        labels = clustering.labels_
        unique_labels = set(labels)
        
        # 找到最大的聚类
        max_cluster_size = 0
        dominant_angle = None
        confidence = 0.0
        
        for label in unique_labels:
            if label == -1:  # 噪声点
                continue
            
            cluster_mask = labels == label
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size > max_cluster_size:
                max_cluster_size = cluster_size
                cluster_angles = np.array(angles)[cluster_mask]
                
                # 计算聚类中心角度
                cluster_angles_rad = cluster_angles * np.pi / 180.0
                mean_cos = np.mean(np.cos(cluster_angles_rad))
                mean_sin = np.mean(np.sin(cluster_angles_rad))
                dominant_angle = math.degrees(math.atan2(mean_sin, mean_cos))
                
                if dominant_angle < 0:
                    dominant_angle += 360
                
                confidence = cluster_size / len(angles)
        
        return dominant_angle, confidence
    
    def cluster_steps(self, steps: List[float]) -> Tuple[Optional[float], float]:
        """对步长进行聚类，找出主导步长"""
        if len(steps) < self.min_cluster_size:
            return None, 0.0
        
        # 将步长转换为二维特征（步长和步长的平方）
        steps_array = np.array(steps).reshape(-1, 1)
        
        # 标准化
        scaler = StandardScaler()
        steps_scaled = scaler.fit_transform(steps_array)
        
        # 使用DBSCAN聚类
        clustering = DBSCAN(eps=self.step_cluster_eps, min_samples=self.min_cluster_size).fit(steps_scaled)
        
        labels = clustering.labels_
        unique_labels = set(labels)
        
        # 找到最大的聚类
        max_cluster_size = 0
        dominant_step = None
        confidence = 0.0
        
        for label in unique_labels:
            if label == -1:  # 噪声点
                continue
            
            cluster_mask = labels == label
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size > max_cluster_size:
                max_cluster_size = cluster_size
                cluster_steps = np.array(steps)[cluster_mask]
                dominant_step = np.median(cluster_steps)  # 使用中位数作为主导步长
                confidence = cluster_size / len(steps)
        
        return dominant_step, confidence
    
    def filter_points_by_pattern(self, frames_data: Dict[int, Dict], 
                               dominant_angle: float, dominant_step: float) -> Dict[int, Dict]:
        """根据主导模式过滤点"""
        filtered_data = {}
        frames = sorted(frames_data.keys())
        
        for i, frame in enumerate(frames):
            coords = frames_data[frame]['coords'].copy()
            filtered_coords = []
            
            for coord in coords:
                # 检查与前一个有效点的角度和步长
                if i > 0:
                    prev_frame = frames[i-1]
                    prev_coords = filtered_data.get(prev_frame, {}).get('coords', [])
                    
                    if prev_coords:
                        # 找到最近的前一个点
                        min_dist = float('inf')
                        best_prev_coord = None
                        
                        for prev_coord in prev_coords:
                            dist = math.sqrt((coord[0] - prev_coord[0])**2 + (coord[1] - prev_coord[1])**2)
                            if dist < min_dist:
                                min_dist = dist
                                best_prev_coord = prev_coord
                        
                        if best_prev_coord:
                            # 计算角度
                            dx = coord[0] - best_prev_coord[0]
                            dy = coord[1] - best_prev_coord[1]
                            angle_rad = math.atan2(dy, dx)
                            angle_deg = math.degrees(angle_rad)
                            if angle_deg < 0:
                                angle_deg += 360
                            
                            # 计算步长
                            step = math.sqrt(dx*dx + dy*dy)
                            
                            # 检查角度差异（考虑周期性）
                            angle_diff = min(abs(angle_deg - dominant_angle), 
                                           abs(angle_deg - dominant_angle + 360),
                                           abs(angle_deg - dominant_angle - 360))
                            
                            # 检查步长差异
                            step_diff = abs(step - dominant_step) / dominant_step
                            
                            # 如果角度和步长都在容差范围内，保留该点
                            if angle_diff <= self.angle_tolerance and step_diff <= self.step_tolerance:
                                filtered_coords.append(coord)
                        else:
                            # 没有前一个点，保留该点
                            filtered_coords.append(coord)
                    else:
                        # 前一个帧没有点，保留该点
                        filtered_coords.append(coord)
                else:
                    # 第一帧，保留所有点
                    filtered_coords.append(coord)
            
            filtered_data[frame] = {
                'coords': filtered_coords,
                'num_objects': len(filtered_coords),
                'image_name': frames_data[frame]['image_name']
            }
        
        return filtered_data
    
    def process_single_sequence(self, sequence_id: int, frames_data: Dict[int, Dict]) -> Tuple[Dict[int, Dict], Dict]:
        """处理单个序列"""
        try:
            # 计算所有点对的角度和步长
            angles, steps = self.calculate_angles_and_steps(frames_data)
            
            if not angles or not steps:
                return frames_data, {'error': 'insufficient_data', 'sequence_id': sequence_id}
            
            # 聚类分析角度
            dominant_angle, angle_confidence = self.cluster_angles(angles)
            
            # 聚类分析步长
            dominant_step, step_confidence = self.cluster_steps(steps)
            
            # 根据主导模式过滤点
            if dominant_angle is not None and dominant_step is not None:
                filtered_data = self.filter_points_by_pattern(frames_data, dominant_angle, dominant_step)
                has_pattern = True
            else:
                filtered_data = frames_data
                has_pattern = False
            
            # 收集统计信息
            stats = self.collect_sequence_stats(frames_data, filtered_data, angles, steps, 
                                              dominant_angle, dominant_step, angle_confidence, step_confidence, has_pattern)
            
            return filtered_data, stats
            
        except Exception as e:
            print(f"处理序列 {sequence_id} 时出错: {e}")
            return frames_data, {'error': str(e), 'sequence_id': sequence_id}
    
    def collect_sequence_stats(self, original: Dict[int, Dict], filtered: Dict[int, Dict],
                             angles: List[float], steps: List[float],
                             dominant_angle: Optional[float], dominant_step: Optional[float],
                             angle_confidence: float, step_confidence: float, has_pattern: bool) -> Dict:
        """收集单个序列的统计信息"""
        stats = {
            'original_points': sum(len(frame_data['coords']) for frame_data in original.values()),
            'filtered_points': sum(len(frame_data['coords']) for frame_data in filtered.values()),
            'original_frames': len([f for f, data in original.items() if data['coords']]),
            'filtered_frames': len([f for f, data in filtered.items() if data['coords']]),
            'total_frames': len(original),
            'removed_points': 0,
            'total_angle_step_pairs': len(angles),
            'dominant_angle': dominant_angle,
            'dominant_step': dominant_step,
            'angle_confidence': angle_confidence,
            'step_confidence': step_confidence,
            'has_pattern': has_pattern,
            'angle_statistics': {
                'mean': np.mean(angles) if angles else 0,
                'std': np.std(angles) if angles else 0,
                'min': np.min(angles) if angles else 0,
                'max': np.max(angles) if angles else 0
            },
            'step_statistics': {
                'mean': np.mean(steps) if steps else 0,
                'std': np.std(steps) if steps else 0,
                'min': np.min(steps) if steps else 0,
                'max': np.max(steps) if steps else 0
            }
        }
        
        # 计算移除的点数
        stats['removed_points'] = stats['original_points'] - stats['filtered_points']
        
        return stats
    
    def load_predictions(self, predictions_path: str) -> Dict:
        """加载预测结果"""
        print(f"正在加载预测结果: {predictions_path}")
        with open(predictions_path, 'r') as f:
            predictions = json.load(f)
        return predictions
    
    def organize_by_sequence(self, predictions: Dict) -> Dict[int, Dict[int, Dict]]:
        """按序列组织预测结果"""
        sequences = {}
        
        for img_name, pred_info in predictions.items():
            parts = img_name.split('_')
            if len(parts) >= 2:
                try:
                    seq_id = int(parts[0])
                    frame = int(parts[1])
                    
                    if seq_id not in sequences:
                        sequences[seq_id] = {}
                    
                    sequences[seq_id][frame] = {
                        'coords': pred_info['coords'],
                        'num_objects': pred_info['num_objects'],
                        'image_name': img_name
                    }
                except ValueError:
                    continue
        
        return sequences
    
    def process_dataset(self, predictions_path: str, output_path: str = None) -> Dict:
        """处理整个数据集"""
        print("=== 开始自适应角度距离数据集处理 ===")
        start_time = time.time()
        
        # 加载预测结果
        predictions = self.load_predictions(predictions_path)
        
        # 按序列组织数据
        sequences = self.organize_by_sequence(predictions)
        self.dataset_stats['total_sequences'] = len(sequences)
        
        print(f"发现 {len(sequences)} 个序列")
        
        # 处理结果
        processed_predictions = {}
        sequence_stats = {}
        
        # 使用tqdm显示进度
        for sequence_id, frames_data in tqdm(sequences.items(), desc="处理序列"):
            self.dataset_stats['processed_sequences'] += 1
            
            # 处理单个序列
            filtered_data, stats = self.process_single_sequence(sequence_id, frames_data)
            
            # 更新数据集统计
            if 'error' not in stats:
                self.dataset_stats['successful_sequences'] += 1
                self.dataset_stats['total_original_points'] += stats['original_points']
                self.dataset_stats['total_filtered_points'] += stats['filtered_points']
                self.dataset_stats['total_removed_points'] += stats['removed_points']
                
                if stats['has_pattern']:
                    self.dataset_stats['sequences_with_patterns'] += 1
                else:
                    self.dataset_stats['sequences_without_patterns'] += 1
            else:
                self.dataset_stats['failed_sequences'] += 1
            
            # 保存序列统计
            sequence_stats[sequence_id] = stats
            
            # 将过滤后的结果添加到输出
            for frame, frame_data in filtered_data.items():
                img_name = frame_data['image_name']
                processed_predictions[img_name] = {
                    'coords': frame_data['coords'],
                    'num_objects': frame_data['num_objects']
                }
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 生成输出文件路径
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(predictions_path))[0]
            output_path = f"results/adaptive_filtered_{base_name}.json"
        
        # 保存处理后的预测结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(processed_predictions, f, indent=2)
        
        # 保存详细统计信息
        stats_path = output_path.replace('.json', '_stats.json')
        detailed_stats = {
            'dataset_stats': self.dataset_stats,
            'sequence_stats': sequence_stats,
            'processing_time': processing_time,
            'parameters': {
                'min_cluster_size': self.min_cluster_size,
                'angle_cluster_eps': self.angle_cluster_eps,
                'step_cluster_eps': self.step_cluster_eps,
                'confidence_threshold': self.confidence_threshold,
                'angle_tolerance': self.angle_tolerance,
                'step_tolerance': self.step_tolerance
            }
        }
        
        with open(stats_path, 'w') as f:
            json.dump(detailed_stats, f, indent=2)
        
        # 打印处理结果
        self.print_processing_summary(processing_time, output_path, stats_path)
        
        return {
            'processed_predictions': processed_predictions,
            'dataset_stats': self.dataset_stats,
            'sequence_stats': sequence_stats,
            'output_path': output_path,
            'stats_path': stats_path,
            'processing_time': processing_time
        }
    
    def print_processing_summary(self, processing_time: float, output_path: str, stats_path: str):
        """打印处理摘要"""
        print(f"\n{'='*60}")
        print(f"自适应角度距离数据集处理完成")
        print(f"{'='*60}")
        
        print(f"处理时间: {processing_time:.2f} 秒")
        print(f"输出文件: {output_path}")
        print(f"统计文件: {stats_path}")
        
        print(f"\n数据集统计:")
        print(f"  总序列数: {self.dataset_stats['total_sequences']}")
        print(f"  成功处理: {self.dataset_stats['successful_sequences']}")
        print(f"  处理失败: {self.dataset_stats['failed_sequences']}")
        print(f"  发现模式: {self.dataset_stats['sequences_with_patterns']}")
        print(f"  无模式: {self.dataset_stats['sequences_without_patterns']}")
        
        print(f"\n点数统计:")
        print(f"  原始点数: {self.dataset_stats['total_original_points']}")
        print(f"  过滤后点数: {self.dataset_stats['total_filtered_points']}")
        print(f"  移除点数: {self.dataset_stats['total_removed_points']}")
        
        if self.dataset_stats['total_original_points'] > 0:
            removal_rate = self.dataset_stats['total_removed_points'] / self.dataset_stats['total_original_points'] * 100
            print(f"  移除率: {removal_rate:.2f}%")
        
        print(f"{'='*60}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='自适应角度距离数据集处理器')
    parser.add_argument('--pred_path', type=str, default='results/spotgeov2-IRSTD/WTNet/predictions_8807.json',
                       help='预测结果文件路径')
    parser.add_argument('--output_path', type=str, default="results/spotgeov2/WTNet/adaptive_filtered_predictions.json",
                       help='输出文件路径（可选）')
    parser.add_argument('--min_cluster_size', type=int, default=3,
                       help='最小聚类大小')
    parser.add_argument('--angle_cluster_eps', type=float, default=5.0,
                       help='角度聚类半径（度）')
    parser.add_argument('--step_cluster_eps', type=float, default=0.3,
                       help='步长聚类半径（比例）')
    parser.add_argument('--confidence_threshold', type=float, default=0.6,
                       help='主导模式置信度阈值')
    parser.add_argument('--angle_tolerance', type=float, default=15.0,
                       help='角度容差（度）')
    parser.add_argument('--step_tolerance', type=float, default=0.5,
                       help='步长容差（比例）')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.pred_path):
        print(f"错误：预测结果文件不存在: {args.pred_path}")
        return
    
    # 创建处理器
    processor = AdaptiveDatasetProcessor(
        min_cluster_size=args.min_cluster_size,
        angle_cluster_eps=args.angle_cluster_eps,
        step_cluster_eps=args.step_cluster_eps,
        confidence_threshold=args.confidence_threshold,
        angle_tolerance=args.angle_tolerance,
        step_tolerance=args.step_tolerance
    )
    
    # 处理数据集
    result = processor.process_dataset(args.pred_path, args.output_path)
    
    if result:
        print(f"\n数据集处理完成！")

if __name__ == '__main__':
    main() 