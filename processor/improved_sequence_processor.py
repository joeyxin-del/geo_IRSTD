import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import copy

class ImprovedSequenceProcessor:
    """改进的序列后处理器，更温和的过滤策略"""
    
    def __init__(self, 
                 distance_threshold: float = 100.0,  # 增大距离阈值
                 temporal_window: int = 3,
                 confidence_threshold: float = 0.1,  # 降低置信度阈值
                 min_track_length: int = 1,          # 降低最小轨迹长度
                 max_frame_gap: int = 2):            # 最大帧间隔
        """
        初始化改进的序列后处理器
        
        Args:
            distance_threshold: 目标匹配的距离阈值
            temporal_window: 时间窗口大小
            confidence_threshold: 置信度阈值
            min_track_length: 最小轨迹长度
            max_frame_gap: 最大帧间隔，用于插值
        """
        self.distance_threshold = distance_threshold
        self.temporal_window = temporal_window
        self.confidence_threshold = confidence_threshold
        self.min_track_length = min_track_length
        self.max_frame_gap = max_frame_gap
    
    def calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """计算两点之间的欧氏距离"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def extract_sequence_info(self, predictions: Dict) -> Dict[int, Dict[int, Dict]]:
        """从预测结果中提取序列信息"""
        sequence_data = defaultdict(dict)
        
        for img_name, pred_info in predictions.items():
            # 解析文件名获取序列ID和帧号
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
    
    def simple_temporal_filtering(self, sequence_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """简单的时序过滤，基于前后帧的一致性"""
        filtered_data = copy.deepcopy(sequence_data)
        sequences = list(filtered_data.keys())
        
        for sequence_id in sequences:
            frames_data = filtered_data[sequence_id]
            frames = sorted(frames_data.keys())
            
            if len(frames) < 2:
                continue
            
            # 对每一帧进行过滤
            for i, frame in enumerate(frames):
                current_coords = frames_data[frame]['coords']
                filtered_coords = []
                
                for pos in current_coords:
                    # 检查前后帧的支持
                    support_count = 0
                    
                    # 检查前一帧
                    if i > 0:
                        prev_frame = frames[i-1]
                        prev_coords = frames_data[prev_frame]['coords']
                        for prev_pos in prev_coords:
                            if self.calculate_distance(pos, prev_pos) <= self.distance_threshold:
                                support_count += 1
                                break
                    
                    # 检查后一帧
                    if i < len(frames) - 1:
                        next_frame = frames[i+1]
                        next_coords = frames_data[next_frame]['coords']
                        for next_pos in next_coords:
                            if self.calculate_distance(pos, next_pos) <= self.distance_threshold:
                                support_count += 1
                                break
                    
                    # 如果检测点得到前后帧的支持，或者没有前后帧，则保留
                    if support_count >= 0:  # 更宽松的条件
                        filtered_coords.append(pos)
                
                frames_data[frame]['coords'] = filtered_coords
                frames_data[frame]['num_objects'] = len(filtered_coords)
        
        return filtered_data
    
    def interpolate_missing_detections(self, sequence_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """基于简单插值补全漏检"""
        interpolated_data = copy.deepcopy(sequence_data)
        
        for sequence_id, frames_data in interpolated_data.items():
            frames = sorted(frames_data.keys())
            
            for i in range(len(frames) - 1):
                frame1 = frames[i]
                frame2 = frames[i + 1]
                
                # 如果帧间隔大于1且小于等于最大间隔
                if 1 < frame2 - frame1 <= self.max_frame_gap:
                    coords1 = frames_data[frame1]['coords']
                    coords2 = frames_data[frame2]['coords']
                    
                    # 简单的线性插值
                    for frame in range(frame1 + 1, frame2):
                        alpha = (frame - frame1) / (frame2 - frame1)
                        
                        # 为每个检测点进行插值
                        interpolated_coords = []
                        for pos1 in coords1:
                            for pos2 in coords2:
                                # 如果两个点距离较近，进行插值
                                if self.calculate_distance(pos1, pos2) <= self.distance_threshold * 2:
                                    interpolated_pos = [
                                        pos1[0] + alpha * (pos2[0] - pos1[0]),
                                        pos1[1] + alpha * (pos2[1] - pos1[1])
                                    ]
                                    interpolated_coords.append(interpolated_pos)
                        
                        # 添加到插值结果中
                        if frame not in frames_data:
                            frames_data[frame] = {
                                'coords': [],
                                'num_objects': 0,
                                'image_name': f"{sequence_id}_{frame}_test"
                            }
                        
                        frames_data[frame]['coords'].extend(interpolated_coords)
                        frames_data[frame]['num_objects'] = len(frames_data[frame]['coords'])
        
        return interpolated_data
    
    def remove_isolated_detections(self, sequence_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """移除孤立的检测点"""
        cleaned_data = copy.deepcopy(sequence_data)
        
        for sequence_id, frames_data in cleaned_data.items():
            frames = sorted(frames_data.keys())
            
            for frame in frames:
                current_coords = frames_data[frame]['coords']
                kept_coords = []
                
                for pos in current_coords:
                    # 检查是否在前后帧中有相近的检测点
                    has_neighbor = False
                    
                    # 检查前一帧
                    if frame > min(frames):
                        prev_frame = max(f for f in frames if f < frame)
                        prev_coords = frames_data[prev_frame]['coords']
                        for prev_pos in prev_coords:
                            if self.calculate_distance(pos, prev_pos) <= self.distance_threshold * 1.5:
                                has_neighbor = True
                                break
                    
                    # 检查后一帧
                    if frame < max(frames):
                        next_frame = min(f for f in frames if f > frame)
                        next_coords = frames_data[next_frame]['coords']
                        for next_pos in next_coords:
                            if self.calculate_distance(pos, next_pos) <= self.distance_threshold * 1.5:
                                has_neighbor = True
                                break
                    
                    # 如果检测点有邻居，或者序列只有一帧，则保留
                    if has_neighbor or len(frames) == 1:
                        kept_coords.append(pos)
                
                frames_data[frame]['coords'] = kept_coords
                frames_data[frame]['num_objects'] = len(kept_coords)
        
        return cleaned_data
    
    def process_sequence(self, predictions: Dict) -> Dict:
        """
        处理整个预测结果，使用更温和的策略
        
        Args:
            predictions: 原始预测结果
            
        Returns:
            processed_predictions: 处理后的预测结果
        """
        print("开始改进的序列后处理...")
        
        # 提取序列信息
        sequence_data = self.extract_sequence_info(predictions)
        print(f"提取了 {len(sequence_data)} 个序列的信息")
        
        processed_predictions = {}
        
        for sequence_id, frames_data in sequence_data.items():
            print(f"处理序列 {sequence_id}，包含 {len(frames_data)} 帧")
            
            # 1. 简单的时序过滤
            filtered_data = self.simple_temporal_filtering({sequence_id: frames_data})
            
            # 2. 插值补全漏检
            interpolated_data = self.interpolate_missing_detections(filtered_data)
            
            # 3. 移除孤立检测点
            cleaned_data = self.remove_isolated_detections(interpolated_data)
            
            # 转换回原始格式
            for frame, frame_data in cleaned_data[sequence_id].items():
                image_name = frame_data['image_name']
                processed_predictions[image_name] = {
                    'coords': frame_data['coords'],
                    'num_objects': frame_data['num_objects']
                }
        
        print(f"改进的序列后处理完成，处理了 {len(processed_predictions)} 张图像")
        return processed_predictions
    
    def evaluate_improvement(self, original_predictions: Dict, processed_predictions: Dict, ground_truth: List[Dict]) -> Dict:
        """评估后处理的效果改善"""
        import sys
        import os
        # 添加父目录到路径，以便导入eval_predictions
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from eval_predictions import calculate_metrics
        
        # 计算原始预测的指标
        original_metrics = calculate_metrics(original_predictions, ground_truth, self.distance_threshold)
        
        # 计算处理后预测的指标
        processed_metrics = calculate_metrics(processed_predictions, ground_truth, self.distance_threshold)
        
        # 计算改善程度
        improvement = {
            'precision_improvement': processed_metrics['precision'] - original_metrics['precision'],
            'recall_improvement': processed_metrics['recall'] - original_metrics['recall'],
            'f1_improvement': processed_metrics['f1'] - original_metrics['f1'],
            'mse_improvement': original_metrics['mse'] - processed_metrics['mse'],
            'original_metrics': original_metrics,
            'processed_metrics': processed_metrics
        }
        
        return improvement

def main():
    """主函数，演示改进的序列后处理"""
    # 加载预测结果和真实标注
    pred_path = 'results/WTNet/predictions.json'
    gt_path = 'datasets/spotgeov2-IRSTD/test_anno.json'
    
    print("正在加载预测结果和真实标注...")
    with open(pred_path, 'r') as f:
        original_predictions = json.load(f)
    
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    # 创建改进的序列后处理器
    processor = ImprovedSequenceProcessor(
        distance_threshold=100.0,  # 更大的距离阈值
        temporal_window=3,
        confidence_threshold=0.1,  # 更低的置信度阈值
        min_track_length=1,        # 更短的最小轨迹长度
        max_frame_gap=2            # 最大帧间隔
    )
    
    # 进行序列后处理
    processed_predictions = processor.process_sequence(original_predictions)
    
    # 评估改善效果
    improvement = processor.evaluate_improvement(original_predictions, processed_predictions, ground_truth)
    
    # 打印结果
    print("\n=== 改进的序列后处理效果评估 ===")
    print(f"Precision 改善: {improvement['precision_improvement']:.4f}")
    print(f"Recall 改善: {improvement['recall_improvement']:.4f}")
    print(f"F1 Score 改善: {improvement['f1_improvement']:.4f}")
    print(f"MSE 改善: {improvement['mse_improvement']:.4f}")
    
    print("\n=== 原始指标 ===")
    print(f"Precision: {improvement['original_metrics']['precision']:.4f}")
    print(f"Recall: {improvement['original_metrics']['recall']:.4f}")
    print(f"F1 Score: {improvement['original_metrics']['f1']:.4f}")
    print(f"MSE: {improvement['original_metrics']['mse']:.4f}")
    
    print("\n=== 处理后指标 ===")
    print(f"Precision: {improvement['processed_metrics']['precision']:.4f}")
    print(f"Recall: {improvement['processed_metrics']['recall']:.4f}")
    print(f"F1 Score: {improvement['processed_metrics']['f1']:.4f}")
    print(f"MSE: {improvement['processed_metrics']['mse']:.4f}")
    
    # 保存处理后的结果
    output_path = 'results/WTNet/improved_processed_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(processed_predictions, f, indent=2)
    print(f"\n改进的处理后的预测结果已保存到: {output_path}")

if __name__ == '__main__':
    main() 