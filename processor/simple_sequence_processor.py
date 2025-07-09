import json
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import copy

class SimpleSequenceProcessor:
    """简化的序列后处理器，使用温和的过滤策略"""
    
    def __init__(self, distance_threshold: float = 100.0):
        self.distance_threshold = distance_threshold
    
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
    
    def simple_temporal_filter(self, sequence_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """简单的时序过滤，只移除明显的孤立检测点"""
        filtered_data = copy.deepcopy(sequence_data)
        
        for sequence_id, frames_data in filtered_data.items():
            frames = sorted(frames_data.keys())
            
            if len(frames) < 3:  # 如果序列太短，不做过滤
                continue
            
            for i, frame in enumerate(frames):
                current_coords = frames_data[frame]['coords']
                kept_coords = []
                
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
                    
                    # 保留检测点（更宽松的条件）
                    kept_coords.append(pos)
                
                frames_data[frame]['coords'] = kept_coords
                frames_data[frame]['num_objects'] = len(kept_coords)
        
        return filtered_data
    
    def interpolate_missing_frames(self, sequence_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """简单的插值补全"""
        interpolated_data = copy.deepcopy(sequence_data)
        
        for sequence_id, frames_data in interpolated_data.items():
            frames = sorted(frames_data.keys())
            
            for i in range(len(frames) - 1):
                frame1 = frames[i]
                frame2 = frames[i + 1]
                
                # 如果帧间隔为2，进行插值
                if frame2 - frame1 == 2:
                    coords1 = frames_data[frame1]['coords']
                    coords2 = frames_data[frame2]['coords']
                    
                    # 简单的线性插值
                    interpolated_coords = []
                    for pos1 in coords1:
                        for pos2 in coords2:
                            if self.calculate_distance(pos1, pos2) <= self.distance_threshold * 2:
                                interpolated_pos = [
                                    (pos1[0] + pos2[0]) / 2,
                                    (pos1[1] + pos2[1]) / 2
                                ]
                                interpolated_coords.append(interpolated_pos)
                    
                    # 添加到中间帧
                    mid_frame = frame1 + 1
                    if mid_frame not in frames_data:
                        frames_data[mid_frame] = {
                            'coords': [],
                            'num_objects': 0,
                            'image_name': f"{sequence_id}_{mid_frame}_test"
                        }
                    
                    frames_data[mid_frame]['coords'].extend(interpolated_coords)
                    frames_data[mid_frame]['num_objects'] = len(frames_data[mid_frame]['coords'])
        
        return interpolated_data
    
    def process_sequence(self, predictions: Dict) -> Dict:
        """处理序列数据"""
        print("开始简化的序列后处理...")
        
        sequence_data = self.extract_sequence_info(predictions)
        print(f"提取了 {len(sequence_data)} 个序列的信息")
        
        processed_predictions = {}
        
        for sequence_id, frames_data in sequence_data.items():
            print(f"处理序列 {sequence_id}，包含 {len(frames_data)} 帧")
            
            # 1. 简单的时序过滤
            filtered_data = self.simple_temporal_filter({sequence_id: frames_data})
            
            # 2. 插值补全
            interpolated_data = self.interpolate_missing_frames(filtered_data)
            
            # 转换回原始格式
            for frame, frame_data in interpolated_data[sequence_id].items():
                image_name = frame_data['image_name']
                processed_predictions[image_name] = {
                    'coords': frame_data['coords'],
                    'num_objects': frame_data['num_objects']
                }
        
        print(f"简化的序列后处理完成，处理了 {len(processed_predictions)} 张图像")
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
    """主函数"""
    # 加载数据
    pred_path = 'results/WTNet/predictions.json'
    gt_path = 'datasets/spotgeov2-IRSTD/test_anno.json'
    
    print("正在加载数据...")
    with open(pred_path, 'r') as f:
        original_predictions = json.load(f)
    
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    # 创建处理器
    processor = SimpleSequenceProcessor(distance_threshold=100.0)
    
    # 处理数据
    processed_predictions = processor.process_sequence(original_predictions)
    
    # 保存结果
    output_path = 'results/WTNet/simple_processed_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(processed_predictions, f, indent=2)
    
    print(f"结果已保存到: {output_path}")
    
    # 简单统计
    original_count = sum(len(pred['coords']) for pred in original_predictions.values())
    processed_count = sum(len(pred['coords']) for pred in processed_predictions.values())
    
    print(f"\n统计信息:")
    print(f"原始检测点总数: {original_count}")
    print(f"处理后检测点总数: {processed_count}")
    print(f"变化: {processed_count - original_count}")

if __name__ == '__main__':
    main() 