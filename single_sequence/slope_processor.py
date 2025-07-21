import json
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple, Optional, Set
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, Counter
import copy
import math
import sys

class SingleSequenceSlopeProcessor:
    """单个序列的斜率处理器，专门处理指定序列的轨迹补全和异常点筛选"""
    
    def __init__(self, 
                 sequence_id: int,
                 base_distance_threshold: float = 80.0,
                 min_track_length: int = 2,
                 expected_sequence_length: int = 5,
                 slope_tolerance: float = 0.1,  # 斜率容差
                 min_slope_count: int = 2,      # 最小斜率出现次数
                 point_distance_threshold: float = 5.0):  # 重合点过滤阈值
        """
        初始化单个序列的斜率处理器
        
        Args:
            sequence_id: 要处理的序列ID
            base_distance_threshold: 基础距离阈值
            min_track_length: 最小轨迹长度
            expected_sequence_length: 期望的序列长度
            slope_tolerance: 斜率容差，用于聚类相似斜率
            min_slope_count: 最小斜率出现次数，用于确定主导斜率
            point_distance_threshold: 重合点过滤阈值
        """
        self.sequence_id = sequence_id
        self.base_distance_threshold = base_distance_threshold
        self.min_track_length = min_track_length
        self.expected_sequence_length = expected_sequence_length
        self.slope_tolerance = slope_tolerance
        self.min_slope_count = min_slope_count
        self.point_distance_threshold = point_distance_threshold
        
        # 添加父目录到路径，以便导入eval_predictions
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from eval_predictions import calculate_metrics
        self.calculate_metrics = calculate_metrics
    
    def calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """计算两点之间的欧氏距离"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_slope(self, point1: List[float], point2: List[float]) -> float:
        """计算两点之间的斜率"""
        if abs(point2[0] - point1[0]) < 1e-6:  # 避免除零
            return float('inf') if point2[1] > point1[1] else float('-inf')
        return (point2[1] - point1[1]) / (point2[0] - point1[0])
    
    def extract_sequence_data(self, predictions: Dict) -> Dict[int, Dict]:
        """从预测结果中提取指定序列的数据"""
        sequence_data = {}
        
        for img_name, pred_info in predictions.items():
            parts = img_name.split('_')
            if len(parts) >= 2:
                try:
                    sequence_id = int(parts[0])
                    frame = int(parts[1])
                    
                    if sequence_id == self.sequence_id:
                        sequence_data[frame] = {
                            'coords': pred_info['coords'],
                            'num_objects': pred_info['num_objects'],
                            'image_name': img_name
                        }
                except ValueError:
                    continue
        
        return sequence_data
    
    def collect_sequence_slopes(self, frames_data: Dict[int, Dict]) -> Dict:
        """收集序列内所有帧间配对的斜率统计"""
        frames = sorted(frames_data.keys())
        
        if len(frames) < 2:
            return {'slopes': [], 'slope_pairs': [], 'dominant_slopes': []}
        
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
                            if distance <= 300:  # 使用较大的阈值
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
                                    'pos2_idx': pos2_idx
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
        
        print(f"序列 {self.sequence_id} 找到 {len(dominant_slopes)} 个主导斜率")
        return {
            'slopes': all_slopes,
            'slope_pairs': slope_pairs,
            'dominant_slopes': dominant_slopes,
            'total_slopes': len(all_slopes),
            'unique_slopes': len(slope_counter),
            'total_pairs_considered': len(slope_pairs),
            'frames_processed': len(frames)
        }
    
    def generate_5_points_from_pair(self, pos1: List[float], pos2: List[float], 
                                   slope: float, frame1: int, frame2: int, 
                                   total_frames: int = 5) -> List[Tuple[int, List[float]]]:
        """从两个点生成缺失帧的点，生成除了原始两个点之外的所有帧点"""
        points = []
        
        # 计算帧间隔
        frame_gap = frame2 - frame1
        
        # 生成除了frame1和frame2之外的所有帧点（1到total_frames）
        for frame in range(1, total_frames + 1):
            # 跳过原始的两个检测点
            if frame == frame1 or frame == frame2:
                continue
            
            if frame1 < frame < frame2:
                # 在两个检测点之间，使用插值
                ratio = (frame - frame1) / frame_gap
                
                if math.isinf(slope):
                    # 垂直线，只改变y坐标
                    x = pos1[0]
                    y = pos1[1] + ratio * (pos2[1] - pos1[1])
                else:
                    # 使用斜率进行插值
                    x = pos1[0] + ratio * (pos2[0] - pos1[0])
                    y = pos1[1] + ratio * (pos2[1] - pos1[1])
                
                points.append((frame, [x, y]))
                print(f"      帧 {frame} 在 {frame1}-{frame2} 之间，使用插值")
                
            elif frame < frame1:
                # 在第一个检测点之前，使用外推（从frame1向前）
                frame_gap_to_frame = frame - frame1
                extrapolated_pos = self.extrapolate_position(pos1, slope, -frame_gap_to_frame, 
                                                           reference_pos=pos2, reference_frame_gap=frame2-frame1)
                points.append((frame, extrapolated_pos))
                print(f"      帧 {frame} 在 {frame1} 之前，使用外推")
                
            elif frame > frame2:
                # 在第二个检测点之后，使用外推（从frame2向后）
                frame_gap_to_frame = frame - frame2
                extrapolated_pos = self.extrapolate_position(pos2, slope, frame_gap_to_frame, 
                                                           reference_pos=pos1, reference_frame_gap=frame2-frame1)
                points.append((frame, extrapolated_pos))
                print(f"      帧 {frame} 在 {frame2} 之后，使用外推")
        
        print(f"      总共生成 {len(points)} 个点：帧 {[p[0] for p in points]}")
        return points
    
    def extrapolate_trajectory(self, frames_data: Dict[int, Dict], 
                             dominant_slopes: List[Tuple[float, int]]) -> List[Tuple[int, List[float]]]:
        """外推轨迹，填补间隙"""
        frames = sorted(frames_data.keys())
        if len(frames) < 2:
            return []
        
        extrapolated_points = []
        
        # 找到所有有检测点的帧
        detected_frames = [f for f in frames if frames_data[f]['coords']]
        
        if len(detected_frames) < 2:
            return extrapolated_points
        
        # 计算主导斜率
        dominant_slope = dominant_slopes[0][0] if dominant_slopes else 0.0
        
        # 策略1: 填补序列开头的间隙
        first_detected = detected_frames[0]
        if first_detected > min(frames):
            # 从第一个检测点向前外推
            first_pos = frames_data[first_detected]['coords'][0]
            missing_frames = range(min(frames), first_detected)
            
            for frame in missing_frames:
                frame_gap = first_detected - frame
                extrapolated_pos = self.extrapolate_position(first_pos, dominant_slope, -frame_gap)
                extrapolated_points.append((frame, extrapolated_pos))
        
        # 策略2: 填补序列中间的间隙
        for i in range(len(detected_frames) - 1):
            frame1 = detected_frames[i]
            frame2 = detected_frames[i + 1]
            
            if frame2 - frame1 > 1:  # 存在间隙
                pos1 = frames_data[frame1]['coords'][0]
                pos2 = frames_data[frame2]['coords'][0]
                
                # 计算原始斜率
                original_slope = self.calculate_slope(pos1, pos2)
                
                # 使用主导斜率或原始斜率
                use_slope = self.find_best_slope_match(original_slope, dominant_slopes)
                
                # 填补间隙
                for frame in range(frame1 + 1, frame2):
                    ratio = (frame - frame1) / (frame2 - frame1)
                    interpolated_pos = self.interpolate_with_slope(pos1, pos2, ratio, use_slope)
                    extrapolated_points.append((frame, interpolated_pos))
        
        # 策略3: 填补序列结尾的间隙
        last_detected = detected_frames[-1]
        if last_detected < max(frames):
            # 从最后一个检测点向后外推
            last_pos = frames_data[last_detected]['coords'][0]
            missing_frames = range(last_detected + 1, max(frames) + 1)
            
            for frame in missing_frames:
                frame_gap = frame - last_detected
                extrapolated_pos = self.extrapolate_position(last_pos, dominant_slope, frame_gap)
                extrapolated_points.append((frame, extrapolated_pos))
        
        return extrapolated_points
    
    def find_best_slope_match(self, target_slope: float, dominant_slopes: List[Tuple[float, int]]) -> float:
        """找到与目标斜率最匹配的主导斜率"""
        if not dominant_slopes:
            return target_slope
        
        best_match = dominant_slopes[0][0]  # 默认使用第一个主导斜率
        min_diff = abs(target_slope - best_match)
        
        for slope, _ in dominant_slopes:
            diff = abs(target_slope - slope)
            if diff < min_diff:
                min_diff = diff
                best_match = slope
        
        return best_match
    
    def interpolate_with_slope(self, pos1: List[float], pos2: List[float], 
                             ratio: float, target_slope: float) -> List[float]:
        """使用目标斜率进行插值"""
        # 计算原始插值位置
        original_x = pos1[0] + ratio * (pos2[0] - pos1[0])
        original_y = pos1[1] + ratio * (pos2[1] - pos1[1])
        
        # 如果目标斜率是无穷大（垂直线），保持x坐标不变
        if math.isinf(target_slope):
            return [original_x, original_y]
        
        # 使用目标斜率调整位置
        # 计算从pos1到目标位置的距离
        total_distance = self.calculate_distance(pos1, pos2)
        target_distance = total_distance * ratio
        
        # 使用目标斜率计算新位置
        dx = target_distance / np.sqrt(1 + target_slope**2)
        dy = target_slope * dx
        
        # 根据方向调整符号
        if pos2[0] < pos1[0]:
            dx = -dx
        if pos2[1] < pos1[1]:
            dy = -dy
        
        estimated_x = pos1[0] + dx
        estimated_y = pos1[1] + dy
        
        return [estimated_x, estimated_y]
    
    def extrapolate_position(self, base_pos: List[float], slope: float, frame_gap: int, 
                           reference_pos: List[float] = None, reference_frame_gap: int = None) -> List[float]:
        """基于斜率和帧间隔外推位置，正确处理方向"""
        if math.isinf(slope):
            # 垂直线
            if reference_pos and reference_frame_gap:
                # 使用参考点计算每帧的位移
                step_size = abs(reference_pos[1] - base_pos[1]) / abs(reference_frame_gap)
                dy = step_size * frame_gap
            else:
                # 使用默认步长
                step_size = 20.0
                dy = step_size * frame_gap
            return [base_pos[0], base_pos[1] + dy]
        
        # 计算位移
        if reference_pos and reference_frame_gap:
            # 使用参考点计算每帧的位移
            # 计算从reference_pos到base_pos的距离和方向
            dx_ref = base_pos[0] - reference_pos[0]
            dy_ref = base_pos[1] - reference_pos[1]
            ref_distance = abs(reference_frame_gap)
            
            # 计算每帧的位移
            step_dx = dx_ref / ref_distance
            step_dy = dy_ref / ref_distance
        else:
            # 使用默认步长
            step_size = 20.0
            # 使用斜率计算每帧的位移
            step_dx = step_size / np.sqrt(1 + slope**2)
            step_dy = slope * step_dx
        
        # 计算总位移
        total_dx = step_dx * frame_gap
        total_dy = step_dy * frame_gap
        
        return [base_pos[0] + total_dx, base_pos[1] + total_dy]
    
    def filter_outliers_by_slope(self, frames_data: Dict[int, Dict], 
                               dominant_slopes: List[Tuple[float, int]]) -> Dict[int, Dict]:
        """基于主导斜率过滤异常点"""
        if not dominant_slopes:
            return frames_data
        
        filtered_frames = copy.deepcopy(frames_data)
        dominant_slope_values = [slope for slope, _ in dominant_slopes]
        
        for frame, frame_data in filtered_frames.items():
            if not frame_data['coords']:
                continue
            
            kept_coords = []
            for pos in frame_data['coords']:
                # 检查该点与其他帧中点的斜率是否符合主导斜率
                slope_matches = 0
                total_checks = 0
                
                for other_frame, other_frame_data in frames_data.items():
                    if other_frame != frame and other_frame_data['coords']:
                        for other_pos in other_frame_data['coords']:
                            slope = self.calculate_slope(pos, other_pos)
                            
                            # 检查是否与主导斜率匹配
                            for dominant_slope in dominant_slope_values:
                                if abs(slope - dominant_slope) <= self.slope_tolerance:
                                    slope_matches += 1
                                    break
                            total_checks += 1
                
                # 如果斜率匹配率足够高，保留该点
                if total_checks > 0 and (slope_matches / total_checks) >= 0.3:
                    kept_coords.append(pos)
            
            frame_data['coords'] = kept_coords
            frame_data['num_objects'] = len(kept_coords)
        
        return filtered_frames
    
    def process_sequence(self, predictions: Dict) -> Tuple[Dict[int, Dict], Dict[int, Dict], List[Tuple[int, List[float]]]]:
        """处理单个序列，返回轨迹补全和异常点筛选的结果"""
        print(f"开始处理序列 {self.sequence_id}...")
        
        # 提取当前序列的数据
        sequence_data = {}
        for img_name, pred_info in predictions.items():
            parts = img_name.split('_')
            if len(parts) >= 2:
                try:
                    sequence_id = int(parts[0])
                    frame = int(parts[1])
                    if sequence_id == self.sequence_id:
                        sequence_data[frame] = {
                            'coords': pred_info['coords'],
                            'num_objects': pred_info['num_objects'],
                            'image_name': img_name
                        }
                except ValueError:
                    continue
        
        if not sequence_data:
            print(f"  序列 {self.sequence_id} 没有找到数据")
            return {}, {}, []
        
        print(f"序列 {self.sequence_id} 包含 {len(sequence_data)} 帧")
        
        # 收集序列内的斜率统计
        slope_stats = self.collect_sequence_slopes(sequence_data)
        
        if not slope_stats['dominant_slopes']:
            print(f"  序列 {self.sequence_id} 没有找到主导斜率，保持原始数据")
            return sequence_data, sequence_data, []
        
        print(f"序列 {self.sequence_id} 找到 {len(slope_stats['dominant_slopes'])} 个主导斜率")
        for slope, count in slope_stats['dominant_slopes'][:3]:
            print(f"  主导斜率 {slope:.4f}: 出现 {count} 次")
        
        # 步骤1: 轨迹补全
        print(f"  开始轨迹补全...")
        completed_sequence, generated_points_from_pairs = self.complete_trajectory(sequence_data, slope_stats['dominant_slopes'])
        
        # 步骤2: 异常点筛选
        print(f"  开始异常点筛选...")
        filtered_sequence = self.filter_outliers_by_slope(completed_sequence, slope_stats['dominant_slopes'])
        
        return completed_sequence, filtered_sequence, generated_points_from_pairs
    
    def complete_trajectory(self, frames_data: Dict[int, Dict], 
                          dominant_slopes: List[Tuple[float, int]]) -> Tuple[Dict[int, Dict], List[Tuple[int, List[float]]]]:
        """轨迹补全"""
        # 收集所有生成的点
        all_generated_points = []
        generated_points_from_pairs = []  # 专门记录从点对生成的点
        
        # 对每个符合主导斜率的点对生成5个点
        slope_stats = self.collect_sequence_slopes(frames_data)
        for pair_info in slope_stats['slope_pairs']:
            slope = pair_info['slope']
            
            # 检查是否属于主导斜率
            is_dominant = False
            for dominant_slope, _ in dominant_slopes:
                if abs(slope - dominant_slope) <= self.slope_tolerance:
                    is_dominant = True
                    break
            
            if is_dominant:
                # 生成缺失帧的点
                generated_points = self.generate_5_points_from_pair(
                    pair_info['pos1'], pair_info['pos2'], 
                    pair_info['slope'], pair_info['frame1'], pair_info['frame2'],
                    total_frames=len(frames_data)  # 传入总帧数
                )
                print(generated_points)
                all_generated_points.extend(generated_points)
                generated_points_from_pairs.extend(generated_points)  # 记录从点对生成的点
                print(f"    从帧 {pair_info['frame1']}-{pair_info['frame2']} 生成 {len(generated_points)} 个缺失帧点")
        
        # 外推轨迹，填补间隙
        extrapolated_points = self.extrapolate_trajectory(frames_data, dominant_slopes)
        all_generated_points.extend(extrapolated_points)
        print(f"    外推生成 {len(extrapolated_points)} 个点")

        
        # 按帧组织点
        completed_frames = copy.deepcopy(frames_data)
        
        # 将生成的点添加到对应帧
        for frame, pos in all_generated_points:
            if frame not in completed_frames:
                completed_frames[frame] = {
                    'coords': [],
                    'num_objects': 0,
                    'image_name': f"{self.sequence_id}_{frame}_test"
                }
            
            # 检查是否与已有点重合
            is_duplicate = False
            for existing_pos in completed_frames[frame]['coords']:
                if self.calculate_distance(pos, existing_pos) <= self.point_distance_threshold:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                completed_frames[frame]['coords'].append(pos)
                completed_frames[frame]['num_objects'] = len(completed_frames[frame]['coords'])
        
        return completed_frames, generated_points_from_pairs
    
    def evaluate_sequence_metrics(self, original_predictions: Dict, processed_predictions: Dict, 
                                ground_truth: List[Dict]) -> Dict:
        """评估序列处理前后的指标变化"""
        # 提取当前序列的原始预测
        original_seq_predictions = {}
        for img_name, pred_info in original_predictions.items():
            parts = img_name.split('_')
            if len(parts) >= 2:
                try:
                    sequence_id = int(parts[0])
                    if sequence_id == self.sequence_id:
                        original_seq_predictions[img_name] = pred_info
                except ValueError:
                    continue
        
        # 提取当前序列的处理后预测
        processed_seq_predictions = {}
        for img_name, pred_info in processed_predictions.items():
            parts = img_name.split('_')
            if len(parts) >= 2:
                try:
                    sequence_id = int(parts[0])
                    if sequence_id == self.sequence_id:
                        processed_seq_predictions[img_name] = pred_info
                except ValueError:
                    continue
        
        # 计算指标
        original_metrics = self.calculate_metrics(original_seq_predictions, ground_truth, self.base_distance_threshold)
        processed_metrics = self.calculate_metrics(processed_seq_predictions, ground_truth, self.base_distance_threshold)
        
        # 计算改善程度
        improvement = {
            'precision_improvement': processed_metrics['precision'] - original_metrics['precision'],
            'recall_improvement': processed_metrics['recall'] - original_metrics['recall'],
            'f1_improvement': processed_metrics['f1'] - original_metrics['f1'],
            'mse_improvement': original_metrics['mse'] - processed_metrics['mse'],
            'original_metrics': original_metrics,
            'processed_metrics': processed_metrics,
            'total_tp_improvement': processed_metrics['total_tp'] - original_metrics['total_tp'],
            'total_fp_improvement': processed_metrics['total_fp'] - original_metrics['total_fp'],
            'total_fn_improvement': processed_metrics['total_fn'] - original_metrics['total_fn'],
        }
        
        return improvement
    
    def create_visualization(self, original_predictions: Dict, completed_predictions: Dict, 
                           filtered_predictions: Dict, ground_truth: List[Dict], 
                           save_dir: str = "single_sequence_results", generated_points: List[Tuple[int, List[float]]] = None):
        """创建可视化对比图"""
        print(f"  创建可视化对比图...")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 提取当前序列的数据
        original_sequence = self.extract_sequence_data(original_predictions)
        completed_sequence = self.extract_sequence_data(completed_predictions)
        filtered_sequence = self.extract_sequence_data(filtered_predictions)
        
        # 提取真实标注中当前序列的数据
        gt_sequence = {}
        for gt_item in ground_truth:
            if gt_item['sequence_id'] == self.sequence_id:
                frame = gt_item['frame']
                gt_sequence[frame] = {
                    'coords': gt_item['object_coords'],
                    'num_objects': gt_item['num_objects']
                }
        
        if not gt_sequence:
            print(f"    警告: 未找到序列 {self.sequence_id} 的真实标注数据")
            return
        
        # 获取生成的点信息
        # generated_points = self.get_generated_points_from_pairs(original_predictions) # 移除此行
        
        # 创建三列并排的可视化
        self.plot_three_stage_comparison(original_sequence, completed_sequence, filtered_sequence, 
                                       gt_sequence, generated_points, save_dir)
        
        print(f"    可视化结果已保存到: {save_dir}")
    
    def plot_three_stage_comparison(self, original_data: Dict, completed_data: Dict, 
                                  filtered_data: Dict, gt_data: Dict, 
                                  generated_points: List[Tuple[int, List[float]]], save_dir: str):
        """绘制三阶段对比图：原始预测、轨迹补全后、异常点筛选后"""
        # 设置图像大小
        img_size = [512, 640]  # 根据实际图像尺寸调整
        
        # 创建三个子图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(f'Three-Stage Processing Comparison - Sequence {self.sequence_id}', 
                    fontsize=18, fontweight='bold')
        
        # 绘制原始数据
        self.plot_sequence_frame(ax1, original_data, gt_data, img_size, "Original Prediction")
        
        # 绘制轨迹补全后的数据，显示生成的点
        self.plot_sequence_frame(ax2, completed_data, gt_data, img_size, "After Trajectory Completion", 
                               show_changes=True, compare_data=original_data, 
                               generated_points=generated_points)
        
        # 绘制异常点筛选后的数据
        self.plot_sequence_frame(ax3, filtered_data, gt_data, img_size, "After Outlier Filtering", 
                               show_changes=True, compare_data=completed_data)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/sequence_{self.sequence_id}_three_stage_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
        print(f"    三阶段对比可视化已保存到: {save_dir}/sequence_{self.sequence_id}_three_stage_comparison.png")
    
    def plot_trajectory_completion(self, original_data: Dict, completed_data: Dict, 
                                 gt_data: Dict, save_dir: str):
        """绘制轨迹补全的可视化"""
        # 设置图像大小
        img_size = [512, 640]  # 根据实际图像尺寸调整
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Trajectory Completion Comparison - Sequence {self.sequence_id}', fontsize=16, fontweight='bold')
        
        # 绘制原始数据
        self.plot_sequence_frame(ax1, original_data, gt_data, img_size, "Original Prediction")
        
        # 绘制轨迹补全后的数据
        self.plot_sequence_frame(ax2, completed_data, gt_data, img_size, "After Trajectory Completion", 
                               show_changes=True, compare_data=original_data)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/sequence_{self.sequence_id}_trajectory_completion.png', 
                   dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
        print(f"    轨迹补全可视化已保存到: {save_dir}/sequence_{self.sequence_id}_trajectory_completion.png")
    
    def plot_outlier_filtering(self, completed_data: Dict, filtered_data: Dict, 
                             gt_data: Dict, save_dir: str):
        """绘制异常点筛选的可视化"""
        # 设置图像大小
        img_size = [512, 640]  # 根据实际图像尺寸调整
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Outlier Filtering Comparison - Sequence {self.sequence_id}', fontsize=16, fontweight='bold')
        
        # 绘制轨迹补全后的数据
        self.plot_sequence_frame(ax1, completed_data, gt_data, img_size, "After Trajectory Completion")
        
        # 绘制异常点筛选后的数据
        self.plot_sequence_frame(ax2, filtered_data, gt_data, img_size, "After Outlier Filtering", 
                               show_changes=True, compare_data=completed_data)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/sequence_{self.sequence_id}_outlier_filtering.png', 
                   dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
        print(f"    异常点筛选可视化已保存到: {save_dir}/sequence_{self.sequence_id}_outlier_filtering.png")
    
    def plot_sequence_frame(self, ax, sequence_data: Dict, gt_data: Dict, img_size: List, 
                          title: str, show_changes: bool = False, compare_data: Dict = None,
                          generated_points: List[Tuple[int, List[float]]] = None):
        """绘制单个序列帧的可视化"""
        # 设置图像大小
        ax.imshow(np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8))
        
        # 用于跟踪图例
        legend_added = {'pred': False, 'gt': False, 'added': False, 'removed': False, 'generated': False}
        
        # 绘制真实标注（ground truth）
        gt_points = []
        for frame, frame_data in gt_data.items():
            if frame_data['coords']:
                gt_points.extend(frame_data['coords'])
        
        if gt_points:
            gt_array = np.array(gt_points)
            ax.scatter(gt_array[:, 0], gt_array[:, 1], 
                      c='yellow', s=80, marker='o', alpha=0.8,
                      label='Ground Truth' if not legend_added['gt'] else "")
            legend_added['gt'] = True
        
        # 绘制每一帧的点
        for frame, frame_data in sequence_data.items():
            pred_points = np.array(frame_data['coords'])
            
            if len(pred_points) > 0:
                # 绘制预测点（红色叉）
                ax.scatter(pred_points[:, 0], pred_points[:, 1], 
                          c='red', s=150, marker='x', linewidth=2,
                          label='Prediction' if not legend_added['pred'] else "")
                legend_added['pred'] = True
                
                # 在每个预测点上添加帧号标注
                for i, point in enumerate(pred_points):
                    ax.annotate(f'F{frame}', 
                              (point[0], point[1]), 
                              xytext=(5, 5), 
                              textcoords='offset points',
                              fontsize=10, 
                              color='white',
                              weight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
                
                # 只在处理后图像上显示变化标注
                if show_changes and compare_data is not None:
                    # 找出变化的点
                    compare_frame_data = compare_data.get(frame, {'coords': []})
                    added_points, removed_points = self.find_changed_points(
                        compare_frame_data['coords'], frame_data['coords']
                    )
                    
                    # 绘制新增的点（绿色中空圆圈）
                    if added_points:
                        added_array = np.array(added_points)
                        for point in added_array:
                            circle = plt.Circle((point[0], point[1]), radius=20, 
                                              fill=False, color='green', linewidth=3)
                            ax.add_patch(circle)
                        # 添加图例
                        if not legend_added['added']:
                            ax.scatter([], [], c='green', s=100, marker='o', linewidth=5, facecolors='none',
                                     label='Added')
                            legend_added['added'] = True
                    
                    # 绘制删除的点（白色中空圆圈）
                    if removed_points:
                        removed_array = np.array(removed_points)
                        for point in removed_array:
                            circle = plt.Circle((point[0], point[1]), radius=20, 
                                              fill=False, color='white', linewidth=3)
                            ax.add_patch(circle)
                        # 添加图例
                        if not legend_added['removed']:
                            ax.scatter([], [], c='white', s=100, marker='o', linewidth=1, facecolors='none',
                                     label='Removed')
                            legend_added['removed'] = True
        
        # 绘制生成的点（蓝色星号）
        if generated_points:
            generated_coords = [point[1] for point in generated_points]  # 提取坐标
            if generated_coords:
                generated_array = np.array(generated_coords)
                ax.scatter(generated_array[:, 0], generated_array[:, 1], 
                          c='blue', s=120, marker='*', linewidth=2, alpha=0.8,
                          label='Generated from Pairs' if not legend_added['generated'] else "")
                legend_added['generated'] = True
                
                # 在每个生成点上添加帧号标注
                for i, point in enumerate(generated_points):
                    frame, coords = point
                    ax.annotate(f'F{frame}', 
                              (coords[0], coords[1]), 
                              xytext=(5, 5), 
                              textcoords='offset points',
                              fontsize=10, 
                              color='white',
                              weight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7))
        
        # 设置坐标轴
        ax.set_xlim(0, img_size[1])
        ax.set_ylim(img_size[0], 0)
        ax.set_title(f"{title} - Sequence {self.sequence_id}", fontsize=12, fontweight='bold', color='white')
        
        # 添加坐标轴标签
        ax.set_xlabel('X coordinate', color='white', fontsize=10)
        ax.set_ylabel('Y coordinate', color='white', fontsize=10)
        
        # 添加网格线以便更好地理解坐标系统
        ax.grid(True, alpha=0.3, color='gray')
        
        # 设置坐标轴颜色
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        
        # ax.axis('off')  # 注释掉，以便显示坐标轴
        
        # 添加图例
        if any(legend_added.values()):
            legend = ax.legend(loc='upper right', frameon=True, 
                             facecolor='black', edgecolor='white',
                             labelcolor='white', fontsize=8)
    
    def find_changed_points(self, original_coords: List, processed_coords: List, 
                          distance_threshold: float = 5.0) -> Tuple[List, List]:
        """找出变化的点"""
        added_points = []
        removed_points = []
        
        # 找出新增的点
        for proc_point in processed_coords:
            is_new = True
            for orig_point in original_coords:
                if self.calculate_distance(proc_point, orig_point) <= distance_threshold:
                    is_new = False
                    break
            if is_new:
                added_points.append(proc_point)
        
        # 找出删除的点
        for orig_point in original_coords:
            is_removed = True
            for proc_point in processed_coords:
                if self.calculate_distance(orig_point, proc_point) <= distance_threshold:
                    is_removed = False
                    break
            if is_removed:
                removed_points.append(orig_point)
        
        return added_points, removed_points
    
    def run_analysis(self, original_predictions: Dict, ground_truth: List[Dict], 
                    save_dir: str = "single_sequence_results") -> Dict:
        """运行完整的序列分析"""
        print(f"开始分析序列 {self.sequence_id}...")
        
        # 处理序列，获得轨迹补全和异常点筛选的结果
        completed_sequence, filtered_sequence, generated_points_from_pairs = self.process_sequence(original_predictions)
        
        # 将结果转换回原始格式用于评估
        completed_predictions = {}
        filtered_predictions = {}
        
        for frame, frame_data in completed_sequence.items():
            image_name = frame_data['image_name']
            completed_predictions[image_name] = {
                'coords': frame_data['coords'],
                'num_objects': frame_data['num_objects']
            }
        
        for frame, frame_data in filtered_sequence.items():
            image_name = frame_data['image_name']
            filtered_predictions[image_name] = {
                'coords': frame_data['coords'],
                'num_objects': frame_data['num_objects']
            }
        
        # 评估轨迹补全的效果
        completion_improvement = self.evaluate_sequence_metrics(
            original_predictions, completed_predictions, ground_truth
        )
        
        # 评估异常点筛选的效果
        filtering_improvement = self.evaluate_sequence_metrics(
            completed_predictions, filtered_predictions, ground_truth
        )
        
        # 创建可视化
        self.create_visualization(original_predictions, completed_predictions, 
                                filtered_predictions, ground_truth, save_dir, generated_points_from_pairs)
        
        # 打印结果
        print(f"\n=== 序列 {self.sequence_id} 处理效果评估 ===")
        print(f"轨迹补全效果:")
        print(f"  F1 Score 改善: {completion_improvement['f1_improvement']:.4f}")
        print(f"  Precision 改善: {completion_improvement['precision_improvement']:.4f}")
        print(f"  Recall 改善: {completion_improvement['recall_improvement']:.4f}")
        print(f"  MSE 改善: {completion_improvement['mse_improvement']:.4f}")
        
        print(f"\n异常点筛选效果:")
        print(f"  F1 Score 改善: {filtering_improvement['f1_improvement']:.4f}")
        print(f"  Precision 改善: {filtering_improvement['precision_improvement']:.4f}")
        print(f"  Recall 改善: {filtering_improvement['recall_improvement']:.4f}")
        print(f"  MSE 改善: {filtering_improvement['mse_improvement']:.4f}")
        
        print(f"\n=== 原始指标 ===")
        print(f"F1 Score: {completion_improvement['original_metrics']['f1']:.4f}")
        print(f"Precision: {completion_improvement['original_metrics']['precision']:.4f}")
        print(f"Recall: {completion_improvement['original_metrics']['recall']:.4f}")
        print(f"MSE: {completion_improvement['original_metrics']['mse']:.4f}")
        
        print(f"\n=== 轨迹补全后指标 ===")
        print(f"F1 Score: {completion_improvement['processed_metrics']['f1']:.4f}")
        print(f"Precision: {completion_improvement['processed_metrics']['precision']:.4f}")
        print(f"Recall: {completion_improvement['processed_metrics']['recall']:.4f}")
        print(f"MSE: {completion_improvement['processed_metrics']['mse']:.4f}")
        
        print(f"\n=== 异常点筛选后指标 ===")
        print(f"F1 Score: {filtering_improvement['processed_metrics']['f1']:.4f}")
        print(f"Precision: {filtering_improvement['processed_metrics']['precision']:.4f}")
        print(f"Recall: {filtering_improvement['processed_metrics']['recall']:.4f}")
        print(f"MSE: {filtering_improvement['processed_metrics']['mse']:.4f}")
        
        return {
            'completion_improvement': completion_improvement,
            'filtering_improvement': filtering_improvement,
            'completed_predictions': completed_predictions,
            'filtered_predictions': filtered_predictions
        }


def main():
    """主函数，演示单个序列处理"""
    # 加载预测结果和真实标注
    pred_path = 'results/WTNet/predictions.json'
    gt_path = 'datasets/spotgeov2-IRSTD/test_anno.json'
    
    print("正在加载预测结果和真实标注...")
    with open(pred_path, 'r') as f:
        original_predictions = json.load(f)
    
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    # 指定要处理的序列ID
    sequence_id = 1  # 可以根据需要修改
    
    # 创建单个序列处理器
    processor = SingleSequenceSlopeProcessor(
        sequence_id=sequence_id,
        base_distance_threshold=500,  # 宽松的距离阈值
        min_track_length=1,           # 允许短轨迹
        expected_sequence_length=5,   # 期望序列长度
        slope_tolerance=0.01,          # 斜率容差
        min_slope_count=2,            # 最小斜率出现次数
        point_distance_threshold=200.0  # 重合点过滤阈值
    )
    
    # 运行分析
    improvement = processor.run_analysis(original_predictions, ground_truth)
    
    print(f"\n序列 {sequence_id} 分析完成！")


if __name__ == '__main__':
    main() 