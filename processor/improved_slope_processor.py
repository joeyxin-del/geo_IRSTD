import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, Counter
import copy
import math
import os

class ImprovedSlopeProcessor:
    """基于改进斜率的轨迹补全处理器，适用于整个数据集"""
    
    def __init__(self, 
                 base_distance_threshold: float = 80.0,
                 min_track_length: int = 2,
                 expected_sequence_length: int = 5,
                 slope_tolerance: float = 0.1,  # 斜率容差
                 min_slope_count: int = 2,      # 最小斜率出现次数
                 point_distance_threshold: float = 5.0):  # 重合点过滤阈值
        """
        初始化改进的斜率处理器
        
        Args:
            base_distance_threshold: 基础距离阈值
            min_track_length: 最小轨迹长度
            expected_sequence_length: 期望的序列长度
            slope_tolerance: 斜率容差，用于聚类相似斜率
            min_slope_count: 最小斜率出现次数，用于确定主导斜率
            point_distance_threshold: 重合点过滤阈值
        """
        self.base_distance_threshold = base_distance_threshold
        self.min_track_length = min_track_length
        self.expected_sequence_length = expected_sequence_length
        self.slope_tolerance = slope_tolerance
        self.min_slope_count = min_slope_count
        self.point_distance_threshold = point_distance_threshold
    
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
    
    def collect_sequence_slopes(self, frames_data: Dict[int, Dict]) -> Dict:
        """收集单个序列内所有帧间配对的斜率统计，并标识参与主导斜率的点"""
        frames = sorted(frames_data.keys())
        
        if len(frames) < 2:
            return {'slopes': [], 'slope_pairs': [], 'dominant_slopes': [], 'point_participation': {}}
        
        all_slopes = []
        slope_pairs = []  # 存储斜率对应的点对信息
        
        # 创建点参与哈希表，用于标识每个点是否参与了主导斜率的形成
        # 格式: {(frame_id, point_idx): {'participated': False, 'point': [x, y], 'min_dominant_distance': float('inf')}}
        point_participation = {}
        
        # 初始化点参与哈希表
        for frame_id, frame_data in frames_data.items():
            if frame_data['coords']:
                for point_idx, point in enumerate(frame_data['coords']):
                    point_key = (frame_id, point_idx)
                    point_participation[point_key] = {
                        'participated': False,
                        'point': point,
                        'frame_id': frame_id,
                        'point_idx': point_idx,
                        'min_dominant_distance': float('inf'),  # 记录与主导斜率点的最短距离
                        'dominant_connections': []  # 记录与主导斜率点的连接信息
                    }
        
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
                            if distance <= 300:  # 使用更大的阈值
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
        
        # 标记参与主导斜率的点
        dominant_slope_values = [slope for slope, _ in dominant_slopes]
        for pair_info in slope_pairs:
            slope = pair_info['slope']
            
            # 检查是否属于主导斜率
            is_dominant = False
            for dominant_slope in dominant_slope_values:
                if abs(slope - dominant_slope) <= self.slope_tolerance:
                    is_dominant = True
                    break
            
            if is_dominant:
                # 标记这两个点参与了主导斜率的形成
                frame1, pos1_idx = pair_info['frame1'], pair_info['pos1_idx']
                frame2, pos2_idx = pair_info['frame2'], pair_info['pos2_idx']
                distance = pair_info['distance']
                
                point_key1 = (frame1, pos1_idx)
                point_key2 = (frame2, pos2_idx)
                
                # 更新点1的信息
                if point_key1 in point_participation:
                    point_participation[point_key1]['participated'] = True
                    point_participation[point_key1]['min_dominant_distance'] = min(
                        point_participation[point_key1]['min_dominant_distance'], distance
                    )
                    point_participation[point_key1]['dominant_connections'].append({
                        'connected_point': point_key2,
                        'distance': distance,
                        'slope': slope
                    })
                
                # 更新点2的信息
                if point_key2 in point_participation:
                    point_participation[point_key2]['participated'] = True
                    point_participation[point_key2]['min_dominant_distance'] = min(
                        point_participation[point_key2]['min_dominant_distance'], distance
                    )
                    point_participation[point_key2]['dominant_connections'].append({
                        'connected_point': point_key1,
                        'distance': distance,
                        'slope': slope
                    })
        
        # 根据距离阈值重新评估参与状态
        distance_threshold = 150  # 距离阈值
        for point_key, point_info in point_participation.items():
            if point_info['participated']:
                # 检查该点与所有主导斜率连接点的距离
                distances = [connection['distance'] for connection in point_info['dominant_connections']]
                min_distance = min(distances) if distances else float('inf')
                
                print(f"    点 {point_key} {point_info['point']} 参与主导斜率，连接距离: {distances}, 最短距离: {min_distance}")
                
                # 如果最短距离大于阈值，则不算参与
                if min_distance > distance_threshold:
                    point_info['participated'] = False
                    print(f"    点 {point_key} 最短距离 {min_distance} 大于阈值 {distance_threshold}，标记为非参与")
                else:
                    print(f"    点 {point_key} 最短距离 {min_distance} 小于等于阈值 {distance_threshold}，保持参与状态")
        
        return {
            'slopes': all_slopes,
            'slope_pairs': slope_pairs,
            'dominant_slopes': dominant_slopes,
            'total_slopes': len(all_slopes),
            'unique_slopes': len(slope_counter),
            'total_pairs_considered': len(slope_pairs),
            'frames_processed': len(frames),
            'point_participation': point_participation
        }
    
    def collect_sequence_slopes_hungarian(self, frames_data: Dict[int, Dict]) -> Dict:
        """收集单个序列内所有帧间配对的斜率统计"""
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
                    # 使用匈牙利算法找到最佳匹配
                    cost_matrix = np.zeros((len(coords1), len(coords2)))
                    for k, pos1 in enumerate(coords1):
                        for l, pos2 in enumerate(coords2):
                            cost_matrix[k, l] = self.calculate_distance(pos1, pos2)
                    
                    if cost_matrix.size > 0:
                        row_indices, col_indices = linear_sum_assignment(cost_matrix)
                        
                        # 记录匹配点之间的斜率
                        for row_idx, col_idx in zip(row_indices, col_indices):
                            pos1 = coords1[row_idx]
                            pos2 = coords2[col_idx]
                            distance = cost_matrix[row_idx, col_idx]
                            
                            # 只考虑距离合理的匹配
                            if distance <= self.base_distance_threshold * 2:
                                slope = self.calculate_slope(pos1, pos2)
                                all_slopes.append(slope)
                                slope_pairs.append({
                                    'frame1': frame1,
                                    'frame2': frame2,
                                    'pos1': pos1,
                                    'pos2': pos2,
                                    'slope': slope,
                                    'distance': distance
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
        
        return {
            'slopes': all_slopes,
            'slope_pairs': slope_pairs,
            'dominant_slopes': dominant_slopes,
            'total_slopes': len(all_slopes),
            'unique_slopes': len(slope_counter)
        }

    def generate_points_from_pair(self, pos1: List[float], pos2: List[float], 
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
                
            elif frame < frame1:
                # 在第一个检测点之前，使用外推（从frame1向前）
                frame_gap_to_frame = frame - frame1
                extrapolated_pos = self.extrapolate_position(pos1, slope, -frame_gap_to_frame, 
                                                           reference_pos=pos2, reference_frame_gap=frame2-frame1)
                points.append((frame, extrapolated_pos))
                
            elif frame > frame2:
                # 在第二个检测点之后，使用外推（从frame2向后）
                frame_gap_to_frame = frame - frame2
                extrapolated_pos = self.extrapolate_position(pos2, slope, frame_gap_to_frame, 
                                                           reference_pos=pos1, reference_frame_gap=frame2-frame1)
                points.append((frame, extrapolated_pos))
        
        return points
    
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
    
    def filter_outliers_by_dominant(self, frames_data: Dict[int, Dict], 
                                  point_participation: Dict) -> Dict[int, Dict]:
        """基于主导斜率参与情况过滤异常点"""
        if not point_participation:
            return frames_data
        
        filtered_frames = copy.deepcopy(frames_data)
        
        for frame_id, frame_data in filtered_frames.items():
            if not frame_data['coords']:
                continue
            
            kept_coords = []
            for point_idx, point in enumerate(frame_data['coords']):
                point_key = (frame_id, point_idx)
                
                # 检查该点是否参与了主导斜率的形成
                if point_key in point_participation:
                    if point_participation[point_key]['participated']:
                        # 该点参与了主导斜率的形成，保留
                        kept_coords.append(point)
                    else:
                        # 该点没有参与主导斜率的形成，可能是异常点
                        print(f"    移除未参与主导斜率的点: 帧{frame_id}, 位置{point}")
                else:
                    # 该点不在参与哈希表中，可能是新增的点，保留
                    kept_coords.append(point)
            
            frame_data['coords'] = kept_coords
            frame_data['num_objects'] = len(kept_coords)
        
        return filtered_frames
    
    def process_sequence(self, sequence_id: int, frames_data: Dict[int, Dict]) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
        """处理单个序列，返回轨迹补全和异常点筛选的结果"""
        # 收集序列内的斜率统计
        slope_stats = self.collect_sequence_slopes(frames_data)
        
        if not slope_stats['dominant_slopes']:
            return frames_data, frames_data
        
        # 步骤1: 轨迹补全
        completed_sequence, generated_points_from_pairs = self.complete_trajectory(frames_data, slope_stats['dominant_slopes'])
        
        # 步骤2: 异常点筛选 - 使用新的基于主导斜率参与情况的过滤方法
        filtered_sequence = self.filter_outliers_by_dominant(completed_sequence, slope_stats['point_participation'])
        # filtered_sequence = completed_sequence

        return completed_sequence, filtered_sequence
    
    def complete_trajectory(self, frames_data: Dict[int, Dict], 
                          dominant_slopes: List[Tuple[float, int]]) -> Tuple[Dict[int, Dict], List[Tuple[int, List[float]]]]:
        """轨迹补全"""
        # 收集所有生成的点
        all_generated_points = []
        generated_points_from_pairs = []  # 专门记录从点对生成的点
        
        # 对每个符合主导斜率的点对生成点
        slope_stats = self.collect_sequence_slopes_hungarian(frames_data)
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
                generated_points = self.generate_points_from_pair(
                    pair_info['pos1'], pair_info['pos2'], 
                    pair_info['slope'], pair_info['frame1'], pair_info['frame2'],
                    total_frames=len(frames_data)  # 传入总帧数
                )
                all_generated_points.extend(generated_points)
                generated_points_from_pairs.extend(generated_points)  # 记录从点对生成的点
        
        # 外推轨迹，填补间隙
        extrapolated_points = self.extrapolate_trajectory(frames_data, dominant_slopes)
        all_generated_points.extend(extrapolated_points)
        
        # 按帧组织点
        completed_frames = copy.deepcopy(frames_data)
        
        # 将生成的点添加到对应帧
        for frame, pos in all_generated_points:
            if frame not in completed_frames:
                completed_frames[frame] = {
                    'coords': [],
                    'num_objects': 0,
                    'image_name': f"sequence_{frame}_test"
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
    
    def process_dataset(self, predictions: Dict) -> Dict:
        """处理整个数据集，使用改进的斜率补全策略"""
        print("开始基于改进斜率的轨迹补全处理...")
        
        # 提取序列信息
        sequence_data = self.extract_sequence_info(predictions)
        print(f"提取了 {len(sequence_data)} 个序列的信息")
        
        # 处理每个序列
        processed_predictions = {}
        
        for sequence_id, frames_data in sequence_data.items():
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
        
        print(f"基于改进斜率的轨迹补全完成，处理了 {len(processed_predictions)} 张图像")
        return processed_predictions
    
    def evaluate_improvement(self, original_predictions: Dict, processed_predictions: Dict, ground_truth: List[Dict]) -> Dict:
        """评估后处理的效果改善"""
        import sys
        import os
        # 添加父目录到路径，以便导入eval_predictions
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from eval_predictions import calculate_metrics
        
        # 计算原始预测的指标
        original_metrics = calculate_metrics(original_predictions, ground_truth, self.base_distance_threshold)
        
        # 计算处理后预测的指标
        processed_metrics = calculate_metrics(processed_predictions, ground_truth, self.base_distance_threshold)
        
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


def main():
    """主函数，演示改进的斜率轨迹补全"""
    import argparse
    import os
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='改进的斜率轨迹补全处理器')
    parser.add_argument('--pred_path', type=str, default='results/spotgeov2-IRSTD/WTNet/predictions_8807.json',
                       help='预测结果文件路径')
    parser.add_argument('--gt_path', type=str, default='datasets/spotgeov2-IRSTD/test_anno.json',
                       help='真实标注文件路径')
    parser.add_argument('--output_path', type=str, default='results/0808/improved_slope_processed_predictions.json',
                       help='输出文件路径')
    parser.add_argument('--base_distance_threshold', type=float, default=500.0,
                       help='基础距离阈值')
    parser.add_argument('--slope_tolerance', type=float, default=0.05,
                       help='斜率容差')
    parser.add_argument('--min_slope_count', type=int, default=2,
                       help='最小斜率出现次数')
    parser.add_argument('--point_distance_threshold', type=float, default=200.0,
                       help='重合点过滤阈值')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
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
    
    # 创建改进的斜率处理器
    processor = ImprovedSlopeProcessor(
        base_distance_threshold=args.base_distance_threshold,
        min_track_length=1,           # 允许短轨迹
        expected_sequence_length=5,   # 期望序列长度
        slope_tolerance=args.slope_tolerance,
        min_slope_count=args.min_slope_count,
        point_distance_threshold=args.point_distance_threshold
    )
    
    # 进行改进的斜率轨迹补全
    processed_predictions = processor.process_dataset(original_predictions)
    
    # 评估改善效果
    try:
        improvement = processor.evaluate_improvement(original_predictions, processed_predictions, ground_truth)
        
        # 打印结果
        print("\n=== 改进的斜率轨迹补全效果评估 ===")
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
        
        print("\n=== 处理后指标 ===")
        print(f"Precision: {improvement['processed_metrics']['precision']:.4f}")
        print(f"Recall: {improvement['processed_metrics']['recall']:.4f}")
        print(f"F1 Score: {improvement['processed_metrics']['f1']:.4f}")
        print(f"MSE: {improvement['processed_metrics']['mse']:.4f}")
        
    except Exception as e:
        print(f"评估过程中出错: {e}")
        print("跳过评估，继续保存结果...")
    
    # 保存处理后的结果
    try:
        with open(args.output_path, 'w') as f:
            json.dump(processed_predictions, f, indent=2)
        print(f"\n改进的斜率轨迹补全结果已保存到: {args.output_path}")
    except Exception as e:
        print(f"保存结果时出错: {e}")

if __name__ == '__main__':
    main() 