import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import copy

class BalancedSequenceProcessor:
    """平衡的序列后处理器，旨在同时提升精确率和召回率"""
    
    def __init__(self, 
                 base_distance_threshold: float = 80.0,
                 temporal_window: int = 3,
                 confidence_threshold: float = 0.05,
                 min_track_length: int = 2,
                 max_frame_gap: int = 3,
                 adaptive_threshold: bool = True,
                 # 基于轨迹距离统计的新参数
                 expected_sequence_length: int = 5,
                 trajectory_mean_distance: float = 116.53,
                 trajectory_std_distance: float = 29.76,
                 trajectory_max_distance: float = 237.87,
                 trajectory_min_distance: float = 6.74,
                 trajectory_median_distance: float = 124.96):
        """
        初始化平衡的序列后处理器
        
        Args:
            base_distance_threshold: 基础距离阈值
            temporal_window: 时间窗口大小
            confidence_threshold: 置信度阈值
            min_track_length: 最小轨迹长度
            max_frame_gap: 最大帧间隔
            adaptive_threshold: 是否使用自适应阈值
            expected_sequence_length: 期望的序列长度（先验知识）
            trajectory_mean_distance: 轨迹平均距离
            trajectory_std_distance: 轨迹距离标准差
            trajectory_max_distance: 轨迹最大距离
            trajectory_min_distance: 轨迹最小距离
            trajectory_median_distance: 轨迹距离中位数
        """
        self.base_distance_threshold = base_distance_threshold
        self.temporal_window = temporal_window
        self.confidence_threshold = confidence_threshold
        self.min_track_length = min_track_length
        self.max_frame_gap = max_frame_gap
        self.adaptive_threshold = adaptive_threshold
        
        # 基于轨迹距离统计的参数
        self.expected_sequence_length = expected_sequence_length
        self.trajectory_mean_distance = trajectory_mean_distance
        self.trajectory_std_distance = trajectory_std_distance
        self.trajectory_max_distance = trajectory_max_distance
        self.trajectory_min_distance = trajectory_min_distance
        self.trajectory_median_distance = trajectory_median_distance
        
        # 计算基于统计的距离阈值
        self.statistical_distance_threshold = min(
            self.trajectory_mean_distance + 2 * self.trajectory_std_distance,  # 2σ范围
            self.trajectory_max_distance * 0.8  # 最大距离的80%
        )
    
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
    
    def adaptive_temporal_filtering(self, sequence_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """自适应时序过滤，根据序列特征调整过滤策略"""
        filtered_data = copy.deepcopy(sequence_data)
        
        for sequence_id, frames_data in filtered_data.items():
            frames = sorted(frames_data.keys())
            
            if len(frames) < 2:
                continue
            
            # 计算序列的统计特征
            all_coords = []
            for frame_data in frames_data.values():
                all_coords.extend(frame_data['coords'])
            
            if not all_coords:
                continue
            
            # 计算平均检测密度
            avg_detections_per_frame = len(all_coords) / len(frames)
            
            # 自适应调整距离阈值
            if self.adaptive_threshold:
                if avg_detections_per_frame > 3:
                    # 高密度序列，使用更严格的阈值
                    distance_threshold = self.base_distance_threshold * 0.8
                elif avg_detections_per_frame < 1:
                    # 低密度序列，使用更宽松的阈值
                    distance_threshold = self.base_distance_threshold * 1.5
                else:
                    distance_threshold = self.base_distance_threshold
            else:
                distance_threshold = self.base_distance_threshold
            
            # 对每一帧进行过滤
            for i, frame in enumerate(frames):
                current_coords = frames_data[frame]['coords']
                filtered_coords = []
                
                for pos in current_coords:
                    # 计算时序一致性分数
                    consistency_score = 0
                    max_possible_score = 0
                    
                    # 检查前后帧的支持
                    for offset in range(-self.temporal_window, self.temporal_window + 1):
                        if offset == 0:
                            continue
                        
                        check_frame = frame + offset
                        if check_frame in frames_data:
                            max_possible_score += 1
                            check_coords = frames_data[check_frame]['coords']
                            
                            # 找到最近的检测点
                            min_dist = float('inf')
                            for check_pos in check_coords:
                                dist = self.calculate_distance(pos, check_pos)
                                min_dist = min(min_dist, dist)
                            
                            # 如果距离在阈值内，增加一致性分数
                            if min_dist <= distance_threshold:
                                consistency_score += 1
                    
                    # 计算一致性比例
                    if max_possible_score > 0:
                        consistency_ratio = consistency_score / max_possible_score
                    else:
                        consistency_ratio = 1.0  # 如果没有其他帧，保持检测点
                    
                    # 根据一致性比例和检测密度决定是否保留
                    if consistency_ratio >= 0.3 or avg_detections_per_frame < 1.5:
                        filtered_coords.append(pos)
                
                frames_data[frame]['coords'] = filtered_coords
                frames_data[frame]['num_objects'] = len(filtered_coords)
        
        return filtered_data
    
    def smart_interpolation(self, sequence_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """智能插值，考虑运动轨迹的连续性"""
        interpolated_data = copy.deepcopy(sequence_data)
        
        for sequence_id, frames_data in interpolated_data.items():
            frames = sorted(frames_data.keys())
            
            for i in range(len(frames) - 1):
                frame1 = frames[i]
                frame2 = frames[i + 1]
                
                # 如果帧间隔在合理范围内，进行插值
                if 1 < frame2 - frame1 <= self.max_frame_gap:
                    coords1 = frames_data[frame1]['coords']
                    coords2 = frames_data[frame2]['coords']
                    
                    # 使用匈牙利算法进行最优匹配
                    if coords1 and coords2:
                        cost_matrix = np.zeros((len(coords1), len(coords2)))
                        for j, pos1 in enumerate(coords1):
                            for k, pos2 in enumerate(coords2):
                                cost_matrix[j, k] = self.calculate_distance(pos1, pos2)
                        
                        # 如果成本矩阵不为空，进行匹配
                        if cost_matrix.size > 0:
                            row_indices, col_indices = linear_sum_assignment(cost_matrix)
                            
                            # 为匹配的检测点进行插值
                            for frame in range(frame1 + 1, frame2):
                                alpha = (frame - frame1) / (frame2 - frame1)
                                interpolated_coords = []
                                
                                for row_idx, col_idx in zip(row_indices, col_indices):
                                    pos1 = coords1[row_idx]
                                    pos2 = coords2[col_idx]
                                    
                                    # 只对距离合理的匹配进行插值
                                    if cost_matrix[row_idx, col_idx] <= self.base_distance_threshold * 2:
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
    
    def remove_noise_detections(self, sequence_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """移除噪声检测点，保留有意义的轨迹"""
        cleaned_data = copy.deepcopy(sequence_data)
        
        for sequence_id, frames_data in cleaned_data.items():
            frames = sorted(frames_data.keys())
            
            if len(frames) < 3:
                continue
            
            for frame in frames:
                current_coords = frames_data[frame]['coords']
                kept_coords = []
                
                for pos in current_coords:
                    # 检查在时间窗口内的支持
                    support_frames = 0
                    total_frames = 0
                    
                    for offset in range(-2, 3):  # 检查前后2帧
                        check_frame = frame + offset
                        if check_frame in frames_data and check_frame != frame:
                            total_frames += 1
                            check_coords = frames_data[check_frame]['coords']
                            
                            # 检查是否有相近的检测点
                            for check_pos in check_coords:
                                if self.calculate_distance(pos, check_pos) <= self.base_distance_threshold * 1.2:
                                    support_frames += 1
                                    break
                    
                    # 如果得到足够的支持，或者序列较短，则保留
                    if total_frames == 0 or support_frames / total_frames >= 0.2 or len(frames) <= 3:
                        kept_coords.append(pos)
                
                frames_data[frame]['coords'] = kept_coords
                frames_data[frame]['num_objects'] = len(kept_coords)
        
        return cleaned_data
    
    def complete_missing_detections(self, sequence_data: Dict[int, Dict], distance_threshold: float = None) -> Dict[int, Dict]:
        """
        利用先验知识补全漏检
        基于轨迹距离统计信息和期望序列长度来补全缺失的检测点
        
        Args:
            sequence_data: 序列数据
            distance_threshold: 距离阈值，如果为None则使用统计阈值
        """
        if distance_threshold is None:
            distance_threshold = self.statistical_distance_threshold
            
        # 使用更宽松的补全阈值，允许更多的补全
        completion_threshold = distance_threshold * 2.0  # 放宽到2倍阈值
        aggressive_threshold = distance_threshold * 3.0  # 激进补全阈值
            
        completed_data = copy.deepcopy(sequence_data)
        
        for sequence_id, frames_data in completed_data.items():
            frames = sorted(frames_data.keys())
            
            # 如果序列长度不足期望长度，尝试补全
            if len(frames) < self.expected_sequence_length:
                print(f"序列 {sequence_id} 只有 {len(frames)} 帧，期望 {self.expected_sequence_length} 帧，尝试补全...")
                
                # 找到缺失的帧
                expected_frames = set(range(min(frames), max(frames) + 1))
                missing_frames = expected_frames - set(frames)
                
                for missing_frame in missing_frames:
                    # 找到前后最近的检测点
                    prev_frame = None
                    next_frame = None
                    
                    for frame in frames:
                        if frame < missing_frame and (prev_frame is None or frame > prev_frame):
                            prev_frame = frame
                        if frame > missing_frame and (next_frame is None or frame < next_frame):
                            next_frame = frame
                    
                    # 如果前后都有检测点，进行插值
                    if prev_frame is not None and next_frame is not None:
                        prev_coords = frames_data[prev_frame]['coords']
                        next_coords = frames_data[next_frame]['coords']
                        
                        if prev_coords and next_coords:
                            # 使用匈牙利算法找到最佳匹配
                            cost_matrix = np.zeros((len(prev_coords), len(next_coords)))
                            for i, pos1 in enumerate(prev_coords):
                                for j, pos2 in enumerate(next_coords):
                                    cost_matrix[i, j] = self.calculate_distance(pos1, pos2)
                            
                            if cost_matrix.size > 0:
                                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                                
                                # 使用更宽松的阈值进行匹配
                                valid_matches = []
                                for row_idx, col_idx in zip(row_indices, col_indices):
                                    distance = cost_matrix[row_idx, col_idx]
                                    # 使用宽松阈值判断是否合理
                                    if distance <= completion_threshold:
                                        valid_matches.append((row_idx, col_idx, distance))
                                
                                if valid_matches:
                                    # 选择距离最小的匹配进行插值
                                    best_match = min(valid_matches, key=lambda x: x[2])
                                    row_idx, col_idx, _ = best_match
                                    
                                    pos1 = prev_coords[row_idx]
                                    pos2 = next_coords[col_idx]
                                    
                                    # 计算插值权重
                                    alpha = (missing_frame - prev_frame) / (next_frame - prev_frame)
                                    
                                    # 插值位置
                                    interpolated_pos = [
                                        pos1[0] + alpha * (pos2[0] - pos1[0]),
                                        pos1[1] + alpha * (pos2[1] - pos1[1])
                                    ]
                                    
                                    # 添加到缺失帧
                                    frames_data[missing_frame] = {
                                        'coords': [interpolated_pos],
                                        'num_objects': 1,
                                        'image_name': f"{sequence_id}_{missing_frame}_test"
                                    }
                                    print(f"  补全帧 {missing_frame}，位置: {interpolated_pos}")
                    
                    # 如果只有前帧或只有后帧，使用增强的外推
                    elif prev_frame is not None:
                        prev_coords = frames_data[prev_frame]['coords']
                        if prev_coords:
                            # 尝试找到更多前帧来预测运动趋势
                            motion_vector = self.estimate_motion_vector(frames_data, prev_frame, look_back=3)
                            if motion_vector is not None:
                                # 使用运动向量进行外推
                                estimated_pos = [
                                    prev_coords[0][0] + motion_vector[0] * (missing_frame - prev_frame),
                                    prev_coords[0][1] + motion_vector[1] * (missing_frame - prev_frame)
                                ]
                            else:
                                # 使用前帧的位置作为估计
                                estimated_pos = prev_coords[0]
                            
                            frames_data[missing_frame] = {
                                'coords': [estimated_pos],
                                'num_objects': 1,
                                'image_name': f"{sequence_id}_{missing_frame}_test"
                            }
                            print(f"  外推帧 {missing_frame}，位置: {estimated_pos}")
                    
                    elif next_frame is not None:
                        next_coords = frames_data[next_frame]['coords']
                        if next_coords:
                            # 尝试找到更多后帧来预测运动趋势
                            motion_vector = self.estimate_motion_vector(frames_data, next_frame, look_forward=3)
                            if motion_vector is not None:
                                # 使用运动向量进行外推
                                estimated_pos = [
                                    next_coords[0][0] + motion_vector[0] * (missing_frame - next_frame),
                                    next_coords[0][1] + motion_vector[1] * (missing_frame - next_frame)
                                ]
                            else:
                                # 使用后帧的位置作为估计
                                estimated_pos = next_coords[0]
                            
                            frames_data[missing_frame] = {
                                'coords': [estimated_pos],
                                'num_objects': 1,
                                'image_name': f"{sequence_id}_{missing_frame}_test"
                            }
                            print(f"  外推帧 {missing_frame}，位置: {estimated_pos}")
            
            # 对于已有完整帧的序列，检查是否有明显的漏检
            else:
                # 检查每帧的检测数量是否合理
                for frame in frames:
                    current_coords = frames_data[frame]['coords']
                    
                    # 如果当前帧没有检测点，尝试从相邻帧推断
                    if not current_coords:
                        # 找到前后帧的检测点
                        prev_frame = None
                        next_frame = None
                        
                        for f in frames:
                            if f < frame and frames_data[f]['coords']:
                                prev_frame = f
                            if f > frame and frames_data[f]['coords']:
                                next_frame = f
                                break
                        
                        if prev_frame is not None and next_frame is not None:
                            # 插值补全
                            prev_coords = frames_data[prev_frame]['coords']
                            next_coords = frames_data[next_frame]['coords']
                            
                            if prev_coords and next_coords:
                                # 找到最佳匹配
                                cost_matrix = np.zeros((len(prev_coords), len(next_coords)))
                                for i, pos1 in enumerate(prev_coords):
                                    for j, pos2 in enumerate(next_coords):
                                        cost_matrix[i, j] = self.calculate_distance(pos1, pos2)
                                
                                if cost_matrix.size > 0:
                                    row_indices, col_indices = linear_sum_assignment(cost_matrix)
                                    
                                    # 选择距离最小的匹配
                                    min_cost = float('inf')
                                    best_match = None
                                    
                                    for row_idx, col_idx in zip(row_indices, col_indices):
                                        if cost_matrix[row_idx, col_idx] < min_cost:
                                            min_cost = cost_matrix[row_idx, col_idx]
                                            best_match = (row_idx, col_idx)
                                    
                                    # 使用更宽松的阈值
                                    if best_match and min_cost <= completion_threshold:
                                        row_idx, col_idx = best_match
                                        pos1 = prev_coords[row_idx]
                                        pos2 = next_coords[col_idx]
                                        
                                        alpha = (frame - prev_frame) / (next_frame - prev_frame)
                                        interpolated_pos = [
                                            pos1[0] + alpha * (pos2[0] - pos1[0]),
                                            pos1[1] + alpha * (pos2[1] - pos1[1])
                                        ]
                                        
                                        frames_data[frame]['coords'] = [interpolated_pos]
                                        frames_data[frame]['num_objects'] = 1
                                        print(f"  补全序列 {sequence_id} 帧 {frame}，位置: {interpolated_pos}")
                        
                        # 如果插值失败，尝试激进补全
                        elif prev_frame is not None or next_frame is not None:
                            estimated_pos = self.aggressive_completion(frames_data, frame, prev_frame, next_frame, aggressive_threshold)
                            if estimated_pos is not None:
                                frames_data[frame]['coords'] = [estimated_pos]
                                frames_data[frame]['num_objects'] = 1
                                print(f"  激进补全序列 {sequence_id} 帧 {frame}，位置: {estimated_pos}")
        
        return completed_data
    
    def estimate_motion_vector(self, frames_data: Dict[int, Dict], reference_frame: int, look_back: int = 3, look_forward: int = 0) -> Optional[List[float]]:
        """
        估计运动向量，用于外推预测
        
        Args:
            frames_data: 帧数据
            reference_frame: 参考帧
            look_back: 向后看的帧数
            look_forward: 向前看的帧数
            
        Returns:
            运动向量 [dx, dy]，如果无法估计则返回None
        """
        frames = sorted(frames_data.keys())
        motion_vectors = []
        
        # 向后看
        for i in range(1, look_back + 1):
            check_frame = reference_frame - i
            if check_frame in frames_data and frames_data[check_frame]['coords']:
                ref_coords = frames_data[reference_frame]['coords']
                check_coords = frames_data[check_frame]['coords']
                
                if ref_coords and check_coords:
                    # 找到最近的匹配
                    min_dist = float('inf')
                    best_pair = None
                    
                    for ref_pos in ref_coords:
                        for check_pos in check_coords:
                            dist = self.calculate_distance(ref_pos, check_pos)
                            if dist < min_dist:
                                min_dist = dist
                                best_pair = (ref_pos, check_pos)
                    
                    if best_pair and min_dist <= self.statistical_distance_threshold * 2:
                        ref_pos, check_pos = best_pair
                        dx = (ref_pos[0] - check_pos[0]) / i
                        dy = (ref_pos[1] - check_pos[1]) / i
                        motion_vectors.append([dx, dy])
        
        # 向前看
        for i in range(1, look_forward + 1):
            check_frame = reference_frame + i
            if check_frame in frames_data and frames_data[check_frame]['coords']:
                ref_coords = frames_data[reference_frame]['coords']
                check_coords = frames_data[check_frame]['coords']
                
                if ref_coords and check_coords:
                    # 找到最近的匹配
                    min_dist = float('inf')
                    best_pair = None
                    
                    for ref_pos in ref_coords:
                        for check_pos in check_coords:
                            dist = self.calculate_distance(ref_pos, check_pos)
                            if dist < min_dist:
                                min_dist = dist
                                best_pair = (ref_pos, check_pos)
                    
                    if best_pair and min_dist <= self.statistical_distance_threshold * 2:
                        ref_pos, check_pos = best_pair
                        dx = (check_pos[0] - ref_pos[0]) / i
                        dy = (check_pos[1] - ref_pos[1]) / i
                        motion_vectors.append([dx, dy])
        
        # 计算平均运动向量
        if motion_vectors:
            avg_dx = np.mean([v[0] for v in motion_vectors])
            avg_dy = np.mean([v[1] for v in motion_vectors])
            return [avg_dx, avg_dy]
        
        return None
    
    def aggressive_completion(self, frames_data: Dict[int, Dict], target_frame: int, prev_frame: Optional[int], next_frame: Optional[int], threshold: float) -> Optional[List[float]]:
        """
        激进补全策略，使用更宽松的条件进行补全
        
        Args:
            frames_data: 帧数据
            target_frame: 目标帧
            prev_frame: 前帧
            next_frame: 后帧
            threshold: 距离阈值
            
        Returns:
            估计的位置，如果无法估计则返回None
        """
        # 策略1: 使用最近的检测点
        if prev_frame is not None and frames_data[prev_frame]['coords']:
            return frames_data[prev_frame]['coords'][0]
        
        if next_frame is not None and frames_data[next_frame]['coords']:
            return frames_data[next_frame]['coords'][0]
        
        # 策略2: 使用序列中所有检测点的中心
        all_coords = []
        for frame, frame_data in frames_data.items():
            if frame_data['coords']:
                all_coords.extend(frame_data['coords'])
        
        if all_coords:
            center_x = np.mean([pos[0] for pos in all_coords])
            center_y = np.mean([pos[1] for pos in all_coords])
            return [center_x, center_y]
        
        return None
    
    def analyze_trajectory_statistics(self, sequence_data: Dict[int, Dict]) -> Dict:
        """
        分析轨迹距离统计信息，为补全策略提供依据
        """
        print("分析轨迹距离统计信息...")
        
        all_distances = []
        sequence_lengths = []
        missing_frames_count = 0
        total_expected_frames = 0
        
        for sequence_id, frames_data in sequence_data.items():
            frames = sorted(frames_data.keys())
            sequence_lengths.append(len(frames))
            
            # 计算期望的帧数
            if frames:
                expected_frames = set(range(min(frames), max(frames) + 1))
                missing_frames_count += len(expected_frames) - len(frames)
                total_expected_frames += len(expected_frames)
            
            # 计算相邻帧之间的距离
            for i in range(len(frames) - 1):
                frame1 = frames[i]
                frame2 = frames[i + 1]
                
                coords1 = frames_data[frame1]['coords']
                coords2 = frames_data[frame2]['coords']
                
                if coords1 and coords2:
                    # 找到最近的两个检测点
                    min_dist = float('inf')
                    for pos1 in coords1:
                        for pos2 in coords2:
                            dist = self.calculate_distance(pos1, pos2)
                            min_dist = min(min_dist, dist)
                    
                    if min_dist != float('inf'):
                        all_distances.append(min_dist)
        
        # 计算统计信息
        if all_distances:
            actual_mean = np.mean(all_distances)
            actual_std = np.std(all_distances)
            actual_median = np.median(all_distances)
            actual_min = np.min(all_distances)
            actual_max = np.max(all_distances)
        else:
            actual_mean = actual_std = actual_median = actual_min = actual_max = 0
        
        if sequence_lengths:
            avg_sequence_length = np.mean(sequence_lengths)
            min_sequence_length = np.min(sequence_lengths)
            max_sequence_length = np.max(sequence_lengths)
        else:
            avg_sequence_length = min_sequence_length = max_sequence_length = 0
        
        statistics = {
            'actual_distances': {
                'mean': actual_mean,
                'std': actual_std,
                'median': actual_median,
                'min': actual_min,
                'max': actual_max,
                'count': len(all_distances)
            },
            'sequence_lengths': {
                'mean': avg_sequence_length,
                'min': min_sequence_length,
                'max': max_sequence_length,
                'count': len(sequence_lengths)
            },
            'missing_frames': {
                'count': missing_frames_count,
                'total_expected': total_expected_frames,
                'missing_ratio': missing_frames_count / max(total_expected_frames, 1)
            },
            'comparison_with_prior': {
                'distance_mean_diff': actual_mean - self.trajectory_mean_distance,
                'distance_std_diff': actual_std - self.trajectory_std_distance,
                'length_diff': avg_sequence_length - self.expected_sequence_length
            }
        }
        
        print(f"实际轨迹距离统计:")
        print(f"  平均距离: {actual_mean:.2f} (先验: {self.trajectory_mean_distance:.2f})")
        print(f"  标准差: {actual_std:.2f} (先验: {self.trajectory_std_distance:.2f})")
        print(f"  中位数: {actual_median:.2f} (先验: {self.trajectory_median_distance:.2f})")
        print(f"  最小值: {actual_min:.2f} (先验: {self.trajectory_min_distance:.2f})")
        print(f"  最大值: {actual_max:.2f} (先验: {self.trajectory_max_distance:.2f})")
        print(f"序列长度统计:")
        print(f"  平均长度: {avg_sequence_length:.2f} (期望: {self.expected_sequence_length})")
        print(f"  最小长度: {min_sequence_length}")
        print(f"  最大长度: {max_sequence_length}")
        print(f"漏检统计:")
        print(f"  缺失帧数: {missing_frames_count}")
        print(f"  总期望帧数: {total_expected_frames}")
        print(f"  缺失比例: {missing_frames_count / max(total_expected_frames, 1):.2%}")
        
        return statistics
    
    def process_sequence(self, predictions: Dict) -> Dict:
        """
        处理整个预测结果，使用平衡的策略
        
        Args:
            predictions: 原始预测结果
            
        Returns:
            processed_predictions: 处理后的预测结果
        """
        print("开始平衡的序列后处理...")
        
        # 提取序列信息
        sequence_data = self.extract_sequence_info(predictions)
        print(f"提取了 {len(sequence_data)} 个序列的信息")
        
        # 分析轨迹距离统计信息
        statistics = self.analyze_trajectory_statistics(sequence_data)
        
        # 根据统计分析结果调整处理参数
        if statistics['comparison_with_prior']['distance_mean_diff'] > 20:
            print("检测到实际距离与先验差异较大，调整距离阈值...")
            # 如果实际距离明显大于先验，适当放宽阈值
            adjusted_threshold = self.statistical_distance_threshold * 1.5  # 更宽松的调整
        elif statistics['comparison_with_prior']['distance_mean_diff'] < -20:
            print("检测到实际距离与先验差异较大，收紧距离阈值...")
            # 如果实际距离明显小于先验，适当收紧阈值
            adjusted_threshold = self.statistical_distance_threshold * 0.9  # 不那么严格的收紧
        else:
            adjusted_threshold = self.statistical_distance_threshold
        
        print(f"使用调整后的距离阈值: {adjusted_threshold:.2f}")
        
        processed_predictions = {}
        
        for sequence_id, frames_data in sequence_data.items():
            print(f"处理序列 {sequence_id}，包含 {len(frames_data)} 帧")
            
            # # 1. 自适应时序过滤
            # filtered_data = self.adaptive_temporal_filtering({sequence_id: frames_data})
            filtered_data = {sequence_id: frames_data}
            # 2. 智能插值
            interpolated_data = self.smart_interpolation(filtered_data)
            
            # 3. 移除噪声检测点
            cleaned_data = self.remove_noise_detections(interpolated_data)
            # cleaned_data = interpolated_data
            # 4. 利用先验知识补全漏检
            completed_data = self.complete_missing_detections(cleaned_data, adjusted_threshold)
            
            # 5. 激进补全：基于序列模式的补全
            aggressive_completed_data = self.aggressive_sequence_completion(completed_data, adjusted_threshold)
            
            # 转换回原始格式
            for frame, frame_data in aggressive_completed_data[sequence_id].items():
                image_name = frame_data['image_name']
                processed_predictions[image_name] = {
                    'coords': frame_data['coords'],
                    'num_objects': frame_data['num_objects']
                }
        
        print(f"平衡的序列后处理完成，处理了 {len(processed_predictions)} 张图像")
        return processed_predictions
    
    def aggressive_sequence_completion(self, sequence_data: Dict[int, Dict], distance_threshold: float) -> Dict[int, Dict]:
        """
        激进序列补全，基于序列模式和统计信息进行更激进的补全
        
        Args:
            sequence_data: 序列数据
            distance_threshold: 距离阈值
            
        Returns:
            补全后的序列数据
        """
        aggressive_data = copy.deepcopy(sequence_data)
        
        for sequence_id, frames_data in aggressive_data.items():
            frames = sorted(frames_data.keys())
            
            # 计算序列的统计特征
            all_coords = []
            for frame_data in frames_data.values():
                all_coords.extend(frame_data['coords'])
            
            if not all_coords:
                continue
            
            # 计算序列中心
            center_x = np.mean([pos[0] for pos in all_coords])
            center_y = np.mean([pos[1] for pos in all_coords])
            
            # 计算序列的边界
            min_x = min([pos[0] for pos in all_coords])
            max_x = max([pos[0] for pos in all_coords])
            min_y = min([pos[1] for pos in all_coords])
            max_y = max([pos[1] for pos in all_coords])
            
            # 计算序列的期望帧范围
            if frames:
                expected_start = min(frames)
                expected_end = max(frames)
                expected_frames = set(range(expected_start, expected_end + 1))
                missing_frames = expected_frames - set(frames)
                
                print(f"序列 {sequence_id} 激进补全: 期望帧 {expected_start}-{expected_end}, 缺失 {len(missing_frames)} 帧")
                
                for missing_frame in missing_frames:
                    # 如果缺失帧还没有检测点，尝试补全
                    if missing_frame not in frames_data or not frames_data[missing_frame]['coords']:
                        estimated_pos = self.estimate_position_from_sequence_pattern(
                            frames_data, missing_frame, center_x, center_y, 
                            min_x, max_x, min_y, max_y, distance_threshold
                        )
                        
                        if estimated_pos is not None:
                            if missing_frame not in frames_data:
                                frames_data[missing_frame] = {
                                    'coords': [],
                                    'num_objects': 0,
                                    'image_name': f"{sequence_id}_{missing_frame}_test"
                                }
                            
                            frames_data[missing_frame]['coords'] = [estimated_pos]
                            frames_data[missing_frame]['num_objects'] = 1
                            print(f"  激进补全帧 {missing_frame}，位置: {estimated_pos}")
        
        return aggressive_data
    
    def estimate_position_from_sequence_pattern(self, frames_data: Dict[int, Dict], target_frame: int, 
                                              center_x: float, center_y: float,
                                              min_x: float, max_x: float, min_y: float, max_y: float,
                                              distance_threshold: float) -> Optional[List[float]]:
        """
        基于序列模式估计位置
        
        Args:
            frames_data: 帧数据
            target_frame: 目标帧
            center_x, center_y: 序列中心
            min_x, max_x, min_y, max_y: 序列边界
            distance_threshold: 距离阈值
            
        Returns:
            估计的位置
        """
        frames = sorted(frames_data.keys())
        
        # 策略1: 基于时间位置的线性插值
        if len(frames) >= 2:
            # 找到目标帧在序列中的相对位置
            frame_positions = []
            frame_coords = []
            
            for frame in frames:
                if frames_data[frame]['coords']:
                    frame_positions.append(frame)
                    frame_coords.append(frames_data[frame]['coords'][0])  # 使用第一个检测点
            
            if len(frame_positions) >= 2:
                # 使用线性回归估计位置
                try:
                    # 对x坐标进行线性回归
                    x_coords = [pos[0] for pos in frame_coords]
                    y_coords = [pos[1] for pos in frame_coords]
                    
                    # 简单的线性插值
                    if target_frame < min(frame_positions):
                        # 外推
                        alpha = (target_frame - min(frame_positions)) / (max(frame_positions) - min(frame_positions))
                        estimated_x = x_coords[0] + alpha * (x_coords[-1] - x_coords[0])
                        estimated_y = y_coords[0] + alpha * (y_coords[-1] - y_coords[0])
                    elif target_frame > max(frame_positions):
                        # 外推
                        alpha = (target_frame - min(frame_positions)) / (max(frame_positions) - min(frame_positions))
                        estimated_x = x_coords[0] + alpha * (x_coords[-1] - x_coords[0])
                        estimated_y = y_coords[0] + alpha * (y_coords[-1] - y_coords[0])
                    else:
                        # 内插
                        # 找到最近的两个帧
                        prev_frame = max([f for f in frame_positions if f < target_frame], default=None)
                        next_frame = min([f for f in frame_positions if f > target_frame], default=None)
                        
                        if prev_frame is not None and next_frame is not None:
                            prev_idx = frame_positions.index(prev_frame)
                            next_idx = frame_positions.index(next_frame)
                            
                            alpha = (target_frame - prev_frame) / (next_frame - prev_frame)
                            estimated_x = x_coords[prev_idx] + alpha * (x_coords[next_idx] - x_coords[prev_idx])
                            estimated_y = y_coords[prev_idx] + alpha * (y_coords[next_idx] - y_coords[prev_idx])
                        else:
                            # 使用序列中心
                            estimated_x = center_x
                            estimated_y = center_y
                    
                    # 确保估计位置在合理范围内
                    estimated_x = max(min_x - distance_threshold, min(max_x + distance_threshold, estimated_x))
                    estimated_y = max(min_y - distance_threshold, min(max_y + distance_threshold, estimated_y))
                    
                    return [estimated_x, estimated_y]
                    
                except Exception as e:
                    print(f"    线性插值失败: {e}")
        
        # 策略2: 使用序列中心
        return [center_x, center_y]
    
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
    """主函数，演示平衡的序列后处理"""
    # 加载预测结果和真实标注
    pred_path = 'results/WTNet/predictions.json'
    gt_path = 'datasets/spotgeov2-IRSTD/test_anno.json'
    
    print("正在加载预测结果和真实标注...")
    with open(pred_path, 'r') as f:
        original_predictions = json.load(f)
    
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    # 创建平衡的序列后处理器，使用更激进的参数
    processor = BalancedSequenceProcessor(
        base_distance_threshold=1000,  # 更宽松的距离阈值
        temporal_window=5,            # 更大的时间窗口
        confidence_threshold=0.0001,   # 较低的置信度阈值
        min_track_length=1,           # 更小的最小轨迹长度，允许更短的轨迹
        max_frame_gap=5,              # 更大的最大帧间隔，允许更大的间隔补全
        adaptive_threshold=True,       # 启用自适应阈值
        # 基于轨迹距离统计的新参数
        expected_sequence_length=5,
        trajectory_mean_distance=116.53,
        trajectory_std_distance=29.76,
        trajectory_max_distance=237.87,
        trajectory_min_distance=6.74,
        trajectory_median_distance=124.96
    )
    
    # 进行序列后处理
    processed_predictions = processor.process_sequence(original_predictions)
    
    # 评估改善效果
    improvement = processor.evaluate_improvement(original_predictions, processed_predictions, ground_truth)
    
    # 打印结果
    print("\n=== 平衡的序列后处理效果评估 ===")
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
    
    # 保存处理后的结果
    output_path = 'results/WTNet/aggressive_balanced_processed_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(processed_predictions, f, indent=2)
    print(f"\n激进平衡的处理后的预测结果已保存到: {output_path}")

if __name__ == '__main__':
    main() 