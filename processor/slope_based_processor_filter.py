import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, Counter
import copy
import math

class SequenceSlopeProcessor:
    """基于序列内主导斜率的轨迹补全处理器"""
    
    def __init__(self, 
                 base_distance_threshold: float = 80.0,
                 min_track_length: int = 2,
                 expected_sequence_length: int = 5,
                 slope_tolerance: float = 0.1,  # 斜率容差
                 min_slope_count: int = 2,      # 最小斜率出现次数
                 point_distance_threshold: float = 5.0):  # 重合点过滤阈值
        """
        初始化基于序列内主导斜率的轨迹处理器
        
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

    def collect_sequence_slopes_from_pair(self, frames_data: Dict[int, Dict]) -> Dict:
        """收集单个序列内所有帧间配对的斜率统计（遍历所有可能的点配对）"""
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
                            
                            # 只考虑距离合理的配对（可以设置一个较大的阈值）
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
                                   slope: float, frame1: int, frame2: int) -> List[Tuple[int, List[float]]]:
        """从两个点生成5个点（包括原始的两个点）"""
        points = []
        
        # 计算帧间隔
        frame_gap = frame2 - frame1
        
        # 生成5个均匀分布的点
        for i in range(5):
            # 计算插值比例 (0, 0.25, 0.5, 0.75, 1.0)
            ratio = i / 4.0
            
            # 计算对应的帧号
            frame = frame1 + ratio * frame_gap
            
            # 插值计算位置
            if math.isinf(slope):
                # 垂直线，只改变y坐标
                x = pos1[0]
                y = pos1[1] + ratio * (pos2[1] - pos1[1])
            else:
                # 使用斜率进行插值
                x = pos1[0] + ratio * (pos2[0] - pos1[0])
                y = pos1[1] + ratio * (pos2[1] - pos1[1])
            
            points.append((int(frame), [x, y]))
        
        return points
    
    def filter_duplicate_points_improved(self, all_points: List[Tuple[int, List[float]]]) -> List[Tuple[int, List[float]]]:
        """改进的重合点过滤算法"""
        if not all_points:
            return []
        
        # 按帧分组
        frame_groups = defaultdict(list)
        for frame, pos in all_points:
            frame_groups[frame].append(pos)
        
        filtered_points = []
        
        # 对每一帧内的点进行聚类过滤
        for frame, positions in frame_groups.items():
            if len(positions) == 1:
                # 只有一个点，直接保留
                filtered_points.append((frame, positions[0]))
                continue
            
            # 使用层次聚类过滤重合点
            filtered_positions = self.cluster_filter_points(positions)
            
            for pos in filtered_positions:
                filtered_points.append((frame, pos))
        
        # 跨帧过滤：检查相邻帧之间的重合点
        filtered_points = self.cross_frame_filter(filtered_points)
        
        return filtered_points
    
    def cluster_filter_points(self, positions: List[List[float]]) -> List[List[float]]:
        """使用聚类方法过滤重合点"""
        if len(positions) <= 1:
            return positions
        
        # 计算距离矩阵
        n = len(positions)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.calculate_distance(positions[i], positions[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        # 使用DBSCAN聚类
        from sklearn.cluster import DBSCAN
        
        # 将位置转换为numpy数组
        positions_array = np.array(positions)
        
        # 使用DBSCAN聚类，eps为距离阈值，min_samples=1表示每个点都可以是一个簇
        clustering = DBSCAN(eps=self.point_distance_threshold, min_samples=1).fit(positions_array)
        
        # 获取聚类标签
        labels = clustering.labels_
        
        # 对每个聚类，选择中心点或最代表性的点
        filtered_positions = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # 噪声点，单独处理
                cluster_points = positions_array[labels == label]
                for point in cluster_points:
                    filtered_positions.append(point.tolist())
            else:
                # 找到聚类中心
                cluster_points = positions_array[labels == label]
                if len(cluster_points) == 1:
                    filtered_positions.append(cluster_points[0].tolist())
                else:
                    # 计算聚类中心
                    center = np.mean(cluster_points, axis=0)
                    filtered_positions.append(center.tolist())
        
        return filtered_positions
    
    def cross_frame_filter(self, points: List[Tuple[int, List[float]]]) -> List[Tuple[int, List[float]]]:
        """跨帧过滤重合点"""
        if len(points) <= 1:
            return points
        
        # 按帧排序
        points.sort(key=lambda x: x[0])
        
        filtered_points = []
        used_positions = set()
        
        for frame, pos in points:
            # 检查是否与已使用的位置重合
            is_duplicate = False
            for used_pos in used_positions:
                if self.calculate_distance(pos, used_pos) <= self.point_distance_threshold * 0.5:  # 更严格的阈值
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_points.append((frame, pos))
                used_positions.add(tuple(pos))
        
        return filtered_points
    
    def adaptive_distance_threshold(self, positions: List[List[float]]) -> float:
        """自适应距离阈值"""
        if len(positions) <= 1:
            return self.point_distance_threshold
        
        # 计算所有点对之间的距离
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = self.calculate_distance(positions[i], positions[j])
                distances.append(dist)
        
        if not distances:
            return self.point_distance_threshold
        
        # 使用距离的25%分位数作为自适应阈值
        adaptive_threshold = np.percentile(distances, 25)
        
        # 限制在合理范围内
        min_threshold = 2.0
        max_threshold = self.point_distance_threshold * 2
        
        return max(min_threshold, min(adaptive_threshold, max_threshold))
    
    def filter_duplicate_points_with_priority(self, all_points: List[Tuple[int, List[float]]]) -> List[Tuple[int, List[float]]]:
        """基于优先级过滤重合点"""
        if not all_points:
            return []
        
        # 按帧分组
        frame_groups = defaultdict(list)
        for frame, pos in all_points:
            frame_groups[frame].append(pos)
        
        filtered_points = []
        
        # 对每一帧内的点进行优先级过滤
        for frame, positions in frame_groups.items():
            if len(positions) == 1:
                filtered_points.append((frame, positions[0]))
                continue
            
            # 计算自适应阈值
            adaptive_threshold = self.adaptive_distance_threshold(positions)
            
            # 使用优先级过滤
            filtered_positions = self.priority_filter_points(positions, adaptive_threshold)
            
            for pos in filtered_positions:
                filtered_points.append((frame, pos))
        
        return filtered_points
    
    def priority_filter_points(self, positions: List[List[float]], threshold: float) -> List[List[float]]:
        """基于优先级的点过滤"""
        if len(positions) <= 1:
            return positions
        
        # 计算每个点的"重要性"分数
        point_scores = []
        for i, pos in enumerate(positions):
            # 计算与其他点的平均距离（距离越远，越重要）
            total_distance = 0
            count = 0
            for j, other_pos in enumerate(positions):
                if i != j:
                    dist = self.calculate_distance(pos, other_pos)
                    total_distance += dist
                    count += 1
            
            if count > 0:
                avg_distance = total_distance / count
                # 距离越远，分数越高
                score = avg_distance
            else:
                score = 0
            
            point_scores.append((score, i, pos))
        
        # 按分数排序（高分优先）
        point_scores.sort(reverse=True)
        
        # 贪心选择点
        selected_points = []
        selected_indices = set()
        
        for score, idx, pos in point_scores:
            # 检查是否与已选择的点重合
            is_duplicate = False
            for selected_pos in selected_points:
                if self.calculate_distance(pos, selected_pos) <= threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                selected_points.append(pos)
                selected_indices.add(idx)
        
        return selected_points
    
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
    
    def extrapolate_trajectory(self, frames_data: Dict[int, Dict], 
                             dominant_slopes: List[Tuple[float, int]], 
                             sequence_id: int) -> List[Tuple[int, List[float]]]:
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
                extrapolated_pos = self.extrapolate_position(first_pos, dominant_slope, -frame_gap)  # 负值表示向后外推
                extrapolated_points.append((frame, extrapolated_pos))
                print(f"    向前外推帧 {frame}，位置: {extrapolated_pos}")
        
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
                    print(f"    填补间隙帧 {frame}，位置: {interpolated_pos}")
        
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
                print(f"    向后外推帧 {frame}，位置: {extrapolated_pos}")
        
        # 策略4: 基于轨迹模式的外推
        extrapolated_points.extend(self.pattern_based_extrapolation(frames_data, dominant_slopes))
        
        return extrapolated_points
    
        
    def remove_noise_detections(self, frames_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """移除噪声检测点，保留有意义的轨迹"""
        print(f"  清理噪声检测点...")
        
        # 策略1: 基于空间分布的噪声过滤
        frames_data = self.spatial_distribution_filter(frames_data)
        
        # 策略2: 基于时序一致性的噪声过滤
        frames_data = self.temporal_consistency_filter(frames_data)
        
        # 策略3: 基于轨迹连续性的噪声过滤
        frames_data = self.trajectory_continuity_filter(frames_data)
        
        # 策略4: 基于密度聚类的噪声过滤
        frames_data = self.density_clustering_filter(frames_data)
        
        return frames_data
    
    def spatial_distribution_filter(self, frames_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """基于空间分布的噪声过滤"""
        frames = sorted(frames_data.keys())
        
        # 收集所有检测点
        all_points = []
        for frame, frame_data in frames_data.items():
            if frame_data['coords']:
                for pos in frame_data['coords']:
                    all_points.append((frame, pos))
        
        if len(all_points) < 3:
            return frames_data
        
        # 计算所有点的空间统计
        x_coords = [pos[0] for _, pos in all_points]
        y_coords = [pos[1] for _, pos in all_points]
        
        x_mean, x_std = np.mean(x_coords), np.std(x_coords)
        y_mean, y_std = np.mean(y_coords), np.std(y_coords)
        
        # 计算马氏距离阈值（2σ原则）
        mahalanobis_threshold = 2.0
        
        # 过滤异常点
        filtered_frames = copy.deepcopy(frames_data)
        
        for frame, frame_data in filtered_frames.items():
            if not frame_data['coords']:
                continue
            
            kept_coords = []
            for pos in frame_data['coords']:
                # 计算马氏距离
                x_score = abs(pos[0] - x_mean) / (x_std + 1e-6)
                y_score = abs(pos[1] - y_mean) / (y_std + 1e-6)
                mahalanobis_dist = np.sqrt(x_score**2 + y_score**2)
                
                # 如果点在分布范围内，保留
                if mahalanobis_dist <= mahalanobis_threshold:
                    kept_coords.append(pos)
                else:
                    print(f"    移除空间异常点: 帧{frame}, 位置{pos}, 马氏距离{mahalanobis_dist:.2f}")
            
            frame_data['coords'] = kept_coords
            frame_data['num_objects'] = len(kept_coords)
        
        return filtered_frames
    
    def temporal_consistency_filter(self, frames_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """基于时序一致性的噪声过滤"""
        frames = sorted(frames_data.keys())
        
        filtered_frames = copy.deepcopy(frames_data)
        
        for frame in frames:
            if not frames_data[frame]['coords']:
                continue
            
            current_coords = frames_data[frame]['coords']
            kept_coords = []
            
            for pos in current_coords:
                # 检查前后帧的支持
                support_count = 0
                total_checks = 0
                
                # 检查前后2帧
                for offset in [-2, -1, 1, 2]:
                    check_frame = frame + offset
                    if check_frame in frames_data and frames_data[check_frame]['coords']:
                        total_checks += 1
                        check_coords = frames_data[check_frame]['coords']
                        
                        # 找到最近的检测点
                        min_dist = float('inf')
                        for check_pos in check_coords:
                            dist = self.calculate_distance(pos, check_pos)
                            min_dist = min(min_dist, dist)
                        
                        # 如果距离合理，增加支持
                        if min_dist <= self.base_distance_threshold * 1.5:
                            support_count += 1
                
                # 计算支持率
                support_ratio = support_count / max(total_checks, 1)
                
                # 如果支持率足够高，保留
                if support_ratio >= 0.3 or total_checks < 2:
                    kept_coords.append(pos)
                else:
                    print(f"    移除时序不一致点: 帧{frame}, 位置{pos}, 支持率{support_ratio:.2f}")
            
            filtered_frames[frame]['coords'] = kept_coords
            filtered_frames[frame]['num_objects'] = len(kept_coords)
        
        return filtered_frames
    
    def trajectory_continuity_filter(self, frames_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """基于轨迹连续性的噪声过滤"""
        frames = sorted(frames_data.keys())
        
        if len(frames) < 3:
            return frames_data
        
        filtered_frames = copy.deepcopy(frames_data)
        
        # 构建轨迹图
        trajectory_graph = defaultdict(list)
        
        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            
            coords1 = frames_data[frame1]['coords']
            coords2 = frames_data[frame2]['coords']
            
            if coords1 and coords2:
                # 找到最佳匹配
                cost_matrix = np.zeros((len(coords1), len(coords2)))
                for j, pos1 in enumerate(coords1):
                    for k, pos2 in enumerate(coords2):
                        cost_matrix[j, k] = self.calculate_distance(pos1, pos2)
                
                if cost_matrix.size > 0:
                    row_indices, col_indices = linear_sum_assignment(cost_matrix)
                    
                    for row_idx, col_idx in zip(row_indices, col_indices):
                        if cost_matrix[row_idx, col_idx] <= self.base_distance_threshold * 2:
                            trajectory_graph[(frame1, tuple(coords1[row_idx]))].append((frame2, tuple(coords2[col_idx])))
        
        # 找到最长的轨迹
        longest_trajectory = self.find_longest_trajectory(trajectory_graph)
        
        # 保留最长轨迹中的点
        trajectory_points = set()
        for frame, pos in longest_trajectory:
            trajectory_points.add((frame, pos))
        
        # 过滤点
        for frame in frames:
            if not frames_data[frame]['coords']:
                continue
            
            kept_coords = []
            for pos in frames_data[frame]['coords']:
                if (frame, tuple(pos)) in trajectory_points:
                    kept_coords.append(pos)
                else:
                    print(f"    移除轨迹外点: 帧{frame}, 位置{pos}")
            
            filtered_frames[frame]['coords'] = kept_coords
            filtered_frames[frame]['num_objects'] = len(kept_coords)
        
        return filtered_frames
    
    def find_longest_trajectory(self, trajectory_graph: Dict) -> List[Tuple[int, Tuple[float, float]]]:
        """找到最长的轨迹"""
        visited = set()
        longest_trajectory = []
        
        def dfs(node, current_trajectory):
            nonlocal longest_trajectory
            
            if len(current_trajectory) > len(longest_trajectory):
                longest_trajectory = current_trajectory.copy()
            
            for next_node in trajectory_graph.get(node, []):
                if next_node not in visited:
                    visited.add(next_node)
                    dfs(next_node, current_trajectory + [next_node])
                    visited.remove(next_node)
        
        # 从每个起始点开始DFS，先收集所有起始点
        start_nodes = list(trajectory_graph.keys())
        
        for start_node in start_nodes:
            if start_node not in visited:
                visited.add(start_node)
                dfs(start_node, [start_node])
                visited.remove(start_node)
        
        return longest_trajectory
    
    def density_clustering_filter(self, frames_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """基于密度聚类的噪声过滤"""
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            print("    DBSCAN不可用，跳过密度聚类过滤")
            return frames_data
        
        frames = sorted(frames_data.keys())
        
        # 收集所有检测点
        all_points = []
        point_to_frame = {}
        
        for frame, frame_data in frames_data.items():
            if frame_data['coords']:
                for i, pos in enumerate(frame_data['coords']):
                    all_points.append(pos)
                    point_to_frame[tuple(pos)] = frame
        
        if len(all_points) < 3:
            return frames_data
        
        # 使用DBSCAN聚类
        points_array = np.array(all_points)
        
        # 自适应eps参数
        eps = self.adaptive_eps_parameter(points_array)
        
        clustering = DBSCAN(eps=eps, min_samples=2).fit(points_array)
        labels = clustering.labels_
        
        # 找到最大的聚类
        unique_labels = set(labels)
        if -1 in unique_labels:  # 移除噪声标签
            unique_labels.remove(-1)
        
        if not unique_labels:
            return frames_data
        
        # 找到最大的聚类
        largest_cluster_label = max(unique_labels, key=lambda x: np.sum(labels == x))
        
        # 保留最大聚类中的点
        filtered_frames = copy.deepcopy(frames_data)
        
        for frame in frames:
            if not frames_data[frame]['coords']:
                continue
            
            kept_coords = []
            for pos in frames_data[frame]['coords']:
                # 找到点在all_points中的索引
                point_idx = None
                for i, p in enumerate(all_points):
                    if tuple(p) == tuple(pos):
                        point_idx = i
                        break
                
                if point_idx is not None and labels[point_idx] == largest_cluster_label:
                    kept_coords.append(pos)
                else:
                    print(f"    移除密度聚类外点: 帧{frame}, 位置{pos}")
            
            filtered_frames[frame]['coords'] = kept_coords
            filtered_frames[frame]['num_objects'] = len(kept_coords)
        
        return filtered_frames
    
    def adaptive_eps_parameter(self, points_array: np.ndarray) -> float:
        """自适应计算DBSCAN的eps参数"""
        if len(points_array) < 2:
            return self.base_distance_threshold
        
        # 计算所有点对之间的距离
        distances = []
        for i in range(len(points_array)):
            for j in range(i + 1, len(points_array)):
                dist = np.linalg.norm(points_array[i] - points_array[j])
                distances.append(dist)
        
        if not distances:
            return self.base_distance_threshold
        
        # 使用距离的75%分位数作为eps
        eps = np.percentile(distances, 75)
        
        # 限制在合理范围内
        min_eps = 5.0
        max_eps = self.base_distance_threshold * 2
        
        return max(min_eps, min(eps, max_eps))
    
    def extrapolate_position(self, base_pos: List[float], slope: float, frame_gap: int) -> List[float]:
        """基于斜率和帧间隔外推位置"""
        if math.isinf(slope):
            # 垂直线
            step_size = 10.0
            dy = step_size * frame_gap
            return [base_pos[0], base_pos[1] + dy]
        
        # 计算位移
        step_size = 10.0  # 每帧移动距离
        total_distance = step_size * abs(frame_gap)
        
        # 使用斜率计算位移
        dx = total_distance / np.sqrt(1 + slope**2)
        dy = slope * dx
        
        # 根据方向调整符号
        if frame_gap < 0:  # 向后外推
            dx = -dx
            dy = -dy
        
        return [base_pos[0] + dx, base_pos[1] + dy]
    
    def pattern_based_extrapolation(self, frames_data: Dict[int, Dict], 
                                  dominant_slopes: List[Tuple[float, int]]) -> List[Tuple[int, List[float]]]:
        """基于轨迹模式的外推"""
        frames = sorted(frames_data.keys())
        detected_frames = [f for f in frames if frames_data[f]['coords']]
        
        if len(detected_frames) < 3:
            return []
        
        extrapolated_points = []
        
        # 分析轨迹的运动模式
        motion_vectors = []
        for i in range(len(detected_frames) - 1):
            frame1 = detected_frames[i]
            frame2 = detected_frames[i + 1]
            pos1 = frames_data[frame1]['coords'][0]
            pos2 = frames_data[frame2]['coords'][0]
            
            vector = self.calculate_vector(pos1, pos2)
            motion_vectors.append((vector, frame2 - frame1))
        
        if not motion_vectors:
            return extrapolated_points
        
        # 计算平均运动向量
        avg_dx = np.mean([v[0] for v, _ in motion_vectors])
        avg_dy = np.mean([v[1] for v, _ in motion_vectors])
        avg_frame_gap = np.mean([gap for _, gap in motion_vectors])
        
        # 计算每帧的平均位移
        if avg_frame_gap > 0:
            per_frame_dx = avg_dx / avg_frame_gap
            per_frame_dy = avg_dy / avg_frame_gap
        else:
            per_frame_dx = per_frame_dy = 0
        
        # 找到轨迹的起点和终点
        start_frame = min(detected_frames)
        end_frame = max(detected_frames)
        start_pos = frames_data[start_frame]['coords'][0]
        end_pos = frames_data[end_frame]['coords'][0]
        
        # 向前外推（从起点开始）
        if start_frame > min(frames):
            for frame in range(min(frames), start_frame):
                frame_gap = start_frame - frame
                extrapolated_pos = [
                    start_pos[0] - per_frame_dx * frame_gap,
                    start_pos[1] - per_frame_dy * frame_gap
                ]
                extrapolated_points.append((frame, extrapolated_pos))
                print(f"    模式外推-向前帧 {frame}，位置: {extrapolated_pos}")
        
        # 向后外推（从终点开始）
        if end_frame < max(frames):
            for frame in range(end_frame + 1, max(frames) + 1):
                frame_gap = frame - end_frame
                extrapolated_pos = [
                    end_pos[0] + per_frame_dx * frame_gap,
                    end_pos[1] + per_frame_dy * frame_gap
                ]
                extrapolated_points.append((frame, extrapolated_pos))
                print(f"    模式外推-向后帧 {frame}，位置: {extrapolated_pos}")
        
        return extrapolated_points
    
    def calculate_vector(self, point1: List[float], point2: List[float]) -> Tuple[float, float]:
        """计算从point1到point2的向量"""
        return (point2[0] - point1[0], point2[1] - point1[1])
    
    def process_sequence_with_slopes(self, sequence_id: int, frames_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """处理单个序列，使用主导斜率补全轨迹"""
        print(f"处理序列 {sequence_id}，包含 {len(frames_data)} 帧")
        
        # 收集序列内的斜率统计
        slope_stats = self.collect_sequence_slopes_from_pair(frames_data)
        
        if not slope_stats['dominant_slopes']:
            print(f"  序列 {sequence_id} 没有找到主导斜率，保持原始数据")
            return frames_data
        
        print(f"  序列 {sequence_id} 找到 {len(slope_stats['dominant_slopes'])} 个主导斜率")
        for slope, count in slope_stats['dominant_slopes'][:3]:
            print(f"    主导斜率 {slope:.4f}: 出现 {count} 次")
        
        # 收集所有生成的点
        all_generated_points = []
        
        # 对每个符合主导斜率的点对生成5个点
        for pair_info in slope_stats['slope_pairs']:
            slope = pair_info['slope']
            
            # 检查是否属于主导斜率
            is_dominant = False
            for dominant_slope, _ in slope_stats['dominant_slopes']:
                if abs(slope - dominant_slope) <= self.slope_tolerance:
                    is_dominant = True
                    break
            
            if is_dominant:
                # 生成5个点
                generated_points = self.generate_5_points_from_pair(
                    pair_info['pos1'], pair_info['pos2'], 
                    pair_info['slope'], pair_info['frame1'], pair_info['frame2']
                )
                all_generated_points.extend(generated_points)
                print(f"    从帧 {pair_info['frame1']}-{pair_info['frame2']} 生成 {len(generated_points)} 个点")
        
        # 外推轨迹，填补间隙
        print(f"    开始外推轨迹...")
        extrapolated_points = self.extrapolate_trajectory(frames_data, slope_stats['dominant_slopes'], sequence_id)
        all_generated_points.extend(extrapolated_points)
        print(f"    外推生成 {len(extrapolated_points)} 个点")
        
        # 使用改进的过滤算法
        print(f"    总共生成 {len(all_generated_points)} 个点，开始过滤...")
        # filtered_points = self.filter_duplicate_points_with_priority(all_generated_points)
        filtered_points = all_generated_points
        print(f"    过滤后剩余 {len(filtered_points)} 个点")
        
        # 按帧组织点
        processed_frames = copy.deepcopy(frames_data)
        
        # 将生成的点添加到对应帧
        for frame, pos in filtered_points:
            if frame not in processed_frames:
                processed_frames[frame] = {
                    'coords': [],
                    'num_objects': 0,
                    'image_name': f"{sequence_id}_{frame}_test"
                }
            
            # 检查是否与已有点重合
            is_duplicate = False
            for existing_pos in processed_frames[frame]['coords']:
                if self.calculate_distance(pos, existing_pos) <= self.point_distance_threshold:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                processed_frames[frame]['coords'].append(pos)
                processed_frames[frame]['num_objects'] = len(processed_frames[frame]['coords'])
        
        # # # 移除噪声检测点
        # processed_frames = self.remove_noise_detections(processed_frames)
        
        return processed_frames
   
    def process_sequence(self, predictions: Dict) -> Dict:
        """处理整个预测结果，使用基于序列内主导斜率的补全策略"""
        print("开始基于序列内主导斜率的轨迹补全处理...")
        
        # 提取序列信息
        sequence_data = self.extract_sequence_info(predictions)
        print(f"提取了 {len(sequence_data)} 个序列的信息")
        
        # 处理每个序列
        processed_predictions = {}
        
        for sequence_id, frames_data in sequence_data.items():
            # 处理单个序列
            processed_frames = self.process_sequence_with_slopes(sequence_id, frames_data)

        # 转换回原始格式
            for frame, frame_data in processed_frames.items():
                image_name = frame_data['image_name']
                processed_predictions[image_name] = {
                    'coords': frame_data['coords'],
                    'num_objects': frame_data['num_objects']
                }
        
        print(f"基于序列内主导斜率的轨迹补全完成，处理了 {len(processed_predictions)} 张图像")
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
    """主函数，演示基于序列内主导斜率的轨迹补全"""
    # 加载预测结果和真实标注
    pred_path = 'results/spotgeov2/WTNet/predictions.json'
    gt_path = 'datasets/spotgeov2-IRSTD/test_anno.json'
    
    print("正在加载预测结果和真实标注...")
    with open(pred_path, 'r') as f:
        original_predictions = json.load(f)
    
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    # 创建基于序列内主导斜率的轨迹处理器
    processor = SequenceSlopeProcessor(
        base_distance_threshold=500,  # 宽松的距离阈值
        min_track_length=1,           # 允许短轨迹
        expected_sequence_length=5,   # 期望序列长度
        slope_tolerance=0.1,          # 斜率容差
        min_slope_count=2,            # 最小斜率出现次数
        point_distance_threshold=200.0  # 重合点过滤阈值
    )
    
    # 进行基于序列内主导斜率的轨迹补全
    processed_predictions = processor.process_sequence(original_predictions)
    
    # 评估改善效果
    improvement = processor.evaluate_improvement(original_predictions, processed_predictions, ground_truth)
    
    # 打印结果
    print("\n=== 基于序列内主导斜率的轨迹补全效果评估 ===")
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
    output_path = 'results/spotgeov2/WTNet/sequence_slope_processed_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(processed_predictions, f, indent=2)
    print(f"\n基于序列内主导斜率的轨迹补全结果已保存到: {output_path}")

if __name__ == '__main__':
    main()
