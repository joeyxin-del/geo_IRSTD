import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, Counter
import copy
import math
import os

class AngleDistanceProcessor:
    """基于角度和步长的轨迹补全处理器，综合考虑角度和距离间隔的相似性"""
    
    def __init__(self, 
                 base_distance_threshold: float = 80.0,
                 min_track_length: int = 2,
                 expected_sequence_length: int = 5,
                 angle_tolerance: float = 10.0,  # 角度容差（度）
                 min_angle_count: int = 2,       # 最小角度出现次数
                 step_tolerance: float = 0.2,    # 步长容差（比例）
                 min_step_count: int = 1,        # 最小步长出现次数
                 max_step_size: float = 40.0,    # 最大步长限制
                 point_distance_threshold: float = 5.0):  # 重合点过滤阈值
        """
        初始化角度距离处理器
        
        Args:
            base_distance_threshold: 基础距离阈值
            min_track_length: 最小轨迹长度
            expected_sequence_length: 期望的序列长度
            angle_tolerance: 角度容差（度），用于聚类相似角度
            min_angle_count: 最小角度出现次数，用于确定主导角度
            step_tolerance: 步长容差（比例），用于聚类相似步长
            min_step_count: 最小步长出现次数，用于确定主导步长
            max_step_size: 最大步长限制，超过此值的步长将被忽略
            point_distance_threshold: 重合点过滤阈值
        """
        self.base_distance_threshold = base_distance_threshold
        self.min_track_length = min_track_length
        self.expected_sequence_length = expected_sequence_length
        self.angle_tolerance = angle_tolerance
        self.min_angle_count = min_angle_count
        self.step_tolerance = step_tolerance
        self.min_step_count = min_step_count
        self.max_step_size = max_step_size
        self.point_distance_threshold = point_distance_threshold
    
    def calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """计算两点之间的欧氏距离"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_angle(self, point1: List[float], point2: List[float]) -> float:
        """计算两点之间的角度（度）"""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        
        # 计算角度（弧度）
        angle_rad = math.atan2(dy, dx)
        
        # 转换为度
        angle_deg = math.degrees(angle_rad)
        
        # 标准化到 [0, 360) 度
        if angle_deg < 0:
            angle_deg += 360
        
        return angle_deg
    
    def calculate_step_size(self, point1: List[float], point2: List[float], frame_gap: int) -> float:
        """计算两点之间的步长（每帧的平均距离）"""
        distance = self.calculate_distance(point1, point2)
        return distance / frame_gap if frame_gap > 0 else 0.0
    
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
    
    def collect_sequence_angles_and_steps(self, frames_data: Dict[int, Dict]) -> Dict:
        """收集单个序列内所有帧间配对的角度和步长统计，并标识参与主导模式的点"""
        frames = sorted(frames_data.keys())
        
        if len(frames) < 2:
            return {
                'angles': [], 'steps': [], 'angle_step_pairs': [], 
                'dominant_angles': [], 'dominant_steps': [], 
                'point_participation': {}
            }
        
        all_angles = []
        all_steps = []
        angle_step_pairs = []  # 存储角度和步长对应的点对信息
        
        # 创建点参与哈希表，用于标识每个点是否参与了主导模式的形成
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
                        'min_dominant_distance': float('inf'),
                        'dominant_connections': [],
                        'angle_score': 0,  # 角度匹配分数
                        'step_score': 0    # 步长匹配分数
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
                            if distance <= 145:  # 使用更大的阈值
                                angle = self.calculate_angle(pos1, pos2)
                                frame_gap = frame2 - frame1
                                step_size = self.calculate_step_size(pos1, pos2, frame_gap)
                                
                                if step_size > 100:
                                    step_size = None
                                all_angles.append(angle)
                                all_steps.append(step_size)
                                angle_step_pairs.append({
                                    'frame1': frame1,
                                    'frame2': frame2,
                                    'pos1': pos1,
                                    'pos2': pos2,
                                    'angle': angle,
                                    'step_size': step_size,
                                    'distance': distance,
                                    'frame_gap': frame_gap,
                                    'pos1_idx': pos1_idx,
                                    'pos2_idx': pos2_idx
                                })
        
        # 统计角度分布
        angle_counter = Counter()
        for angle in all_angles:
            # 将角度聚类到相近的区间
            angle_key = round(angle / self.angle_tolerance) * self.angle_tolerance
            angle_counter[angle_key] += 1
            print(f"    角度 {angle:.1f}° 聚类到 {angle_key:.1f}°")
        
        # 找到主导角度
        dominant_angles = []
        for angle, count in angle_counter.most_common():
            if count >= self.min_angle_count:
                dominant_angles.append((angle, count))
        
        # 基于主导角度计算主导步长
        dominant_angle_values = [angle for angle, _ in dominant_angles]
        dominant_step_counter = Counter()
        
        dominant_angle_values = dominant_angle_values[:1]

        print(f"    主导角度: {dominant_angle_values}, 出现次数: {dominant_angles}")
        print(f"    开始基于主导角度筛选步长...")
        
        dominant_angle_pairs = 0  # 统计贡献主导角度的点对数量
        valid_step_pairs = 0      # 统计有效步长的点对数量
        
        for pair_info in angle_step_pairs:
            angle = pair_info['angle']
            step_size = pair_info['step_size']
            
            # 检查是否属于主导角度
            is_dominant_angle = False
            for dominant_angle in dominant_angle_values:
                # 处理角度环绕问题（0度和360度是相同的）
                angle_diff = min(abs(angle - dominant_angle), 
                               abs(angle - dominant_angle + 360),
                               abs(angle - dominant_angle - 360))
                if angle_diff <= self.angle_tolerance:
                    is_dominant_angle = True
                    break
            
            if is_dominant_angle:
                dominant_angle_pairs += 1
                
                # 只有贡献了主导角度的点对才参与步长统计
                if step_size is not None and step_size > 0 and step_size <= 40:
                    valid_step_pairs += 1
                    # 将步长聚类到相近的区间（按比例）
                    step_key = round(step_size / (step_size * self.step_tolerance)) * (step_size * self.step_tolerance)
                    # 修复聚类算法：使用固定的步长区间
                    step_key = round(step_size / 2.0) * 2.0  # 每2个单位一个区间
                    dominant_step_counter[step_key] += 1
                    print(f"      角度 {angle:.1f}° 步长 {step_size:.2f} -> 聚类到 {step_key:.2f}")
        
        print(f"    贡献主导角度的点对: {dominant_angle_pairs}")
        print(f"    有效步长的点对: {valid_step_pairs}")
        
        # 找到主导步长
        dominant_steps = []
        print(f"    步长分布: {dict(dominant_step_counter)}")
        for step, count in dominant_step_counter.most_common():
            if count >= self.min_step_count:
                dominant_steps.append((step, count))
                print(f"      主导步长 {step:.2f}: 出现 {count} 次")
        
        print(f"    最终主导步长: {dominant_steps}")
        
        # 标记参与主导模式的点
        dominant_angle_values = [angle for angle, _ in dominant_angles]
        dominant_step_values = [step for step, _ in dominant_steps]
        
        for pair_info in angle_step_pairs:
            angle = pair_info['angle']
            step_size = pair_info['step_size']
            
            # 检查角度是否属于主导角度
            angle_score = 0
            for dominant_angle in dominant_angle_values:
                # 处理角度环绕问题（0度和360度是相同的）
                angle_diff = min(abs(angle - dominant_angle), 
                               abs(angle - dominant_angle + 360),
                               abs(angle - dominant_angle - 360))
                if angle_diff <= self.angle_tolerance:
                    angle_score = 1 - (angle_diff / self.angle_tolerance)  # 归一化分数
                    break
            
            # 检查步长是否属于主导步长
            step_score = 0
            if step_size is not None and step_size > 0 and step_size <= 40:  # 步长必须在有效范围内
                for dominant_step in dominant_step_values:
                    if dominant_step > 0 and dominant_step <= 40:  # 主导步长也不能超过40
                        step_ratio = step_size / dominant_step
                        if 0.9 <= step_ratio <= 1.1:  # 在0.9-1.1倍范围内
                            step_score = 1 - abs(step_ratio - 1) / 0.1  # 归一化分数
                            break
            
            # 综合评分：角度和步长的加权平均
            combined_score = (angle_score + step_score) / 2
            
            # 如果综合分数足够高，标记为参与
            if combined_score >= 0.3:  # 阈值可调整
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
                    point_participation[point_key1]['angle_score'] = max(
                        point_participation[point_key1]['angle_score'], angle_score
                    )
                    point_participation[point_key1]['step_score'] = max(
                        point_participation[point_key1]['step_score'], step_score
                    )
                    point_participation[point_key1]['dominant_connections'].append({
                        'connected_point': point_key2,
                        'distance': distance,
                        'angle': angle,
                        'step_size': step_size,
                        'combined_score': combined_score
                    })
                
                # 更新点2的信息
                if point_key2 in point_participation:
                    point_participation[point_key2]['participated'] = True
                    point_participation[point_key2]['min_dominant_distance'] = min(
                        point_participation[point_key2]['min_dominant_distance'], distance
                    )
                    point_participation[point_key2]['angle_score'] = max(
                        point_participation[point_key2]['angle_score'], angle_score
                    )
                    point_participation[point_key2]['step_score'] = max(
                        point_participation[point_key2]['step_score'], step_score
                    )
                    point_participation[point_key2]['dominant_connections'].append({
                        'connected_point': point_key1,
                        'distance': distance,
                        'angle': angle,
                        'step_size': step_size,
                        'combined_score': combined_score
                    })
        
        # 根据距离阈值重新评估参与状态
        distance_threshold = 120  # 距离阈值
        for point_key, point_info in point_participation.items():
            if point_info['participated']:
                # 检查该点与所有主导模式连接点的距离
                distances = [connection['distance'] for connection in point_info['dominant_connections']]
                min_distance = min(distances) if distances else float('inf')
                
                print(f"    点 {point_key} {point_info['point']} 参与主导模式，连接距离: {distances}, 最短距离: {min_distance}")
                print(f"    角度分数: {point_info['angle_score']:.3f}, 步长分数: {point_info['step_score']:.3f}")
                
                # 如果最短距离大于阈值，则不算参与
                if min_distance > distance_threshold:
                    point_info['participated'] = False
                    print(f"    点 {point_key} 最短距离 {min_distance} 大于阈值 {distance_threshold}，标记为非参与")
                else:
                    print(f"    点 {point_key} 最短距离 {min_distance} 小于等于阈值 {distance_threshold}，保持参与状态")
        
        return {
            'angles': all_angles,
            'steps': all_steps,
            'angle_step_pairs': angle_step_pairs,
            'dominant_angles': dominant_angles,
            'dominant_steps': dominant_steps,
            'total_angles': len(all_angles),
            'total_steps': len(all_steps),
            'unique_angles': len(angle_counter),
            'unique_steps': len(dominant_step_counter),
            'total_pairs_considered': len(angle_step_pairs),
            'frames_processed': len(frames),
            'point_participation': point_participation
        }
    
    def generate_points_from_pair_with_steps(self, pos1: List[float], pos2: List[float], 
                                           angle: float, step_size: float, frame1: int, frame2: int, 
                                           total_frames: int = 5) -> List[Tuple[int, List[float]]]:
        """从两个点生成缺失帧的点，考虑步长信息"""
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
                
                # 使用线性插值
                x = pos1[0] + ratio * (pos2[0] - pos1[0])
                y = pos1[1] + ratio * (pos2[1] - pos1[1])
                
                points.append((frame, [x, y]))
                
            elif frame < frame1:
                # 在第一个检测点之前，使用外推（从frame1向前）
                frame_gap_to_frame = frame - frame1
                extrapolated_pos = self.extrapolate_position_by_angle_and_step(
                    pos1, angle, step_size, -frame_gap_to_frame, 
                    reference_pos=pos2, reference_frame_gap=frame2-frame1
                )
                points.append((frame, extrapolated_pos))
                
            elif frame > frame2:
                # 在第二个检测点之后，使用外推（从frame2向后）
                frame_gap_to_frame = frame - frame2
                extrapolated_pos = self.extrapolate_position_by_angle_and_step(
                    pos2, angle, step_size, frame_gap_to_frame, 
                    reference_pos=pos1, reference_frame_gap=frame2-frame1
                )
                points.append((frame, extrapolated_pos))
        
        return points
    
    def extrapolate_position_by_angle_and_step(self, base_pos: List[float], angle_deg: float, 
                                             step_size: float, frame_gap: int, 
                                             reference_pos: List[float] = None, 
                                             reference_frame_gap: int = None) -> List[float]:
        """基于角度和步长外推位置"""
        if reference_pos and reference_frame_gap:
            # 使用参考点计算每帧的位移
            dx_ref = base_pos[0] - reference_pos[0]
            dy_ref = base_pos[1] - reference_pos[1]
            ref_distance = abs(reference_frame_gap)
            
            # 计算每帧的位移
            step_dx = dx_ref / ref_distance
            step_dy = dy_ref / ref_distance
        else:
            # 使用步长和角度计算每帧的位移
            angle_rad = math.radians(angle_deg)
            step_dx = step_size * math.cos(angle_rad)
            step_dy = step_size * math.sin(angle_rad)
        
        # 计算总位移
        total_dx = step_dx * frame_gap
        total_dy = step_dy * frame_gap
        
        return [base_pos[0] + total_dx, base_pos[1] + total_dy]
    
    def extrapolate_trajectory_with_steps(self, frames_data: Dict[int, Dict], 
                                        dominant_angles: List[Tuple[float, int]],
                                        dominant_steps: List[Tuple[float, int]]) -> List[Tuple[int, List[float]]]:
        """外推轨迹，填补间隙，考虑步长信息"""
        frames = sorted(frames_data.keys())
        if len(frames) < 2:
            return []
        
        extrapolated_points = []
        
        # 找到所有有检测点的帧
        detected_frames = [f for f in frames if frames_data[f]['coords']]
        
        if len(detected_frames) < 2:
            return extrapolated_points
        
        # 计算主导角度和步长
        dominant_angle = dominant_angles[0][0] if dominant_angles else 0.0
        dominant_step = dominant_steps[0][0] if dominant_steps else 20.0
        
        # 策略1: 填补序列开头的间隙
        first_detected = detected_frames[0]
        if first_detected > min(frames):
            # 从第一个检测点向前外推
            first_pos = frames_data[first_detected]['coords'][0]
            missing_frames = range(min(frames), first_detected)
            
            for frame in missing_frames:
                frame_gap = first_detected - frame
                extrapolated_pos = self.extrapolate_position_by_angle_and_step(
                    first_pos, dominant_angle, dominant_step, -frame_gap
                )
                extrapolated_points.append((frame, extrapolated_pos))
        
        # 策略2: 填补序列中间的间隙
        for i in range(len(detected_frames) - 1):
            frame1 = detected_frames[i]
            frame2 = detected_frames[i + 1]
            
            if frame2 - frame1 > 1:  # 存在间隙
                pos1 = frames_data[frame1]['coords'][0]
                pos2 = frames_data[frame2]['coords'][0]
                
                # 计算原始角度和步长
                original_angle = self.calculate_angle(pos1, pos2)
                original_step = self.calculate_step_size(pos1, pos2, frame2 - frame1)
                
                # 使用主导角度和步长或原始值
                use_angle = self.find_best_angle_match(original_angle, dominant_angles)
                use_step = self.find_best_step_match(original_step, dominant_steps)
                
                # 填补间隙
                for frame in range(frame1 + 1, frame2):
                    ratio = (frame - frame1) / (frame2 - frame1)
                    interpolated_pos = self.interpolate_with_angle_and_step(
                        pos1, pos2, ratio, use_angle, use_step
                    )
                    extrapolated_points.append((frame, interpolated_pos))
        
        # 策略3: 填补序列结尾的间隙
        last_detected = detected_frames[-1]
        if last_detected < max(frames):
            # 从最后一个检测点向后外推
            last_pos = frames_data[last_detected]['coords'][0]
            missing_frames = range(last_detected + 1, max(frames) + 1)
            
            for frame in missing_frames:
                frame_gap = frame - last_detected
                extrapolated_pos = self.extrapolate_position_by_angle_and_step(
                    last_pos, dominant_angle, dominant_step, frame_gap
                )
                extrapolated_points.append((frame, extrapolated_pos))
        
        return extrapolated_points
    
    def find_best_angle_match(self, target_angle: float, dominant_angles: List[Tuple[float, int]]) -> float:
        """找到与目标角度最匹配的主导角度"""
        if not dominant_angles:
            return target_angle
        
        best_match = dominant_angles[0][0]  # 默认使用第一个主导角度
        min_diff = min(abs(target_angle - best_match), 
                      abs(target_angle - best_match + 360),
                      abs(target_angle - best_match - 360))
        
        for angle, _ in dominant_angles:
            diff = min(abs(target_angle - angle), 
                      abs(target_angle - angle + 360),
                      abs(target_angle - angle - 360))
            if diff < min_diff:
                min_diff = diff
                best_match = angle
        
        return best_match
    
    def find_best_step_match(self, target_step: float, dominant_steps: List[Tuple[float, int]]) -> float:
        """找到与目标步长最匹配的主导步长"""
        if not dominant_steps:
            return target_step
        
        best_match = dominant_steps[0][0]  # 默认使用第一个主导步长
        min_ratio_diff = abs(target_step / best_match - 1) if best_match > 0 else float('inf')
        
        for step, _ in dominant_steps:
            if step > 0:
                ratio_diff = abs(target_step / step - 1)
                if ratio_diff < min_ratio_diff:
                    min_ratio_diff = ratio_diff
                    best_match = step
        
        return best_match
    
    def interpolate_with_angle_and_step(self, pos1: List[float], pos2: List[float], 
                                      ratio: float, target_angle: float, target_step: float) -> List[float]:
        """使用目标角度和步长进行插值"""
        # 计算原始插值位置
        original_x = pos1[0] + ratio * (pos2[0] - pos1[0])
        original_y = pos1[1] + ratio * (pos2[1] - pos1[1])
        
        # 使用目标角度和步长调整位置
        # 计算从pos1到目标位置的距离
        total_distance = self.calculate_distance(pos1, pos2)
        target_distance = total_distance * ratio
        
        # 使用目标角度和步长计算新位置
        angle_rad = math.radians(target_angle)
        dx = target_distance * math.cos(angle_rad)
        dy = target_distance * math.sin(angle_rad)
        
        estimated_x = pos1[0] + dx
        estimated_y = pos1[1] + dy
        
        return [estimated_x, estimated_y]
    
    def filter_outliers_by_dominant_patterns(self, frames_data: Dict[int, Dict], 
                                          point_participation: Dict) -> Dict[int, Dict]:
        """基于主导模式参与情况过滤异常点"""
        if not point_participation:
            return frames_data
        
        filtered_frames = copy.deepcopy(frames_data)
        
        for frame_id, frame_data in filtered_frames.items():
            if not frame_data['coords']:
                continue
            
            kept_coords = []
            for point_idx, point in enumerate(frame_data['coords']):
                point_key = (frame_id, point_idx)
                
                # 检查该点是否参与了主导模式的形成
                if point_key in point_participation:
                    if point_participation[point_key]['participated']:
                        # 该点参与了主导模式的形成，保留
                        kept_coords.append(point)
                    else:
                        # 该点没有参与主导模式的形成，可能是异常点
                        print(f"    移除未参与主导模式的点: 帧{frame_id}, 位置{point}")
                else:
                    # 该点不在参与哈希表中，可能是新增的点，保留
                    kept_coords.append(point)
            
            frame_data['coords'] = kept_coords
            frame_data['num_objects'] = len(kept_coords)
        
        return filtered_frames
    
    def process_sequence(self, sequence_id: int, frames_data: Dict[int, Dict]) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
        """处理单个序列，返回轨迹补全和异常点筛选的结果"""
        # 收集序列内的角度和步长统计
        pattern_stats = self.collect_sequence_angles_and_steps(frames_data)
        
        if not pattern_stats['dominant_angles'] and not pattern_stats['dominant_steps']:
            return frames_data, frames_data
        
        # 步骤1: 轨迹补全
        completed_sequence, generated_points_from_pairs = self.complete_trajectory_with_patterns(
            frames_data, pattern_stats['dominant_angles'], pattern_stats['dominant_steps']
        )
        
        # 步骤2: 异常点筛选 - 使用基于主导模式参与情况的过滤方法
        filtered_sequence = self.filter_outliers_by_dominant_patterns(
            completed_sequence, pattern_stats['point_participation']
        )

        return completed_sequence, filtered_sequence
    
    def complete_trajectory_with_patterns(self, frames_data: Dict[int, Dict], 
                                       dominant_angles: List[Tuple[float, int]],
                                       dominant_steps: List[Tuple[float, int]]) -> Tuple[Dict[int, Dict], List[Tuple[int, List[float]]]]:
        """轨迹补全，考虑角度和步长模式"""
        # 收集所有生成的点
        all_generated_points = []
        generated_points_from_pairs = []  # 专门记录从点对生成的点
        
        # 对每个符合主导模式的点对生成点
        pattern_stats = self.collect_sequence_angles_and_steps(frames_data)
        for pair_info in pattern_stats['angle_step_pairs']:
            angle = pair_info['angle']
            step_size = pair_info['step_size']
            
            # 检查是否属于主导模式（角度和步长都符合）
            angle_match = False
            step_match = False
            
            # 检查角度匹配
            for dominant_angle, _ in dominant_angles:
                angle_diff = min(abs(angle - dominant_angle), 
                               abs(angle - dominant_angle + 360),
                               abs(angle - dominant_angle - 360))
                if angle_diff <= self.angle_tolerance:
                    angle_match = True
                    break
            
            # 检查步长匹配
            if step_size is not None and step_size > 0 and step_size <= 40:  # 步长必须在有效范围内
                for dominant_step, _ in dominant_steps:
                    if dominant_step > 0 and dominant_step <= 40:  # 主导步长也不能超过40
                        step_ratio = step_size / dominant_step
                        if 0.9 <= step_ratio <= 1.1:
                            step_match = True
                            break
            
            # 如果角度和步长都匹配，则生成点
            if angle_match and step_match:
                # 生成缺失帧的点
                generated_points = self.generate_points_from_pair_with_steps(
                    pair_info['pos1'], pair_info['pos2'], 
                    pair_info['angle'], pair_info['step_size'], 
                    pair_info['frame1'], pair_info['frame2'],
                    total_frames=len(frames_data)  # 传入总帧数
                )
                all_generated_points.extend(generated_points)
                generated_points_from_pairs.extend(generated_points)  # 记录从点对生成的点
        
        # 外推轨迹，填补间隙
        extrapolated_points = self.extrapolate_trajectory_with_steps(
            frames_data, dominant_angles, dominant_steps
        )
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
        """处理整个数据集，使用角度和步长补全策略"""
        print("开始基于角度和步长的轨迹补全处理...")
        
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
        
        print(f"基于角度和步长的轨迹补全完成，处理了 {len(processed_predictions)} 张图像")
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
    """主函数，演示角度距离轨迹补全"""
    import argparse
    import os
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='角度距离轨迹补全处理器')
    parser.add_argument('--pred_path', type=str, default='results/spotgeov2-IRSTD/WTNet/predictions_8807.json',
                       help='预测结果文件路径')
    parser.add_argument('--gt_path', type=str, default='datasets/spotgeov2-IRSTD/test_anno.json',
                       help='真实标注文件路径')
    parser.add_argument('--output_path', type=str, default='results/spotgeov2/WTNet/angle_distance_processed_predictions.json',
                       help='输出文件路径')
    parser.add_argument('--base_distance_threshold', type=float, default=1000,
                       help='基础距离阈值')
    parser.add_argument('--angle_tolerance', type=float, default=15,
                       help='角度容差（度）')
    parser.add_argument('--min_angle_count', type=int, default=2,
                       help='最小角度出现次数')
    parser.add_argument('--step_tolerance', type=float, default=0.2,
                       help='步长容差（比例）')
    parser.add_argument('--min_step_count', type=int, default=1,
                       help='最小步长出现次数')
    parser.add_argument('--point_distance_threshold', type=float, default=94,
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
    
    # 创建角度距离处理器
    processor = AngleDistanceProcessor(
        base_distance_threshold=args.base_distance_threshold,
        min_track_length=1,           # 允许短轨迹
        expected_sequence_length=5,   # 期望序列长度
        angle_tolerance=args.angle_tolerance,
        min_angle_count=args.min_angle_count,
        step_tolerance=args.step_tolerance,
        min_step_count=args.min_step_count,
        point_distance_threshold=args.point_distance_threshold
    )
    
    # 进行角度距离轨迹补全
    processed_predictions = processor.process_dataset(original_predictions)
    
    # 评估改善效果
    try:
        improvement = processor.evaluate_improvement(original_predictions, processed_predictions, ground_truth)
        
        # 打印结果
        print("\n=== 角度距离轨迹补全效果评估 ===")
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
        print(f"\n角度距离轨迹补全结果已保存到: {args.output_path}")
    except Exception as e:
        print(f"保存结果时出错: {e}")

if __name__ == '__main__':
    main()