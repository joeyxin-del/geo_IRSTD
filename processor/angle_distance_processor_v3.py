import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import copy
import math
import os


class GeometryUtils:
    """几何计算工具类"""
    
    @staticmethod
    def calculate_distance(point1: List[float], point2: List[float]) -> float:
        """计算两点之间的欧氏距离"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    @staticmethod
    def calculate_angle(point1: List[float], point2: List[float]) -> float:
        """计算两点之间的角度（度）"""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        return angle_deg + 360 if angle_deg < 0 else angle_deg
    
    @staticmethod
    def calculate_step_size(point1: List[float], point2: List[float], frame_gap: int) -> float:
        """计算两点之间的步长（每帧的平均距离）"""
        distance = GeometryUtils.calculate_distance(point1, point2)
        return distance / frame_gap if frame_gap > 0 else 0.0


class SequenceExtractor:
    """序列信息提取器"""
    
    @staticmethod
    def extract_sequence_info(predictions: Dict) -> Dict[int, Dict[int, Dict]]:
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


class PatternAnalyzer:
    """模式分析器 - 分析角度和步长模式"""
    
    def __init__(self, angle_tolerance: float = 10.0, min_angle_count: int = 2, 
                 min_step_count: int = 1):
        self.angle_tolerance = angle_tolerance
        self.min_angle_count = min_angle_count
        self.min_step_count = min_step_count
    
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
        angle_step_pairs = []
        
        # 创建点参与哈希表
        point_participation = self._initialize_point_participation(frames_data)
        
        # 计算所有可能的帧间配对
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                frame1 = frames[i]
                frame2 = frames[j]
                
                coords1 = frames_data[frame1]['coords']
                coords2 = frames_data[frame2]['coords']
                
                if coords1 and coords2:
                    for pos1_idx, pos1 in enumerate(coords1):
                        for pos2_idx, pos2 in enumerate(coords2):
                            distance = GeometryUtils.calculate_distance(pos1, pos2)
                            
                            # 使用原始版本的距离阈值145
                            if distance <= 145:
                                angle = GeometryUtils.calculate_angle(pos1, pos2)
                                frame_gap = frame2 - frame1
                                step_size = GeometryUtils.calculate_step_size(pos1, pos2, frame_gap)
                                
                                # 使用原始版本的步长过滤条件
                                if step_size > 100:
                                    step_size = None
                                
                                all_angles.append(angle)
                                all_steps.append(step_size)
                                angle_step_pairs.append({
                                    'frame1': frame1, 'frame2': frame2,
                                    'pos1': pos1, 'pos2': pos2,
                                    'angle': angle, 'step_size': step_size,
                                    'distance': distance, 'frame_gap': frame_gap,
                                    'pos1_idx': pos1_idx, 'pos2_idx': pos2_idx
                                })
        
        # 分析主导模式
        dominant_angles = self._find_dominant_angles(all_angles)
        dominant_steps = self._find_dominant_steps(angle_step_pairs, dominant_angles)
        
        # 标记参与主导模式的点
        self._mark_point_participation(angle_step_pairs, point_participation, 
                                     dominant_angles, dominant_steps)
        
        # 根据距离阈值重新评估参与状态
        self._reevaluate_participation_by_distance(point_participation)
        
        return {
            'angles': all_angles,
            'steps': all_steps,
            'angle_step_pairs': angle_step_pairs,
            'dominant_angles': dominant_angles,
            'dominant_steps': dominant_steps,
            'point_participation': point_participation
        }
    
    def _initialize_point_participation(self, frames_data: Dict[int, Dict]) -> Dict:
        """初始化点参与哈希表"""
        point_participation = {}
        
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
                        'angle_score': 0,
                        'step_score': 0
                    }
        
        return point_participation
    
    def _find_dominant_angles(self, all_angles: List[float]) -> List[Tuple[float, int]]:
        """找到主导角度"""
        angle_counter = Counter()
        for angle in all_angles:
            angle_key = round(angle / self.angle_tolerance) * self.angle_tolerance
            angle_counter[angle_key] += 1
        
        dominant_angles = []
        for angle, count in angle_counter.most_common():
            if count >= self.min_angle_count:
                dominant_angles.append((angle, count))
        
        return dominant_angles
    
    def _find_dominant_steps(self, angle_step_pairs: List[Dict], 
                           dominant_angles: List[Tuple[float, int]]) -> List[Tuple[float, int]]:
        """基于主导角度计算主导步长"""
        dominant_angle_values = [angle for angle, _ in dominant_angles]
        dominant_step_counter = Counter()
        
        # 只取第一个主导角度
        dominant_angle_values = dominant_angle_values[:1]
        
        for pair_info in angle_step_pairs:
            angle = pair_info['angle']
            step_size = pair_info['step_size']
            
            # 检查是否属于主导角度
            is_dominant_angle = False
            for dominant_angle in dominant_angle_values:
                angle_diff = min(abs(angle - dominant_angle), 
                               abs(angle - dominant_angle + 360),
                               abs(angle - dominant_angle - 360))
                if angle_diff <= self.angle_tolerance:
                    is_dominant_angle = True
                    break
            
            if is_dominant_angle:
                if step_size is not None and step_size > 0 and step_size <= 40:
                    step_key = round(step_size / 2.0) * 2.0
                    dominant_step_counter[step_key] += 1
        
        # 找到主导步长
        dominant_steps = []
        for step, count in dominant_step_counter.most_common():
            if count >= self.min_step_count:
                dominant_steps.append((step, count))
        
        return dominant_steps
    
    def _mark_point_participation(self, angle_step_pairs: List[Dict], 
                                point_participation: Dict,
                                dominant_angles: List[Tuple[float, int]],
                                dominant_steps: List[Tuple[float, int]]):
        """标记参与主导模式的点"""
        dominant_angle_values = [angle for angle, _ in dominant_angles]
        dominant_step_values = [step for step, _ in dominant_steps]
        
        for pair_info in angle_step_pairs:
            angle = pair_info['angle']
            step_size = pair_info['step_size']
            
            # 计算角度分数
            angle_score = 0
            for dominant_angle in dominant_angle_values:
                angle_diff = min(abs(angle - dominant_angle), 
                               abs(angle - dominant_angle + 360),
                               abs(angle - dominant_angle - 360))
                if angle_diff <= self.angle_tolerance:
                    angle_score = 1 - (angle_diff / self.angle_tolerance)
                    break
            
            # 计算步长分数
            step_score = 0
            if step_size is not None and step_size > 0 and step_size <= 40:
                for dominant_step in dominant_step_values:
                    if dominant_step > 0 and dominant_step <= 40:
                        step_ratio = step_size / dominant_step
                        if 0.9 <= step_ratio <= 1.1:
                            step_score = 1 - abs(step_ratio - 1) / 0.1
                            break
            
            # 综合评分
            combined_score = (angle_score + step_score) / 2
            
            if combined_score >= 0.3:
                frame1, pos1_idx = pair_info['frame1'], pair_info['pos1_idx']
                frame2, pos2_idx = pair_info['frame2'], pair_info['pos2_idx']
                distance = pair_info['distance']
                
                # 更新点1的信息
                point_key1 = (frame1, pos1_idx)
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
                
                # 更新点2的信息
                point_key2 = (frame2, pos2_idx)
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
    
    def _reevaluate_participation_by_distance(self, point_participation: Dict):
        """根据距离阈值重新评估参与状态"""
        distance_threshold = 120
        for point_key, point_info in point_participation.items():
            if point_info['participated'] and point_info['min_dominant_distance'] > distance_threshold:
                point_info['participated'] = False


class TrajectoryGenerator:
    """轨迹生成器"""
    
    @staticmethod
    def generate_points_from_pair_with_steps(pos1: List[float], pos2: List[float], 
                                           angle: float, step_size: float, frame1: int, frame2: int, 
                                           total_frames: int) -> List[Tuple[int, List[float]]]:
        """从两个点生成缺失帧的点，考虑步长信息"""
        points = []
        frame_gap = frame2 - frame1
        
        for frame in range(1, total_frames + 1):
            if frame == frame1 or frame == frame2:
                continue
            
            if frame1 < frame < frame2:
                # 插值
                ratio = (frame - frame1) / frame_gap
                x = pos1[0] + ratio * (pos2[0] - pos1[0])
                y = pos1[1] + ratio * (pos2[1] - pos1[1])
                points.append((frame, [x, y]))
                
            elif frame < frame1:
                # 向前外推
                frame_gap_to_frame = frame - frame1
                extrapolated_pos = TrajectoryGenerator.extrapolate_position_by_angle_and_step(
                    pos1, angle, step_size, -frame_gap_to_frame, 
                    reference_pos=pos2, reference_frame_gap=frame2-frame1
                )
                points.append((frame, extrapolated_pos))
                
            elif frame > frame2:
                # 向后外推
                frame_gap_to_frame = frame - frame2
                extrapolated_pos = TrajectoryGenerator.extrapolate_position_by_angle_and_step(
                    pos2, angle, step_size, frame_gap_to_frame, 
                    reference_pos=pos1, reference_frame_gap=frame2-frame1
                )
                points.append((frame, extrapolated_pos))
        
        return points
    
    @staticmethod
    def extrapolate_position_by_angle_and_step(base_pos: List[float], angle_deg: float, 
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
    
    @staticmethod
    def extrapolate_trajectory_with_steps(frames_data: Dict[int, Dict], 
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
            first_pos = frames_data[first_detected]['coords'][0]
            missing_frames = range(min(frames), first_detected)
            
            for frame in missing_frames:
                frame_gap = first_detected - frame
                extrapolated_pos = TrajectoryGenerator.extrapolate_position_by_angle_and_step(
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
                original_angle = GeometryUtils.calculate_angle(pos1, pos2)
                original_step = GeometryUtils.calculate_step_size(pos1, pos2, frame2 - frame1)
                
                # 使用主导角度和步长或原始值
                use_angle = TrajectoryGenerator.find_best_angle_match(original_angle, dominant_angles)
                use_step = TrajectoryGenerator.find_best_step_match(original_step, dominant_steps)
                
                # 填补间隙
                for frame in range(frame1 + 1, frame2):
                    ratio = (frame - frame1) / (frame2 - frame1)
                    interpolated_pos = TrajectoryGenerator.interpolate_with_angle_and_step(
                        pos1, pos2, ratio, use_angle, use_step
                    )
                    extrapolated_points.append((frame, interpolated_pos))
        
        # 策略3: 填补序列结尾的间隙
        last_detected = detected_frames[-1]
        if last_detected < max(frames):
            last_pos = frames_data[last_detected]['coords'][0]
            missing_frames = range(last_detected + 1, max(frames) + 1)
            
            for frame in missing_frames:
                frame_gap = frame - last_detected
                extrapolated_pos = TrajectoryGenerator.extrapolate_position_by_angle_and_step(
                    last_pos, dominant_angle, dominant_step, frame_gap
                )
                extrapolated_points.append((frame, extrapolated_pos))
        
        return extrapolated_points
    
    @staticmethod
    def find_best_angle_match(target_angle: float, dominant_angles: List[Tuple[float, int]]) -> float:
        """找到与目标角度最匹配的主导角度"""
        if not dominant_angles:
            return target_angle
        
        best_match = dominant_angles[0][0]
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
    
    @staticmethod
    def find_best_step_match(target_step: float, dominant_steps: List[Tuple[float, int]]) -> float:
        """找到与目标步长最匹配的主导步长"""
        if not dominant_steps:
            return target_step
        
        best_match = dominant_steps[0][0]
        min_ratio_diff = abs(target_step / best_match - 1) if best_match > 0 else float('inf')
        
        for step, _ in dominant_steps:
            if step > 0:
                ratio_diff = abs(target_step / step - 1)
                if ratio_diff < min_ratio_diff:
                    min_ratio_diff = ratio_diff
                    best_match = step
        
        return best_match
    
    @staticmethod
    def interpolate_with_angle_and_step(pos1: List[float], pos2: List[float], 
                                      ratio: float, target_angle: float, target_step: float) -> List[float]:
        """使用目标角度和步长进行插值"""
        # 计算原始插值位置
        original_x = pos1[0] + ratio * (pos2[0] - pos1[0])
        original_y = pos1[1] + ratio * (pos2[1] - pos1[1])
        
        # 使用目标角度和步长调整位置
        total_distance = GeometryUtils.calculate_distance(pos1, pos2)
        target_distance = total_distance * ratio
        
        # 使用目标角度和步长计算新位置
        angle_rad = math.radians(target_angle)
        dx = target_distance * math.cos(angle_rad)
        dy = target_distance * math.sin(angle_rad)
        
        estimated_x = pos1[0] + dx
        estimated_y = pos1[1] + dy
        
        return [estimated_x, estimated_y]


class TrajectoryCompleter:
    """轨迹补全器"""
    
    def __init__(self, point_distance_threshold: float = 5.0):
        self.point_distance_threshold = point_distance_threshold
    
    def complete_trajectory_with_patterns(self, frames_data: Dict[int, Dict], 
                                       dominant_angles: List[Tuple[float, int]],
                                       dominant_steps: List[Tuple[float, int]]) -> Tuple[Dict[int, Dict], List[Tuple[int, List[float]]]]:
        """轨迹补全，考虑角度和步长模式"""
        all_generated_points = []
        generated_points_from_pairs = []
        
        # 对每个符合主导模式的点对生成点
        pattern_stats = PatternAnalyzer().collect_sequence_angles_and_steps(frames_data)
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
                if angle_diff <= 10.0:  # 使用默认角度容差
                    angle_match = True
                    break
            
            # 检查步长匹配
            if step_size is not None and step_size > 0 and step_size <= 40:
                for dominant_step, _ in dominant_steps:
                    if dominant_step > 0 and dominant_step <= 40:
                        step_ratio = step_size / dominant_step
                        if 0.9 <= step_ratio <= 1.1:
                            step_match = True
                            break
            
            # 如果角度和步长都匹配，则生成点
            if angle_match and step_match:
                generated_points = TrajectoryGenerator.generate_points_from_pair_with_steps(
                    pair_info['pos1'], pair_info['pos2'], 
                    pair_info['angle'], pair_info['step_size'], 
                    pair_info['frame1'], pair_info['frame2'],
                    total_frames=len(frames_data)
                )
                all_generated_points.extend(generated_points)
                generated_points_from_pairs.extend(generated_points)
        
        # 外推轨迹，填补间隙
        extrapolated_points = TrajectoryGenerator.extrapolate_trajectory_with_steps(
            frames_data, dominant_angles, dominant_steps
        )
        all_generated_points.extend(extrapolated_points)
        
        # 按帧组织点
        completed_frames = copy.deepcopy(frames_data)
        
        for frame, pos in all_generated_points:
            if frame not in completed_frames:
                completed_frames[frame] = {
                    'coords': [], 'num_objects': 0,
                    'image_name': f"sequence_{frame}_test"
                }
            
            # 检查是否与已有点重合
            is_duplicate = False
            for existing_pos in completed_frames[frame]['coords']:
                if GeometryUtils.calculate_distance(pos, existing_pos) <= self.point_distance_threshold:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                completed_frames[frame]['coords'].append(pos)
                completed_frames[frame]['num_objects'] = len(completed_frames[frame]['coords'])
        
        return completed_frames, generated_points_from_pairs


class OutlierFilter:
    """异常点过滤器"""
    
    @staticmethod
    def filter_outliers_by_dominant_patterns(frames_data: Dict[int, Dict], 
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
                
                if point_key in point_participation:
                    if point_participation[point_key]['participated']:
                        kept_coords.append(point)
                    else:
                        print(f"    移除未参与主导模式的点: 帧{frame_id}, 位置{point}")
                else:
                    kept_coords.append(point)
            
            frame_data['coords'] = kept_coords
            frame_data['num_objects'] = len(kept_coords)
        
        return filtered_frames


class AngleDistanceProcessorV3:
    """基于角度和步长的轨迹补全处理器 - 重构后的核心版本"""
    
    def __init__(self, 
                 angle_tolerance: float = 10.0,
                 min_angle_count: int = 2,
                 step_tolerance: float = 0.2,
                 min_step_count: int = 1,
                 max_step_size: float = 40.0,
                 point_distance_threshold: float = 5.0):
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
        
        # 初始化组件
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
            frames_data, pattern_stats['dominant_angles'], pattern_stats['dominant_steps']
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
                       default='results/spotgeov2-IRSTD/WTNet/predictions_8807.json',
                       help='预测结果文件路径')
    parser.add_argument('--gt_path', type=str, 
                       default='datasets/spotgeov2-IRSTD/test_anno.json',
                       help='真实标注文件路径')
    parser.add_argument('--output_path', type=str, 
                       default='results/spotgeov2/WTNet/angle_distance_processed_predictions_v3.json',
                       help='输出文件路径')
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
        point_distance_threshold=args.point_distance_threshold
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