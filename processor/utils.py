import numpy as np
import math
import copy
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
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
                        # print(f"    移除未参与主导模式的点: 帧{frame_id}, 位置{point}")
                        pass
                else:
                    kept_coords.append(point)
            
            frame_data['coords'] = kept_coords
            frame_data['num_objects'] = len(kept_coords)
        
        return filtered_frames


class TrajectoryCompleter:
    """轨迹补全器"""
    
    def __init__(self, point_distance_threshold: float = 5.0):
        self.point_distance_threshold = point_distance_threshold
    
    def complete_trajectory_with_patterns(self, frames_data: Dict[int, Dict], 
                                       dominant_angles: List[Tuple[float, int]],
                                       dominant_steps: List[Tuple[float, int]],
                                       pattern_analyzer=None) -> Tuple[Dict[int, Dict], List[Tuple[int, List[float]]]]:
        """轨迹补全，考虑角度和步长模式"""
        all_generated_points = []
        generated_points_from_pairs = []
        
        # 对每个符合主导模式的点对生成点
        if pattern_analyzer is not None:
            pattern_stats = pattern_analyzer.collect_sequence_angles_and_steps(frames_data)
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



def visualize_angle_clustering(angles: List[float], sequence_id: int, 
                                clusters_info: List[Tuple[float, float, int]], 
                                merged_clusters: List[Tuple[float, float, int]],
                                output_dir: str = 'clustering_visualizations'):
    """可视化角度聚类过程"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Sequence {sequence_id} Angle Clustering Visualization', fontsize=16, fontweight='bold')
    
    # 1. 原始角度分布直方图
    axes[0, 0].hist(angles, bins=36, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Original Angle Distribution')
    axes[0, 0].set_xlabel('Angle (degrees)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 角度在单位圆上的分布
    angles_rad = np.array(angles) * np.pi / 180.0
    x_coords = np.cos(angles_rad)
    y_coords = np.sin(angles_rad)
    
    # 绘制单位圆
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].add_patch(circle)
    
    # 绘制角度点
    axes[0, 1].scatter(x_coords, y_coords, alpha=0.6, s=30, color='blue')
    axes[0, 1].set_title('Angles on Unit Circle')
    axes[0, 1].set_xlabel('cos(angle)')
    axes[0, 1].set_ylabel('sin(angle)')
    axes[0, 1].set_xlim(-1.2, 1.2)
    axes[0, 1].set_ylim(-1.2, 1.2)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_aspect('equal')
    
    # 3. 聚类结果（合并前）
    if clusters_info:
        cluster_angles = [angle for angle, _, _ in clusters_info]
        cluster_sizes = [size for _, _, size in clusters_info]
        cluster_confs = [conf for _, conf, _ in clusters_info]
        
        # 使用不同颜色表示不同聚类
        colors = plt.cm.Set3(np.linspace(0, 1, len(clusters_info)))
        
        for i, (angle, conf, size) in enumerate(clusters_info):
            # 在单位圆上标记聚类中心
            angle_rad = angle * np.pi / 180.0
            x = np.cos(angle_rad)
            y = np.sin(angle_rad)
            axes[1, 0].scatter(x, y, s=size*10, c=[colors[i]], alpha=0.8, 
                                label=f'Cluster {i+1}: {angle:.1f}° (size={size})')
            
            # 绘制聚类范围
            circle_radius = self.angle_tolerance * np.pi / 180.0
            cluster_circle = plt.Circle((x, y), circle_radius, fill=False, 
                                        color=colors[i], linestyle='-', alpha=0.7)
            axes[1, 0].add_patch(cluster_circle)
        
        # 绘制单位圆
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].add_patch(circle)
        
        axes[1, 0].set_title('Clusters Before Merging')
        axes[1, 0].set_xlabel('cos(angle)')
        axes[1, 0].set_ylabel('sin(angle)')
        axes[1, 0].set_xlim(-1.2, 1.2)
        axes[1, 0].set_ylim(-1.2, 1.2)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_aspect('equal')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. 聚类结果（合并后）
    if merged_clusters:
        colors = plt.cm.Set1(np.linspace(0, 1, len(merged_clusters)))
        
        for i, (angle, conf, size) in enumerate(merged_clusters):
            # 在单位圆上标记合并后的聚类中心
            angle_rad = angle * np.pi / 180.0
            x = np.cos(angle_rad)
            y = np.sin(angle_rad)
            axes[1, 1].scatter(x, y, s=size*10, c=[colors[i]], alpha=0.8,
                                label=f'Cluster {i+1}: {angle:.1f}° (size={size}, conf={conf:.3f})')
            
            # 绘制聚类范围
            circle_radius = self.angle_tolerance * np.pi / 180.0
            cluster_circle = plt.Circle((x, y), circle_radius, fill=False, 
                                        color=colors[i], linestyle='-', alpha=0.7)
            axes[1, 1].add_patch(cluster_circle)
        
        # 绘制单位圆
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].add_patch(circle)
        
        axes[1, 1].set_title('Clusters After Merging')
        axes[1, 1].set_xlabel('cos(angle)')
        axes[1, 1].set_ylabel('sin(angle)')
        axes[1, 1].set_xlim(-1.2, 1.2)
        axes[1, 1].set_ylim(-1.2, 1.2)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_aspect('equal')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(output_dir, f'sequence_{sequence_id}_angle_clustering.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"角度聚类可视化已保存到: {output_path}")
    
    plt.close()
    
    return output_path

