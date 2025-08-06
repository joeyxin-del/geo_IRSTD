import json
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import copy
import math
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class SimpleAngleProcessor:
    """简易角度处理器，只通过聚类得出主导角度进行后处理"""
    
    def __init__(self, 
                 base_distance_threshold: float = 80.0,
                 min_cluster_size: int = 2,
                 angle_cluster_eps: float = 15.0,
                 angle_tolerance: float = 15.0,
                 max_distance: float = 145.0,
                 point_distance_threshold: float = 5.0,
                 confidence_threshold: float = 0.1,
                 max_step_size: float = 40.0):
        """
        初始化简易角度处理器
        
        Args:
            base_distance_threshold: 基础距离阈值
            min_cluster_size: 最小聚类大小
            angle_cluster_eps: 角度聚类半径（度）
            angle_tolerance: 角度容差（度）
            max_distance: 最大距离阈值
            point_distance_threshold: 重合点过滤阈值
            confidence_threshold: 主导模式置信度阈值
            max_step_size: 最大步长限制
        """
        self.base_distance_threshold = base_distance_threshold
        self.min_cluster_size = min_cluster_size
        self.angle_cluster_eps = angle_cluster_eps
        self.angle_tolerance = angle_tolerance
        self.max_distance = max_distance
        self.point_distance_threshold = point_distance_threshold
        self.confidence_threshold = confidence_threshold
        self.max_step_size = max_step_size
    
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
    
    def collect_all_angles(self, frames_data: Dict[int, Dict]) -> Tuple[List[float], List[Dict]]:
        """收集序列中所有点对的角度"""
        frames = sorted(frames_data.keys())
        
        if len(frames) < 2:
            return [], []
        
        all_angles = []
        angle_pairs = []
        
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
                            if distance <= self.max_distance:
                                angle = self.calculate_angle(pos1, pos2)
                                frame_gap = frame2 - frame1
                                step_size = self.calculate_step_size(pos1, pos2, frame_gap)
                                
                                # 过滤无效的步长
                                if step_size is not None and step_size > 0 and step_size <= self.max_step_size:
                                    all_angles.append(angle)
                                    angle_pairs.append({
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
        
        return all_angles, angle_pairs
    
    def cluster_angles(self, angles: List[float]) -> Tuple[List[float], List[float], List[Tuple[float, float, int]]]:
        """对角度进行聚类，找出主导角度"""
        if len(angles) < self.min_cluster_size:
            return [], [], []
        
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
        
        # 收集所有聚类信息
        clusters_info = []
        
        for label in unique_labels:
            if label == -1:  # 噪声点
                continue
            
            cluster_mask = labels == label
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size >= self.min_cluster_size:
                cluster_angles = np.array(angles)[cluster_mask]
                
                # 计算聚类中心角度
                cluster_angles_rad = cluster_angles * np.pi / 180.0
                mean_cos = np.mean(np.cos(cluster_angles_rad))
                mean_sin = np.mean(np.sin(cluster_angles_rad))
                dominant_angle = math.degrees(math.atan2(mean_sin, mean_cos))
                
                if dominant_angle < 0:
                    dominant_angle += 360
                
                confidence = cluster_size / len(angles)
                
                clusters_info.append((dominant_angle, confidence, cluster_size))
        
        print(clusters_info)
        # 按聚类大小排序，选择最大的聚类作为主导角度
        clusters_info.sort(key=lambda x: x[1], reverse=True)
        
        dominant_angles = []
        confidences = []
        
        for angle, conf, size in clusters_info:
            if conf >= self.confidence_threshold:
                dominant_angles.append(angle)
                confidences.append(conf)
        
        return dominant_angles, confidences, clusters_info
    
    def generate_points_from_pattern(self, pos1: List[float], pos2: List[float], 
                                   dominant_angle: float, 
                                   frame1: int, frame2: int, total_frames: int) -> List[Tuple[int, List[float]]]:
        """基于主导角度生成缺失帧的点"""
        points = []
        
        # 计算两点之间的实际距离和步长
        actual_distance = self.calculate_distance(pos1, pos2)
        frame_gap = frame2 - frame1
        actual_step_size = actual_distance / frame_gap if frame_gap > 0 else 0.0
        
        # 生成除了frame1和frame2之外的所有帧点
        for frame in range(1, total_frames + 1):
            if frame == frame1 or frame == frame2:
                continue
            
            if frame1 < frame < frame2:
                # 在两个检测点之间，使用插值
                ratio = (frame - frame1) / (frame2 - frame1)
                x = pos1[0] + ratio * (pos2[0] - pos1[0])
                y = pos1[1] + ratio * (pos2[1] - pos1[1])
                points.append((frame, [x, y]))
                
            elif frame < frame1:
                # 在第一个检测点之前，使用外推
                frame_gap = frame1 - frame
                extrapolated_pos = self.extrapolate_position_by_angle(
                    pos1, dominant_angle, actual_step_size, -frame_gap
                )
                points.append((frame, extrapolated_pos))
                
            elif frame > frame2:
                # 在第二个检测点之后，使用外推
                frame_gap = frame - frame2
                extrapolated_pos = self.extrapolate_position_by_angle(
                    pos2, dominant_angle, actual_step_size, frame_gap
                )
                points.append((frame, extrapolated_pos))
        
        return points
    
    def extrapolate_position_by_angle(self, base_pos: List[float], dominant_angle: float, 
                                    step_size: float, frame_gap: int) -> List[float]:
        """基于主导角度外推位置"""
        angle_rad = math.radians(dominant_angle)
        step_dx = step_size * math.cos(angle_rad)
        step_dy = step_size * math.sin(angle_rad)
        
        total_dx = step_dx * frame_gap
        total_dy = step_dy * frame_gap
        
        return [base_pos[0] + total_dx, base_pos[1] + total_dy]
    
    def complete_trajectory_with_angles(self, frames_data: Dict[int, Dict], 
                                     dominant_angles: List[float],
                                     angle_pairs: List[Dict]) -> Tuple[Dict[int, Dict], List[Tuple[int, List[float]]]]:
        """基于主导角度进行轨迹补全"""
        all_generated_points = []
        generated_points_from_pairs = []
        
        # 对每个符合主导角度的点对生成点
        for pair_info in angle_pairs:
            angle = pair_info['angle']
            
            # 检查是否属于任一主导角度
            angle_match = False
            best_angle = None
            
            # 检查角度匹配（与所有主导角度比较，取最佳匹配）
            best_angle_score = 0
            for dominant_angle in dominant_angles:
                angle_diff = min(abs(angle - dominant_angle), 
                               abs(angle - dominant_angle + 360),
                               abs(angle - dominant_angle - 360))
                if angle_diff <= self.angle_tolerance:
                    angle_score = 1 - (angle_diff / self.angle_tolerance)
                    if angle_score > best_angle_score:
                        best_angle_score = angle_score
                        angle_match = True
                        best_angle = dominant_angle
            
            # 如果角度匹配，则生成点
            if angle_match:
                # 使用最佳匹配的主导角度生成点
                generated_points = self.generate_points_from_pattern(
                    pair_info['pos1'], pair_info['pos2'], 
                    best_angle,
                    pair_info['frame1'], pair_info['frame2'],
                    len(frames_data)
                )
                all_generated_points.extend(generated_points)
                generated_points_from_pairs.extend(generated_points)
        
        # 外推轨迹，填补间隙（使用第一个主导角度）
        if dominant_angles:
            extrapolated_points = self.extrapolate_trajectory_with_angles(
                frames_data, dominant_angles[0]
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
    
    def extrapolate_trajectory_with_angles(self, frames_data: Dict[int, Dict], 
                                        dominant_angle: float) -> List[Tuple[int, List[float]]]:
        """基于主导角度外推轨迹"""
        frames = sorted(frames_data.keys())
        if len(frames) < 2:
            return []
        
        extrapolated_points = []
        
        # 找到所有有检测点的帧
        detected_frames = [f for f in frames if frames_data[f]['coords']]
        
        if len(detected_frames) < 2:
            return extrapolated_points
        
        # 填补序列开头的间隙
        first_detected = detected_frames[0]
        if first_detected > min(frames):
            first_pos = frames_data[first_detected]['coords'][0]
            missing_frames = range(min(frames), first_detected)
            
            # 计算平均步长
            if len(detected_frames) >= 2:
                total_distance = 0
                total_gap = 0
                for i in range(len(detected_frames) - 1):
                    pos1 = frames_data[detected_frames[i]]['coords'][0]
                    pos2 = frames_data[detected_frames[i + 1]]['coords'][0]
                    distance = self.calculate_distance(pos1, pos2)
                    gap = detected_frames[i + 1] - detected_frames[i]
                    total_distance += distance
                    total_gap += gap
                
                avg_step_size = total_distance / total_gap if total_gap > 0 else 10.0
            else:
                avg_step_size = 10.0  # 默认步长
            
            for frame in missing_frames:
                frame_gap = first_detected - frame
                extrapolated_pos = self.extrapolate_position_by_angle(
                    first_pos, dominant_angle, avg_step_size, -frame_gap
                )
                extrapolated_points.append((frame, extrapolated_pos))
        
        # 填补序列结尾的间隙
        last_detected = detected_frames[-1]
        if last_detected < max(frames):
            last_pos = frames_data[last_detected]['coords'][0]
            missing_frames = range(last_detected + 1, max(frames) + 1)
            
            # 计算平均步长
            if len(detected_frames) >= 2:
                total_distance = 0
                total_gap = 0
                for i in range(len(detected_frames) - 1):
                    pos1 = frames_data[detected_frames[i]]['coords'][0]
                    pos2 = frames_data[detected_frames[i + 1]]['coords'][0]
                    distance = self.calculate_distance(pos1, pos2)
                    gap = detected_frames[i + 1] - detected_frames[i]
                    total_distance += distance
                    total_gap += gap
                
                avg_step_size = total_distance / total_gap if total_gap > 0 else 10.0
            else:
                avg_step_size = 10.0  # 默认步长
            
            for frame in missing_frames:
                frame_gap = frame - last_detected
                extrapolated_pos = self.extrapolate_position_by_angle(
                    last_pos, dominant_angle, avg_step_size, frame_gap
                )
                extrapolated_points.append((frame, extrapolated_pos))
        
        return extrapolated_points
    
    def visualize_angle_clustering(self, angles: List[float], sequence_id: int, 
                                 clusters_info: List[Tuple[float, float, int]],
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
        
        # 3. 聚类结果
        if clusters_info:
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
            
            axes[1, 0].set_title('Clusters')
            axes[1, 0].set_xlabel('cos(angle)')
            axes[1, 0].set_ylabel('sin(angle)')
            axes[1, 0].set_xlim(-1.2, 1.2)
            axes[1, 0].set_ylim(-1.2, 1.2)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_aspect('equal')
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. 聚类统计信息
        if clusters_info:
            cluster_angles = [angle for angle, _, _ in clusters_info]
            cluster_sizes = [size for _, _, size in clusters_info]
            cluster_confs = [conf for _, conf, _ in clusters_info]
            
            # 创建柱状图
            x_pos = np.arange(len(clusters_info))
            colors = plt.cm.Set3(np.linspace(0, 1, len(clusters_info)))
            bars = axes[1, 1].bar(x_pos, cluster_sizes, alpha=0.7, color=colors)
            
            # 添加数值标签
            for i, (bar, size, conf) in enumerate(zip(bars, cluster_sizes, cluster_confs)):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{size}\n({conf:.3f})', ha='center', va='bottom')
            
            axes[1, 1].set_title('Cluster Statistics')
            axes[1, 1].set_xlabel('Cluster Index')
            axes[1, 1].set_ylabel('Cluster Size')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels([f'C{i+1}' for i in range(len(clusters_info))])
            axes[1, 1].grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        output_path = os.path.join(output_dir, f'sequence_{sequence_id}_angle_clustering.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"角度聚类可视化已保存到: {output_path}")
        
        plt.close()
        
        return output_path
    
    def filter_points_by_angle(self, frames_data: Dict[int, Dict], 
                             dominant_angles: List[float], 
                             angle_pairs: List[Dict]) -> Dict[int, Dict]:
        """基于主导角度过滤点"""
        if not dominant_angles:
            return frames_data
        
        # 创建点参与哈希表
        point_participation = {}
        
        for pair_info in angle_pairs:
            frame1, pos1_idx = pair_info['frame1'], pair_info['pos1_idx']
            frame2, pos2_idx = pair_info['frame2'], pair_info['pos2_idx']
            
            point_key1 = (frame1, pos1_idx)
            point_key2 = (frame2, pos2_idx)
            
            if point_key1 not in point_participation:
                point_participation[point_key1] = {
                    'participated': False,
                    'point': pair_info['pos1'],
                    'frame_id': frame1,
                    'point_idx': pos1_idx,
                    'best_angle_match': None,
                    'angle_score': 0
                }
            
            if point_key2 not in point_participation:
                point_participation[point_key2] = {
                    'participated': False,
                    'point': pair_info['pos2'],
                    'frame_id': frame2,
                    'point_idx': pos2_idx,
                    'best_angle_match': None,
                    'angle_score': 0
                }
        
        # 评估每个点对的模式匹配度
        for pair_info in angle_pairs:
            angle = pair_info['angle']
            
            # 计算角度匹配分数（与所有主导角度比较，取最佳匹配）
            best_angle_score = 0
            best_angle_match = None
            for dominant_angle in dominant_angles:
                angle_diff = min(abs(angle - dominant_angle), 
                               abs(angle - dominant_angle + 360),
                               abs(angle - dominant_angle - 360))
                angle_score = max(0, 1 - (angle_diff / self.angle_tolerance))
                if angle_score > best_angle_score:
                    best_angle_score = angle_score
                    best_angle_match = dominant_angle
            
            # 如果角度匹配分数足够高，标记为参与主导模式
            if best_angle_score >= 0.3:  # 阈值可调整
                frame1, pos1_idx = pair_info['frame1'], pair_info['pos1_idx']
                frame2, pos2_idx = pair_info['frame2'], pair_info['pos2_idx']
                
                point_key1 = (frame1, pos1_idx)
                point_key2 = (frame2, pos2_idx)
                
                # 更新点1的信息
                if point_key1 in point_participation:
                    point_participation[point_key1]['participated'] = True
                    point_participation[point_key1]['angle_score'] = max(
                        point_participation[point_key1]['angle_score'], best_angle_score
                    )
                    point_participation[point_key1]['best_angle_match'] = best_angle_match
                
                # 更新点2的信息
                if point_key2 in point_participation:
                    point_participation[point_key2]['participated'] = True
                    point_participation[point_key2]['angle_score'] = max(
                        point_participation[point_key2]['angle_score'], best_angle_score
                    )
                    point_participation[point_key2]['best_angle_match'] = best_angle_match
        
        # 过滤点
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
    
    def process_sequence(self, sequence_id: int, frames_data: Dict[int, Dict], 
                        save_visualization: bool = True) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
        """处理单个序列，返回轨迹补全和异常点筛选的结果"""
        print(f"开始处理序列 {sequence_id}...")
        
        # 收集所有点对的角度
        all_angles, angle_pairs = self.collect_all_angles(frames_data)
        
        if not all_angles:
            print(f"序列 {sequence_id} 没有足够的点对数据")
            return frames_data, frames_data
        
        print(f"序列 {sequence_id} 计算了 {len(all_angles)} 个角度")
        
        # 聚类分析角度
        dominant_angles, confidences, clusters_info = self.cluster_angles(all_angles)
        
        # 可视化角度聚类过程
        if save_visualization and clusters_info:
            try:
                self.visualize_angle_clustering(all_angles, sequence_id, clusters_info)
            except Exception as e:
                print(f"可视化角度聚类时出错: {e}")
        
        if dominant_angles:
            print(f"主导角度: {dominant_angles[0]:.1f}° (置信度: {confidences[0]:.3f})")
            if len(dominant_angles) > 1:
                print(f"次主导角度: {dominant_angles[1]:.1f}° (置信度: {confidences[1]:.3f})")
        else:
            print(f"主导角度: 未发现")
            return frames_data, frames_data
        
        # 步骤1: 轨迹补全
        completed_sequence, generated_points = self.complete_trajectory_with_angles(
            frames_data, dominant_angles, angle_pairs
        )
        
        # 步骤2: 异常点筛选
        filtered_sequence = self.filter_points_by_angle(
            completed_sequence, dominant_angles, angle_pairs
        )
        
        return completed_sequence, filtered_sequence
    
    def process_dataset(self, predictions: Dict, save_visualization: bool = True) -> Dict:
        """处理整个数据集"""
        print("开始简易角度处理...")
        
        # 提取序列信息
        sequence_data = self.extract_sequence_info(predictions)
        print(f"提取了 {len(sequence_data)} 个序列的信息")
        
        # 处理每个序列
        processed_predictions = {}
        
        for sequence_id, frames_data in sequence_data.items():
            # 处理单个序列
            completed_sequence, filtered_sequence = self.process_sequence(
                sequence_id, frames_data, save_visualization=save_visualization
            )
            
            # 使用过滤后的结果
            processed_sequence = filtered_sequence
            
            # 转换回原始格式
            for frame, frame_data in processed_sequence.items():
                image_name = frame_data['image_name']
                processed_predictions[image_name] = {
                    'coords': frame_data['coords'],
                    'num_objects': frame_data['num_objects']
                }
        
        print(f"简易角度处理完成，处理了 {len(processed_predictions)} 张图像")
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
    """主函数，演示简易角度处理"""
    import argparse
    import os
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='简易角度处理器')
    parser.add_argument('--pred_path', type=str, default='results/spotgeov2-IRSTD/WTNet/predictions_8807.json',
                       help='预测结果文件路径')
    parser.add_argument('--gt_path', type=str, default='datasets/spotgeov2-IRSTD/test_anno.json',
                       help='真实标注文件路径')
    parser.add_argument('--output_path', type=str, default='results/spotgeov2/WTNet/angle_cluster_predictions.json',
                       help='输出文件路径')
    parser.add_argument('--base_distance_threshold', type=float, default=1000,
                       help='基础距离阈值')
    parser.add_argument('--min_cluster_size', type=int, default=2,
                       help='最小聚类大小')
    parser.add_argument('--angle_cluster_eps', type=float, default=15.0,
                       help='角度聚类半径（度）')
    parser.add_argument('--angle_tolerance', type=float, default=1.0,
                       help='角度容差（度）')
    parser.add_argument('--max_distance', type=float, default=94.0,
                       help='最大距离阈值')
    parser.add_argument('--point_distance_threshold', type=float, default=10,
                       help='重合点过滤阈值')
    parser.add_argument('--confidence_threshold', type=float, default=0.1,
                       help='主导模式置信度阈值')
    parser.add_argument('--max_step_size', type=float, default=40.0,
                       help='最大步长限制')
    parser.add_argument('--no_visualization', default=True,
                       help='不保存可视化结果')
    
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
    
    # 创建简易角度处理器
    processor = SimpleAngleProcessor(
        base_distance_threshold=args.base_distance_threshold,
        min_cluster_size=args.min_cluster_size,
        angle_cluster_eps=args.angle_cluster_eps,
        angle_tolerance=args.angle_tolerance,
        max_distance=args.max_distance,
        point_distance_threshold=args.point_distance_threshold,
        confidence_threshold=args.confidence_threshold,
        max_step_size=args.max_step_size
    )
    
    # 进行简易角度处理
    processed_predictions = processor.process_dataset(
        original_predictions, save_visualization=not args.no_visualization
    )
    
    # 评估改善效果
    try:
        improvement = processor.evaluate_improvement(original_predictions, processed_predictions, ground_truth)
        
        # 打印结果
        print("\n=== 简易角度处理效果评估 ===")
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
        print(f"\n简易角度处理结果已保存到: {args.output_path}")
    except Exception as e:
        print(f"保存结果时出错: {e}")


if __name__ == '__main__':
    main()