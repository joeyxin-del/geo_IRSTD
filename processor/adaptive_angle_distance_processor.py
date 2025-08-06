import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, Counter
import copy
import math
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class AdaptiveAngleDistanceProcessor:
    """自适应角度距离处理器，通过聚类自动发现主导角度和主导步长"""
    
    def __init__(self, 
                 base_distance_threshold: float = 80.0,
                 min_track_length: int = 2,
                 expected_sequence_length: int = 5,
                 min_cluster_size: int = 3,
                 angle_cluster_eps: float = 5.0,
                 step_cluster_eps: float = 0.3,
                 confidence_threshold: float = 0.6,
                 angle_tolerance: float = 15.0,
                 step_tolerance: float = 0.5,
                 max_step_size: float = 40.0,
                 point_distance_threshold: float = 5.0,
                 dominant_ratio_threshold: float = 0.7,
                 secondary_ratio_threshold: float = 0.2,
                 max_dominant_patterns: int = 3,
                 use_angle_clustering: bool = True,
                 use_step_clustering: bool = True,
                 angle_weight: float = 0.65,
                 step_weight: float = 0.35):
        """
        初始化自适应角度距离处理器
        
        Args:
            base_distance_threshold: 基础距离阈值
            min_track_length: 最小轨迹长度
            expected_sequence_length: 期望的序列长度
            min_cluster_size: 最小聚类大小
            angle_cluster_eps: 角度聚类半径（度）
            step_cluster_eps: 步长聚类半径（比例）
            confidence_threshold: 主导模式置信度阈值
            angle_tolerance: 角度容差（度）
            step_tolerance: 步长容差（比例）
            max_step_size: 最大步长限制
            point_distance_threshold: 重合点过滤阈值
            dominant_ratio_threshold: 主导模式占比阈值（0.7表示70%）
            secondary_ratio_threshold: 次主导模式占比阈值（0.2表示20%）
            max_dominant_patterns: 最大主导模式数量
            use_angle_clustering: 是否使用角度聚类
            use_step_clustering: 是否使用步长聚类
            angle_weight: 角度在综合评分中的权重
            step_weight: 步长在综合评分中的权重
        """
        self.base_distance_threshold = base_distance_threshold
        self.min_track_length = min_track_length
        self.expected_sequence_length = expected_sequence_length
        self.min_cluster_size = min_cluster_size
        self.angle_cluster_eps = angle_cluster_eps
        self.step_cluster_eps = step_cluster_eps
        self.confidence_threshold = confidence_threshold
        self.angle_tolerance = angle_tolerance
        self.step_tolerance = step_tolerance
        self.max_step_size = max_step_size
        self.point_distance_threshold = point_distance_threshold
        self.dominant_ratio_threshold = dominant_ratio_threshold
        self.secondary_ratio_threshold = secondary_ratio_threshold
        self.max_dominant_patterns = max_dominant_patterns
        self.use_angle_clustering = use_angle_clustering
        self.use_step_clustering = use_step_clustering
        self.angle_weight = angle_weight
        self.step_weight = step_weight
        
        # 验证权重设置
        if abs(angle_weight + step_weight - 1.0) > 1e-6:
            print(f"警告：角度权重({angle_weight}) + 步长权重({step_weight}) != 1.0，将自动归一化")
            total_weight = angle_weight + step_weight
            self.angle_weight = angle_weight / total_weight
            self.step_weight = step_weight / total_weight
    
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
    
    def collect_all_angle_step_pairs(self, frames_data: Dict[int, Dict]) -> Tuple[List[float], List[float], List[Dict]]:
        """收集序列中所有点对的角度和步长，采用angle_distance_processor.py的成功策略"""
        frames = sorted(frames_data.keys())
        
        if len(frames) < 2:
            return [], [], []
        
        all_angles = []
        all_steps = []
        angle_step_pairs = []
        
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
                            if distance <= 145:  # 使用较大的阈值
                                angle = self.calculate_angle(pos1, pos2)
                                frame_gap = frame2 - frame1
                                step_size = self.calculate_step_size(pos1, pos2, frame_gap)
                                
                                # 过滤无效的步长
                                if step_size is not None and step_size > 0 and step_size <= self.max_step_size:
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
        
        return all_angles, all_steps, angle_step_pairs
    
    def cluster_angles(self, angles: List[float]) -> Tuple[List[Tuple[float, float]], List[float], List[Tuple[float, float, int]], List[Tuple[float, float, int]]]:
        """对角度进行聚类，自适应确定主导角度数量"""
        if len(angles) < 1:
            return [], [], [], []
        
        # 分类处理策略
        if len(angles) == 1:
            # 只有一个角度，直接使用该角度
            single_angle = angles[0]
            dominant_angles = [(single_angle, 1.0)]  # 置信度为1.0
            confidences = [1.0]
            clusters_info = [(single_angle, 1.0, 1)]
            merged_clusters = [(single_angle, 1.0, 1)]
            print(f"检测到单一角度: {single_angle:.1f}° (置信度: 1.000)")
            return dominant_angles, confidences, clusters_info, merged_clusters
        
        elif len(angles) == 2:
            # 只有两个角度，检查是否相似
            angle_diff = min(abs(angles[0] - angles[1]), 
                           abs(angles[0] - angles[1] + 360),
                           abs(angles[0] - angles[1] - 360))
            
            if angle_diff <= self.angle_tolerance:
                # 两个角度相似，取平均值
                avg_angle = (angles[0] + angles[1]) / 2
                if avg_angle < 0:
                    avg_angle += 360
                dominant_angles = [(avg_angle, 1.0)]
                confidences = [1.0]
                clusters_info = [(avg_angle, 1.0, 2)]
                merged_clusters = [(avg_angle, 1.0, 2)]
                print(f"检测到相似双角度，取平均值: {avg_angle:.1f}° (置信度: 1.000)")
                return dominant_angles, confidences, clusters_info, merged_clusters
            else:
                # 两个角度不相似，都保留
                dominant_angles = [(angles[0], 0.5), (angles[1], 0.5)]
                confidences = [0.5, 0.5]
                clusters_info = [(angles[0], 0.5, 1), (angles[1], 0.5, 1)]
                merged_clusters = [(angles[0], 0.5, 1), (angles[1], 0.5, 1)]
                print(f"检测到双角度模式: {angles[0]:.1f}° 和 {angles[1]:.1f}° (置信度: 0.500)")
                return dominant_angles, confidences, clusters_info, merged_clusters
        
        # 多个角度，使用聚类
        if len(angles) < self.min_cluster_size:
            # 角度数量少于最小聚类大小，但大于2，使用简单统计
            angle_counter = Counter(angles)
            most_common_angle = angle_counter.most_common(1)[0][0]
            confidence = angle_counter[most_common_angle] / len(angles)
            dominant_angles = [(most_common_angle, confidence)]
            confidences = [confidence]
            clusters_info = [(most_common_angle, confidence, angle_counter[most_common_angle])]
            merged_clusters = [(most_common_angle, confidence, angle_counter[most_common_angle])]
            print(f"角度数量较少，使用最常见角度: {most_common_angle:.1f}° (置信度: {confidence:.3f})")
            return dominant_angles, confidences, clusters_info, merged_clusters
        
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
        
        # 合并相似的角度聚类
        merged_clusters = self.merge_similar_angle_clusters(clusters_info)
        
        # 自适应确定主导角度数量
        dominant_angles, confidences = self.adaptive_select_dominant_angles(merged_clusters, len(angles))
        
        return dominant_angles, confidences, clusters_info, merged_clusters
    
    def adaptive_select_dominant_angles(self, merged_clusters: List[Tuple[float, float, int]], total_samples: int) -> Tuple[List[Tuple[float, float]], List[float]]:
        """自适应选择主导角度数量"""
        if not merged_clusters:
            return [], []
        
        # 按聚类大小排序
        merged_clusters.sort(key=lambda x: x[2], reverse=True)
        
        # 计算主要聚类的统计信息
        largest_cluster_size = merged_clusters[0][2]
        largest_cluster_ratio = largest_cluster_size / total_samples
        
        # 自适应规则：
        # 1. 如果最大聚类占比 > dominant_ratio_threshold，只取1个主导角度
        # 2. 如果最大聚类占比 > 50% 且第二大聚类占比 > secondary_ratio_threshold，取2个主导角度
        # 3. 如果最大聚类占比 < 50%，取所有聚类（最多max_dominant_patterns个）
        
        if largest_cluster_ratio > self.dominant_ratio_threshold:
            # 明显的主导模式，只取1个
            selected_clusters = merged_clusters[:1]
            print(f"检测到单一主导模式，占比: {largest_cluster_ratio:.1%}")
            
        elif largest_cluster_ratio > 0.5 and len(merged_clusters) > 1:
            second_cluster_ratio = merged_clusters[1][2] / total_samples
            if second_cluster_ratio > self.secondary_ratio_threshold:
                # 双主导模式
                selected_clusters = merged_clusters[:2]
                print(f"检测到双主导模式，主模式占比: {largest_cluster_ratio:.1%}，次模式占比: {second_cluster_ratio:.1%}")
            else:
                # 虽然有两个聚类，但第二个太小，只取1个
                selected_clusters = merged_clusters[:1]
                print(f"检测到单一主导模式（次模式太小），主模式占比: {largest_cluster_ratio:.1%}")
                
        else:
            # 多模式或模式不明显，取前max_dominant_patterns个
            selected_clusters = merged_clusters[:min(self.max_dominant_patterns, len(merged_clusters))]
            print(f"检测到多模式或模式不明显，选择前{len(selected_clusters)}个聚类")
        
        # 返回选中的主导角度和置信度
        dominant_angles = [(angle, conf) for angle, conf, _ in selected_clusters]
        confidences = [conf for _, conf, _ in selected_clusters]
        
        return dominant_angles, confidences
    
    def merge_similar_angle_clusters(self, clusters_info: List[Tuple[float, float, int]]) -> List[Tuple[float, float, int]]:
        """合并相似的角度聚类"""
        if len(clusters_info) <= 1:
            return clusters_info
        
        # 按角度排序
        clusters_info.sort(key=lambda x: x[0])
        
        merged_clusters = []
        i = 0
        
        while i < len(clusters_info):
            current_angle, current_conf, current_size = clusters_info[i]
            merged_angles = [current_angle]
            merged_confs = [current_conf]
            merged_sizes = [current_size]
            
            # 检查后续的聚类是否与当前聚类相似
            j = i + 1
            while j < len(clusters_info):
                next_angle, next_conf, next_size = clusters_info[j]
                
                # 计算角度差异（考虑周期性）
                angle_diff = min(abs(next_angle - current_angle), 
                               abs(next_angle - current_angle + 360),
                               abs(next_angle - current_angle - 360))
                
                # 如果角度差异小于阈值，合并聚类
                if angle_diff <= self.angle_tolerance:
                    merged_angles.append(next_angle)
                    merged_confs.append(next_conf)
                    merged_sizes.append(next_size)
                    j += 1
                else:
                    break
            
            # 计算合并后的聚类中心角度
            if len(merged_angles) > 1:
                # 使用加权平均计算合并后的角度
                total_size = sum(merged_sizes)
                weighted_angle = sum(angle * size for angle, size in zip(merged_angles, merged_sizes)) / total_size
                weighted_conf = sum(conf * size for conf, size in zip(merged_confs, merged_sizes)) / total_size
                merged_size = total_size
            else:
                weighted_angle = current_angle
                weighted_conf = current_conf
                merged_size = current_size
            
            merged_clusters.append((weighted_angle, weighted_conf, merged_size))
            i = j
        
        return merged_clusters
    
    def cluster_steps(self, steps: List[float]) -> Tuple[List[Tuple[float, float]], List[float]]:
        """对步长进行聚类，自适应确定主导步长数量"""
        if len(steps) < 1:
            return [], []
        
        # 分类处理策略
        if len(steps) == 1:
            # 只有一个步长，直接使用该步长
            single_step = steps[0]
            dominant_steps = [(single_step, 1.0)]  # 置信度为1.0
            confidences = [1.0]
            print(f"检测到单一步长: {single_step:.1f} (置信度: 1.000)")
            return dominant_steps, confidences
        
        elif len(steps) == 2:
            # 只有两个步长，检查是否相似
            if steps[0] > 0 and steps[1] > 0:
                step_ratio = steps[1] / steps[0]
                step_diff = abs(step_ratio - 1)
                
                if step_diff <= self.step_tolerance:
                    # 两个步长相似，取平均值
                    avg_step = (steps[0] + steps[1]) / 2
                    dominant_steps = [(avg_step, 1.0)]
                    confidences = [1.0]
                    print(f"检测到相似双步长，取平均值: {avg_step:.1f} (置信度: 1.000)")
                    return dominant_steps, confidences
                else:
                    # 两个步长不相似，都保留
                    dominant_steps = [(steps[0], 0.5), (steps[1], 0.5)]
                    confidences = [0.5, 0.5]
                    print(f"检测到双步长模式: {steps[0]:.1f} 和 {steps[1]:.1f} (置信度: 0.500)")
                    return dominant_steps, confidences
            else:
                # 有无效步长，使用有效的一个
                valid_steps = [s for s in steps if s > 0]
                if len(valid_steps) == 1:
                    dominant_steps = [(valid_steps[0], 1.0)]
                    confidences = [1.0]
                    print(f"检测到单一有效步长: {valid_steps[0]:.1f} (置信度: 1.000)")
                    return dominant_steps, confidences
                else:
                    # 没有有效步长
                    return [], []
        
        # 多个步长，使用聚类
        if len(steps) < self.min_cluster_size:
            # 步长数量少于最小聚类大小，但大于2，使用简单统计
            valid_steps = [s for s in steps if s > 0]
            if not valid_steps:
                return [], []
            
            step_counter = Counter(valid_steps)
            most_common_step = step_counter.most_common(1)[0][0]
            confidence = step_counter[most_common_step] / len(valid_steps)
            dominant_steps = [(most_common_step, confidence)]
            confidences = [confidence]
            print(f"步长数量较少，使用最常见步长: {most_common_step:.1f} (置信度: {confidence:.3f})")
            return dominant_steps, confidences
        
        # 将步长转换为二维特征
        valid_steps = [s for s in steps if s > 0]
        if not valid_steps:
            return [], []
        
        steps_array = np.array(valid_steps).reshape(-1, 1)
        
        # 标准化
        scaler = StandardScaler()
        steps_scaled = scaler.fit_transform(steps_array)
        
        # 使用DBSCAN聚类
        clustering = DBSCAN(eps=self.step_cluster_eps, min_samples=self.min_cluster_size).fit(steps_scaled)
        
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
                cluster_steps = np.array(valid_steps)[cluster_mask]
                dominant_step = np.median(cluster_steps)  # 使用中位数作为主导步长
                confidence = cluster_size / len(valid_steps)
                
                clusters_info.append((dominant_step, confidence, cluster_size))
        
        # 合并相似的步长聚类
        merged_clusters = self.merge_similar_step_clusters(clusters_info)
        
        # 自适应确定主导步长数量
        dominant_steps, confidences = self.adaptive_select_dominant_steps(merged_clusters, len(valid_steps))
        
        return dominant_steps, confidences
    
    def adaptive_select_dominant_steps(self, merged_clusters: List[Tuple[float, float, int]], total_samples: int) -> Tuple[List[Tuple[float, float]], List[float]]:
        """自适应选择主导步长数量"""
        if not merged_clusters:
            return [], []
        
        # 按聚类大小排序
        merged_clusters.sort(key=lambda x: x[2], reverse=True)
        
        # 计算主要聚类的统计信息
        largest_cluster_size = merged_clusters[0][2]
        largest_cluster_ratio = largest_cluster_size / total_samples
        
        # 自适应规则（与角度类似）：
        # 1. 如果最大聚类占比 > dominant_ratio_threshold，只取1个主导步长
        # 2. 如果最大聚类占比 > 50% 且第二大聚类占比 > secondary_ratio_threshold，取2个主导步长
        # 3. 如果最大聚类占比 < 50%，取所有聚类（最多max_dominant_patterns个）
        
        if largest_cluster_ratio > self.dominant_ratio_threshold:
            # 明显的主导模式，只取1个
            selected_clusters = merged_clusters[:1]
            print(f"检测到单一主导步长，占比: {largest_cluster_ratio:.1%}")
            
        elif largest_cluster_ratio > 0.5 and len(merged_clusters) > 1:
            second_cluster_ratio = merged_clusters[1][2] / total_samples
            if second_cluster_ratio > self.secondary_ratio_threshold:
                # 双主导模式
                selected_clusters = merged_clusters[:2]
                print(f"检测到双主导步长，主模式占比: {largest_cluster_ratio:.1%}，次模式占比: {second_cluster_ratio:.1%}")
            else:
                # 虽然有两个聚类，但第二个太小，只取1个
                selected_clusters = merged_clusters[:1]
                print(f"检测到单一主导步长（次模式太小），主模式占比: {largest_cluster_ratio:.1%}")
                
        else:
            # 多模式或模式不明显，取前max_dominant_patterns个
            selected_clusters = merged_clusters[:min(self.max_dominant_patterns, len(merged_clusters))]
            print(f"检测到多模式或模式不明显，选择前{len(selected_clusters)}个步长聚类")
        
        # 返回选中的主导步长和置信度
        dominant_steps = [(step, conf) for step, conf, _ in selected_clusters]
        confidences = [conf for _, conf, _ in selected_clusters]
        
        return dominant_steps, confidences
    
    def merge_similar_step_clusters(self, clusters_info: List[Tuple[float, float, int]]) -> List[Tuple[float, float, int]]:
        """合并相似的步长聚类"""
        if len(clusters_info) <= 1:
            return clusters_info
        
        # 按步长排序
        clusters_info.sort(key=lambda x: x[0])
        
        merged_clusters = []
        i = 0
        
        while i < len(clusters_info):
            current_step, current_conf, current_size = clusters_info[i]
            merged_steps = [current_step]
            merged_confs = [current_conf]
            merged_sizes = [current_size]
            
            # 检查后续的聚类是否与当前聚类相似
            j = i + 1
            while j < len(clusters_info):
                next_step, next_conf, next_size = clusters_info[j]
                
                # 计算步长差异（使用比例差异）
                if current_step > 0:
                    step_ratio = next_step / current_step
                    step_diff = abs(step_ratio - 1)
                else:
                    step_diff = float('inf')
                
                # 如果步长差异小于阈值，合并聚类
                if step_diff <= self.step_tolerance:
                    merged_steps.append(next_step)
                    merged_confs.append(next_conf)
                    merged_sizes.append(next_size)
                    j += 1
                else:
                    break
            
            # 计算合并后的聚类中心步长
            if len(merged_steps) > 1:
                # 使用加权平均计算合并后的步长
                total_size = sum(merged_sizes)
                weighted_step = sum(step * size for step, size in zip(merged_steps, merged_sizes)) / total_size
                weighted_conf = sum(conf * size for conf, size in zip(merged_confs, merged_sizes)) / total_size
                merged_size = total_size
            else:
                weighted_step = current_step
                weighted_conf = current_conf
                merged_size = current_size
            
            merged_clusters.append((weighted_step, weighted_conf, merged_size))
            i = j
        
        return merged_clusters
    
    def identify_pattern_participation(self, angle_step_pairs: List[Dict], 
                                     dominant_angles: List[Tuple[float, float]], 
                                     dominant_steps: List[Tuple[float, float]]) -> Dict:
        """识别每个点对主导模式的参与情况，支持多个主导角度和步长"""
        point_participation = {}
        
        # 初始化点参与哈希表
        for pair_info in angle_step_pairs:
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
                    'min_dominant_distance': float('inf'),
                    'dominant_connections': [],
                    'angle_score': 0,
                    'step_score': 0,
                    'best_angle_match': None,
                    'best_step_match': None
                }
            
            if point_key2 not in point_participation:
                point_participation[point_key2] = {
                    'participated': False,
                    'point': pair_info['pos2'],
                    'frame_id': frame2,
                    'point_idx': pos2_idx,
                    'min_dominant_distance': float('inf'),
                    'dominant_connections': [],
                    'angle_score': 0,
                    'step_score': 0,
                    'best_angle_match': None,
                    'best_step_match': None
                }
        
        # 评估每个点对的模式匹配度
        for pair_info in angle_step_pairs:
            angle = pair_info['angle']
            step_size = pair_info['step_size']
            distance = pair_info['distance']
            
            # 计算角度匹配分数（与所有主导角度比较，取最佳匹配）
            best_angle_score = 0
            best_angle_match = None
            if self.use_angle_clustering and dominant_angles:
                for dominant_angle, _ in dominant_angles:
                    angle_diff = min(abs(angle - dominant_angle), 
                                   abs(angle - dominant_angle + 360),
                                   abs(angle - dominant_angle - 360))
                    angle_score = max(0, 1 - (angle_diff / self.angle_tolerance))
                    if angle_score > best_angle_score:
                        best_angle_score = angle_score
                        best_angle_match = dominant_angle
            else:
                # 如果不使用角度聚类，给所有角度一个默认分数
                best_angle_score = 1.0
                best_angle_match = angle
            
            # 计算步长匹配分数（与所有主导步长比较，取最佳匹配）
            best_step_score = 0
            best_step_match = None
            if self.use_step_clustering and dominant_steps and step_size is not None and step_size > 0:
                for dominant_step, _ in dominant_steps:
                    if dominant_step > 0:
                        step_ratio = step_size / dominant_step
                        if 0.9 <= step_ratio <= 1.1:  # 使用更宽松的步长匹配范围，参考angle_distance_processor
                            step_score = max(0, 1 - abs(step_ratio - 1) / 0.1)
                            if step_score > best_step_score:
                                best_step_score = step_score
                                best_step_match = dominant_step
            else:
                # 如果不使用步长聚类，给所有步长一个默认分数
                best_step_score = 1.0
                best_step_match = step_size
            
            # 根据启用的聚类类型调整综合评分
            if self.use_angle_clustering and self.use_step_clustering:
                # 使用角度和步长的加权平均
                combined_score = (best_angle_score * self.angle_weight + best_step_score * self.step_weight)
                threshold = 0.3  # 降低阈值，采用更保守的过滤策略
            elif self.use_angle_clustering:
                # 只使用角度评分
                combined_score = best_angle_score
                threshold = 0.4  # 降低阈值
            elif self.use_step_clustering:
                # 只使用步长评分
                combined_score = best_step_score
                threshold = 0.4  # 降低阈值
            else:
                # 都不使用，给默认分数
                combined_score = 1.0
                threshold = 0.3
            
            # 如果综合分数足够高，标记为参与主导模式
            if combined_score >= threshold:
                frame1, pos1_idx = pair_info['frame1'], pair_info['pos1_idx']
                frame2, pos2_idx = pair_info['frame2'], pair_info['pos2_idx']
                
                point_key1 = (frame1, pos1_idx)
                point_key2 = (frame2, pos2_idx)
                
                # 更新点1的信息
                if point_key1 in point_participation:
                    point_participation[point_key1]['participated'] = True
                    point_participation[point_key1]['min_dominant_distance'] = min(
                        point_participation[point_key1]['min_dominant_distance'], distance
                    )
                    point_participation[point_key1]['angle_score'] = max(
                        point_participation[point_key1]['angle_score'], best_angle_score
                    )
                    point_participation[point_key1]['step_score'] = max(
                        point_participation[point_key1]['step_score'], best_step_score
                    )
                    point_participation[point_key1]['best_angle_match'] = best_angle_match
                    point_participation[point_key1]['best_step_match'] = best_step_match
                    point_participation[point_key1]['dominant_connections'].append({
                        'connected_point': point_key2,
                        'distance': distance,
                        'angle': angle,
                        'step_size': step_size,
                        'combined_score': combined_score,
                        'matched_angle': best_angle_match,
                        'matched_step': best_step_match
                    })
                
                # 更新点2的信息
                if point_key2 in point_participation:
                    point_participation[point_key2]['participated'] = True
                    point_participation[point_key2]['min_dominant_distance'] = min(
                        point_participation[point_key2]['min_dominant_distance'], distance
                    )
                    point_participation[point_key2]['angle_score'] = max(
                        point_participation[point_key2]['angle_score'], best_angle_score
                    )
                    point_participation[point_key2]['step_score'] = max(
                        point_participation[point_key2]['step_score'], best_step_score
                    )
                    point_participation[point_key2]['best_angle_match'] = best_angle_match
                    point_participation[point_key2]['best_step_match'] = best_step_match
                    point_participation[point_key2]['dominant_connections'].append({
                        'connected_point': point_key1,
                        'distance': distance,
                        'angle': angle,
                        'step_size': step_size,
                        'combined_score': combined_score,
                        'matched_angle': best_angle_match,
                        'matched_step': best_step_match
                    })
        
        # 根据距离阈值重新评估参与状态（参考angle_distance_processor的做法）
        distance_threshold = 120  # 使用更大的距离阈值
        for point_key, point_info in point_participation.items():
            if point_info['participated']:
                # 检查该点与所有主导模式连接点的距离
                distances = [connection['distance'] for connection in point_info['dominant_connections']]
                min_distance = min(distances) if distances else float('inf')
                
                # 如果最短距离大于阈值，则不算参与
                if min_distance > distance_threshold:
                    point_info['participated'] = False
                    print(f"    点 {point_key} 最短距离 {min_distance:.1f} 大于阈值 {distance_threshold}，标记为非参与")
                else:
                    print(f"    点 {point_key} {point_info['point']} 参与主导模式，连接距离: {[f'{d:.1f}' for d in distances]}, 最短距离: {min_distance:.1f}")
                    print(f"    角度分数: {point_info['angle_score']:.3f}, 步长分数: {point_info['step_score']:.3f}")
        
        return point_participation
    
    def generate_points_from_pattern(self, pos1: List[float], pos2: List[float], 
                                   dominant_angle: float, dominant_step: float, 
                                   frame1: int, frame2: int, total_frames: int) -> List[Tuple[int, List[float]]]:
        """基于主导模式生成缺失帧的点"""
        points = []
        
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
                extrapolated_pos = self.extrapolate_position_by_pattern(
                    pos1, dominant_angle, dominant_step, -frame_gap
                )
                points.append((frame, extrapolated_pos))
                
            elif frame > frame2:
                # 在第二个检测点之后，使用外推
                frame_gap = frame - frame2
                extrapolated_pos = self.extrapolate_position_by_pattern(
                    pos2, dominant_angle, dominant_step, frame_gap
                )
                points.append((frame, extrapolated_pos))
        
        return points
    
    def extrapolate_position_by_pattern(self, base_pos: List[float], dominant_angle: float, 
                                      dominant_step: float, frame_gap: int) -> List[float]:
        """基于主导模式外推位置"""
        angle_rad = math.radians(dominant_angle)
        step_dx = dominant_step * math.cos(angle_rad)
        step_dy = dominant_step * math.sin(angle_rad)
        
        total_dx = step_dx * frame_gap
        total_dy = step_dy * frame_gap
        
        return [base_pos[0] + total_dx, base_pos[1] + total_dy]
    
    def complete_trajectory_with_patterns(self, frames_data: Dict[int, Dict], 
                                       dominant_angles: List[Tuple[float, float]], 
                                       dominant_steps: List[Tuple[float, float]],
                                       angle_step_pairs: List[Dict]) -> Tuple[Dict[int, Dict], List[Tuple[int, List[float]]]]:
        """基于多个主导模式进行轨迹补全"""
        all_generated_points = []
        generated_points_from_pairs = []
        
        # 对每个符合主导模式的点对生成点
        for pair_info in angle_step_pairs:
            angle = pair_info['angle']
            step_size = pair_info['step_size']
            
            # 检查是否属于任一主导模式
            angle_match = False
            step_match = False
            best_angle = None
            best_step = None
            
            # 检查角度匹配（与所有主导角度比较，取最佳匹配）
            best_angle_score = 0
            if self.use_angle_clustering and dominant_angles:
                for dominant_angle, _ in dominant_angles:
                    angle_diff = min(abs(angle - dominant_angle), 
                                   abs(angle - dominant_angle + 360),
                                   abs(angle - dominant_angle - 360))
                    if angle_diff <= self.angle_tolerance:
                        angle_score = 1 - (angle_diff / self.angle_tolerance)
                        if angle_score > best_angle_score:
                            best_angle_score = angle_score
                            angle_match = True
                            best_angle = dominant_angle
            else:
                # 如果不使用角度聚类，默认匹配
                angle_match = True
                best_angle = angle
                best_angle_score = 1.0
            
            # 检查步长匹配（与所有主导步长比较，取最佳匹配）
            best_step_score = 0
            if self.use_step_clustering and dominant_steps and step_size is not None and step_size > 0:
                for dominant_step, _ in dominant_steps:
                    if dominant_step > 0:
                        step_ratio = step_size / dominant_step
                        if 0.9 <= step_ratio <= 1.1:  # 使用更宽松的步长匹配范围
                            step_score = max(0, 1 - abs(step_ratio - 1) / 0.1)
                            if step_score > best_step_score:
                                best_step_score = step_score
                                step_match = True
                                best_step = dominant_step
            else:
                # 如果不使用步长聚类，默认匹配
                step_match = True
                best_step = step_size
                best_step_score = 1.0
            
            # 根据启用的聚类类型调整综合评分和匹配条件
            if self.use_angle_clustering and self.use_step_clustering:
                # 使用角度和步长的加权平均
                combined_score = (best_angle_score * self.angle_weight + best_step_score * self.step_weight)
                should_generate = angle_match and step_match and combined_score >= 0.3  # 降低阈值
            elif self.use_angle_clustering:
                # 只使用角度评分
                combined_score = best_angle_score
                should_generate = angle_match and combined_score >= 0.4  # 降低阈值
            elif self.use_step_clustering:
                # 只使用步长评分
                combined_score = best_step_score
                should_generate = step_match and combined_score >= 0.4  # 降低阈值
            else:
                # 都不使用，默认生成
                combined_score = 1.0
                should_generate = True
            
            # 如果综合分数足够高，则生成点
            if should_generate:
                # 使用最佳匹配的主导模式生成点
                generated_points = self.generate_points_from_pattern(
                    pair_info['pos1'], pair_info['pos2'], 
                    best_angle, best_step,
                    pair_info['frame1'], pair_info['frame2'],
                    len(frames_data)
                )
                all_generated_points.extend(generated_points)
                generated_points_from_pairs.extend(generated_points)
        
        # 外推轨迹，填补间隙（使用第一个主导模式）
        if (self.use_angle_clustering and dominant_angles) and (self.use_step_clustering and dominant_steps):
            # 两者都启用，使用第一个主导模式
            extrapolated_points = self.extrapolate_trajectory_with_patterns(
                frames_data, dominant_angles[0][0], dominant_steps[0][0]
            )
            all_generated_points.extend(extrapolated_points)
        elif self.use_angle_clustering and dominant_angles:
            # 只启用角度聚类，使用默认步长
            default_step = 10.0  # 默认步长
            extrapolated_points = self.extrapolate_trajectory_with_patterns(
                frames_data, dominant_angles[0][0], default_step
            )
            all_generated_points.extend(extrapolated_points)
        elif self.use_step_clustering and dominant_steps:
            # 只启用步长聚类，使用默认角度
            default_angle = 0.0  # 默认角度
            extrapolated_points = self.extrapolate_trajectory_with_patterns(
                frames_data, default_angle, dominant_steps[0][0]
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
    
    def extrapolate_trajectory_with_patterns(self, frames_data: Dict[int, Dict], 
                                          dominant_angle: float, dominant_step: float) -> List[Tuple[int, List[float]]]:
        """基于主导模式外推轨迹"""
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
            
            for frame in missing_frames:
                frame_gap = first_detected - frame
                extrapolated_pos = self.extrapolate_position_by_pattern(
                    first_pos, dominant_angle, dominant_step, -frame_gap
                )
                extrapolated_points.append((frame, extrapolated_pos))
        
        # 填补序列结尾的间隙
        last_detected = detected_frames[-1]
        if last_detected < max(frames):
            last_pos = frames_data[last_detected]['coords'][0]
            missing_frames = range(last_detected + 1, max(frames) + 1)
            
            for frame in missing_frames:
                frame_gap = frame - last_detected
                extrapolated_pos = self.extrapolate_position_by_pattern(
                    last_pos, dominant_angle, dominant_step, frame_gap
                )
                extrapolated_points.append((frame, extrapolated_pos))
        
        return extrapolated_points 
    
    def filter_outliers_by_patterns(self, frames_data: Dict[int, Dict], 
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
    
    def process_sequence(self, sequence_id: int, frames_data: Dict[int, Dict], 
                        save_visualization: bool = True) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
        """处理单个序列，返回轨迹补全和异常点筛选的结果"""
        print(f"开始自适应处理序列 {sequence_id}...")
        print(f"  角度聚类: {'启用' if self.use_angle_clustering else '禁用'}")
        print(f"  步长聚类: {'启用' if self.use_step_clustering else '禁用'}")
        if self.use_angle_clustering and self.use_step_clustering:
            print(f"  权重设置: 角度={self.angle_weight:.2f}, 步长={self.step_weight:.2f}")
        
        # 收集所有点对的角度和步长
        all_angles, all_steps, angle_step_pairs = self.collect_all_angle_step_pairs(frames_data)
        
        if not all_angles or not all_steps:
            print(f"序列 {sequence_id} 没有足够的点对数据")
            return frames_data, frames_data
        
        print(f"序列 {sequence_id} 计算了 {len(all_angles)} 个角度和步长对")
        
        # 聚类分析角度
        dominant_angles = []
        angle_confidences = []
        clusters_info = []
        merged_clusters = []
        
        if self.use_angle_clustering:
            dominant_angles, angle_confidences, clusters_info, merged_clusters = self.cluster_angles(all_angles)
            
            # 可视化角度聚类过程
            if save_visualization and clusters_info:
                try:
                    self.visualize_angle_clustering(all_angles, sequence_id, clusters_info, merged_clusters)
                except Exception as e:
                    print(f"可视化角度聚类时出错: {e}")
        else:
            print("  跳过角度聚类分析")
        
        # 聚类分析步长
        dominant_steps = []
        step_confidences = []
        
        if self.use_step_clustering:
            dominant_steps, step_confidences = self.cluster_steps(all_steps)
        else:
            print("  跳过步长聚类分析")
        
        # 输出主导模式信息
        if self.use_angle_clustering and dominant_angles:
            print(f"主导角度: {dominant_angles[0][0]:.1f}° (置信度: {dominant_angles[0][1]:.3f})")
            if len(dominant_angles) > 1:
                print(f"次主导角度: {dominant_angles[1][0]:.1f}° (置信度: {dominant_angles[1][1]:.3f})")
        elif self.use_angle_clustering:
            print(f"主导角度: 未发现 (置信度: {angle_confidences[0] if angle_confidences else 0.0:.3f})")
            
        if self.use_step_clustering and dominant_steps:
            print(f"主导步长: {dominant_steps[0][0]:.1f} (置信度: {dominant_steps[0][1]:.3f})")
            if len(dominant_steps) > 1:
                print(f"次主导步长: {dominant_steps[1][0]:.1f} (置信度: {dominant_steps[1][1]:.3f})")
        elif self.use_step_clustering:
            print(f"主导步长: 未发现 (置信度: {step_confidences[0] if step_confidences else 0.0:.3f})")
        
        # 检查置信度
        if (self.use_angle_clustering and angle_confidences and angle_confidences[0] < self.confidence_threshold) or \
           (self.use_step_clustering and step_confidences and step_confidences[0] < self.confidence_threshold):
            print(f"警告：主导模式置信度较低，可能影响处理效果")
        
        # 如果无法确定主导模式且启用了聚类，返回原始数据
        if ((self.use_angle_clustering and not dominant_angles) or 
            (self.use_step_clustering and not dominant_steps)):
            print(f"无法确定主导模式，返回原始数据")
            return frames_data, frames_data
        
        # 识别模式参与情况
        point_participation = self.identify_pattern_participation(
            angle_step_pairs, dominant_angles, dominant_steps
        )
        
        # 步骤1: 轨迹补全
        completed_sequence, generated_points = self.complete_trajectory_with_patterns(
            frames_data, dominant_angles, dominant_steps, angle_step_pairs
        )
        
        # 步骤2: 异常点筛选
        filtered_sequence = self.filter_outliers_by_patterns(
            completed_sequence, point_participation
        )
        
        return completed_sequence, filtered_sequence
    
    def process_dataset(self, predictions: Dict, save_visualization: bool = True) -> Dict:
        """处理整个数据集"""
        print("开始自适应角度距离处理...")
        
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
        
        print(f"自适应角度距离处理完成，处理了 {len(processed_predictions)} 张图像")
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

    def visualize_angle_clustering(self, angles: List[float], sequence_id: int, 
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


def main():
    """主函数，演示自适应角度距离处理"""
    import argparse
    import os
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='自适应角度距离处理器')
    parser.add_argument('--pred_path', type=str, default='results/spotgeov2-IRSTD/WTNet/predictions_8807.json',
                       help='预测结果文件路径')
    parser.add_argument('--gt_path', type=str, default='datasets/spotgeov2-IRSTD/test_anno.json',
                       help='真实标注文件路径')
    parser.add_argument('--output_path', type=str, default='results/spotgeov2/WTNet/adaptive_processed_predictions.json',
                       help='输出文件路径')
    parser.add_argument('--base_distance_threshold', type=float, default=1000,
                       help='基础距离阈值')
    parser.add_argument('--min_cluster_size', type=int, default=1,
                       help='最小聚类大小')
    parser.add_argument('--angle_cluster_eps', type=float, default=15.0,  # 调整为15度，参考angle_distance_processor
                       help='角度聚类半径（度）')
    parser.add_argument('--step_cluster_eps', type=float, default=0.2,  # 调整为0.2，参考angle_distance_processor
                       help='步长聚类半径（比例）')
    parser.add_argument('--confidence_threshold', type=float, default=0.35,
                       help='主导模式置信度阈值')
    parser.add_argument('--angle_tolerance', type=float, default=15.0,  # 调整为15度
                       help='角度容差（度）')
    parser.add_argument('--step_tolerance', type=float, default=0.2,  # 调整为0.2
                       help='步长容差（比例）')
    parser.add_argument('--point_distance_threshold', type=float, default=5.0,
                       help='重合点过滤阈值')
    parser.add_argument('--dominant_ratio_threshold', type=float, default=0.7,
                       help='主导模式占比阈值（0.7表示70%）')
    parser.add_argument('--secondary_ratio_threshold', type=float, default=0.15,
                       help='次主导模式占比阈值（0.2表示20%）')
    parser.add_argument('--max_dominant_patterns', type=int, default=2,
                       help='最大主导模式数量')
    parser.add_argument('--no_visualization', action='store_true',
                       help='不保存可视化结果')
    parser.add_argument('--use_angle_clustering', action='store_true', default=True,
                       help='启用角度聚类')
    parser.add_argument('--no_angle_clustering', action='store_true',
                       help='禁用角度聚类')
    parser.add_argument('--use_step_clustering', action='store_true', default=True,
                       help='启用步长聚类')
    parser.add_argument('--no_step_clustering', action='store_true',
                       help='禁用步长聚类')
    parser.add_argument('--angle_weight', type=float, default=0.5,  # 调整为0.5，平衡角度和步长
                       help='角度在综合评分中的权重')
    parser.add_argument('--step_weight', type=float, default=0.5,  # 调整为0.5，平衡角度和步长
                       help='步长在综合评分中的权重')
    
    args = parser.parse_args()
    
    # 处理角度和步长聚类的启用/禁用逻辑
    use_angle_clustering = args.use_angle_clustering and not args.no_angle_clustering
    use_step_clustering = args.use_step_clustering and not args.no_step_clustering
    
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
    
    # 创建自适应角度距离处理器
    processor = AdaptiveAngleDistanceProcessor(
        base_distance_threshold=args.base_distance_threshold,
        min_track_length=1,
        expected_sequence_length=5,
        min_cluster_size=args.min_cluster_size,
        angle_cluster_eps=args.angle_cluster_eps,
        step_cluster_eps=args.step_cluster_eps,
        confidence_threshold=args.confidence_threshold,
        angle_tolerance=args.angle_tolerance,
        step_tolerance=args.step_tolerance,
        point_distance_threshold=args.point_distance_threshold,
        dominant_ratio_threshold=args.dominant_ratio_threshold,
        secondary_ratio_threshold=args.secondary_ratio_threshold,
        max_dominant_patterns=args.max_dominant_patterns,
        use_angle_clustering=use_angle_clustering,
        use_step_clustering=use_step_clustering,
        angle_weight=args.angle_weight,
        step_weight=args.step_weight
    )
    
    # 打印处理模式
    print(f"\n=== 处理模式配置 ===")
    print(f"角度聚类: {'启用' if use_angle_clustering else '禁用'}")
    print(f"步长聚类: {'启用' if use_step_clustering else '禁用'}")
    if use_angle_clustering and use_step_clustering:
        print(f"权重设置: 角度={args.angle_weight:.2f}, 步长={args.step_weight:.2f}")
    elif use_angle_clustering:
        print("模式: 仅使用角度聚类")
    elif use_step_clustering:
        print("模式: 仅使用步长聚类")
    else:
        print("模式: 不使用聚类（仅进行基础处理）")
    
    # 进行自适应角度距离处理
    processed_predictions = processor.process_dataset(
        original_predictions, save_visualization=not args.no_visualization
    )
    
    # 评估改善效果
    try:
        improvement = processor.evaluate_improvement(original_predictions, processed_predictions, ground_truth)
        
        # 打印结果
        print("\n=== 自适应角度距离处理效果评估 ===")
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
        print(f"\n自适应角度距离处理结果已保存到: {args.output_path}")
    except Exception as e:
        print(f"保存结果时出错: {e}")

if __name__ == '__main__':
    main() 