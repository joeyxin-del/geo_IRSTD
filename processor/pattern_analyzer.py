from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.neighbors import kneighbors_graph
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from collections import Counter
from utils import GeometryUtils
# 新增导入用于von Mises分布和异常值处理
from scipy.stats import vonmises
from scipy.optimize import minimize
from sklearn.mixture import BayesianGaussianMixture

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
        distance_threshold = 130
        for point_key, point_info in point_participation.items():
            if point_info['participated'] and point_info['min_dominant_distance'] > distance_threshold:
                point_info['participated'] = False


class ClusteringPatternAnalyzer(PatternAnalyzer):
    """基于角度聚类的模式分析器 - 继承自PatternAnalyzer"""
    
    def __init__(self, angle_tolerance: float = 5.0, min_angle_count: int = 2, 
                 min_step_count: int = 1, angle_cluster_eps: float = 4.0,
                 min_cluster_size: int = 1, confidence_threshold: float = 0.12):
        """
        初始化基于角度聚类的模式分析器
        
        Args:
            angle_tolerance: 角度容差（度）
            min_angle_count: 最小角度数量
            min_step_count: 最小步长数量
            angle_cluster_eps: 角度聚类半径（度）
            min_cluster_size: 最小聚类大小
            confidence_threshold: 主导模式置信度阈值（降低默认值）
        """
        super().__init__(angle_tolerance, min_angle_count, min_step_count)
        self.angle_cluster_eps = angle_cluster_eps
        self.min_cluster_size = min_cluster_size
        self.confidence_threshold = confidence_threshold
    
    def _find_dominant_angles(self, all_angles: List[float]) -> List[Tuple[float, int]]:
        """
        通过聚类的方式找到主导角度
        
        Args:
            all_angles: 所有角度列表
            
        Returns:
            主导角度列表，每个元素为(角度, 数量)的元组
        """
        if len(all_angles) < self.min_cluster_size:
            return []
        
        # 将角度转换为二维坐标（考虑角度的周期性）
        angles_rad = np.array(all_angles) * np.pi / 180.0
        X = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 动态调整聚类参数
        eps = self._adjust_cluster_eps(len(all_angles))
        min_samples = max(2, min(self.min_cluster_size, len(all_angles) // 10))
        
        # 使用DBSCAN聚类
        clustering = DBSCAN(
            eps=eps, 
            min_samples=min_samples
        ).fit(X_scaled)
        
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
                cluster_angles = np.array(all_angles)[cluster_mask]
                
                # 计算聚类中心角度
                cluster_angles_rad = cluster_angles * np.pi / 180.0
                mean_cos = np.mean(np.cos(cluster_angles_rad))
                mean_sin = np.mean(np.sin(cluster_angles_rad))
                dominant_angle = math.degrees(math.atan2(mean_sin, mean_cos))
                
                if dominant_angle < 0:
                    dominant_angle += 360
                
                confidence = cluster_size / len(all_angles)
                
                if confidence >= self.confidence_threshold:
                    clusters_info.append((dominant_angle, cluster_size, confidence))
        
        # 按聚类大小排序
        clusters_info.sort(key=lambda x: x[1], reverse=True)
        
        # 转换为返回格式
        dominant_angles = [(angle, count) for angle, count, _ in clusters_info]
        
        # 如果没有找到聚类，回退到原始方法
        if not dominant_angles:
            return self._fallback_find_dominant_angles(all_angles)
        
        return dominant_angles
    
    def _adjust_cluster_eps(self, num_angles: int) -> float:
        """
        根据角度数量动态调整聚类半径
        """
        # 基础半径
        base_eps = self.angle_cluster_eps / 180.0 * np.pi
        
        # 根据数据量调整
        if num_angles < 10:
            return base_eps * 2.0  # 数据少时大幅放宽条件
        elif num_angles < 50:
            return base_eps * 1.5  # 数据较少时放宽条件
        elif num_angles > 200:
            return base_eps * 0.7  # 数据多时收紧条件
        elif num_angles > 100:
            return base_eps * 0.8  # 数据较多时稍微收紧条件
        else:
            return base_eps
    
    def _fallback_find_dominant_angles(self, all_angles: List[float]) -> List[Tuple[float, int]]:
        """
        当聚类失败时的回退方法，使用原始的分箱方法
        """
        angle_counter = Counter()
        for angle in all_angles:
            angle_key = round(angle / self.angle_tolerance) * self.angle_tolerance
            angle_counter[angle_key] += 1
        
        dominant_angles = []
        for angle, count in angle_counter.most_common():
            if count >= self.min_angle_count:
                dominant_angles.append((angle, count))
        
        return dominant_angles
    
    def cluster_angles_with_info(self, all_angles: List[float]) -> Tuple[List[Tuple[float, int]], List[Tuple[float, float, int]]]:
        """
        对角度进行聚类并返回详细信息
        
        Args:
            all_angles: 所有角度列表
            
        Returns:
            (主导角度列表, 聚类详细信息列表)
        """
        if len(all_angles) < self.min_cluster_size:
            return [], []
        
        # 将角度转换为二维坐标（考虑角度的周期性）
        angles_rad = np.array(all_angles) * np.pi / 180.0
        X = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 动态调整聚类参数
        eps = self._adjust_cluster_eps(len(all_angles))
        min_samples = max(2, min(self.min_cluster_size, len(all_angles) // 10))
        
        # 使用DBSCAN聚类
        clustering = DBSCAN(
            eps=eps, 
            min_samples=min_samples
        ).fit(X_scaled)
        
        labels = clustering.labels_
        unique_labels = set(labels)
        
        # 收集所有聚类信息
        clusters_info = []
        dominant_angles = []
        
        for label in unique_labels:
            if label == -1:  # 噪声点
                continue
            
            cluster_mask = labels == label
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size >= self.min_cluster_size:
                cluster_angles = np.array(all_angles)[cluster_mask]
                
                # 计算聚类中心角度
                cluster_angles_rad = cluster_angles * np.pi / 180.0
                mean_cos = np.mean(np.cos(cluster_angles_rad))
                mean_sin = np.mean(np.sin(cluster_angles_rad))
                dominant_angle = math.degrees(math.atan2(mean_sin, mean_cos))
                
                if dominant_angle < 0:
                    dominant_angle += 360
                
                confidence = cluster_size / len(all_angles)
                
                clusters_info.append((dominant_angle, confidence, cluster_size))
                
                if confidence >= self.confidence_threshold:
                    dominant_angles.append((dominant_angle, cluster_size))
        
        # 按聚类大小排序
        dominant_angles.sort(key=lambda x: x[1], reverse=True)
        clusters_info.sort(key=lambda x: x[2], reverse=True)
        
        return dominant_angles, clusters_info


class GMMPatternAnalyzer(PatternAnalyzer):
    """基于高斯混合模型(GMM)和von Mises分布的角度聚类模式分析器 - 继承自PatternAnalyzer"""
    
    def __init__(self, angle_tolerance: float = 5.0, min_angle_count: int = 2, 
                 min_step_count: int = 1, max_components: int = 10,
                 min_cluster_size: int = 1, confidence_threshold: float = 0.08,
                 covariance_type: str = 'full', random_state: int = 42,
                 use_von_mises: bool = True, outlier_eps: float = 1.5,
                 outlier_min_samples: int = 1):
        """
        初始化基于GMM的角度聚类模式分析器
        
        Args:
            angle_tolerance: 角度容差（度）
            min_angle_count: 最小角度数量
            min_step_count: 最小步长数量
            max_components: 最大高斯组件数量
            min_cluster_size: 最小聚类大小
            confidence_threshold: 主导模式置信度阈值（降低到0.08以提高recall）
            covariance_type: 协方差类型 ('full', 'tied', 'diag', 'spherical')
            random_state: 随机种子
            use_von_mises: 是否使用von Mises分布（推荐True）
            outlier_eps: DBSCAN异常值检测的eps参数（增加到1.5，更宽松的过滤）
            outlier_min_samples: DBSCAN异常值检测的最小样本数（降低到1，更宽松）
        """
        super().__init__(angle_tolerance, min_angle_count, min_step_count)
        self.max_components = max_components
        self.min_cluster_size = min_cluster_size
        self.confidence_threshold = confidence_threshold
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.use_von_mises = use_von_mises
        self.outlier_eps = outlier_eps
        self.outlier_min_samples = outlier_min_samples
    
    def _find_dominant_angles(self, all_angles: List[float]) -> List[Tuple[float, int]]:
        """
        通过GMM聚类或von Mises混合模型的方式找到主导角度
        
        Args:
            all_angles: 所有角度列表
            
        Returns:
            主导角度列表，每个元素为(角度, 数量)的元组
        """
        # 检查样本数量是否足够
        if len(all_angles) < 2:
            print(f"Clustering skipped: insufficient angles ({len(all_angles)} < 2)")
            return self._fallback_find_dominant_angles(all_angles)
        
        if len(all_angles) < self.min_cluster_size:
            print(f"Clustering skipped: insufficient angles for clustering ({len(all_angles)} < {self.min_cluster_size})")
            return self._fallback_find_dominant_angles(all_angles)
        
        # 异常值过滤
        filtered_angles = self._filter_outliers(all_angles)
        if len(filtered_angles) < self.min_cluster_size:
            print(f"After outlier filtering, insufficient angles ({len(filtered_angles)} < {self.min_cluster_size})")
            return self._fallback_find_dominant_angles(all_angles)
        
        # 尝试不同的聚类方法，如果失败则降低阈值重试
        dominant_angles = []
        confidence_thresholds = [self.confidence_threshold, 0.05, 0.03, 0.02]  # 逐步降低阈值
        
        for threshold in confidence_thresholds:
            try:
                if self.use_von_mises:
                    # 使用von Mises混合模型
                    dominant_angles = self._von_mises_cluster_angles_with_threshold(filtered_angles, threshold)
                else:
                    # 使用改进的GMM聚类
                    dominant_angles = self._improved_gmm_cluster_angles_with_threshold(filtered_angles, threshold)
                
                if dominant_angles:
                    print(f"Clustering successful with threshold {threshold}")
                    break
                    
            except Exception as e:
                print(f"Clustering failed with threshold {threshold}: {e}")
                continue
        
        # 如果没有找到聚类，回退到原始方法
        if not dominant_angles:
            print("All clustering methods failed, falling back to binning method")
            return self._fallback_find_dominant_angles(all_angles)
        
        return dominant_angles
    
    def _filter_outliers(self, all_angles: List[float]) -> List[float]:
        """
        使用DBSCAN过滤异常值（优化版本，更宽松的过滤）
        
        Args:
            all_angles: 所有角度列表
            
        Returns:
            过滤后的角度列表
        """
        if len(all_angles) < self.outlier_min_samples:
            return all_angles
        
        # 如果数据量较小，直接返回原始数据
        if len(all_angles) < 10:
            return all_angles
        
        # 将角度转换为二维坐标（考虑角度的周期性）
        angles_rad = np.array(all_angles) * np.pi / 180.0
        X = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 使用更宽松的DBSCAN参数
        eps = max(self.outlier_eps, 0.5)  # 确保eps至少为0.5
        min_samples = max(self.outlier_min_samples, 1)  # 确保min_samples至少为1
        
        # 使用DBSCAN检测异常值
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        # 保留非异常值（标签不为-1的点）
        filtered_indices = labels != -1
        filtered_angles = [all_angles[i] for i in range(len(all_angles)) if filtered_indices[i]]
        
        # 如果过滤后数据量减少太多，返回原始数据
        if len(filtered_angles) < len(all_angles) * 0.7:  # 如果过滤掉超过30%的数据
            print(f"Outlier filtering too aggressive ({len(filtered_angles)}/{len(all_angles)}), returning original data")
            return all_angles
        
        if len(filtered_angles) < len(all_angles):
            print(f"Filtered {len(all_angles) - len(filtered_angles)} outliers from {len(all_angles)} angles")
        
        return filtered_angles
    
    def _von_mises_cluster_angles_with_threshold(self, all_angles: List[float], confidence_threshold: float) -> List[Tuple[float, int]]:
        """
        使用von Mises混合模型对角度进行聚类（支持自定义阈值）
        
        Args:
            all_angles: 所有角度列表
            confidence_threshold: 置信度阈值
            
        Returns:
            主导角度列表
        """
        n_samples = len(all_angles)
        
        # 确定最优的组件数量
        best_n_components = self._find_optimal_von_mises_components(all_angles, n_samples)
        
        if best_n_components is None or best_n_components < 1:
            return []
        
        try:
            # 使用von Mises混合模型
            clusters_info = self._fit_von_mises_mixture_with_threshold(all_angles, best_n_components, confidence_threshold)
            
            # 转换为返回格式
            dominant_angles = [(angle, count) for angle, count, _ in clusters_info]
            
            return dominant_angles
            
        except Exception as e:
            print(f"Von Mises clustering failed: {e}")
            return []
    
    def _fit_von_mises_mixture_with_threshold(self, all_angles: List[float], n_components: int, confidence_threshold: float) -> List[Tuple[float, int, float]]:
        """
        拟合von Mises混合模型（支持自定义阈值）
        
        Args:
            all_angles: 所有角度列表
            n_components: 组件数量
            confidence_threshold: 置信度阈值
            
        Returns:
            聚类信息列表，每个元素为(角度, 数量, 置信度)的元组
        """
        if len(all_angles) < n_components:
            # 如果样本数量少于组件数量，减少组件数量
            n_components = max(1, len(all_angles) // 2)
        
        angles_rad = np.array(all_angles) * np.pi / 180.0
        
        # 初始化参数
        weights = np.ones(n_components) / n_components
        means = np.linspace(0, 2*np.pi, n_components, endpoint=False)
        kappas = np.ones(n_components) * 2.0  # 集中度参数
        
        # 使用EM算法拟合von Mises混合模型
        max_iter = 100
        tolerance = 1e-6
        prev_log_likelihood = -np.inf
        
        for iteration in range(max_iter):
            try:
                # E步：计算后验概率
                responsibilities = self._compute_von_mises_responsibilities(angles_rad, means, kappas, weights)
                
                # 检查responsibilities是否有效
                if not np.all(np.isfinite(responsibilities)):
                    print(f"Invalid responsibilities in iteration {iteration}")
                    break
                
                # M步：更新参数
                weights, means, kappas = self._update_von_mises_parameters(angles_rad, responsibilities)
                
                # 检查参数是否有效
                if (not np.all(np.isfinite(weights)) or 
                    not np.all(np.isfinite(means)) or 
                    not np.all(np.isfinite(kappas))):
                    print(f"Invalid parameters in iteration {iteration}")
                    break
                
                # 计算对数似然
                current_log_likelihood = self._compute_von_mises_log_likelihood(angles_rad, means, kappas, weights)
                
                # 检查收敛
                if np.isfinite(current_log_likelihood) and np.isfinite(prev_log_likelihood):
                    if abs(current_log_likelihood - prev_log_likelihood) < tolerance:
                        break
                
                prev_log_likelihood = current_log_likelihood
                
            except Exception as e:
                print(f"Error in von Mises fitting iteration {iteration}: {e}")
                break
        
        # 收集聚类信息
        clusters_info = []
        try:
            for i in range(n_components):
                cluster_mask = responsibilities[:, i] > 0.5  # 硬分配
                cluster_size = np.sum(cluster_mask)
                
                if cluster_size >= self.min_cluster_size:
                    cluster_angles = np.array(all_angles)[cluster_mask]
                    cluster_probs = responsibilities[cluster_mask, i]
                    
                    if len(cluster_angles) > 0:
                        # 计算聚类中心角度（加权平均）
                        cluster_angles_rad = cluster_angles * np.pi / 180.0
                        weighted_cos = np.average(np.cos(cluster_angles_rad), weights=cluster_probs)
                        weighted_sin = np.average(np.sin(cluster_angles_rad), weights=cluster_probs)
                        dominant_angle = math.degrees(math.atan2(weighted_sin, weighted_cos))
                        
                        if dominant_angle < 0:
                            dominant_angle += 360
                        
                        # 计算置信度
                        avg_probability = np.mean(cluster_probs)
                        confidence = (cluster_size / len(all_angles)) * avg_probability
                        
                        if confidence >= confidence_threshold:
                            clusters_info.append((dominant_angle, cluster_size, confidence))
        except Exception as e:
            print(f"Error collecting cluster info: {e}")
        
        # 按聚类大小排序
        clusters_info.sort(key=lambda x: x[1], reverse=True)
        
        return clusters_info
    
    def _compute_von_mises_responsibilities(self, angles_rad: np.ndarray, 
                                          means: np.ndarray, kappas: np.ndarray, 
                                          weights: np.ndarray) -> np.ndarray:
        """
        计算von Mises混合模型的后验概率
        
        Args:
            angles_rad: 角度（弧度）
            means: 均值参数
            kappas: 集中度参数
            weights: 混合权重
            
        Returns:
            后验概率矩阵
        """
        n_samples = len(angles_rad)
        n_components = len(means)
        
        # 计算每个组件的概率密度
        log_probs = np.zeros((n_samples, n_components))
        
        for i in range(n_components):
            try:
                # 使用scipy的vonmises分布，添加数值稳定性检查
                kappa = max(kappas[i], 1e-6)  # 确保kappa不为零
                log_probs[:, i] = vonmises.logpdf(angles_rad, kappa, loc=means[i])
                
                # 处理无效值
                log_probs[:, i] = np.where(np.isfinite(log_probs[:, i]), 
                                         log_probs[:, i], -1e10)
            except Exception:
                # 如果vonmises计算失败，使用简单的余弦相似度
                log_probs[:, i] = kappa * np.cos(angles_rad - means[i]) - np.log(2 * np.pi)
        
        # 添加权重（确保权重为正）
        weights_safe = np.maximum(weights, 1e-10)
        log_probs += np.log(weights_safe)
        
        # 计算归一化常数（使用更稳定的方法）
        max_log_probs = np.max(log_probs, axis=1, keepdims=True)
        log_probs_shifted = log_probs - max_log_probs
        exp_probs = np.exp(log_probs_shifted)
        sum_exp_probs = np.sum(exp_probs, axis=1, keepdims=True)
        
        # 计算后验概率
        responsibilities = exp_probs / np.maximum(sum_exp_probs, 1e-10)
        
        return responsibilities
    
    def _update_von_mises_parameters(self, angles_rad: np.ndarray, 
                                   responsibilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        更新von Mises混合模型的参数
        
        Args:
            angles_rad: 角度（弧度）
            responsibilities: 后验概率矩阵
            
        Returns:
            更新后的权重、均值和集中度参数
        """
        n_samples, n_components = responsibilities.shape
        
        # 更新权重
        weights = np.sum(responsibilities, axis=0) / n_samples
        weights = np.maximum(weights, 1e-10)  # 确保权重不为零
        
        # 更新均值
        means = np.zeros(n_components)
        for i in range(n_components):
            # 计算加权平均角度
            weighted_cos = np.average(np.cos(angles_rad), weights=responsibilities[:, i])
            weighted_sin = np.average(np.sin(angles_rad), weights=responsibilities[:, i])
            means[i] = np.arctan2(weighted_sin, weighted_cos)
            if means[i] < 0:
                means[i] += 2 * np.pi
        
        # 更新集中度参数
        kappas = np.zeros(n_components)
        for i in range(n_components):
            try:
                # 使用最大似然估计更新kappa
                r_cos = np.average(np.cos(angles_rad - means[i]), weights=responsibilities[:, i])
                r_sin = np.average(np.sin(angles_rad - means[i]), weights=responsibilities[:, i])
                r = np.sqrt(r_cos**2 + r_sin**2)
                
                # 使用更稳定的kappa估计方法
                if r > 0.95:
                    # 当r接近1时，使用近似公式
                    kappas[i] = 1.0 / (1.0 - r)
                elif r > 0.5:
                    # 使用改进的kappa估计公式
                    kappas[i] = r * (2.0 - r**2) / (1.0 - r**2)
                elif r > 0.1:
                    # 使用低浓度近似
                    kappas[i] = 2.0 * r / (1.0 - r**2)
                else:
                    # 默认值
                    kappas[i] = 2.0
                
                # 限制kappa的范围
                kappas[i] = np.clip(kappas[i], 0.1, 100.0)
                
            except Exception:
                # 如果计算失败，使用默认值
                kappas[i] = 2.0
        
        return weights, means, kappas
    
    def _find_optimal_von_mises_components(self, all_angles: List[float], n_samples: int) -> Optional[int]:
        """
        为von Mises混合模型找到最优的组件数量
        
        Args:
            all_angles: 所有角度列表
            n_samples: 样本数量
            
        Returns:
            最优的组件数量
        """
        if n_samples < 2:
            return 1
        
        # 限制最大组件数量
        max_components = min(self.max_components, n_samples // 2, n_samples - 1)
        if max_components < 1:
            return 1
        
        # 尝试不同的组件数量，使用BIC准则
        n_components_range = range(1, max_components + 1)
        bic_scores = []
        
        for n_components in n_components_range:
            try:
                # 拟合von Mises混合模型
                angles_rad = np.array(all_angles) * np.pi / 180.0
                weights = np.ones(n_components) / n_components
                means = np.linspace(0, 2*np.pi, n_components, endpoint=False)
                kappas = np.ones(n_components) * 2.0
                
                # 简单拟合（减少迭代次数以提高效率）
                for _ in range(20):
                    try:
                        responsibilities = self._compute_von_mises_responsibilities(angles_rad, means, kappas, weights)
                        if not np.all(np.isfinite(responsibilities)):
                            break
                        weights, means, kappas = self._update_von_mises_parameters(angles_rad, responsibilities)
                        if (not np.all(np.isfinite(weights)) or 
                            not np.all(np.isfinite(means)) or 
                            not np.all(np.isfinite(kappas))):
                            break
                    except Exception:
                        break
                
                # 计算BIC
                try:
                    log_likelihood = self._compute_von_mises_log_likelihood(angles_rad, means, kappas, weights)
                    if np.isfinite(log_likelihood):
                        n_params = 3 * n_components - 1  # 权重、均值、集中度参数
                        bic = -2 * log_likelihood + n_params * np.log(n_samples)
                        bic_scores.append(bic)
                    else:
                        bic_scores.append(float('inf'))
                except Exception:
                    bic_scores.append(float('inf'))
                
            except Exception as e:
                print(f"Von Mises with {n_components} components failed: {e}")
                bic_scores.append(float('inf'))
        
        if not bic_scores or all(score == float('inf') for score in bic_scores):
            return 1
        
        # 选择BIC最小的组件数量
        best_n_components = n_components_range[np.argmin(bic_scores)]
        
        return best_n_components
    
    def _compute_von_mises_log_likelihood(self, angles_rad: np.ndarray, means: np.ndarray, 
                                        kappas: np.ndarray, weights: np.ndarray) -> float:
        """
        计算von Mises混合模型的对数似然
        
        Args:
            angles_rad: 角度（弧度）
            means: 均值参数
            kappas: 集中度参数
            weights: 混合权重
            
        Returns:
            对数似然
        """
        log_probs = np.zeros((len(angles_rad), len(means)))
        
        for i in range(len(means)):
            try:
                # 使用scipy的vonmises分布，添加数值稳定性检查
                kappa = max(kappas[i], 1e-6)  # 确保kappa不为零
                log_probs[:, i] = vonmises.logpdf(angles_rad, kappa, loc=means[i])
                
                # 处理无效值
                log_probs[:, i] = np.where(np.isfinite(log_probs[:, i]), 
                                         log_probs[:, i], -1e10)
            except Exception:
                # 如果vonmises计算失败，使用简单的余弦相似度
                log_probs[:, i] = kappa * np.cos(angles_rad - means[i]) - np.log(2 * np.pi)
        
        # 添加权重（确保权重为正）
        weights_safe = np.maximum(weights, 1e-10)
        log_probs += np.log(weights_safe)
        
        # 使用更稳定的方法计算对数似然
        max_log_probs = np.max(log_probs, axis=1)
        log_probs_shifted = log_probs - max_log_probs[:, np.newaxis]
        exp_probs = np.exp(log_probs_shifted)
        sum_exp_probs = np.sum(exp_probs, axis=1)
        
        # 计算对数似然
        log_likelihood = np.sum(max_log_probs + np.log(np.maximum(sum_exp_probs, 1e-10)))
        
        return log_likelihood
    
    def _improved_gmm_cluster_angles_with_threshold(self, all_angles: List[float], confidence_threshold: float) -> List[Tuple[float, int]]:
        """
        使用改进的GMM聚类对角度进行聚类（支持自定义阈值）
        
        Args:
            all_angles: 所有角度列表
            confidence_threshold: 置信度阈值
            
        Returns:
            主导角度列表
        """
        n_samples = len(all_angles)
        
        # 将角度转换为二维坐标（考虑角度的周期性）
        angles_rad = np.array(all_angles) * np.pi / 180.0
        X = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 确定最优的组件数量
        best_n_components = self._find_optimal_components(X_scaled, n_samples)
        
        if best_n_components is None or best_n_components < 1:
            return []
        
        try:
            # 使用GMM聚类
            gmm = GaussianMixture(
                n_components=best_n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                n_init=5,  # 减少初始化次数以提高效率
                max_iter=200
            )
            
            # 训练模型
            gmm.fit(X_scaled)
            
            # 获取每个点的聚类标签
            labels = gmm.predict(X_scaled)
            
            # 检查实际聚类数量
            unique_labels = set(labels)
            if len(unique_labels) < 2:
                print(f"GMM clustering resulted in only {len(unique_labels)} cluster(s), skipping")
                return []
            
            # 获取每个点属于每个聚类的概率
            probabilities = gmm.predict_proba(X_scaled)
            
            # 收集聚类信息
            clusters_info = []
            
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_size = np.sum(cluster_mask)
                
                if cluster_size >= self.min_cluster_size:
                    cluster_angles = np.array(all_angles)[cluster_mask]
                    cluster_probs = probabilities[cluster_mask, label]
                    
                    # 计算聚类中心角度（考虑周期性）
                    dominant_angle = self._compute_circular_mean(cluster_angles, cluster_probs)
                    
                    # 计算置信度
                    avg_probability = np.mean(cluster_probs)
                    confidence = (cluster_size / len(all_angles)) * avg_probability
                    
                    if confidence >= confidence_threshold:
                        clusters_info.append((dominant_angle, cluster_size, confidence))
            
            # 按聚类大小排序
            clusters_info.sort(key=lambda x: x[1], reverse=True)
            
            # 转换为返回格式
            dominant_angles = [(angle, count) for angle, count, _ in clusters_info]
            
            return dominant_angles
            
        except Exception as e:
            print(f"GMM clustering failed: {e}")
            return []
    
    def _compute_circular_mean(self, angles: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """
        计算角度的加权圆形均值（考虑周期性）
        
        Args:
            angles: 角度数组（度）
            weights: 权重数组（可选）
            
        Returns:
            圆形均值角度（度）
        """
        angles_rad = angles * np.pi / 180.0
        
        if weights is None:
            weights = np.ones(len(angles))
        
        # 计算加权平均
        weighted_cos = np.average(np.cos(angles_rad), weights=weights)
        weighted_sin = np.average(np.sin(angles_rad), weights=weights)
        
        # 计算角度
        mean_angle = math.degrees(math.atan2(weighted_sin, weighted_cos))
        
        # 确保角度在[0, 360)范围内
        if mean_angle < 0:
            mean_angle += 360
        
        return mean_angle
    
    def _find_optimal_components(self, X_scaled: np.ndarray, n_samples: int) -> Optional[int]:
        """
        使用BIC准则找到最优的组件数量（移除轮廓系数干预）
        
        Args:
            X_scaled: 标准化后的数据
            n_samples: 样本数量
            
        Returns:
            最优的组件数量
        """
        if n_samples < 2:
            return 1
        
        # 限制最大组件数量
        max_components = min(self.max_components, n_samples // 2, n_samples - 1)
        if max_components < 1:
            return 1
        
        # 尝试不同的组件数量，只使用BIC准则
        n_components_range = range(1, max_components + 1)
        bic_scores = []
        
        for n_components in n_components_range:
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type=self.covariance_type,
                    random_state=self.random_state,
                    n_init=3,  # 减少初始化次数
                    max_iter=100
                )
                gmm.fit(X_scaled)
                
                # 检查实际聚类数量
                labels = gmm.predict(X_scaled)
                unique_labels = set(labels)
                
                if len(unique_labels) < n_components:
                    bic_scores.append(float('inf'))
                    continue
                
                # 计算BIC
                bic_scores.append(gmm.bic(X_scaled))
                
            except Exception as e:
                print(f"GMM with {n_components} components failed: {e}")
                bic_scores.append(float('inf'))
        
        if not bic_scores or all(score == float('inf') for score in bic_scores):
            return 1
        
        # 选择BIC最小的组件数量
        best_n_components = n_components_range[np.argmin(bic_scores)]
        
        return best_n_components
    
    def _fallback_find_dominant_angles(self, all_angles: List[float]) -> List[Tuple[float, int]]:
        """
        当聚类失败时的回退方法，使用原始的分箱方法
        """
        angle_counter = Counter()
        for angle in all_angles:
            angle_key = round(angle / self.angle_tolerance) * self.angle_tolerance
            angle_counter[angle_key] += 1
        
        dominant_angles = []
        for angle, count in angle_counter.most_common():
            if count >= self.min_angle_count:
                dominant_angles.append((angle, count))
        
        return dominant_angles
    
    def cluster_angles_with_info(self, all_angles: List[float]) -> Tuple[List[Tuple[float, int]], List[Tuple[float, float, int]]]:
        """
        对角度进行聚类并返回详细信息
        
        Args:
            all_angles: 所有角度列表
            
        Returns:
            (主导角度列表, 聚类详细信息列表)
        """
        if len(all_angles) < self.min_cluster_size:
            return [], []
        
        # 异常值过滤
        filtered_angles = self._filter_outliers(all_angles)
        if len(filtered_angles) < self.min_cluster_size:
            return [], []
        
        if self.use_von_mises:
            # 使用von Mises混合模型
            dominant_angles, clusters_info = self._von_mises_cluster_with_info(filtered_angles)
        else:
            # 使用改进的GMM聚类
            dominant_angles, clusters_info = self._improved_gmm_cluster_with_info(filtered_angles)
        
        return dominant_angles, clusters_info
    
    def _von_mises_cluster_with_info(self, all_angles: List[float]) -> Tuple[List[Tuple[float, int]], List[Tuple[float, float, int]]]:
        """
        使用von Mises混合模型聚类并返回详细信息
        """
        n_samples = len(all_angles)
        best_n_components = self._find_optimal_von_mises_components(all_angles, n_samples)
        
        if best_n_components is None:
            return [], []
        
        clusters_info = self._fit_von_mises_mixture(all_angles, best_n_components)
        dominant_angles = [(angle, count) for angle, count, _ in clusters_info]
        
        return dominant_angles, clusters_info
    
    def _improved_gmm_cluster_with_info(self, all_angles: List[float]) -> Tuple[List[Tuple[float, int]], List[Tuple[float, float, int]]]:
        """
        使用改进的GMM聚类并返回详细信息
        """
        angles_rad = np.array(all_angles) * np.pi / 180.0
        X = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_samples = len(all_angles)
        best_n_components = self._find_optimal_components(X_scaled, n_samples)
        
        if best_n_components is None:
            return [], []
        
        gmm = GaussianMixture(
            n_components=best_n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=5
        )
        
        gmm.fit(X_scaled)
        labels = gmm.predict(X_scaled)
        probabilities = gmm.predict_proba(X_scaled)
        
        clusters_info = []
        dominant_angles = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size >= self.min_cluster_size:
                cluster_angles = np.array(all_angles)[cluster_mask]
                cluster_probs = probabilities[cluster_mask, label]
                
                dominant_angle = self._compute_circular_mean(cluster_angles, cluster_probs)
                
                avg_probability = np.mean(cluster_probs)
                confidence = (cluster_size / len(all_angles)) * avg_probability
                
                clusters_info.append((dominant_angle, confidence, cluster_size))
                
                if confidence >= self.confidence_threshold:
                    dominant_angles.append((dominant_angle, cluster_size))
        
        dominant_angles.sort(key=lambda x: x[1], reverse=True)
        clusters_info.sort(key=lambda x: x[2], reverse=True)
        
        return dominant_angles, clusters_info


class SpectralPatternAnalyzer(PatternAnalyzer):
    """基于谱聚类的角度聚类模式分析器 - 继承自PatternAnalyzer"""
    
    def __init__(self, angle_tolerance: float = 5.0, min_angle_count: int = 2, 
                 min_step_count: int = 1, max_clusters: int = 10,
                 min_cluster_size: int = 1, confidence_threshold: float = 0.05,
                 affinity: str = 'cosine', random_state: int = 42):
        """
        初始化基于谱聚类的角度聚类模式分析器
        
        Args:
            angle_tolerance: 角度容差（度）
            min_angle_count: 最小角度数量
            min_step_count: 最小步长数量
            max_clusters: 最大聚类数量
            min_cluster_size: 最小聚类大小
            confidence_threshold: 主导模式置信度阈值
            affinity: 相似性度量方法 ('cosine', 'rbf', 'nearest_neighbors')
            random_state: 随机种子
        """
        super().__init__(angle_tolerance, min_angle_count, min_step_count)
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.confidence_threshold = confidence_threshold
        self.affinity = affinity
        self.random_state = random_state
    
    def _find_dominant_angles(self, all_angles: List[float]) -> List[Tuple[float, int]]:
        """
        通过谱聚类的方式找到主导角度
        
        Args:
            all_angles: 所有角度列表
            
        Returns:
            主导角度列表，每个元素为(角度, 数量)的元组
        """
        # 检查样本数量是否足够
        if len(all_angles) < 2:
            print(f"Spectral clustering skipped: insufficient angles ({len(all_angles)} < 2)")
            return self._fallback_find_dominant_angles(all_angles)
        
        if len(all_angles) < self.min_cluster_size:
            print(f"Spectral clustering skipped: insufficient angles for clustering ({len(all_angles)} < {self.min_cluster_size})")
            return self._fallback_find_dominant_angles(all_angles)
        
        # 检查数据多样性
        unique_angles = set(all_angles)
        if len(unique_angles) < 2:
            print(f"Spectral clustering skipped: insufficient unique angles ({len(unique_angles)} < 2)")
            return self._fallback_find_dominant_angles(all_angles)
        
        # 将角度转换为二维坐标（考虑角度的周期性）
        angles_rad = np.array(all_angles) * np.pi / 180.0
        X = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])
        
        # 检查数据是否包含NaN或inf
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("Spectral clustering skipped: data contains NaN or inf values")
            return self._fallback_find_dominant_angles(all_angles)
        
        # 标准化
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 再次检查标准化后的数据
            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                print("Spectral clustering skipped: scaled data contains NaN or inf values")
                return self._fallback_find_dominant_angles(all_angles)
        except Exception as e:
            print(f"Spectral clustering skipped: scaling failed: {e}")
            return self._fallback_find_dominant_angles(all_angles)
        
        # 使用谱聚类
        dominant_angles = self._spectral_cluster_angles(X_scaled, all_angles)
        
        # 如果没有找到聚类，回退到原始方法
        if not dominant_angles:
            print("Spectral clustering failed, falling back to binning method")
            return self._fallback_find_dominant_angles(all_angles)
        
        return dominant_angles
    
    def _spectral_cluster_angles(self, 
                                X_scaled: np.ndarray, 
                                all_angles: List[float]) -> List[Tuple[float, int]]:
        """
        使用谱聚类对角度进行聚类
        
        Args:
            X_scaled: 标准化后的二维坐标
            all_angles: 原始角度列表
            
        Returns:
            主导角度列表
        """
        n_samples = len(all_angles)
        
        # 检查样本数量是否足够
        if n_samples < 2:
            print(f"Spectral clustering skipped: insufficient samples ({n_samples} < 2)")
            return []
        
        # 检查数据形状
        if X_scaled.shape[0] != n_samples:
            print(f"Spectral clustering skipped: shape mismatch (X_scaled: {X_scaled.shape}, n_samples: {n_samples})")
            return []
        
        # 确定最优的聚类数量
        best_n_clusters = self._find_optimal_clusters(X_scaled, n_samples)
        
        if best_n_clusters is None or best_n_clusters < 1:
            return []
        
        try:
            # 计算相似性矩阵
            similarity_matrix = self._compute_similarity_matrix(X_scaled)
            
            # 检查相似性矩阵是否包含NaN或inf
            if np.any(np.isnan(similarity_matrix)) or np.any(np.isinf(similarity_matrix)):
                print("Spectral clustering failed: similarity matrix contains NaN or inf values")
                return []
            
            # 检查相似性矩阵是否全为零
            if np.all(similarity_matrix == 0):
                print("Spectral clustering failed: similarity matrix is all zeros")
                return []
            
            # 使用自定义的谱聚类实现，避免scipy的数值问题
            labels = self._custom_spectral_clustering(similarity_matrix, best_n_clusters)
            
            if labels is None:
                print("Spectral clustering failed: custom implementation failed")
                return []
            
            # 检查实际聚类数量
            unique_labels = set(labels)
            if len(unique_labels) < 2:
                print(f"Spectral clustering resulted in only {len(unique_labels)} cluster(s), skipping")
                return []
            
            # 收集聚类信息
            clusters_info = []
            
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_size = np.sum(cluster_mask)
                
                if cluster_size >= self.min_cluster_size:
                    cluster_angles = np.array(all_angles)[cluster_mask]
                    
                    # 计算聚类中心角度
                    cluster_angles_rad = cluster_angles * np.pi / 180.0
                    mean_cos = np.mean(np.cos(cluster_angles_rad))
                    mean_sin = np.mean(np.sin(cluster_angles_rad))
                    dominant_angle = math.degrees(math.atan2(mean_sin, mean_cos))
                    
                    if dominant_angle < 0:
                        dominant_angle += 360
                    
                    # 计算置信度
                    confidence = cluster_size / len(all_angles)
                    
                    if confidence >= self.confidence_threshold:
                        clusters_info.append((dominant_angle, cluster_size, confidence))
            
            # 按聚类大小排序
            clusters_info.sort(key=lambda x: x[1], reverse=True)
            
            # 转换为返回格式
            dominant_angles = [(angle, count) for angle, count, _ in clusters_info]
            
            return dominant_angles
            
        except Exception as e:
            print(f"Spectral clustering failed: {e}")
            return []
    
    def _custom_spectral_clustering(self, similarity_matrix: np.ndarray, n_clusters: int) -> Optional[np.ndarray]:
        """
        自定义谱聚类实现，避免scipy的数值问题
        
        Args:
            similarity_matrix: 相似性矩阵
            n_clusters: 聚类数量
            
        Returns:
            聚类标签
        """
        try:
            n_samples = len(similarity_matrix)
            
            # 确保相似性矩阵是对称的
            similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
            
            # 计算度矩阵
            degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
            
            # 计算拉普拉斯矩阵
            laplacian_matrix = degree_matrix - similarity_matrix
            
            # 检查拉普拉斯矩阵是否有效
            if np.any(np.isnan(laplacian_matrix)) or np.any(np.isinf(laplacian_matrix)):
                print("Custom spectral clustering failed: invalid Laplacian matrix")
                return None
            
            # 确保拉普拉斯矩阵是对称的
            laplacian_matrix = (laplacian_matrix + laplacian_matrix.T) / 2
            
            # 添加小的对角项以确保正定性
            min_eigenval = np.linalg.eigvals(laplacian_matrix).min()
            if min_eigenval < 1e-8:
                laplacian_matrix += np.eye(n_samples) * 1e-8
            
            # 计算特征值和特征向量
            try:
                eigenvals, eigenvecs = np.linalg.eigh(laplacian_matrix)
            except Exception as e:
                print(f"Eigenvalue decomposition failed: {e}")
                return None
            
            # 检查特征值和特征向量是否有效
            if np.any(np.isnan(eigenvals)) or np.any(np.isinf(eigenvals)):
                print("Custom spectral clustering failed: invalid eigenvalues")
                return None
            
            if np.any(np.isnan(eigenvecs)) or np.any(np.isinf(eigenvecs)):
                print("Custom spectral clustering failed: invalid eigenvectors")
                return None
            
            # 选择前n_clusters个最小的非零特征值对应的特征向量
            # 跳过第一个特征值（通常为0）
            if n_clusters >= n_samples:
                n_clusters = n_samples - 1
            
            # 选择特征向量
            selected_eigenvecs = eigenvecs[:, 1:n_clusters+1]
            
            # 对特征向量进行归一化
            row_norms = np.linalg.norm(selected_eigenvecs, axis=1, keepdims=True)
            row_norms = np.where(row_norms == 0, 1, row_norms)  # 避免除零
            selected_eigenvecs = selected_eigenvecs / row_norms
            
            # 使用K-means对特征向量进行聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(selected_eigenvecs)
            
            return labels
            
        except Exception as e:
            print(f"Custom spectral clustering failed: {e}")
            return None
    
    def _compute_similarity_matrix(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        计算相似性矩阵
        
        Args:
            X_scaled: 标准化后的数据
            
        Returns:
            相似性矩阵
        """
        try:
            n_samples = len(X_scaled)
            
            # 检查数据是否包含NaN或inf
            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                print("Warning: input data contains NaN or inf values")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if self.affinity == 'cosine':
                # 使用余弦相似性（适合单位圆上的点）
                # 添加小的扰动以避免数值不稳定
                X_normalized = X_scaled / (np.linalg.norm(X_scaled, axis=1, keepdims=True) + 1e-8)
                similarity_matrix = np.dot(X_normalized, X_normalized.T)
                
                # 确保相似性矩阵在有效范围内
                similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
                
            elif self.affinity == 'rbf':
                # 使用RBF核相似性
                similarity_matrix = rbf_kernel(X_scaled, gamma=1.0)
                
            elif self.affinity == 'nearest_neighbors':
                # 使用k近邻相似性
                n_neighbors = min(5, len(X_scaled)-1)
                if n_neighbors < 1:
                    n_neighbors = 1
                similarity_matrix = kneighbors_graph(X_scaled, n_neighbors=n_neighbors, 
                                                   mode='connectivity', include_self=True).toarray()
            else:
                # 默认使用余弦相似性
                X_normalized = X_scaled / (np.linalg.norm(X_scaled, axis=1, keepdims=True) + 1e-8)
                similarity_matrix = np.dot(X_normalized, X_normalized.T)
                similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
            
            # 检查相似性矩阵是否包含NaN或inf
            if np.any(np.isnan(similarity_matrix)) or np.any(np.isinf(similarity_matrix)):
                print("Warning: similarity matrix contains NaN or inf values, attempting cleanup")
                similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0, posinf=1.0, neginf=0.0)
            
            # 确保相似性矩阵是对称的
            similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
            
            # 确保对角线元素为1（对于余弦相似性）
            if self.affinity in ['cosine', 'rbf']:
                np.fill_diagonal(similarity_matrix, 1.0)
            
            # 确保相似性矩阵是正定的（添加小的对角项）
            min_eigenval = np.linalg.eigvals(similarity_matrix).min()
            if min_eigenval < 1e-8:
                similarity_matrix += np.eye(n_samples) * 1e-8
            
            # 最终检查：确保没有NaN或inf
            if np.any(np.isnan(similarity_matrix)) or np.any(np.isinf(similarity_matrix)):
                print("Warning: similarity matrix still contains NaN or inf values after cleanup")
                # 创建一个简单的单位矩阵作为fallback
                similarity_matrix = np.eye(n_samples)
            
            return similarity_matrix
            
        except Exception as e:
            print(f"Similarity matrix computation failed: {e}")
            # 返回单位矩阵作为fallback
            return np.eye(len(X_scaled))
    
    def _find_optimal_clusters(self, X_scaled: np.ndarray, n_samples: int) -> Optional[int]:
        """
        使用多种方法找到最优的聚类数量
        
        Args:
            X_scaled: 标准化后的数据
            n_samples: 样本数量
            
        Returns:
            最优的聚类数量
        """
        # 检查样本数量
        if n_samples < 2:
            return 1  # 样本不足时返回1个聚类
        
        # 限制最大聚类数量
        max_clusters = min(self.max_clusters, n_samples // 2, n_samples - 1)
        if max_clusters < 1:
            return 1  # 至少返回1个聚类
        
        # 尝试不同的聚类数量
        n_clusters_range = range(1, max_clusters + 1)
        silhouette_scores = []
        eigengap_scores = []
        
        for n_clusters in n_clusters_range:
            try:
                # 计算相似性矩阵
                similarity_matrix = self._compute_similarity_matrix(X_scaled)
                
                # 检查相似性矩阵是否有效
                if np.any(np.isnan(similarity_matrix)) or np.any(np.isinf(similarity_matrix)):
                    print(f"Spectral clustering with {n_clusters} clusters skipped: invalid similarity matrix")
                    silhouette_scores.append(0)
                    eigengap_scores.append(0)
                    continue
                
                # 检查相似性矩阵是否全为零或全为1
                if np.all(similarity_matrix == 0) or np.all(similarity_matrix == 1):
                    print(f"Spectral clustering with {n_clusters} clusters skipped: similarity matrix is uniform")
                    silhouette_scores.append(0)
                    eigengap_scores.append(0)
                    continue
                
                # 使用自定义谱聚类
                try:
                    labels = self._custom_spectral_clustering(similarity_matrix, n_clusters)
                    if labels is None:
                        print(f"Spectral clustering with {n_clusters} clusters failed: custom implementation failed")
                        silhouette_scores.append(0)
                        eigengap_scores.append(0)
                        continue
                except Exception as e:
                    print(f"Spectral clustering with {n_clusters} clusters failed: {e}")
                    silhouette_scores.append(0)
                    eigengap_scores.append(0)
                    continue
                
                # 计算轮廓系数
                if n_clusters > 1 and len(set(labels)) > 1:
                    try:
                        silhouette_scores.append(silhouette_score(X_scaled, labels))
                    except:
                        silhouette_scores.append(0)
                else:
                    silhouette_scores.append(0)
                
                # 计算特征值间隙（eigengap）
                if n_clusters > 1:
                    eigengap_score = self._compute_eigengap_score(similarity_matrix, n_clusters)
                    eigengap_scores.append(eigengap_score)
                else:
                    eigengap_scores.append(0)
                    
            except Exception as e:
                print(f"Spectral clustering with {n_clusters} clusters failed: {e}")
                silhouette_scores.append(0)
                eigengap_scores.append(0)
        
        # 选择最优聚类数量
        if not silhouette_scores or all(score == 0 for score in silhouette_scores):
            return 1  # 默认返回1个聚类
        
        # 使用轮廓系数选择最优聚类数量
        best_silhouette_idx = np.argmax(silhouette_scores)
        best_n_clusters = n_clusters_range[best_silhouette_idx]
        
        # 如果轮廓系数选择1个聚类，但数据量较大，尝试使用特征值间隙
        if best_n_clusters == 1 and n_samples > 20:
            # 找到特征值间隙最大的聚类数量（至少2个聚类）
            valid_eigengap_scores = [(i, score) for i, score in enumerate(eigengap_scores) 
                                   if score > 0 and n_clusters_range[i] > 1]
            if valid_eigengap_scores:
                best_eigengap_idx = max(valid_eigengap_scores, key=lambda x: x[1])[0]
                best_n_clusters = n_clusters_range[best_eigengap_idx]
        
        return best_n_clusters
    
    def _compute_eigengap_score(self, similarity_matrix: np.ndarray, n_clusters: int) -> float:
        """
        计算特征值间隙分数
        
        Args:
            similarity_matrix: 相似性矩阵
            n_clusters: 聚类数量
            
        Returns:
            特征值间隙分数
        """
        try:
            # 检查相似性矩阵是否有效
            if np.any(np.isnan(similarity_matrix)) or np.any(np.isinf(similarity_matrix)):
                return 0
            
            # 确保相似性矩阵是对称的
            similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
            
            # 计算拉普拉斯矩阵
            degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
            laplacian_matrix = degree_matrix - similarity_matrix
            
            # 检查拉普拉斯矩阵是否有效
            if np.any(np.isnan(laplacian_matrix)) or np.any(np.isinf(laplacian_matrix)):
                return 0
            
            # 确保拉普拉斯矩阵是对称的
            laplacian_matrix = (laplacian_matrix + laplacian_matrix.T) / 2
            
            # 添加小的对角项以确保正定性
            min_eigenval = np.linalg.eigvals(laplacian_matrix).min()
            if min_eigenval < 1e-8:
                laplacian_matrix += np.eye(len(laplacian_matrix)) * 1e-8
            
            # 计算特征值
            try:
                eigenvals = np.linalg.eigvals(laplacian_matrix)
                eigenvals = np.sort(eigenvals)
            except Exception as e:
                print(f"Eigenvalue computation failed: {e}")
                return 0
            
            # 检查特征值是否有效
            if np.any(np.isnan(eigenvals)) or np.any(np.isinf(eigenvals)):
                return 0
            
            # 计算特征值间隙
            if n_clusters < len(eigenvals):
                eigengap = eigenvals[n_clusters] - eigenvals[n_clusters - 1]
                return eigengap
            else:
                return 0
        except Exception as e:
            print(f"Eigengap computation failed: {e}")
            return 0
    
    def _fallback_find_dominant_angles(self, all_angles: List[float]) -> List[Tuple[float, int]]:
        """
        当谱聚类失败时的回退方法，使用原始的分箱方法
        """
        angle_counter = Counter()
        for angle in all_angles:
            angle_key = round(angle / self.angle_tolerance) * self.angle_tolerance
            angle_counter[angle_key] += 1
        
        # 过滤掉数量不足的角度
        dominant_angles = [(angle, count) for angle, count in angle_counter.items() 
                          if count >= self.min_angle_count]
        
        # 按数量排序
        dominant_angles.sort(key=lambda x: x[1], reverse=True)
        
        return dominant_angles
    
    def cluster_angles_with_info(self, all_angles: List[float]) -> Tuple[List[Tuple[float, int]], List[Tuple[float, float, int]]]:
        """
        对角度进行谱聚类并返回详细信息
        
        Args:
            all_angles: 所有角度列表
            
        Returns:
            (主导角度列表, 聚类详细信息列表)
        """
        if len(all_angles) < self.min_cluster_size:
            return [], []
        
        # 将角度转换为二维坐标（考虑角度的周期性）
        angles_rad = np.array(all_angles) * np.pi / 180.0
        X = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 使用谱聚类
        n_samples = len(all_angles)
        best_n_clusters = self._find_optimal_clusters(X_scaled, n_samples)
        
        if best_n_clusters is None:
            return [], []
        
        spectral = SpectralClustering(
            n_clusters=best_n_clusters,
            affinity='precomputed',
            random_state=self.random_state,
            n_init=10
        )
        
        labels = spectral.fit_predict(similarity_matrix)
        
        # 收集聚类信息
        clusters_info = []
        dominant_angles = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size >= self.min_cluster_size:
                cluster_angles = np.array(all_angles)[cluster_mask]
                
                # 计算聚类中心角度
                cluster_angles_rad = cluster_angles * np.pi / 180.0
                mean_cos = np.mean(np.cos(cluster_angles_rad))
                mean_sin = np.mean(np.sin(cluster_angles_rad))
                dominant_angle = math.degrees(math.atan2(mean_sin, mean_cos))
                
                if dominant_angle < 0:
                    dominant_angle += 360
                
                # 计算置信度
                confidence = cluster_size / len(all_angles)
                
                clusters_info.append((dominant_angle, confidence, cluster_size))
                
                if confidence >= self.confidence_threshold:
                    dominant_angles.append((dominant_angle, cluster_size))
        
        # 按聚类大小排序
        dominant_angles.sort(key=lambda x: x[1], reverse=True)
        clusters_info.sort(key=lambda x: x[2], reverse=True)
        
        return dominant_angles, clusters_info

    def _von_mises_cluster_angles(self, all_angles: List[float]) -> List[Tuple[float, int]]:
        """
        使用von Mises混合模型对角度进行聚类（向后兼容）
        
        Args:
            all_angles: 所有角度列表
            
        Returns:
            主导角度列表
        """
        return self._von_mises_cluster_angles_with_threshold(all_angles, self.confidence_threshold)
    
    def _improved_gmm_cluster_angles(self, all_angles: List[float]) -> List[Tuple[float, int]]:
        """
        使用改进的GMM对角度进行聚类（向后兼容）
        
        Args:
            all_angles: 所有角度列表
            
        Returns:
            主导角度列表
        """
        return self._improved_gmm_cluster_angles_with_threshold(all_angles, self.confidence_threshold)
    
    def _fit_von_mises_mixture(self, all_angles: List[float], n_components: int) -> List[Tuple[float, int, float]]:
        """
        拟合von Mises混合模型（向后兼容）
        
        Args:
            all_angles: 所有角度列表
            n_components: 组件数量
            
        Returns:
            聚类信息列表，每个元素为(角度, 数量, 置信度)的元组
        """
        return self._fit_von_mises_mixture_with_threshold(all_angles, n_components, self.confidence_threshold)

