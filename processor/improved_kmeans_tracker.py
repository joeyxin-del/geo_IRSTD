import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import copy
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

class ImprovedKMeansTracker:
    """改进的K-means轨迹跟踪器，重点提升Recall"""
    
    def __init__(self, 
                 max_association_distance: float = 150.0,  # 增加距离阈值
                 expected_sequence_length: int = 5,
                 trajectory_mean_distance: float = 116.53,
                 trajectory_std_distance: float = 29.76,
                 trajectory_max_distance: float = 237.87,
                 trajectory_min_distance: float = 6.74,
                 min_track_length: int = 1,  # 降低最小轨迹长度要求
                 max_frame_gap: int = 5,     # 增加最大帧间隔
                 kmeans_n_init: int = 15,    # 增加初始化次数
                 kmeans_max_iter: int = 500, # 增加最大迭代次数
                 silhouette_threshold: float = 0.1,  # 降低轮廓系数阈值
                 aggressive_completion: bool = True,  # 启用积极补全
                 use_dbscan_fallback: bool = True):   # 使用DBSCAN作为备选
        """
        初始化改进的K-means轨迹跟踪器
        
        Args:
            max_association_distance: 轨迹关联的最大距离阈值
            expected_sequence_length: 期望的序列长度
            trajectory_mean_distance: 轨迹平均距离
            trajectory_std_distance: 轨迹距离标准差
            trajectory_max_distance: 轨迹最大距离
            trajectory_min_distance: 轨迹最小距离
            min_track_length: 最小轨迹长度
            max_frame_gap: 最大帧间隔
            kmeans_n_init: K-means初始化次数
            kmeans_max_iter: K-means最大迭代次数
            silhouette_threshold: 轮廓系数阈值
            aggressive_completion: 是否启用积极补全
            use_dbscan_fallback: 是否使用DBSCAN作为备选聚类方法
        """
        self.max_association_distance = max_association_distance
        self.expected_sequence_length = expected_sequence_length
        self.trajectory_mean_distance = trajectory_mean_distance
        self.trajectory_std_distance = trajectory_std_distance
        self.trajectory_max_distance = trajectory_max_distance
        self.trajectory_min_distance = trajectory_min_distance
        self.min_track_length = min_track_length
        self.max_frame_gap = max_frame_gap
        self.kmeans_n_init = kmeans_n_init
        self.kmeans_max_iter = kmeans_max_iter
        self.silhouette_threshold = silhouette_threshold
        self.aggressive_completion = aggressive_completion
        self.use_dbscan_fallback = use_dbscan_fallback
        
        # 计算更宽松的统计阈值
        self.distance_threshold = min(
            self.trajectory_mean_distance + 3 * self.trajectory_std_distance,  # 3σ范围
            self.trajectory_max_distance * 0.9  # 最大距离的90%
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
    
    def cluster_detections_improved(self, frames_data: Dict[int, Dict]) -> List[List[Tuple[int, List[float]]]]:
        """
        改进的检测点聚类方法，结合K-means和DBSCAN
        
        Args:
            frames_data: 序列帧数据
            
        Returns:
            trajectories: 基于聚类结果的轨迹列表
        """
        frames = sorted(frames_data.keys())
        if not frames:
            return []
        
        # 收集所有检测点及其帧信息
        all_points = []
        point_to_frame = []
        
        for frame in frames:
            coords = frames_data[frame]['coords']
            for point in coords:
                all_points.append(point)
                point_to_frame.append(frame)
        
        if len(all_points) < 2:
            return []
        
        # 转换为numpy数组
        points_array = np.array(all_points)
        
        # 首先尝试K-means聚类
        trajectories = self._kmeans_clustering(points_array, point_to_frame)
        
        # 如果K-means聚类结果不理想，尝试DBSCAN
        if self.use_dbscan_fallback and (len(trajectories) == 0 or 
                                        any(len(traj) < self.min_track_length for traj in trajectories)):
            print("  K-means聚类结果不理想，尝试DBSCAN聚类...")
            dbscan_trajectories = self._dbscan_clustering(points_array, point_to_frame)
            if dbscan_trajectories:
                trajectories = dbscan_trajectories
        
        # 如果仍然没有好的结果，使用基于距离的简单聚类
        if not trajectories:
            print("  聚类方法失败，使用基于距离的简单聚类...")
            trajectories = self._distance_based_clustering(points_array, point_to_frame)
        
        # 过滤掉过短的轨迹
        valid_trajectories = [traj for traj in trajectories if len(traj) >= self.min_track_length]
        
        print(f"  改进聚类完成，生成 {len(valid_trajectories)} 条有效轨迹")
        return valid_trajectories
    
    def _kmeans_clustering(self, points_array: np.ndarray, point_to_frame: List[int]) -> List[List[Tuple[int, List[float]]]]:
        """K-means聚类"""
        max_clusters = min(len(points_array), 15)  # 增加最大聚类数
        best_k = 1
        best_silhouette = -1
        
        # 尝试不同的聚类数量
        for k in range(1, max_clusters + 1):
            if k >= len(points_array):
                break
                
            try:
                kmeans = KMeans(n_clusters=k, n_init=self.kmeans_n_init, 
                              max_iter=self.kmeans_max_iter, random_state=42)
                cluster_labels = kmeans.fit_predict(points_array)
                
                # 计算轮廓系数
                if k > 1:
                    silhouette_avg = silhouette_score(points_array, cluster_labels)
                    print(f"    K={k}, 轮廓系数={silhouette_avg:.3f}")
                    
                    if silhouette_avg > best_silhouette:
                        best_silhouette = silhouette_avg
                        best_k = k
                else:
                    best_k = 1
                    break
                    
            except Exception as e:
                print(f"    K={k} 聚类失败: {e}")
                continue
        
        print(f"  选择最佳聚类数: K={best_k}, 轮廓系数={best_silhouette:.3f}")
        
        # 使用最佳聚类数进行最终聚类
        final_kmeans = KMeans(n_clusters=best_k, n_init=self.kmeans_n_init, 
                            max_iter=self.kmeans_max_iter, random_state=42)
        final_labels = final_kmeans.fit_predict(points_array)
        
        # 将聚类结果转换为轨迹
        trajectories = [[] for _ in range(best_k)]
        
        for i, (point, frame, label) in enumerate(zip(points_array, point_to_frame, final_labels)):
            trajectories[label].append((frame, point.tolist()))
        
        # 对每个轨迹按帧排序
        for i in range(len(trajectories)):
            trajectories[i].sort(key=lambda x: x[0])
        
        return trajectories
    
    def _dbscan_clustering(self, points_array: np.ndarray, point_to_frame: List[int]) -> List[List[Tuple[int, List[float]]]]:
        """DBSCAN聚类作为备选方法"""
        # 尝试不同的eps值
        eps_values = [50, 100, 150, 200]
        
        for eps in eps_values:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=1)
                cluster_labels = dbscan.fit_predict(points_array)
                
                # 统计聚类结果
                unique_labels = set(cluster_labels)
                n_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
                
                if n_clusters > 0:
                    print(f"    DBSCAN eps={eps}, 聚类数={n_clusters}")
                    
                    # 将聚类结果转换为轨迹
                    trajectories = [[] for _ in range(n_clusters)]
                    cluster_idx = 0
                    
                    for i, (point, frame, label) in enumerate(zip(points_array, point_to_frame, cluster_labels)):
                        if label != -1:  # 忽略噪声点
                            trajectories[label].append((frame, point.tolist()))
                    
                    # 对每个轨迹按帧排序
                    for i in range(len(trajectories)):
                        trajectories[i].sort(key=lambda x: x[0])
                    
                    return trajectories
                    
            except Exception as e:
                print(f"    DBSCAN eps={eps} 聚类失败: {e}")
                continue
        
        return []
    
    def _distance_based_clustering(self, points_array: np.ndarray, point_to_frame: List[int]) -> List[List[Tuple[int, List[float]]]]:
        """基于距离的简单聚类"""
        trajectories = []
        used_points = set()
        
        for i, (point, frame) in enumerate(zip(points_array, point_to_frame)):
            if i in used_points:
                continue
            
            # 开始新的轨迹
            trajectory = [(frame, point.tolist())]
            used_points.add(i)
            
            # 寻找相近的点
            for j, (other_point, other_frame) in enumerate(zip(points_array, point_to_frame)):
                if j in used_points:
                    continue
                
                # 检查距离和帧间隔
                distance = self.calculate_distance(point.tolist(), other_point.tolist())
                frame_gap = abs(frame - other_frame)
                
                if distance <= self.max_association_distance and frame_gap <= self.max_frame_gap:
                    trajectory.append((other_frame, other_point.tolist()))
                    used_points.add(j)
            
            # 按帧排序
            trajectory.sort(key=lambda x: x[0])
            trajectories.append(trajectory)
        
        return trajectories
    
    def aggressive_trajectory_completion(self, trajectories: List[List[Tuple[int, List[float]]]], 
                                       frames_data: Dict[int, Dict]) -> List[List[Tuple[int, List[float]]]]:
        """
        积极的轨迹补全策略，重点提升Recall
        
        Args:
            trajectories: 原始轨迹列表
            frames_data: 序列帧数据
            
        Returns:
            completed_trajectories: 补充后的轨迹列表
        """
        frames = sorted(frames_data.keys())
        completed_trajectories = []
        
        # 如果没有轨迹，创建基于单帧检测的轨迹
        if not trajectories:
            print("  没有检测到轨迹，创建基于单帧检测的轨迹...")
            for frame in frames:
                coords = frames_data[frame]['coords']
                for point in coords:
                    # 为每个检测点创建完整的5帧轨迹
                    trajectory = self._create_complete_trajectory_from_single_point(frame, point, frames)
                    completed_trajectories.append(trajectory)
            return completed_trajectories
        
        for trajectory in trajectories:
            if len(trajectory) >= self.expected_sequence_length:
                # 轨迹已经足够长，直接保留
                completed_trajectories.append(trajectory)
                continue
            
            # 积极的补全策略
            completed_trajectory = self._aggressive_complete_single_trajectory(trajectory, frames)
            completed_trajectories.append(completed_trajectory)
        
        # 额外的补全：为缺失的帧创建新轨迹
        if self.aggressive_completion:
            additional_trajectories = self._create_additional_trajectories(frames_data, completed_trajectories)
            completed_trajectories.extend(additional_trajectories)
        
        print(f"  积极轨迹补全完成，共 {len(completed_trajectories)} 条完整轨迹")
        return completed_trajectories
    
    def _create_complete_trajectory_from_single_point(self, frame: int, point: List[float], frames: List[int]) -> List[Tuple[int, List[float]]]:
        """从单点创建完整轨迹"""
        trajectory = []
        
        # 确定轨迹的时间范围
        start_frame = max(frame - 2, min(frames))
        end_frame = min(frame + 2, max(frames))
        
        for f in range(start_frame, end_frame + 1):
            if f == frame:
                trajectory.append((f, point))
            else:
                # 使用原始点作为估计（保守策略）
                trajectory.append((f, point))
        
        return trajectory
    
    def _aggressive_complete_single_trajectory(self, trajectory: List[Tuple[int, List[float]]], frames: List[int]) -> List[Tuple[int, List[float]]]:
        """积极补全单个轨迹"""
        trajectory_frames = [frame for frame, _ in trajectory]
        trajectory_points = [point for _, point in trajectory]
        
        # 确定轨迹的时间范围
        min_frame = min(trajectory_frames)
        max_frame = max(trajectory_frames)
        
        # 更积极的扩展策略
        if len(trajectory) == 1:
            # 单点轨迹，向两边扩展更多
            start_frame = max(min_frame - 3, min(frames))
            end_frame = min(max_frame + 3, max(frames))
        else:
            # 多点轨迹，基于现有范围扩展
            frame_range = max_frame - min_frame
            if frame_range < self.expected_sequence_length - 1:
                # 扩展范围到期望长度
                extension = self.expected_sequence_length - 1 - frame_range
                start_frame = max(min_frame - extension, min(frames))
                end_frame = min(max_frame + extension, max(frames))
            else:
                start_frame = min_frame
                end_frame = max_frame
        
        # 创建完整的轨迹
        completed_trajectory = []
        
        # 使用线性插值补充缺失帧
        if len(trajectory_points) >= 2:
            # 有多个点，使用线性插值
            x_coords = [p[0] for p in trajectory_points]
            y_coords = [p[1] for p in trajectory_points]
            
            for frame in range(start_frame, end_frame + 1):
                if frame in trajectory_frames:
                    # 原始帧，直接使用
                    idx = trajectory_frames.index(frame)
                    completed_trajectory.append((frame, trajectory_points[idx]))
                else:
                    # 缺失帧，使用线性插值
                    x_interp = np.interp(frame, trajectory_frames, x_coords)
                    y_interp = np.interp(frame, trajectory_frames, y_coords)
                    completed_trajectory.append((frame, [x_interp, y_interp]))
        else:
            # 单点轨迹，使用外推
            single_point = trajectory_points[0]
            
            for frame in range(start_frame, end_frame + 1):
                completed_trajectory.append((frame, single_point))
        
        return completed_trajectory
    
    def _create_additional_trajectories(self, frames_data: Dict[int, Dict], 
                                      existing_trajectories: List[List[Tuple[int, List[float]]]]) -> List[List[Tuple[int, List[float]]]]:
        """创建额外的轨迹以补全缺失的检测"""
        additional_trajectories = []
        frames = sorted(frames_data.keys())
        
        # 检查每个帧是否有足够的检测点
        for frame in frames:
            coords = frames_data[frame]['coords']
            
            # 统计当前帧在现有轨迹中的检测点数量
            existing_points_in_frame = 0
            for trajectory in existing_trajectories:
                for traj_frame, _ in trajectory:
                    if traj_frame == frame:
                        existing_points_in_frame += 1
            
            # 如果现有轨迹中的检测点少于原始检测点，创建补充轨迹
            if existing_points_in_frame < len(coords):
                missing_count = len(coords) - existing_points_in_frame
                
                # 为缺失的检测点创建新轨迹
                for i in range(missing_count):
                    if i < len(coords):
                        point = coords[i]
                        trajectory = self._create_complete_trajectory_from_single_point(frame, point, frames)
                        additional_trajectories.append(trajectory)
        
        return additional_trajectories
    
    def process_sequence(self, predictions: Dict) -> Dict:
        """
        处理整个预测结果，实现改进的轨迹跟踪
        
        Args:
            predictions: 原始预测结果
            
        Returns:
            processed_predictions: 处理后的预测结果
        """
        print("开始改进的轨迹跟踪处理...")
        
        # 提取序列信息
        sequence_data = self.extract_sequence_info(predictions)
        print(f"提取了 {len(sequence_data)} 个序列的信息")
        
        processed_predictions = {}
        
        for sequence_id, frames_data in sequence_data.items():
            print(f"处理序列 {sequence_id}，包含 {len(frames_data)} 帧")
            
            # 1. 使用改进的聚类方法
            print("  使用改进的聚类方法...")
            trajectories = self.cluster_detections_improved(frames_data)
            
            # 2. 积极的轨迹补全
            print("  进行积极的轨迹补全...")
            completed_trajectories = self.aggressive_trajectory_completion(trajectories, frames_data)
            
            print(f"  最终生成 {len(completed_trajectories)} 条轨迹")
            
            # 3. 转换回原始格式
            for trajectory in completed_trajectories:
                for frame, point in trajectory:
                    image_name = f"{sequence_id}_{frame}_test"
                    if image_name not in processed_predictions:
                        processed_predictions[image_name] = {
                            'coords': [],
                            'num_objects': 0
                        }
                    processed_predictions[image_name]['coords'].append(point)
                    processed_predictions[image_name]['num_objects'] = len(processed_predictions[image_name]['coords'])
        
        # 确保所有原始图像都有对应的处理结果
        for img_name, pred_info in predictions.items():
            if img_name not in processed_predictions:
                processed_predictions[img_name] = {
                    'coords': pred_info['coords'],
                    'num_objects': pred_info['num_objects']
                }
        
        print(f"改进的轨迹跟踪处理完成，处理了 {len(processed_predictions)} 张图像")
        return processed_predictions

def main():
    """主函数，演示改进的轨迹跟踪器"""
    # 加载预测结果和真实标注
    pred_path = 'results/spotgeov2-IRSTD/WTNet/predictions_8807.json'
    gt_path = 'datasets/spotgeov2-IRSTD/test_anno.json'
    
    print("正在加载预测结果和真实标注...")
    with open(pred_path, 'r') as f:
        original_predictions = json.load(f)
    
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    # 创建改进的轨迹跟踪器
    tracker = ImprovedKMeansTracker(
        max_association_distance=150.0,  # 增加距离阈值
        expected_sequence_length=5,
        trajectory_mean_distance=116.53,
        trajectory_std_distance=29.76,
        trajectory_max_distance=237.87,
        trajectory_min_distance=6.74,
        min_track_length=1,              # 降低最小轨迹长度要求
        max_frame_gap=5,                 # 增加最大帧间隔
        kmeans_n_init=15,                # 增加初始化次数
        kmeans_max_iter=500,             # 增加最大迭代次数
        silhouette_threshold=0.1,        # 降低轮廓系数阈值
        aggressive_completion=True,      # 启用积极补全
        use_dbscan_fallback=True         # 使用DBSCAN作为备选
    )
    
    # 进行轨迹跟踪处理
    processed_predictions = tracker.process_sequence(original_predictions)
    
    # 检查处理结果是否为空
    if not processed_predictions:
        print("警告：处理后的预测结果为空！")
        return
    
    # 评估改善效果
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eval_predictions import calculate_metrics
    
    try:
        original_metrics = calculate_metrics(original_predictions, ground_truth, 1000)
        processed_metrics = calculate_metrics(processed_predictions, ground_truth, 1000)
        
        improvement = {
            'precision_improvement': processed_metrics['precision'] - original_metrics['precision'],
            'recall_improvement': processed_metrics['recall'] - original_metrics['recall'],
            'f1_improvement': processed_metrics['f1'] - original_metrics['f1'],
            'mse_improvement': original_metrics['mse'] - processed_metrics['mse'],
            'tp_improvement': processed_metrics['total_tp'] - original_metrics['total_tp'],
            'fp_improvement': processed_metrics['total_fp'] - original_metrics['total_fp'],
            'fn_improvement': processed_metrics['total_fn'] - original_metrics['total_fn'],
        }
        
        # 打印结果
        print("\n=== 改进的轨迹跟踪处理效果评估 ===")
        print(f"Precision 改善: {improvement['precision_improvement']:.4f}")
        print(f"Recall 改善: {improvement['recall_improvement']:.4f}")
        print(f"F1 Score 改善: {improvement['f1_improvement']:.4f}")
        print(f"MSE 改善: {improvement['mse_improvement']:.4f}")
        print(f"TP 改善: {improvement['tp_improvement']}")
        print(f"FP 改善: {improvement['fp_improvement']}")
        print(f"FN 改善: {improvement['fn_improvement']}")
        
        print("\n=== 原始指标 ===")
        print(f"Precision: {original_metrics['precision']:.4f}")
        print(f"Recall: {original_metrics['recall']:.4f}")
        print(f"F1 Score: {original_metrics['f1']:.4f}")
        print(f"MSE: {original_metrics['mse']:.4f}")
        
        print("\n=== 处理后指标 ===")
        print(f"Precision: {processed_metrics['precision']:.4f}")
        print(f"Recall: {processed_metrics['recall']:.4f}")
        print(f"F1 Score: {processed_metrics['f1']:.4f}")
        print(f"MSE: {processed_metrics['mse']:.4f}")
        
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        print("跳过评估，直接保存结果")
    
    # 保存处理后的结果
    output_path = 'results/0808/improved_kmeans_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(processed_predictions, f, indent=2)
    print(f"\n改进的轨迹跟踪处理后的预测结果已保存到: {output_path}")

if __name__ == '__main__':
    main() 