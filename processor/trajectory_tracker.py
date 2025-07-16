import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import copy
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN

class TrajectoryTracker:
    """精确的轨迹跟踪器，实现轨迹关联、补全和异常检测过滤"""
    
    def __init__(self, 
                 max_association_distance: float = 100.0,  # 增大关联距离
                 expected_sequence_length: int = 5,
                 trajectory_mean_distance: float = 116.53,
                 trajectory_std_distance: float = 29.76,
                 trajectory_max_distance: float = 237.87,
                 trajectory_min_distance: float = 6.74,
                 outlier_threshold: float = 3.0,  # 放宽异常检测阈值
                 min_track_length: int = 1,       # 降低最小轨迹长度
                 max_frame_gap: int = 3):         # 增大最大帧间隔
        """
        初始化轨迹跟踪器
        
        Args:
            max_association_distance: 轨迹关联的最大距离阈值
            expected_sequence_length: 期望的序列长度
            trajectory_mean_distance: 轨迹平均距离
            trajectory_std_distance: 轨迹距离标准差
            trajectory_max_distance: 轨迹最大距离
            trajectory_min_distance: 轨迹最小距离
            outlier_threshold: 异常检测阈值（σ倍数）
            min_track_length: 最小轨迹长度
            max_frame_gap: 最大帧间隔
        """
        self.max_association_distance = max_association_distance
        self.expected_sequence_length = expected_sequence_length
        self.trajectory_mean_distance = trajectory_mean_distance
        self.trajectory_std_distance = trajectory_std_distance
        self.trajectory_max_distance = trajectory_max_distance
        self.trajectory_min_distance = trajectory_min_distance
        self.outlier_threshold = outlier_threshold
        self.min_track_length = min_track_length
        self.max_frame_gap = max_frame_gap
        
        # 计算统计阈值
        self.distance_threshold = min(
            self.trajectory_mean_distance + 2 * self.trajectory_std_distance,
            self.trajectory_max_distance * 0.8
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
    
    def associate_trajectories(self, frames_data: Dict[int, Dict]) -> List[List[Tuple[int, List[float]]]]:
        """
        精确的轨迹关联算法
        为每个检测点确定其所属的轨迹
        
        Args:
            frames_data: 序列帧数据 {frame: {'coords': [[x,y], ...], ...}}
            
        Returns:
            trajectories: 轨迹列表，每个轨迹是[(frame, [x,y]), ...]
        """
        frames = sorted(frames_data.keys())
        if not frames:
            return []
        
        # 初始化轨迹：第一帧的每个点作为新轨迹的起点
        trajectories = []
        for point in frames_data[frames[0]]['coords']:
            trajectories.append([(frames[0], point)])
        
        print(f"  初始化 {len(trajectories)} 条轨迹")
        
        # 逐帧进行轨迹关联
        for i in range(1, len(frames)):
            current_frame = frames[i]
            current_points = frames_data[current_frame]['coords']
            
            if not current_points:
                continue
            
            # 为当前帧的每个点找到最佳匹配的轨迹
            unassigned_points = list(range(len(current_points)))
            trajectory_extended = [False] * len(trajectories)
            
            # 第一轮：尝试关联到现有轨迹
            for traj_idx, trajectory in enumerate(trajectories):
                if trajectory_extended[traj_idx]:
                    continue
                
                last_frame, last_point = trajectory[-1]
                
                # 只关联相邻帧或间隔较小的帧
                frame_gap = current_frame - last_frame
                if frame_gap > self.max_frame_gap:
                    continue
                
                # 找到最佳匹配点
                best_point_idx = -1
                min_distance = float('inf')
                
                for point_idx in unassigned_points:
                    point = current_points[point_idx]
                    distance = self.calculate_distance(last_point, point)
                    
                    # 检查距离是否合理
                    if distance <= self.max_association_distance:
                        # 对于间隔帧，检查距离是否在合理范围内
                        if frame_gap > 1:
                            expected_distance = self.trajectory_mean_distance * frame_gap
                            distance_tolerance = self.trajectory_std_distance * frame_gap * 2  # 放宽容差
                            if abs(distance - expected_distance) > distance_tolerance:
                                continue
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_point_idx = point_idx
                
                # 如果找到匹配点，扩展轨迹
                if best_point_idx != -1:
                    trajectory.append((current_frame, current_points[best_point_idx]))
                    unassigned_points.remove(best_point_idx)
                    trajectory_extended[traj_idx] = True
                    print(f"    轨迹 {traj_idx} 扩展到帧 {current_frame}")
            
            # 第二轮：为未分配的点创建新轨迹
            for point_idx in unassigned_points:
                new_trajectory = [(current_frame, current_points[point_idx])]
                trajectories.append(new_trajectory)
                print(f"    创建新轨迹，帧 {current_frame}")
        
        # 过滤掉过短的轨迹
        valid_trajectories = [traj for traj in trajectories if len(traj) >= self.min_track_length]
        print(f"  过滤后保留 {len(valid_trajectories)} 条有效轨迹")
        
        return valid_trajectories
    
    def complete_trajectories(self, trajectories: List[List[Tuple[int, List[float]]]], 
                            frames_data: Dict[int, Dict]) -> List[List[Tuple[int, List[float]]]]:
        """
        将轨迹补充成完整的5帧序列
        
        Args:
            trajectories: 原始轨迹列表
            frames_data: 序列帧数据
            
        Returns:
            completed_trajectories: 补充后的轨迹列表
        """
        frames = sorted(frames_data.keys())
        completed_trajectories = []
        
        for trajectory in trajectories:
            if len(trajectory) >= self.expected_sequence_length:
                # 轨迹已经足够长，直接保留
                completed_trajectories.append(trajectory)
                continue
            
            # 需要补充的轨迹
            trajectory_frames = [frame for frame, _ in trajectory]
            trajectory_points = [point for _, point in trajectory]
            
            # 确定轨迹的时间范围
            min_frame = min(trajectory_frames)
            max_frame = max(trajectory_frames)
            
            # 计算期望的完整帧范围
            if len(trajectory) == 1:
                # 单点轨迹，向两边扩展
                start_frame = max(min_frame - 2, min(frames))
                end_frame = min(max_frame + 2, max(frames))
            else:
                # 多点轨迹，基于现有范围扩展
                frame_range = max_frame - min_frame
                if frame_range < self.expected_sequence_length - 1:
                    # 扩展范围到期望长度
                    extension = self.expected_sequence_length - 1 - frame_range
                    start_frame = max(min_frame - extension // 2, min(frames))
                    end_frame = min(max_frame + extension // 2, max(frames))
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
                        print(f"    补充帧 {frame}，插值位置: [{x_interp:.1f}, {y_interp:.1f}]")
            else:
                # 单点轨迹，使用外推
                single_point = trajectory_points[0]
                single_frame = trajectory_frames[0]
                
                for frame in range(start_frame, end_frame + 1):
                    if frame == single_frame:
                        completed_trajectory.append((frame, single_point))
                    else:
                        # 使用单点作为估计（简单外推）
                        completed_trajectory.append((frame, single_point))
                        print(f"    补充帧 {frame}，外推位置: {single_point}")
            
            completed_trajectories.append(completed_trajectory)
        
        print(f"  轨迹补充完成，共 {len(completed_trajectories)} 条完整轨迹")
        return completed_trajectories
    
    def filter_outliers(self, trajectories: List[List[Tuple[int, List[float]]]]) -> List[List[Tuple[int, List[float]]]]:
        """
        使用统计方法筛掉分布外的检测点
        
        Args:
            trajectories: 轨迹列表
            
        Returns:
            filtered_trajectories: 过滤后的轨迹列表
        """
        filtered_trajectories = []
        
        for trajectory in trajectories:
            if len(trajectory) < 3:
                # 短轨迹直接保留
                filtered_trajectories.append(trajectory)
                continue
            
            # 计算轨迹的统计特征
            points = np.array([point for _, point in trajectory])
            
            # 使用更宽松的DBSCAN参数
            eps = max(self.trajectory_std_distance * 3, 50)  # 增大eps
            clustering = DBSCAN(eps=eps, min_samples=1).fit(points)  # 降低min_samples
            labels = clustering.labels_
            
            # 保留非异常点（标签不为-1的点）
            filtered_trajectory = []
            for i, (frame, point) in enumerate(trajectory):
                if labels[i] != -1:  # 不是异常点
                    filtered_trajectory.append((frame, point))
                else:
                    print(f"    过滤异常点: 帧 {frame}, 位置 {point}")
            
            # 如果过滤后轨迹仍然有效，则保留
            if len(filtered_trajectory) >= self.min_track_length:
                filtered_trajectories.append(filtered_trajectory)
            else:
                print(f"    轨迹过滤后过短，丢弃")
        
        print(f"  异常检测过滤完成，保留 {len(filtered_trajectories)} 条轨迹")
        return filtered_trajectories
    
    def validate_trajectory_consistency(self, trajectory: List[Tuple[int, List[float]]]) -> bool:
        """
        验证轨迹的一致性
        
        Args:
            trajectory: 轨迹数据
            
        Returns:
            is_consistent: 是否一致
        """
        if len(trajectory) < 2:
            return True
        
        # 检查相邻帧间的距离是否合理
        for i in range(len(trajectory) - 1):
            frame1, point1 = trajectory[i]
            frame2, point2 = trajectory[i + 1]
            
            distance = self.calculate_distance(point1, point2)
            frame_gap = frame2 - frame1
            
            # 计算期望距离
            expected_distance = self.trajectory_mean_distance * frame_gap
            distance_tolerance = self.trajectory_std_distance * frame_gap * self.outlier_threshold
            
            if abs(distance - expected_distance) > distance_tolerance:
                return False
        
        return True
    
    def process_sequence(self, predictions: Dict) -> Dict:
        """
        处理整个预测结果，实现精确的轨迹跟踪
        
        Args:
            predictions: 原始预测结果
            
        Returns:
            processed_predictions: 处理后的预测结果
        """
        print("开始精确轨迹跟踪处理...")
        
        # 提取序列信息
        sequence_data = self.extract_sequence_info(predictions)
        print(f"提取了 {len(sequence_data)} 个序列的信息")
        
        processed_predictions = {}
        
        for sequence_id, frames_data in sequence_data.items():
            print(f"处理序列 {sequence_id}，包含 {len(frames_data)} 帧")
            
            # 1. 精确轨迹关联
            print("  进行轨迹关联...")
            trajectories = self.associate_trajectories(frames_data)
            
            # 如果轨迹关联失败，保留原始检测点
            if not trajectories:
                print("  轨迹关联失败，保留原始检测点")
                for frame, frame_data in frames_data.items():
                    image_name = frame_data['image_name']
                    processed_predictions[image_name] = {
                        'coords': frame_data['coords'],
                        'num_objects': frame_data['num_objects']
                    }
                continue
            
            # 2. 轨迹补全
            print("  补全轨迹到完整序列...")
            completed_trajectories = self.complete_trajectories(trajectories, frames_data)
            
            # 3. 异常检测过滤
            print("  过滤异常检测点...")
            filtered_trajectories = self.filter_outliers(completed_trajectories)
            
            # 4. 轨迹一致性验证
            print("  验证轨迹一致性...")
            valid_trajectories = []
            for trajectory in filtered_trajectories:
                if self.validate_trajectory_consistency(trajectory):
                    valid_trajectories.append(trajectory)
                else:
                    print(f"    轨迹一致性验证失败，丢弃")
            
            print(f"  最终保留 {len(valid_trajectories)} 条有效轨迹")
            
            # 如果所有轨迹都被过滤掉，保留原始检测点
            if not valid_trajectories:
                print("  所有轨迹都被过滤掉，保留原始检测点")
                for frame, frame_data in frames_data.items():
                    image_name = frame_data['image_name']
                    processed_predictions[image_name] = {
                        'coords': frame_data['coords'],
                        'num_objects': frame_data['num_objects']
                    }
                continue
            
            # 5. 转换回原始格式
            for trajectory in valid_trajectories:
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
        
        print(f"精确轨迹跟踪处理完成，处理了 {len(processed_predictions)} 张图像")
        return processed_predictions
    
    def analyze_trajectory_quality(self, trajectories: List[List[Tuple[int, List[float]]]]) -> Dict:
        """
        分析轨迹质量统计
        
        Args:
            trajectories: 轨迹列表
            
        Returns:
            statistics: 统计信息
        """
        if not trajectories:
            return {}
        
        lengths = [len(traj) for traj in trajectories]
        distances = []
        
        for trajectory in trajectories:
            for i in range(len(trajectory) - 1):
                _, point1 = trajectory[i]
                _, point2 = trajectory[i + 1]
                distances.append(self.calculate_distance(point1, point2))
        
        statistics = {
            'total_trajectories': len(trajectories),
            'avg_length': np.mean(lengths) if lengths else 0,
            'min_length': np.min(lengths) if lengths else 0,
            'max_length': np.max(lengths) if lengths else 0,
            'avg_distance': np.mean(distances) if distances else 0,
            'std_distance': np.std(distances) if distances else 0,
            'complete_trajectories': sum(1 for l in lengths if l >= self.expected_sequence_length),
            'incomplete_trajectories': sum(1 for l in lengths if l < self.expected_sequence_length)
        }
        
        return statistics

def main():
    """主函数，演示轨迹跟踪器"""
    # 加载预测结果和真实标注
    pred_path = 'results/WTNet/predictions.json'
    gt_path = 'datasets/spotgeov2-IRSTD/test_anno.json'
    
    print("正在加载预测结果和真实标注...")
    with open(pred_path, 'r') as f:
        original_predictions = json.load(f)
    
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    # 创建轨迹跟踪器
    tracker = TrajectoryTracker(
        max_association_distance=100.0,  # 增大关联距离
        expected_sequence_length=5,
        trajectory_mean_distance=116.53,
        trajectory_std_distance=29.76,
        trajectory_max_distance=237.87,
        trajectory_min_distance=6.74,
        outlier_threshold=3.0,  # 放宽异常检测阈值
        min_track_length=1,     # 降低最小轨迹长度
        max_frame_gap=3         # 增大最大帧间隔
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
        print("\n=== 轨迹跟踪处理效果评估 ===")
        print(f"Precision 改善: {improvement['precision_improvement']:.4f}")
        print(f"Recall 改善: {improvement['recall_improvement']:.4f}")
        print(f"F1 Score 改善: {improvement['f1_improvement']:.4f}")
        print(f"MSE 改善: {improvement['mse_improvement']:.4f}")
        print(f"TP 改善: {improvement['tp_improvement']}")
        print(f"FP 改善: {improvement['fp_improvement']}")
        print(f"FN 改善: {improvement['fn_improvement']}")
        
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        print("跳过评估，直接保存结果")
    
    # 保存处理后的结果
    output_path = 'results/WTNet/trajectory_tracked_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(processed_predictions, f, indent=2)
    print(f"\n轨迹跟踪处理后的预测结果已保存到: {output_path}")

if __name__ == '__main__':
    main() 