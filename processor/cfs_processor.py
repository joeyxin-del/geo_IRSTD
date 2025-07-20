import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import copy
from sklearn.linear_model import LinearRegression
import math

class CFSProcessor:
    """基于CFS（Candidate Filtration and Supplement）的序列后处理器"""
    
    def __init__(self, 
                 r_min: float = 50.0,      # 最小间隔距离（进一步调小）
                 r_max: float = 140.0,     # 最大间隔距离（适当放宽）
                 n_points: int = 3,        # 每组最大点数
                 max_frame_gap: int = 2,   # 最大帧间隔
                 sequence_length: int = 5, # 序列长度
                 confidence_threshold: float = 0.05,
                 # 基于统计的先验参数
                 trajectory_mean_distance: float = 116.53,
                 trajectory_std_distance: float = 29.76):
        """
        初始化CFS处理器
        
        Args:
            r_min: 最小间隔距离
            r_max: 最大间隔距离
            n_points: 每组最大点数
            max_frame_gap: 最大帧间隔
            sequence_length: 序列长度
            confidence_threshold: 置信度阈值
            trajectory_mean_distance: 轨迹平均距离
            trajectory_std_distance: 轨迹距离标准差
        """
        self.r_min = r_min
        self.r_max = r_max
        self.n_points = n_points
        self.max_frame_gap = max_frame_gap
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.trajectory_mean_distance = trajectory_mean_distance
        self.trajectory_std_distance = trajectory_std_distance
        
        # 更宽松的搜索窗口
        self.search_window_min = max(r_min, trajectory_mean_distance - 2 * trajectory_std_distance)
        self.search_window_max = min(r_max, trajectory_mean_distance + 2 * trajectory_std_distance)
        
        print(f"CFS处理器初始化完成:")
        print(f"  搜索窗口: [{self.search_window_min:.2f}, {self.search_window_max:.2f}]")
        print(f"  每组最大点数: {self.n_points}")
        print(f"  最大帧间隔: {self.max_frame_gap}")
        print(f"  序列长度: {self.sequence_length}")
    
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
    
    def find_candidate_trajectories_continuous(self, frames_data: Dict[int, Dict]) -> List[List[Tuple[int, List[float]]]]:
        """查找连续帧的候选轨迹"""
        frames = sorted(frames_data.keys())
        candidate_trajectories = []
        
        # 从起始帧开始搜索
        for start_frame_idx in range(len(frames) - 2):  # 至少需要3帧
            start_frame = frames[start_frame_idx]
            start_coords = frames_data[start_frame]['coords']
            
            for start_point in start_coords:
                trajectory = [(start_frame, start_point)]
                current_frame_idx = start_frame_idx
                current_point = start_point
                
                # 逐帧搜索后续点
                while len(trajectory) < self.n_points and current_frame_idx < len(frames) - 1:
                    next_frame_idx = current_frame_idx + 1
                    next_frame = frames[next_frame_idx]
                    next_coords = frames_data[next_frame]['coords']
                    
                    # 在环形搜索窗口内寻找下一个点
                    best_next_point = None
                    min_distance_diff = float('inf')
                    
                    for next_point in next_coords:
                        distance = self.calculate_distance(current_point, next_point)
                        
                        # 检查是否在搜索窗口内
                        if self.search_window_min <= distance <= self.search_window_max:
                            # 选择距离最接近期望距离的点
                            distance_diff = abs(distance - self.trajectory_mean_distance)
                            if distance_diff < min_distance_diff:
                                min_distance_diff = distance_diff
                                best_next_point = next_point
                    
                    if best_next_point is not None:
                        trajectory.append((next_frame, best_next_point))
                        current_point = best_next_point
                        current_frame_idx = next_frame_idx
                    else:
                        break
                
                # 如果轨迹长度足够，添加到候选列表
                if len(trajectory) >= 2:
                    candidate_trajectories.append(trajectory)
        
        return candidate_trajectories
    
    def find_candidate_trajectories_inconsecutive(self, frames_data: Dict[int, Dict]) -> List[List[Tuple[int, List[float]]]]:
        """查找非连续帧的候选轨迹（更严格的版本）"""
        frames = sorted(frames_data.keys())
        candidate_trajectories = []
        
        # 只处理间隔2帧的情况，更严格
        for start_frame_idx in range(len(frames) - 2):
            start_frame = frames[start_frame_idx]
            start_coords = frames_data[start_frame]['coords']
            
            for start_point in start_coords:
                # 只搜索间隔2帧的点
                if start_frame_idx + 2 < len(frames):
                    target_frame = frames[start_frame_idx + 2]
                    target_coords = frames_data[target_frame]['coords']
                    
                    for target_point in target_coords:
                        distance = self.calculate_distance(start_point, target_point)
                        
                        # 更严格的距离检查
                        expected_distance = self.trajectory_mean_distance * 2
                        distance_tolerance = self.trajectory_std_distance * 1.5  # 更严格的容差
                        
                        if abs(distance - expected_distance) <= distance_tolerance:
                            # 尝试找到中间帧的点
                            middle_trajectory = [(start_frame, start_point)]
                            
                            # 寻找中间帧的插值点
                            middle_frame_idx = start_frame_idx + 1
                            if middle_frame_idx < len(frames):
                                middle_frame = frames[middle_frame_idx]
                                middle_coords = frames_data[middle_frame]['coords']
                                
                                # 计算插值位置
                                alpha = 0.5  # 中间位置
                                interpolated_point = [
                                    start_point[0] + alpha * (target_point[0] - start_point[0]),
                                    start_point[1] + alpha * (target_point[1] - start_point[1])
                                ]
                                
                                # 在中间帧寻找最接近插值点的检测点
                                best_middle_point = None
                                min_distance = float('inf')
                                
                                for middle_point in middle_coords:
                                    dist = self.calculate_distance(interpolated_point, middle_point)
                                    if dist < min_distance and dist <= self.search_window_max * 0.8:  # 更严格的阈值
                                        min_distance = dist
                                        best_middle_point = middle_point
                                
                                if best_middle_point is not None:
                                    middle_trajectory.append((middle_frame, best_middle_point))
                            
                            middle_trajectory.append((target_frame, target_point))
                            
                            if len(middle_trajectory) >= 2:
                                candidate_trajectories.append(middle_trajectory)
        
        return candidate_trajectories
    
    def linear_fitting_validation(self, trajectory: List[Tuple[int, List[float]]]) -> Tuple[bool, float, List[float]]:
        """对轨迹进行线性拟合验证（放宽条件）"""
        if len(trajectory) < 2:
            return False, 0.0, []
        
        # 提取坐标
        coords = np.array([point for _, point in trajectory])
        
        # 计算轨迹的直线性指标
        if len(coords) >= 3:
            # 使用线性回归拟合
            x = coords[:, 0].reshape(-1, 1)
            y = coords[:, 1]
            
            reg = LinearRegression()
            reg.fit(x, y)
            
            # 计算拟合误差
            y_pred = reg.predict(x)
            mse = np.mean((y - y_pred) ** 2)
            
            # 计算R²分数
            r2_score = reg.score(x, y)
            
            # 放宽的判断条件
            is_linear = r2_score > 0.7 and mse < 200  # 降低阈值
            
            return is_linear, r2_score, [reg.coef_[0], reg.intercept_]
        else:
            # 只有2个点，直接计算直线
            p1, p2 = coords[0], coords[1]
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] != p1[0] else float('inf')
            intercept = p1[1] - slope * p1[0] if slope != float('inf') else p1[0]
            
            return True, 1.0, [slope, intercept]
    
    def calculate_uniform_speed(self, trajectory: List[Tuple[int, List[float]]]) -> Tuple[bool, float]:
        """计算轨迹的匀速性（放宽条件）"""
        if len(trajectory) < 3:
            return True, 0.0
        
        speeds = []
        for i in range(len(trajectory) - 1):
            frame1, point1 = trajectory[i]
            frame2, point2 = trajectory[i + 1]
            
            distance = self.calculate_distance(point1, point2)
            time_gap = frame2 - frame1
            
            if time_gap > 0:
                speed = distance / time_gap
                speeds.append(speed)
        
        if not speeds:
            return True, 0.0
        
        # 计算速度的标准差
        speed_std = np.std(speeds)
        speed_mean = np.mean(speeds)
        
        # 放宽的匀速判断（速度变化小于25%）
        is_uniform = speed_std / max(speed_mean, 1e-6) < 0.25
        
        return is_uniform, speed_mean
    
    def filter_and_validate_trajectories(self, candidate_trajectories: List[List[Tuple[int, List[float]]]]) -> List[List[Tuple[int, List[float]]]]:
        """过滤和验证候选轨迹（放宽条件）"""
        valid_trajectories = []
        
        for trajectory in candidate_trajectories:
            # 1. 线性拟合验证
            is_linear, r2_score, line_params = self.linear_fitting_validation(trajectory)
            
            # 2. 匀速性验证
            is_uniform, avg_speed = self.calculate_uniform_speed(trajectory)
            
            # 3. 放宽的几何约束
            is_geometric_valid = True
            if len(trajectory) >= 3:
                # 检查轨迹的几何合理性
                coords = np.array([point for _, point in trajectory])
                # 计算轨迹的总长度
                total_length = 0
                for i in range(len(coords) - 1):
                    total_length += self.calculate_distance(coords[i], coords[i+1])
                
                # 放宽的长度检查
                expected_length = self.trajectory_mean_distance * (len(coords) - 1)
                if total_length > expected_length * 3 or total_length < expected_length * 0.3:
                    is_geometric_valid = False
            
            # 4. 放宽的综合判断
            if is_linear and is_uniform and is_geometric_valid:
                valid_trajectories.append(trajectory)
                print(f"  验证通过 - 轨迹长度: {len(trajectory)}, R²: {r2_score:.3f}, 平均速度: {avg_speed:.2f}")
        
        return valid_trajectories
    
    def supplement_missing_detections(self, frames_data: Dict[int, Dict], valid_trajectories: List[List[Tuple[int, List[float]]]]) -> Dict[int, Dict]:
        """基于有效轨迹补全缺失检测（更保守）"""
        supplemented_data = copy.deepcopy(frames_data)
        frames = sorted(frames_data.keys())
        
        # 为每个有效轨迹计算完整的轨迹线
        for trajectory in valid_trajectories:
            if len(trajectory) < 2:
                continue
            
            # 提取轨迹参数
            coords = np.array([point for _, point in trajectory])
            x = coords[:, 0].reshape(-1, 1)
            y = coords[:, 1]
            
            # 线性拟合
            reg = LinearRegression()
            reg.fit(x, y)
            
            # 为缺失的帧补全检测点（更保守的策略）
            for frame in frames:
                if frame not in [f for f, _ in trajectory]:
                    # 计算该帧在轨迹上的位置
                    frame_times = [f for f, _ in trajectory]
                    x_coords = [p[0] for _, p in trajectory]
                    
                    # 线性插值估计x坐标
                    if len(frame_times) >= 2:
                        x_interp = np.interp(frame, frame_times, x_coords)
                        
                        # 使用拟合的直线计算y坐标
                        y_interp = reg.predict([[x_interp]])[0]
                        
                        # 检查补全的点是否合理
                        interpolated_point = [x_interp, y_interp]
                        
                        # 更严格的验证补全点的合理性
                        is_reasonable = True
                        for existing_frame, existing_point in trajectory:
                            distance = self.calculate_distance(interpolated_point, existing_point)
                            frame_diff = abs(frame - existing_frame)
                            
                            # 更严格的距离检查
                            expected_distance = self.trajectory_mean_distance * frame_diff
                            if distance > expected_distance * 1.5:  # 更严格的阈值
                                is_reasonable = False
                                break
                        
                        # 额外的约束：检查是否与现有检测点冲突
                        if is_reasonable and frame in supplemented_data:
                            existing_coords = supplemented_data[frame]['coords']
                            for existing_coord in existing_coords:
                                if self.calculate_distance(interpolated_point, existing_coord) < 30:  # 避免重复
                                    is_reasonable = False
                                    break
                        
                        if is_reasonable:
                            # 添加到补全数据中
                            if frame not in supplemented_data:
                                supplemented_data[frame] = {
                                    'coords': [],
                                    'num_objects': 0,
                                    'image_name': f"supplemented_{frame}"
                                }
                            
                            supplemented_data[frame]['coords'].append(interpolated_point)
                            supplemented_data[frame]['num_objects'] = len(supplemented_data[frame]['coords'])
                            print(f"  补全帧 {frame}，位置: {interpolated_point}")
        
        return supplemented_data
    
    def remove_duplicate_detections(self, frames_data: Dict[int, Dict], distance_threshold: float = 15.0) -> Dict[int, Dict]:
        """移除重复检测点（更严格的去重）"""
        cleaned_data = copy.deepcopy(frames_data)
        
        for frame, frame_data in cleaned_data.items():
            coords = frame_data['coords']
            if len(coords) <= 1:
                continue
            
            # 使用距离矩阵找到重复点
            kept_coords = []
            for i, coord1 in enumerate(coords):
                is_duplicate = False
                for j, coord2 in enumerate(coords):
                    if i != j and self.calculate_distance(coord1, coord2) < distance_threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    kept_coords.append(coord1)
            
            cleaned_data[frame]['coords'] = kept_coords
            cleaned_data[frame]['num_objects'] = len(kept_coords)
        
        return cleaned_data
    
    def process_sequence(self, predictions: Dict) -> Dict:
        """
        使用CFS方法处理序列
        
        Args:
            predictions: 原始预测结果
            
        Returns:
            processed_predictions: 处理后的预测结果
        """
        print("开始CFS序列后处理...")
        
        # 1. 提取序列信息
        sequence_data = self.extract_sequence_info(predictions)
        print(f"提取了 {len(sequence_data)} 个序列的信息")
        
        processed_predictions = {}
        
        for sequence_id, frames_data in sequence_data.items():
            print(f"处理序列 {sequence_id}，包含 {len(frames_data)} 帧")
            
            # 2. 查找连续帧候选轨迹
            print("  查找连续帧候选轨迹...")
            continuous_trajectories = self.find_candidate_trajectories_continuous(frames_data)
            print(f"  找到 {len(continuous_trajectories)} 个连续帧候选轨迹")
            
            # 3. 查找非连续帧候选轨迹
            print("  查找非连续帧候选轨迹...")
            inconsecutive_trajectories = self.find_candidate_trajectories_inconsecutive(frames_data)
            print(f"  找到 {len(inconsecutive_trajectories)} 个非连续帧候选轨迹")
            
            # 4. 合并所有候选轨迹
            all_candidates = continuous_trajectories + inconsecutive_trajectories
            print(f"  总共 {len(all_candidates)} 个候选轨迹")
            
            # 5. 过滤和验证轨迹
            print("  过滤和验证轨迹...")
            valid_trajectories = self.filter_and_validate_trajectories(all_candidates)
            print(f"  验证通过 {len(valid_trajectories)} 个有效轨迹")
            
            # 6. 基于有效轨迹补全缺失检测
            print("  补全缺失检测...")
            supplemented_data = self.supplement_missing_detections(frames_data, valid_trajectories)
            
            # 7. 转换回原始格式，保留所有原始检测结果
            for frame, frame_data in supplemented_data.items():
                image_name = frame_data['image_name']
                if not image_name.startswith('supplemented_'):
                    # 使用原始图像名称
                    for orig_name, orig_data in predictions.items():
                        if f"{sequence_id}_{frame}_" in orig_name:
                            image_name = orig_name
                            break
                
                # 保留原始检测结果，只添加补全的检测点
                if image_name in predictions:
                    # 合并原始检测和补全检测
                    original_coords = predictions[image_name]['coords']
                    supplemented_coords = frame_data['coords']
                    
                    # 去重合并
                    all_coords = original_coords.copy()
                    for supp_coord in supplemented_coords:
                        # 检查是否与原始检测点重复
                        is_duplicate = False
                        for orig_coord in original_coords:
                            if self.calculate_distance(supp_coord, orig_coord) < 20:  # 20像素内的认为是重复
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            all_coords.append(supp_coord)
                    
                    processed_predictions[image_name] = {
                        'coords': all_coords,
                        'num_objects': len(all_coords)
                    }
                else:
                    # 如果原始预测中没有这个图像，直接使用补全结果
                    processed_predictions[image_name] = {
                        'coords': frame_data['coords'],
                        'num_objects': frame_data['num_objects']
                    }
        
        print(f"CFS序列后处理完成，处理了 {len(processed_predictions)} 张图像")
        return processed_predictions
    
    def evaluate_improvement(self, original_predictions: Dict, processed_predictions: Dict, ground_truth: List[Dict]) -> Dict:
        """评估CFS处理的效果改善"""
        import sys
        import os
        # 添加父目录到路径，以便导入eval_predictions
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from eval_predictions import calculate_metrics
        
        # 计算原始预测的指标
        original_metrics = calculate_metrics(original_predictions, ground_truth, 1000)
        
        # 计算处理后预测的指标
        processed_metrics = calculate_metrics(processed_predictions, ground_truth, 1000 )
        
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
    """主函数，演示CFS序列后处理"""
    # 加载预测结果和真实标注
    pred_path = 'results/WTNet/predictions.json'
    gt_path = 'datasets/spotgeov2-IRSTD/test_anno.json'
    
    print("正在加载预测结果和真实标注...")
    with open(pred_path, 'r') as f:
        original_predictions = json.load(f)
    
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    # 创建CFS处理器（优化参数）
    processor = CFSProcessor(
        r_min=50.0,                     # 更小的最小间隔距离
        r_max=140.0,                    # 适当放宽的最大间隔距离
        n_points=3,                     # 每组最大点数
        max_frame_gap=2,                # 最大帧间隔
        sequence_length=5,              # 序列长度
        confidence_threshold=0.05,      # 置信度阈值
        trajectory_mean_distance=116.53,  # 轨迹平均距离
        trajectory_std_distance=29.76     # 轨迹距离标准差
    )
    
    # 进行CFS序列后处理
    processed_predictions = processor.process_sequence(original_predictions)
    
    # 评估改善效果
    improvement = processor.evaluate_improvement(original_predictions, processed_predictions, ground_truth)
    
    # 打印结果
    print("\n=== CFS序列后处理效果评估 ===")
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
    output_path = 'results/WTNet/cfs_processed_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(processed_predictions, f, indent=2)
    print(f"\nCFS处理后的预测结果已保存到: {output_path}")

if __name__ == '__main__':
    main() 