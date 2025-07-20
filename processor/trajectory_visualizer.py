import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import os

class TrajectoryVisualizer:
    """轨迹可视化工具，用于分析和展示轨迹跟踪的效果"""
    
    def __init__(self, save_dir: str = 'visualization_results'):
        """
        初始化轨迹可视化器
        
        Args:
            save_dir: 保存可视化结果的目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置颜色映射
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
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
    
    def visualize_trajectories_static(self, trajectories: List[List[Tuple[int, List[float]]]], 
                                    sequence_id: int, title: str = "轨迹可视化"):
        """
        静态可视化轨迹
        
        Args:
            trajectories: 轨迹列表
            sequence_id: 序列ID
            title: 图表标题
        """
        plt.figure(figsize=(12, 8))
        
        # 绘制每个轨迹
        for i, trajectory in enumerate(trajectories):
            if len(trajectory) < 2:
                continue
            
            frames = [frame for frame, _ in trajectory]
            points = [point for _, point in trajectory]
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            # 绘制轨迹线
            color = self.colors[i % len(self.colors)]
            plt.plot(x_coords, y_coords, 'o-', color=color, linewidth=2, 
                    markersize=6, label=f'轨迹 {i+1} (长度: {len(trajectory)})')
            
            # 标注帧号
            for j, (frame, point) in enumerate(trajectory):
                plt.annotate(f'{frame}', (point[0], point[1]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.7)
        
        plt.xlabel('X 坐标')
        plt.ylabel('Y 坐标')
        plt.title(f'{title} - 序列 {sequence_id}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 保存图片
        save_path = os.path.join(self.save_dir, f'trajectory_static_seq_{sequence_id}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"静态轨迹图已保存到: {save_path}")
    
    def visualize_trajectories_animated(self, trajectories: List[List[Tuple[int, List[float]]]], 
                                      sequence_id: int, title: str = "轨迹动画"):
        """
        动态可视化轨迹
        
        Args:
            trajectories: 轨迹列表
            sequence_id: 序列ID
            title: 图表标题
        """
        if not trajectories:
            return
        
        # 获取所有帧的范围
        all_frames = set()
        for trajectory in trajectories:
            for frame, _ in trajectory:
                all_frames.add(frame)
        
        frames = sorted(all_frames)
        if not frames:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 设置坐标轴范围
        all_points = []
        for trajectory in trajectories:
            for _, point in trajectory:
                all_points.append(point)
        
        if all_points:
            all_points = np.array(all_points)
            x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
            y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
            
            # 添加一些边距
            margin = 50
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)
        
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')
        ax.set_title(f'{title} - 序列 {sequence_id}')
        ax.grid(True, alpha=0.3)
        
        # 初始化散点图
        scatters = []
        for i in range(len(trajectories)):
            scatter = ax.scatter([], [], c=[self.colors[i % len(self.colors)]], 
                               s=100, alpha=0.7, label=f'轨迹 {i+1}')
            scatters.append(scatter)
        
        # 初始化轨迹线
        lines = []
        for i in range(len(trajectories)):
            line, = ax.plot([], [], 'o-', color=self.colors[i % len(self.colors)], 
                          linewidth=2, markersize=4, alpha=0.8)
            lines.append(line)
        
        ax.legend()
        
        def animate(frame_idx):
            current_frame = frames[frame_idx]
            
            # 更新每个轨迹
            for i, trajectory in enumerate(trajectories):
                # 找到当前帧及之前的所有点
                current_points = []
                for frame, point in trajectory:
                    if frame <= current_frame:
                        current_points.append(point)
                
                if current_points:
                    current_points = np.array(current_points)
                    x_coords = current_points[:, 0]
                    y_coords = current_points[:, 1]
                    
                    # 更新散点图
                    scatters[i].set_offsets(current_points)
                    
                    # 更新轨迹线
                    lines[i].set_data(x_coords, y_coords)
                else:
                    # 清空数据
                    scatters[i].set_offsets(np.empty((0, 2)))
                    lines[i].set_data([], [])
            
            ax.set_title(f'{title} - 序列 {sequence_id} - 帧 {current_frame}')
            return scatters + lines
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=len(frames), 
                                     interval=500, blit=True, repeat=True)
        
        # 保存动画
        save_path = os.path.join(self.save_dir, f'trajectory_animated_seq_{sequence_id}.gif')
        anim.save(save_path, writer='pillow', fps=2)
        plt.close()
        
        print(f"轨迹动画已保存到: {save_path}")
    
    def visualize_clustering_process(self, frames_data: Dict[int, Dict], 
                                   trajectories: List[List[Tuple[int, List[float]]]], 
                                   sequence_id: int):
        """
        可视化聚类过程
        
        Args:
            frames_data: 序列帧数据
            trajectories: 聚类后的轨迹
            sequence_id: 序列ID
        """
        # 收集所有检测点
        all_points = []
        all_frames = []
        
        for frame, frame_data in frames_data.items():
            for point in frame_data['coords']:
                all_points.append(point)
                all_frames.append(frame)
        
        if not all_points:
            return
        
        all_points = np.array(all_points)
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左图：原始检测点
        ax1.scatter(all_points[:, 0], all_points[:, 1], c='blue', alpha=0.6, s=50)
        ax1.set_xlabel('X 坐标')
        ax1.set_ylabel('Y 坐标')
        ax1.set_title(f'原始检测点 - 序列 {sequence_id}')
        ax1.grid(True, alpha=0.3)
        
        # 右图：聚类后的轨迹
        for i, trajectory in enumerate(trajectories):
            if len(trajectory) < 2:
                continue
            
            frames = [frame for frame, _ in trajectory]
            points = [point for _, point in trajectory]
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            color = self.colors[i % len(self.colors)]
            ax2.plot(x_coords, y_coords, 'o-', color=color, linewidth=2, 
                    markersize=6, label=f'轨迹 {i+1}')
        
        ax2.set_xlabel('X 坐标')
        ax2.set_ylabel('Y 坐标')
        ax2.set_title(f'聚类后的轨迹 - 序列 {sequence_id}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.save_dir, f'clustering_comparison_seq_{sequence_id}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"聚类对比图已保存到: {save_path}")
    
    def visualize_trajectory_statistics(self, trajectories: List[List[Tuple[int, List[float]]]], 
                                      sequence_id: int):
        """
        可视化轨迹统计信息
        
        Args:
            trajectories: 轨迹列表
            sequence_id: 序列ID
        """
        if not trajectories:
            return
        
        # 计算统计信息
        lengths = [len(traj) for traj in trajectories]
        distances = []
        
        for trajectory in trajectories:
            for i in range(len(trajectory) - 1):
                _, point1 = trajectory[i]
                _, point2 = trajectory[i + 1]
                dist = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
                distances.append(dist)
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 轨迹长度分布
        ax1.hist(lengths, bins=min(10, len(set(lengths))), alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('轨迹长度')
        ax1.set_ylabel('频次')
        ax1.set_title('轨迹长度分布')
        ax1.grid(True, alpha=0.3)
        
        # 距离分布
        if distances:
            ax2.hist(distances, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.set_xlabel('相邻帧距离')
            ax2.set_ylabel('频次')
            ax2.set_title('相邻帧距离分布')
            ax2.grid(True, alpha=0.3)
        
        # 轨迹数量统计
        trajectory_types = ['短轨迹(1-2帧)', '中等轨迹(3-4帧)', '长轨迹(5+帧)']
        short_count = sum(1 for l in lengths if l <= 2)
        medium_count = sum(1 for l in lengths if 3 <= l <= 4)
        long_count = sum(1 for l in lengths if l >= 5)
        counts = [short_count, medium_count, long_count]
        
        ax3.bar(trajectory_types, counts, color=['red', 'orange', 'green'], alpha=0.7)
        ax3.set_ylabel('轨迹数量')
        ax3.set_title('轨迹类型分布')
        ax3.grid(True, alpha=0.3)
        
        # 轨迹质量评分
        quality_scores = []
        for trajectory in trajectories:
            if len(trajectory) < 2:
                quality_scores.append(0)
                continue
            
            # 计算轨迹的直线性
            frames = [frame for frame, _ in trajectory]
            points = [point for _, point in trajectory]
            
            # 计算轨迹的总长度和直线距离
            total_length = 0
            for i in range(len(points) - 1):
                dist = np.sqrt((points[i][0] - points[i+1][0])**2 + (points[i][1] - points[i+1][1])**2)
                total_length += dist
            
            # 直线距离
            start_point = points[0]
            end_point = points[-1]
            straight_distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
            
            # 直线性评分（直线距离/总长度）
            if total_length > 0:
                linearity = straight_distance / total_length
            else:
                linearity = 0
            
            quality_scores.append(linearity)
        
        ax4.hist(quality_scores, bins=10, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_xlabel('直线性评分')
        ax4.set_ylabel('频次')
        ax4.set_title('轨迹直线性分布')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.save_dir, f'trajectory_statistics_seq_{sequence_id}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"轨迹统计图已保存到: {save_path}")
    
    def compare_original_vs_processed(self, original_predictions: Dict, 
                                    processed_predictions: Dict, 
                                    sequence_id: int):
        """
        比较原始预测和处理后的预测
        
        Args:
            original_predictions: 原始预测结果
            processed_predictions: 处理后的预测结果
            sequence_id: 序列ID
        """
        # 提取序列数据
        original_sequence = self.extract_sequence_info(original_predictions)
        processed_sequence = self.extract_sequence_info(processed_predictions)
        
        if sequence_id not in original_sequence or sequence_id not in processed_sequence:
            print(f"序列 {sequence_id} 在预测结果中不存在")
            return
        
        original_frames = original_sequence[sequence_id]
        processed_frames = processed_sequence[sequence_id]
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左图：原始预测
        for frame, frame_data in original_frames.items():
            for point in frame_data['coords']:
                ax1.scatter(point[0], point[1], c='red', alpha=0.6, s=50)
                ax1.annotate(f'{frame}', (point[0], point[1]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.7)
        
        ax1.set_xlabel('X 坐标')
        ax1.set_ylabel('Y 坐标')
        ax1.set_title(f'原始预测 - 序列 {sequence_id}')
        ax1.grid(True, alpha=0.3)
        
        # 右图：处理后预测
        for frame, frame_data in processed_frames.items():
            for point in frame_data['coords']:
                ax2.scatter(point[0], point[1], c='green', alpha=0.6, s=50)
                ax2.annotate(f'{frame}', (point[0], point[1]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.7)
        
        ax2.set_xlabel('X 坐标')
        ax2.set_ylabel('Y 坐标')
        ax2.set_title(f'处理后预测 - 序列 {sequence_id}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.save_dir, f'comparison_seq_{sequence_id}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"对比图已保存到: {save_path}")

def main():
    """主函数，演示轨迹可视化工具"""
    # 加载预测结果
    pred_path = 'results/WTNet/predictions.json'
    processed_path = 'results/WTNet/kmeans_trajectory_predictions.json'
    
    print("正在加载预测结果...")
    with open(pred_path, 'r') as f:
        original_predictions = json.load(f)
    
    try:
        with open(processed_path, 'r') as f:
            processed_predictions = json.load(f)
    except FileNotFoundError:
        print(f"处理后的预测结果文件 {processed_path} 不存在")
        processed_predictions = None
    
    # 创建可视化器
    visualizer = TrajectoryVisualizer()
    
    # 提取序列信息
    sequence_data = visualizer.extract_sequence_info(original_predictions)
    
    # 为前几个序列创建可视化
    for sequence_id in list(sequence_data.keys())[:3]:  # 只处理前3个序列
        print(f"为序列 {sequence_id} 创建可视化...")
        
        frames_data = sequence_data[sequence_id]
        
        # 这里需要从K-means轨迹跟踪器获取轨迹数据
        # 由于可视化器是独立的，我们创建一个简单的轨迹示例
        # 在实际使用中，您需要将轨迹数据传递给可视化器
        
        # 创建简单的轨迹示例（基于检测点的时序关联）
        trajectories = []
        frames = sorted(frames_data.keys())
        
        for frame in frames:
            coords = frames_data[frame]['coords']
            for point in coords:
                trajectories.append([(frame, point)])
        
        # 创建可视化
        visualizer.visualize_trajectories_static(trajectories, sequence_id, "检测点分布")
        
        if processed_predictions:
            visualizer.compare_original_vs_processed(original_predictions, processed_predictions, sequence_id)
    
    print("可视化完成！请查看 visualization_results 目录中的结果。")

if __name__ == '__main__':
    main() 