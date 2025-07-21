import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端，避免Qt依赖
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple, Optional, Set
import sys
import os
from PIL import Image
import cv2

# 添加父目录到路径，以便导入angle_distance_processor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processor.angle_distance_processor import AngleDistanceProcessor

class SingleSequenceAngleDistanceProcessor:
    """单序列角度距离处理器，用于分析和可视化单个序列的角度距离后处理效果"""
    
    def __init__(self, 
                 base_distance_threshold: float = 1000.0,
                 angle_tolerance: float = 2.5,
                 min_angle_count: int = 2,
                 step_tolerance: float = 0.2,
                 min_step_count: int = 1,
                 point_distance_threshold: float = 200.0):
        """
        初始化单序列角度距离处理器
        
        Args:
            base_distance_threshold: 基础距离阈值
            angle_tolerance: 角度容差（度）
            min_angle_count: 最小角度出现次数
            step_tolerance: 步长容差（比例）
            min_step_count: 最小步长出现次数
            point_distance_threshold: 重合点过滤阈值
        """
        self.processor = AngleDistanceProcessor(
            base_distance_threshold=base_distance_threshold,
            angle_tolerance=angle_tolerance,
            min_angle_count=min_angle_count,
            step_tolerance=step_tolerance,
            min_step_count=min_step_count,
            point_distance_threshold=point_distance_threshold
        )
        
        # 可视化参数
        self.image_size = (800, 600)
        self.point_size = 50
        self.line_width = 2
        self.colors = {
            'original': 'red',
            'completed': 'blue', 
            'filtered': 'green',
            'background': 'black'
        }
    
    def load_sequence_data(self, predictions_path: str, sequence_id: int) -> Dict[int, Dict]:
        """加载指定序列的数据"""
        print(f"正在加载序列 {sequence_id} 的数据...")
        
        with open(predictions_path, 'r') as f:
            predictions = json.load(f)
        
        # 提取指定序列的数据
        sequence_data = {}
        for img_name, pred_info in predictions.items():
            parts = img_name.split('_')
            if len(parts) >= 2:
                try:
                    seq_id = int(parts[0])
                    frame = int(parts[1])
                    if seq_id == sequence_id:
                        sequence_data[frame] = {
                            'coords': pred_info['coords'],
                            'num_objects': pred_info['num_objects'],
                            'image_name': img_name
                        }
                except ValueError:
                    continue
        
        print(f"序列 {sequence_id} 包含 {len(sequence_data)} 帧")
        return sequence_data
    
    def process_single_sequence(self, sequence_id: int, frames_data: Dict[int, Dict]) -> Tuple[Dict[int, Dict], Dict[int, Dict], Dict]:
        """处理单个序列，返回原始、补全和过滤后的结果"""
        print(f"开始处理序列 {sequence_id}...")
        
        # 使用angle_distance_processor的process_sequence方法
        completed_sequence, filtered_sequence = self.processor.process_sequence(sequence_id, frames_data)
        
        # 收集处理统计信息
        stats = self.collect_processing_stats(frames_data, completed_sequence, filtered_sequence)
        
        return frames_data, completed_sequence, filtered_sequence, stats
    
    def collect_processing_stats(self, original: Dict[int, Dict], 
                               completed: Dict[int, Dict], 
                               filtered: Dict[int, Dict]) -> Dict:
        """收集处理统计信息"""
        stats = {
            'original_points': sum(len(frame_data['coords']) for frame_data in original.values()),
            'completed_points': sum(len(frame_data['coords']) for frame_data in completed.values()),
            'filtered_points': sum(len(frame_data['coords']) for frame_data in filtered.values()),
            'original_frames': len([f for f, data in original.items() if data['coords']]),
            'completed_frames': len([f for f, data in completed.items() if data['coords']]),
            'filtered_frames': len([f for f, data in filtered.items() if data['coords']]),
            'total_frames': len(original),
            'added_points': 0,
            'removed_points': 0
        }
        
        # 计算添加和移除的点数
        for frame in original.keys():
            original_count = len(original.get(frame, {}).get('coords', []))
            completed_count = len(completed.get(frame, {}).get('coords', []))
            filtered_count = len(filtered.get(frame, {}).get('coords', []))
            
            stats['added_points'] += max(0, completed_count - original_count)
            stats['removed_points'] += max(0, original_count - filtered_count)
        
        return stats
    
    def create_visualization(self, original: Dict[int, Dict], 
                           completed: Dict[int, Dict], 
                           filtered: Dict[int, Dict],
                           sequence_id: int,
                           stats: Dict) -> None:
        """创建三排可视化：原图、填充后、过滤后，参考analyze_trajectory_changes.py的风格"""
        
        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Sequence {sequence_id} Angle-Distance Post-processing Analysis', fontsize=16, fontweight='bold', color='white')
        fig.patch.set_facecolor('black')
        
        # 设置坐标轴范围
        all_points = []
        for data in [original, completed, filtered]:
            for frame_data in data.values():
                all_points.extend(frame_data['coords'])
        
        if all_points:
            all_points = np.array(all_points)
            x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
            y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
            
            # 添加边距
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            x_min -= x_margin
            x_max += x_margin
            y_min -= y_margin
            y_max += y_margin
        else:
            x_min, x_max, y_min, y_max = 0, 1000, 0, 1000
        
        # 绘制原始数据
        self.plot_sequence_improved(axes[0], original, 'Original Detection', 'red', x_min, x_max, y_min, y_max, sequence_id, show_gt=True)
        axes[0].set_title(f'Original Detection\n{stats["original_points"]} points, {stats["original_frames"]} frames', 
                         fontsize=12, fontweight='bold', color='white')
        
        # 绘制补全后数据
        self.plot_sequence_improved(axes[1], completed, 'Trajectory Completion', 'blue', x_min, x_max, y_min, y_max, sequence_id, show_gt=True)
        axes[1].set_title(f'Trajectory Completion\n{stats["completed_points"]} points, {stats["completed_frames"]} frames\nAdded {stats["added_points"]} points', 
                         fontsize=12, fontweight='bold', color='white')
        
        # 绘制过滤后数据
        self.plot_sequence_improved(axes[2], filtered, 'Outlier Filtering', 'green', x_min, x_max, y_min, y_max, sequence_id, show_gt=True)
        axes[2].set_title(f'Outlier Filtering\n{stats["filtered_points"]} points, {stats["filtered_frames"]} frames\nRemoved {stats["removed_points"]} points', 
                         fontsize=12, fontweight='bold', color='white')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像到single_sequence_results目录
        output_path = f'single_sequence_results/sequence_{sequence_id}_angle_distance_analysis.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"角度距离分析可视化结果已保存到: {output_path}")
        
        # 关闭图形以释放内存
        plt.close()
    
    def load_ground_truth(self, gt_path: str = 'datasets/spotgeov2-IRSTD/test_anno.json'):
        """加载真实标注数据"""
        try:
            with open(gt_path, 'r') as f:
                annotations = json.load(f)
            
            # 按序列和帧组织标注数据
            ground_truth = {}
            for anno in annotations:
                seq_id = anno['sequence_id']
                frame_id = anno['frame']
                if seq_id not in ground_truth:
                    ground_truth[seq_id] = {}
                ground_truth[seq_id][frame_id] = anno['object_coords']
            
            return ground_truth
        except Exception as e:
            print(f"加载真实标注失败: {e}")
            return {}
    
    def plot_sequence_improved(self, ax, frames_data: Dict[int, Dict], title: str, color: str, 
                             x_min: float, x_max: float, y_min: float, y_max: float, 
                             sequence_id: int, show_gt: bool = True):
        """改进的序列绘制函数，参考analyze_trajectory_changes.py的风格"""
        # 创建黑色背景，固定尺寸为640×480
        img_size = [480, 640]  # [height, width]
        ax.imshow(np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8))
        
        # 用于跟踪图例
        legend_added = {'pred': False, 'gt': False}
        
        frames = sorted(frames_data.keys())
        
        # 绘制真实标注（ground truth）
        if show_gt:
            gt_data = self.load_ground_truth()
            if sequence_id in gt_data:
                gt_points = []
                for frame in frames:
                    if frame in gt_data[sequence_id]:
                        gt_points.extend(gt_data[sequence_id][frame])
                
                if gt_points:
                    gt_array = np.array(gt_points)
                    ax.scatter(gt_array[:, 0], gt_array[:, 1], 
                              c='yellow', s=80, marker='o', alpha=0.8,
                              label='Ground Truth' if not legend_added['gt'] else "")
                    legend_added['gt'] = True
                    print(f"  序列 {sequence_id} 绘制了 {len(gt_points)} 个真实标注点")
        
        # 绘制所有检测点（不连线）
        all_points = []
        for frame in frames:
            if frames_data[frame]['coords']:
                for coord in frames_data[frame]['coords']:
                    all_points.append(coord)
                    # 绘制点（使用叉号标记）
                    ax.scatter(coord[0], coord[1], c=color, s=150, marker='x', linewidth=2, alpha=0.8,
                              label='Prediction' if not legend_added['pred'] else "")
                    legend_added['pred'] = True
        
        # 添加帧号标注
        for frame in frames:
            if frames_data[frame]['coords']:
                for i, coord in enumerate(frames_data[frame]['coords']):
                    ax.annotate(f'F{frame}', (coord[0], coord[1]), 
                              xytext=(5, 5), textcoords='offset points', 
                              fontsize=10, color='white', weight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
        
        # 设置坐标轴，固定为640×480
        ax.set_xlim(0, img_size[1])
        ax.set_ylim(img_size[0], 0)  # 反转Y轴以匹配图像坐标系
        ax.set_xlabel('X coordinate', color='white', fontsize=10)
        ax.set_ylabel('Y coordinate', color='white', fontsize=10)
        
        # 添加网格线
        ax.grid(True, alpha=0.3, color='gray')
        
        # 设置坐标轴颜色
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        
        # 设置背景色
        ax.set_facecolor('black')
        
        # 添加图例
        if any(legend_added.values()):
            legend = ax.legend(loc='upper right', frameon=True, 
                             facecolor='black', edgecolor='white',
                             labelcolor='white', fontsize=8)
    
    def print_detailed_analysis(self, original: Dict[int, Dict], 
                              completed: Dict[int, Dict], 
                              filtered: Dict[int, Dict],
                              sequence_id: int,
                              stats: Dict) -> None:
        """打印详细的分析信息，参考angle_processor.py的输出风格"""
        print(f"\n{'='*60}")
        print(f"序列 {sequence_id} 角度距离分析详细报告")
        print(f"{'='*60}")
        
        # 帧级详细信息
        print(f"\n帧级检测详情:")
        print(f"{'帧号':<6} {'原始点数':<8} {'补全后点数':<10} {'过滤后点数':<10} {'状态':<10}")
        print("-" * 50)
        
        all_frames = sorted(set(original.keys()) | set(completed.keys()) | set(filtered.keys()))
        for frame in all_frames:
            original_count = len(original.get(frame, {}).get('coords', []))
            completed_count = len(completed.get(frame, {}).get('coords', []))
            filtered_count = len(filtered.get(frame, {}).get('coords', []))
            
            # 确定状态
            if original_count == 0 and completed_count > 0:
                status = "新增"
            elif original_count > 0 and filtered_count == 0:
                status = "移除"
            elif original_count > filtered_count:
                status = "部分移除"
            elif completed_count > original_count:
                status = "部分新增"
            else:
                status = "保持"
            
            print(f"{frame:<6} {original_count:<8} {completed_count:<10} {filtered_count:<10} {status:<10}")
        
        # 统计摘要
        print(f"\n{'='*60}")
        print(f"角度距离处理效果摘要:")
        print(f"{'='*60}")
        print(f"总帧数: {stats['total_frames']}")
        print(f"原始检测: {stats['original_points']} 个点, {stats['original_frames']} 帧")
        print(f"轨迹补全: {stats['completed_points']} 个点, {stats['completed_frames']} 帧")
        print(f"异常点过滤: {stats['filtered_points']} 个点, {stats['filtered_frames']} 帧")
        print(f"新增点数: {stats['added_points']} (+{stats['added_points']/max(stats['original_points'], 1)*100:.1f}%)")
        print(f"移除点数: {stats['removed_points']} (-{stats['removed_points']/max(stats['original_points'], 1)*100:.1f}%)")
        
        # 角度距离处理建议
        print(f"\n{'='*60}")
        print(f"角度距离处理建议:")
        print(f"{'='*60}")
        if stats['added_points'] > 0:
            print(f"✓ 基于角度距离的轨迹补全成功，新增了 {stats['added_points']} 个检测点")
        if stats['removed_points'] > 0:
            print(f"✓ 基于角度距离的异常点过滤有效，移除了 {stats['removed_points']} 个异常点")
        if stats['filtered_points'] == stats['original_points']:
            print("✓ 所有原始检测点都被保留，无异常点")
        if stats['completed_points'] == stats['original_points']:
            print("⚠ 角度距离轨迹补全未添加新点，可能需要调整角度或步长容差参数")
        
        print(f"{'='*60}")
    
    def plot_sequence(self, ax, frames_data: Dict[int, Dict], title: str, color: str):
        """绘制单个序列的轨迹"""
        frames = sorted(frames_data.keys())
        
        # 绘制所有检测点
        all_points = []
        for frame in frames:
            if frames_data[frame]['coords']:
                for coord in frames_data[frame]['coords']:
                    all_points.append(coord)
                    # 绘制点
                    ax.scatter(coord[0], coord[1], c=color, s=self.point_size, alpha=0.7, edgecolors='white', linewidth=1)
        
        # 绘制轨迹线
        if len(all_points) > 1:
            all_points = np.array(all_points)
            ax.plot(all_points[:, 0], all_points[:, 1], color=color, linewidth=self.line_width, alpha=0.5)
        
        # 添加帧号标注
        for frame in frames:
            if frames_data[frame]['coords']:
                for i, coord in enumerate(frames_data[frame]['coords']):
                    ax.annotate(f'{frame}', (coord[0], coord[1]), 
                              xytext=(5, 5), textcoords='offset points', 
                              fontsize=8, alpha=0.7)
        
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')
        ax.grid(True, alpha=0.3)
    
    def analyze_sequence(self, predictions_path: str, sequence_id: int, 
                        save_visualization: bool = True) -> Dict:
        """分析单个序列的角度距离后处理效果"""
        print(f"=== 开始角度距离分析序列 {sequence_id} ===")
        
        # 加载序列数据
        frames_data = self.load_sequence_data(predictions_path, sequence_id)
        
        if not frames_data:
            print(f"序列 {sequence_id} 没有找到数据")
            return {}
        
        # 处理序列
        original, completed, filtered, stats = self.process_single_sequence(sequence_id, frames_data)
        
        # 打印详细分析信息
        self.print_detailed_analysis(original, completed, filtered, sequence_id, stats)
        
        # 创建可视化
        if save_visualization:
            self.create_visualization(original, completed, filtered, sequence_id, stats)
        
        return {
            'sequence_id': sequence_id,
            'stats': stats,
            'original': original,
            'completed': completed,
            'filtered': filtered
        }

def main():
    """主函数，演示单序列角度距离分析"""
    import argparse

    parser = argparse.ArgumentParser(description='单序列角度距离后处理分析')
    parser.add_argument('--pred_path', type=str, default='results/spotgeov2/WTNet/predictions.json',
                       help='预测结果文件路径')
    parser.add_argument('--sequence_id', type=int, default=69,
                       help='要分析的序列ID')
    parser.add_argument('--base_distance_threshold', type=float, default=1000.0,
                       help='基础距离阈值')
    parser.add_argument('--angle_tolerance', type=float, default=3,
                       help='角度容差（度）')
    parser.add_argument('--min_angle_count', type=int, default=2,
                       help='最小角度出现次数')
    parser.add_argument('--step_tolerance', type=float, default=0.1,
                       help='步长容差（比例）')
    parser.add_argument('--min_step_count', type=int, default=1,
                       help='最小步长出现次数')
    parser.add_argument('--point_distance_threshold', type=float, default=200.0,
                       help='重合点过滤阈值')
    parser.add_argument('--no_visualization', action='store_true',
                       help='不保存可视化结果')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.pred_path):
        print(f"错误：预测结果文件不存在: {args.pred_path}")
        return
    
    # 创建单序列角度距离处理器
    processor = SingleSequenceAngleDistanceProcessor(
        base_distance_threshold=args.base_distance_threshold,
        angle_tolerance=args.angle_tolerance,
        min_angle_count=args.min_angle_count,
        step_tolerance=args.step_tolerance,
        min_step_count=args.min_step_count,
        point_distance_threshold=args.point_distance_threshold
    )
    
    # 分析序列
    result = processor.analyze_sequence(
        args.pred_path, 
        args.sequence_id,
        save_visualization=not args.no_visualization
    )
    
    if result:
        print(f"\n序列 {args.sequence_id} 角度距离分析完成！")

if __name__ == '__main__':
    main() 