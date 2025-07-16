import json
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from typing import Dict, List, Tuple
import sys

# 添加路径以便导入评估函数
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eval_predictions import calculate_metrics

class TrajectoryChangeAnalyzer:
    """轨迹变化分析器"""
    
    def __init__(self, 
                 original_pred_path: str = 'results/WTNet/predictions.json',
                #  processed_pred_path: str = 'results/WTNet/kmeans_trajectory_predictions.json',
                #  processed_pred_path: str = 'results/WTNet/aggressive_balanced_processed_predictions.json',
                 processed_pred_path: str = 'results/WTNet/sequence_slope_processed_predictions.json',
                 gt_path: str = 'datasets/spotgeov2-IRSTD/test_anno.json',
                #  save_dir: str = 'trajectory_analysis_results'):
                #  save_dir: str = 'aggressive_balanced_trajectory_analysis_results'):
                 save_dir: str = 'sequence_slope_based_trajectory_analysis_results'):
        """
        初始化分析器
        
        Args:
            original_pred_path: 原始预测结果路径
            processed_pred_path: 处理后预测结果路径
            gt_path: 真实标注路径
            save_dir: 保存结果的目录
        """
        self.original_pred_path = original_pred_path
        self.processed_pred_path = processed_pred_path
        self.gt_path = gt_path
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'improved_sequences'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'degraded_sequences'), exist_ok=True)
        
        # 加载数据
        self.load_data()
        
    def load_data(self):
        """加载所有数据"""
        print("正在加载数据...")
        
        # 加载预测结果
        with open(self.original_pred_path, 'r') as f:
            self.original_predictions = json.load(f)
        
        with open(self.processed_pred_path, 'r') as f:
            self.processed_predictions = json.load(f)
        
        # 加载真实标注并按序列和帧组织
        self.ground_truth = {}
        with open(self.gt_path, 'r') as f:
            annotations = json.load(f)
            # 按序列和帧组织标注数据
            for anno in annotations:
                seq_id = anno['sequence_id']
                frame_id = anno['frame']
                if seq_id not in self.ground_truth:
                    self.ground_truth[seq_id] = {}
                self.ground_truth[seq_id][frame_id] = anno['object_coords']
        
        print(f"加载完成：原始预测 {len(self.original_predictions)} 张图像")
        print(f"处理后预测 {len(self.processed_predictions)} 张图像")
        print(f"真实标注 {len(self.ground_truth)} 个序列")
        
        # 打印一些调试信息
        print("真实标注序列ID范围:", min(self.ground_truth.keys()), "到", max(self.ground_truth.keys()))
        print("前5个序列的帧数:")
        for i, seq_id in enumerate(sorted(self.ground_truth.keys())[:5]):
            print(f"  序列 {seq_id}: {len(self.ground_truth[seq_id])} 帧")
    
    def extract_sequence_info(self, predictions: Dict) -> Dict[int, Dict]:
        """从预测结果中提取序列信息"""
        sequence_data = defaultdict(lambda: {'frames': {}, 'img_size': [512, 512]})
        
        for img_name, pred_info in predictions.items():
            parts = img_name.split('_')
            if len(parts) >= 2:
                try:
                    sequence_id = int(parts[0])
                    frame = int(parts[1])
                    sequence_data[sequence_id]['frames'][frame] = {
                        'coords': pred_info['coords'],
                        'num_objects': pred_info['num_objects']
                    }
                except ValueError:
                    continue
        
        return dict(sequence_data)
    
    def calculate_sequence_metrics(self, sequence_id: int, predictions: Dict) -> Dict:
        """计算单个序列的指标"""
        # 提取该序列的预测结果
        sequence_predictions = {}
        for img_name, pred_info in predictions.items():
            parts = img_name.split('_')
            if len(parts) >= 2:
                try:
                    seq_id = int(parts[0])
                    if seq_id == sequence_id:
                        sequence_predictions[img_name] = pred_info
                except ValueError:
                    continue
        
        # 提取该序列的真实标注
        sequence_gt = []
        if sequence_id in self.ground_truth:
            for frame_id, coords in self.ground_truth[sequence_id].items():
                sequence_gt.append({
                    'sequence_id': sequence_id,
                    'frame': frame_id,
                    'object_coords': coords
                })
        
        if not sequence_predictions or not sequence_gt:
            return {'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': 0}
        
        # 计算指标
        metrics = calculate_metrics(sequence_predictions, sequence_gt, 1000)
        return metrics
    
    def analyze_sequence_changes(self) -> Dict:
        """分析所有序列的变化"""
        print("正在分析序列变化...")
        
        # 提取序列信息
        original_sequences = self.extract_sequence_info(self.original_predictions)
        processed_sequences = self.extract_sequence_info(self.processed_predictions)
        
        # 获取所有序列ID
        all_sequence_ids = set(original_sequences.keys()) | set(processed_sequences.keys())
        
        changes_analysis = {
            'improved_sequences': [],
            'degraded_sequences': [],
            'unchanged_sequences': [],
            'sequence_details': {}
        }
        
        for seq_id in sorted(all_sequence_ids):
            print(f"分析序列 {seq_id}...")
            
            # 计算原始指标
            original_metrics = self.calculate_sequence_metrics(seq_id, self.original_predictions)
            
            # 计算处理后指标
            processed_metrics = self.calculate_sequence_metrics(seq_id, self.processed_predictions)
            
            # 计算变化
            f1_change = processed_metrics['f1'] - original_metrics['f1']
            precision_change = processed_metrics['precision'] - original_metrics['precision']
            recall_change = processed_metrics['recall'] - original_metrics['recall']
            
            # 记录详细信息
            seq_detail = {
                'sequence_id': seq_id,
                'original': original_metrics,
                'processed': processed_metrics,
                'changes': {
                    'f1': f1_change,
                    'precision': precision_change,
                    'recall': recall_change,
                    'tp': processed_metrics['total_tp'] - original_metrics['total_tp'],
                    'fp': processed_metrics['total_fp'] - original_metrics['total_fp'],
                    'fn': processed_metrics['total_fn'] - original_metrics['total_fn']
                }
            }
            
            changes_analysis['sequence_details'][seq_id] = seq_detail
            
            # 分类序列
            if f1_change > 0.01:  # F1提升超过1%
                changes_analysis['improved_sequences'].append(seq_detail)
            elif f1_change < -0.01:  # F1下降超过1%
                changes_analysis['degraded_sequences'].append(seq_detail)
            else:
                changes_analysis['unchanged_sequences'].append(seq_detail)
        
        # 打印统计信息
        print(f"\n=== 序列变化统计 ===")
        print(f"改善的序列数量: {len(changes_analysis['improved_sequences'])}")
        print(f"下降的序列数量: {len(changes_analysis['degraded_sequences'])}")
        print(f"无变化的序列数量: {len(changes_analysis['unchanged_sequences'])}")
        
        return changes_analysis
    
    def find_changed_points(self, original_coords: List, processed_coords: List, distance_threshold: float = 5.0) -> Tuple[List, List]:
        """找出变化的检测点，使用距离阈值判断点是否相同"""
        if not original_coords and not processed_coords:
            return [], []
        
        # 转换为numpy数组
        original_array = np.array(original_coords) if original_coords else np.array([]).reshape(0, 2)
        processed_array = np.array(processed_coords) if processed_coords else np.array([]).reshape(0, 2)
        
        if len(original_array) == 0:
            # 原始没有点，处理后有点，都是新增的
            return processed_coords, []
        
        if len(processed_array) == 0:
            # 原始有点，处理后没有点，都是删除的
            return [], original_coords
        
        # 计算所有点对之间的距离
        added_points = []
        removed_points = []
        
        # 检查每个处理后的点是否在原始点中有对应
        for proc_point in processed_array:
            min_distance = float('inf')
            for orig_point in original_array:
                distance = np.sqrt((proc_point[0] - orig_point[0])**2 + (proc_point[1] - orig_point[1])**2)
                min_distance = min(min_distance, distance)
            
            # 如果最小距离超过阈值，认为是新增的点
            if min_distance > distance_threshold:
                added_points.append(proc_point.tolist())
        
        # 检查每个原始点是否在处理后的点中有对应
        for orig_point in original_array:
            min_distance = float('inf')
            for proc_point in processed_array:
                distance = np.sqrt((orig_point[0] - proc_point[0])**2 + (orig_point[1] - proc_point[1])**2)
                min_distance = min(min_distance, distance)
            
            # 如果最小距离超过阈值，认为是删除的点
            if min_distance > distance_threshold:
                removed_points.append(orig_point.tolist())
        
        return added_points, removed_points
    
    def create_sequence_visualization(self, sequence_id: int, 
                                    original_sequences: Dict, 
                                    processed_sequences: Dict,
                                    category: str):
        """为单个序列创建可视化对比图"""
        print(f"创建序列 {sequence_id} 的可视化...")
        
        # 获取序列数据
        original_seq = original_sequences.get(sequence_id, {'frames': {}, 'img_size': [480, 640]})
        processed_seq = processed_sequences.get(sequence_id, {'frames': {}, 'img_size': [480, 640]})
        
        # 添加序列ID到序列数据中
        original_seq['sequence_id'] = sequence_id
        processed_seq['sequence_id'] = sequence_id
        
        # 固定图像尺寸为640×480
        img_size = [480, 640]  # [height, width]
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左图：后处理前的预测结果（原始预测）
        self.plot_sequence_frame(ax1, original_seq, None, img_size, "Before Processing", show_changes=False)
        
        # 右图：后处理后的预测结果（处理后预测）
        self.plot_sequence_frame(ax2, processed_seq, original_seq, img_size, "After Processing", show_changes=True)
        
        # 保存图像
        save_path = os.path.join(self.save_dir, f"{category}_sequences", f"sequence_{sequence_id}_comparison.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
        
        print(f"序列 {sequence_id} 可视化已保存到: {save_path}")
    
    def plot_sequence_frame(self, ax, current_seq: Dict, compare_seq: Dict, img_size: List, title: str, show_changes: bool = False):
        """绘制单个序列帧"""
        # 创建黑色背景
        ax.imshow(np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8))
        
        # 用于跟踪图例
        legend_added = {'pred': False, 'gt': False, 'added': False, 'removed': False}
        
        # 获取序列ID - 从序列数据的键中获取
        sequence_id = None
        if hasattr(current_seq, 'get') and 'sequence_id' in current_seq:
            sequence_id = current_seq['sequence_id']
        else:
            # 从帧ID推断序列ID
            for frame_id in current_seq['frames'].keys():
                if isinstance(frame_id, str) and '_' in frame_id:
                    sequence_id = int(frame_id.split('_')[0])
                else:
                    sequence_id = int(str(frame_id).split('_')[0]) if '_' in str(frame_id) else frame_id
                break
        
        # 绘制真实标注（ground truth）
        if sequence_id is not None:
            gt_points = []
            # 遍历当前序列的所有帧
            for frame_id in current_seq['frames'].keys():
                # 提取帧号
                if isinstance(frame_id, str) and '_' in frame_id:
                    frame_num = int(frame_id.split('_')[1])
                else:
                    frame_num = frame_id
                
                # 获取对应帧的真实标注
                if sequence_id in self.ground_truth and frame_num in self.ground_truth[sequence_id]:
                    # print(sequence_id, "1111111111111111")
                    # print(self.ground_truth[sequence_id][frame_num])
                    gt_points.extend(self.ground_truth[sequence_id][frame_num])
            
            if gt_points:
                gt_array = np.array(gt_points)
                ax.scatter(gt_array[:, 0], gt_array[:, 1], 
                          c='yellow', s=80, marker='o', alpha=0.8,
                          label='Ground Truth' if not legend_added['gt'] else "")
                legend_added['gt'] = True
                print(f"  序列 {sequence_id} 绘制了 {len(gt_points)} 个真实标注点")
            else:
                print(f"  序列 {sequence_id} 没有找到对应的真实标注")
        
        # 绘制每一帧的点
        for frame_id, frame_data in current_seq['frames'].items():
            pred_points = np.array(frame_data['coords'])
            
            if len(pred_points) > 0:
                # 绘制预测点（红色叉）
                ax.scatter(pred_points[:, 0], pred_points[:, 1], 
                          c='red', s=150, marker='x', linewidth=2,
                          label='Prediction' if not legend_added['pred'] else "")
                legend_added['pred'] = True
                
                # 只在处理后图像上显示变化标注
                if show_changes and compare_seq is not None:
                    # 找出变化的点
                    compare_frame_data = compare_seq['frames'].get(frame_id, {'coords': []})
                    added_points, removed_points = self.find_changed_points(
                        compare_frame_data['coords'], frame_data['coords']
                    )
                    
                    # 绘制新增的点（绿色中空圆圈）
                    if added_points:
                        added_array = np.array(added_points)
                        for point in added_array:
                            circle = plt.Circle((point[0], point[1]), radius=20, 
                                              fill=False, color='green', linewidth=3)
                            ax.add_patch(circle)
                        # 添加图例
                        if not legend_added['added']:
                            ax.scatter([], [], c='green', s=100, marker='o', linewidth=5, facecolors='none',
                                     label='Added')
                            legend_added['added'] = True
                    
                    # 绘制删除的点（白色中空圆圈）
                    if removed_points:
                        removed_array = np.array(removed_points)
                        for point in removed_array:
                            circle = plt.Circle((point[0], point[1]), radius=20, 
                                              fill=False, color='white', linewidth=3)
                            ax.add_patch(circle)
                        # 添加图例
                        if not legend_added['removed']:
                            ax.scatter([], [], c='white', s=100, marker='o', linewidth=1, facecolors='none',
                                     label='Removed')
                            legend_added['removed'] = True
        
        # 设置坐标轴
        ax.set_xlim(0, img_size[1])
        ax.set_ylim(img_size[0], 0)
        ax.set_title(title, fontsize=12, fontweight='bold', color='white')
        ax.axis('off')
        
        # 添加图例
        if any(legend_added.values()):
            legend = ax.legend(loc='upper right', frameon=True, 
                             facecolor='black', edgecolor='white',
                             labelcolor='white', fontsize=8)
    
    def create_summary_report(self, changes_analysis: Dict):
        """创建总结报告"""
        print("\n正在创建总结报告...")
        
        # 计算总体统计
        improved_count = len(changes_analysis['improved_sequences'])
        degraded_count = len(changes_analysis['degraded_sequences'])
        unchanged_count = len(changes_analysis['unchanged_sequences'])
        total_count = improved_count + degraded_count + unchanged_count
        
        # 计算平均变化
        improved_f1_changes = [seq['changes']['f1'] for seq in changes_analysis['improved_sequences']]
        degraded_f1_changes = [seq['changes']['f1'] for seq in changes_analysis['degraded_sequences']]
        
        avg_improved_f1 = np.mean(improved_f1_changes) if improved_f1_changes else 0
        avg_degraded_f1 = np.mean(degraded_f1_changes) if degraded_f1_changes else 0
        
        # 创建报告
        report = f"""
# K-means轨迹跟踪效果分析报告

## 总体统计
- 总序列数: {total_count}
- 改善的序列数: {improved_count} ({improved_count/total_count*100:.1f}%)
- 下降的序列数: {degraded_count} ({degraded_count/total_count*100:.1f}%)
- 无变化的序列数: {unchanged_count} ({unchanged_count/total_count*100:.1f}%)

## 变化统计
- 平均F1改善: {avg_improved_f1:.4f}
- 平均F1下降: {avg_degraded_f1:.4f}

## 改善最显著的序列 (Top 10)
"""
        
        # 添加改善最显著的序列
        improved_sorted = sorted(changes_analysis['improved_sequences'], 
                               key=lambda x: x['changes']['f1'], reverse=True)
        
        for i, seq in enumerate(improved_sorted[:10]):
            report += f"{i+1}. 序列 {seq['sequence_id']}: F1 +{seq['changes']['f1']:.4f} "
            report += f"(P: {seq['changes']['precision']:+.4f}, R: {seq['changes']['recall']:+.4f})\n"
        
        report += "\n## 下降最显著的序列 (Top 10)\n"
        
        # 添加下降最显著的序列
        degraded_sorted = sorted(changes_analysis['degraded_sequences'], 
                               key=lambda x: x['changes']['f1'])
        
        for i, seq in enumerate(degraded_sorted[:10]):
            report += f"{i+1}. 序列 {seq['sequence_id']}: F1 {seq['changes']['f1']:.4f} "
            report += f"(P: {seq['changes']['precision']:+.4f}, R: {seq['changes']['recall']:+.4f})\n"
        
        # 保存报告
        report_path = os.path.join(self.save_dir, 'analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"分析报告已保存到: {report_path}")
        
        # 打印报告内容
        print(report)
    
    def run_analysis(self):
        """运行完整分析"""
        print("开始K-means轨迹跟踪效果分析...")
        
        # 分析序列变化
        changes_analysis = self.analyze_sequence_changes()
        
        # 提取序列信息用于可视化
        original_sequences = self.extract_sequence_info(self.original_predictions)
        processed_sequences = self.extract_sequence_info(self.processed_predictions)
        
        # 为改善的序列创建可视化
        print(f"\n为 {len(changes_analysis['improved_sequences'])} 个改善的序列创建可视化...")
        for seq_detail in changes_analysis['improved_sequences'][:20]:  # 限制数量
            seq_id = seq_detail['sequence_id']
            self.create_sequence_visualization(seq_id, original_sequences, processed_sequences, 'improved')
        
        # 为下降的序列创建可视化
        print(f"\n为 {len(changes_analysis['degraded_sequences'])} 个下降的序列创建可视化...")
        for seq_detail in changes_analysis['degraded_sequences'][:20]:  # 限制数量
            seq_id = seq_detail['sequence_id']
            self.create_sequence_visualization(seq_id, original_sequences, processed_sequences, 'degraded')
        
        # 创建总结报告
        self.create_summary_report(changes_analysis)
        
        print(f"\n分析完成！结果保存在: {self.save_dir}")

def main():
    """主函数"""
    analyzer = TrajectoryChangeAnalyzer()
    analyzer.run_analysis()

if __name__ == '__main__':
    main() 