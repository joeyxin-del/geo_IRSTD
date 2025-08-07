#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化F1分数低的序列 - 5帧映射到一张图，左右对比显示
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class LowF1SequenceVisualizer:
    def __init__(self, 
                 low_f1_file: str,
                 predictions_file: str,
                 processed_predictions_file: str,
                 gt_file: str,
                 save_dir: str = './low_f1_visualizations'):
        """
        初始化可视化器
        
        Args:
            low_f1_file: F1分数低的序列文件路径
            predictions_file: 原始预测结果文件路径
            processed_predictions_file: 后处理预测结果文件路径
            gt_file: 真实标注文件路径
            save_dir: 保存目录
        """
        self.low_f1_file = low_f1_file
        self.predictions_file = predictions_file
        self.processed_predictions_file = processed_predictions_file
        self.gt_file = gt_file
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 加载数据
        self.load_data()
    
    def load_data(self):
        """加载所有数据"""
        print("正在加载数据...")
        
        # 加载F1分数低的序列
        with open(self.low_f1_file, 'r', encoding='utf-8') as f:
            self.low_f1_data = json.load(f)
        
        # 加载原始预测结果
        with open(self.predictions_file, 'r', encoding='utf-8') as f:
            self.original_predictions = json.load(f)
        
        # 加载后处理预测结果
        if os.path.exists(self.processed_predictions_file):
            with open(self.processed_predictions_file, 'r', encoding='utf-8') as f:
                self.processed_predictions = json.load(f)
            print(f"  后处理预测结果数: {len(self.processed_predictions)}")
        else:
            print(f"  后处理预测文件不存在: {self.processed_predictions_file}")
            self.processed_predictions = self.original_predictions
        
        # 加载真实标注
        with open(self.gt_file, 'r', encoding='utf-8') as f:
            self.ground_truth = json.load(f)
        
        # 组织真实标注数据
        self.organized_gt = {}
        for anno in self.ground_truth:
            seq_id = anno['sequence_id']
            frame_id = anno['frame']
            if seq_id not in self.organized_gt:
                self.organized_gt[seq_id] = {}
            self.organized_gt[seq_id][frame_id] = anno['object_coords']
        
        print(f"加载完成：")
        print(f"  F1分数低的序列数: {len(self.low_f1_data['sequences'])}")
        print(f"  原始预测结果数: {len(self.original_predictions)}")
        print(f"  真实标注序列数: {len(self.organized_gt)}")
        
        # 过滤掉真实标注为空的序列
        self.filtered_sequences = []
        empty_sequences = []
        
        for seq_info in self.low_f1_data['sequences']:
            sequence_id = seq_info['sequence_id']
            f1_score = seq_info['f1_score']
            
            # 检查该序列是否有真实标注点
            has_gt_points = False
            if sequence_id in self.organized_gt:
                for frame_id, gt_points in self.organized_gt[sequence_id].items():
                    if gt_points:  # 如果有真实标注点
                        has_gt_points = True
                        break
            
            if has_gt_points:
                self.filtered_sequences.append(seq_info)
            else:
                empty_sequences.append(seq_info)
        
        print(f"\n序列过滤结果：")
        print(f"  有真实标注点的序列数: {len(self.filtered_sequences)}")
        print(f"  真实标注为空的序列数: {len(empty_sequences)}")
        print(f"  过滤比例: {len(empty_sequences)/len(self.low_f1_data['sequences'])*100:.1f}%")
        
        # 更新low_f1_data
        self.low_f1_data['sequences'] = self.filtered_sequences
        self.low_f1_data['total_sequences_below_0_5'] = len(self.filtered_sequences)
        self.low_f1_data['percentage_below_0_5'] = (len(self.filtered_sequences) / len(self.organized_gt)) * 100
        
        print(f"  过滤后的F1分数低的序列数: {len(self.filtered_sequences)}")
        print(f"  过滤后的占比: {self.low_f1_data['percentage_below_0_5']:.2f}%")
    
    def extract_sequence_data(self, sequence_id: int, predictions: Dict) -> Dict:
        """提取指定序列的数据"""
        sequence_data = {
            'sequence_id': sequence_id,
            'frames': {}
        }
        
        # 从预测结果中提取该序列的数据
        for img_name, pred_info in predictions.items():
            parts = img_name.split('_')
            if len(parts) >= 2:
                try:
                    seq_id = int(parts[0])
                    frame_id = int(parts[1])
                    if seq_id == sequence_id:
                        sequence_data['frames'][frame_id] = {
                            'coords': pred_info['coords'],
                            'num_objects': pred_info['num_objects']
                        }
                except ValueError:
                    continue
        
        return sequence_data
    
    def plot_sequence_frame(self, ax, sequence_data: Dict, title: str, img_size: List[int] = [480, 640]):
        """绘制单个序列帧 - 将5帧映射到一张图中"""
        # 创建黑色背景
        ax.imshow(np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8))
        
        # 用于跟踪图例
        legend_added = {'pred': False, 'gt': False}
        
        sequence_id = sequence_data['sequence_id']
        
        # 绘制真实标注（ground truth）
        gt_points = []
        for frame_id in sequence_data['frames'].keys():
            if sequence_id in self.organized_gt and frame_id in self.organized_gt[sequence_id]:
                frame_gt_points = self.organized_gt[sequence_id][frame_id]
                if frame_gt_points:  # 检查是否有真实标注点
                    gt_points.extend(frame_gt_points)
        
        if gt_points:
            gt_array = np.array(gt_points)
            ax.scatter(gt_array[:, 0], gt_array[:, 1], 
                      c='yellow', s=80, marker='o', alpha=0.8,
                      label='Ground Truth' if not legend_added['gt'] else "")
            legend_added['gt'] = True
        
        # 绘制每一帧的预测点，使用不同颜色区分帧
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        total_pred_points = 0
        
        for i, (frame_id, frame_data) in enumerate(sorted(sequence_data['frames'].items())):
            pred_points = np.array(frame_data['coords'])
            
            if len(pred_points) > 0:
                total_pred_points += len(pred_points)
                # 使用不同颜色绘制不同帧的预测点
                color = colors[i % len(colors)]
                ax.scatter(pred_points[:, 0], pred_points[:, 1], 
                          c=color, s=150, marker='x', linewidth=2,
                          label=f'Frame {frame_id}' if not legend_added['pred'] else "")
                legend_added['pred'] = True
                
                # 在每个预测点上添加帧号标注
                for j, point in enumerate(pred_points):
                    ax.annotate(f'F{frame_id}', 
                              (point[0], point[1]), 
                              xytext=(5, 5), 
                              textcoords='offset points',
                              fontsize=10, 
                              color='white',
                              weight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
        
        # 设置坐标轴
        ax.set_xlim(0, img_size[1])
        ax.set_ylim(img_size[0], 0)
        
        # 更新标题以包含更多信息
        if total_pred_points > 0:
            title_with_info = f"{title} - Sequence {sequence_id} (Pred: {total_pred_points} points, GT: {len(gt_points)} points)"
        else:
            title_with_info = f"{title} - Sequence {sequence_id} (No predictions, GT: {len(gt_points)} points)"
        
        ax.set_title(title_with_info, fontsize=12, fontweight='bold', color='white')
        
        # 添加坐标轴标签
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
        
        # 添加图例
        if any(legend_added.values()):
            legend = ax.legend(loc='upper right', frameon=True, 
                             facecolor='black', edgecolor='white',
                             labelcolor='white', fontsize=8)
        
        # 如果没有预测点，添加说明文字
        if total_pred_points == 0:
            ax.text(0.5, 0.5, 'No predictions\nfor this sequence', 
                   ha='center', va='center', transform=ax.transAxes, 
                   color='white', fontsize=14, weight='bold')
    
    def get_sequence_gt_points_count(self, sequence_id: int) -> int:
        """获取指定序列的真实标注点数量"""
        total_gt_points = 0
        if sequence_id in self.organized_gt:
            for frame_id, gt_points in self.organized_gt[sequence_id].items():
                total_gt_points += len(gt_points)
        return total_gt_points
    
    def visualize_sequence_comparison(self, sequence_id: int, f1_score: float):
        """可视化单个序列的对比图"""
        print(f"正在可视化序列 {sequence_id} (F1={f1_score:.4f})...")
        
        # 检查序列是否有真实标注点
        gt_points_count = self.get_sequence_gt_points_count(sequence_id)
        if gt_points_count == 0:
            print(f"  序列 {sequence_id} 没有真实标注点，跳过")
            return
        
        # 提取原始序列数据
        original_sequence_data = self.extract_sequence_data(sequence_id, self.original_predictions)
        
        # 提取后处理序列数据
        processed_sequence_data = self.extract_sequence_data(sequence_id, self.processed_predictions)
        
        if not original_sequence_data['frames']:
            print(f"  序列 {sequence_id} 没有找到预测数据")
            return
        
        # 创建图形 - 左右对比，适应640x480的图像尺寸
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
        fig.patch.set_facecolor('black')
        
        # 绘制原始预测结果（左图）
        self.plot_sequence_frame(ax1, original_sequence_data, "Original Detection")
        
        # 绘制后处理结果（右图）
        self.plot_sequence_frame(ax2, processed_sequence_data, "Post-processed Detection")
        
        # 添加总标题
        fig.suptitle(f'Sequence {sequence_id} Comparison (F1={f1_score:.4f}, GT points: {gt_points_count})', 
                    fontsize=16, fontweight='bold', color='white')
        
        # 保存图像
        save_path = os.path.join(self.save_dir, f'sequence_{sequence_id}_comparison_f1_{f1_score:.4f}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
        print(f"  对比图像已保存到: {save_path}")
    
    def visualize_multiple_sequences(self, max_sequences: int = 5):
        """可视化多个F1分数低的序列（只包含有真实标注点的序列）"""
        print(f"开始可视化F1分数低的序列（过滤空序列后）...")
        
        # 获取F1分数最低的序列（已经过滤过空序列）
        low_f1_sequences = sorted(self.low_f1_data['sequences'], 
                                 key=lambda x: x['f1_score'])[:max_sequences]
        
        if not low_f1_sequences:
            print("  没有找到有真实标注点的F1分数低的序列")
            return
        
        for i, seq_info in enumerate(low_f1_sequences):
            sequence_id = seq_info['sequence_id']
            f1_score = seq_info['f1_score']
            gt_points_count = self.get_sequence_gt_points_count(sequence_id)
            
            print(f"\n[{i+1}/{len(low_f1_sequences)}] 可视化序列 {sequence_id} (GT points: {gt_points_count})")
            self.visualize_sequence_comparison(sequence_id, f1_score)
    
    def create_comparison_visualization(self, sequence_ids: List[int] = None, max_sequences: int = 3):
        """创建对比可视化 - 每个序列显示原始和后处理的对比（只包含有真实标注点的序列）"""
        if sequence_ids is None:
            # 选择F1分数最低的几个序列（已经过滤过空序列）
            low_f1_sequences = sorted(self.low_f1_data['sequences'], 
                                     key=lambda x: x['f1_score'])[:max_sequences]
            sequence_ids = [seq['sequence_id'] for seq in low_f1_sequences]
        
        if not sequence_ids:
            print("  没有找到有真实标注点的F1分数低的序列")
            return
        
        print(f"创建对比可视化，包含 {len(sequence_ids)} 个序列（过滤空序列后）...")
        
        # 计算子图布局 - 每个序列占用2列（原始+后处理）
        n_sequences = len(sequence_ids)
        cols = 2  # 每个序列2列
        rows = n_sequences
        
        # 创建图形，适应640x480的图像尺寸
        fig, axes = plt.subplots(rows, cols, figsize=(24, 8*rows))
        fig.patch.set_facecolor('black')
        
        if n_sequences == 1:
            axes = axes.reshape(1, -1)
        
        # 绘制每个序列
        for i, sequence_id in enumerate(sequence_ids):
            if i >= rows:
                break
                
            # 获取F1分数和真实标注点数量
            f1_score = next((seq['f1_score'] for seq in self.low_f1_data['sequences'] 
                           if seq['sequence_id'] == sequence_id), 0.0)
            gt_points_count = self.get_sequence_gt_points_count(sequence_id)
            
            # 提取序列数据
            original_sequence_data = self.extract_sequence_data(sequence_id, self.original_predictions)
            processed_sequence_data = self.extract_sequence_data(sequence_id, self.processed_predictions)
            
            # 绘制原始预测结果（左列）
            ax1 = axes[i, 0] if rows > 1 else axes[0]
            if original_sequence_data['frames']:
                self.plot_sequence_frame(ax1, original_sequence_data, f"Original - F1={f1_score:.4f}")
            else:
                ax1.text(0.5, 0.5, f'No data for\nsequence {sequence_id}', 
                        ha='center', va='center', transform=ax1.transAxes, color='white')
                ax1.set_title(f"Sequence {sequence_id} - No Data", color='white')
            
            # 绘制后处理结果（右列）
            ax2 = axes[i, 1] if rows > 1 else axes[1]
            if processed_sequence_data['frames']:
                self.plot_sequence_frame(ax2, processed_sequence_data, f"Post-processed - F1={f1_score:.4f}")
            else:
                ax2.text(0.5, 0.5, f'No data for\nsequence {sequence_id}', 
                        ha='center', va='center', transform=ax2.transAxes, color='white')
                ax2.set_title(f"Sequence {sequence_id} - No Data", color='white')
        
        # 保存图像
        save_path = os.path.join(self.save_dir, f'low_f1_sequences_comparison_grid_filtered.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
        print(f"对比可视化已保存到: {save_path}")
    
    def create_filtered_sequences_report(self):
        """创建过滤后的序列报告"""
        print(f"\n=== 过滤后的序列统计报告 ===")
        
        # 统计F1分数分布
        f1_scores = [seq['f1_score'] for seq in self.low_f1_data['sequences']]
        
        if f1_scores:
            print(f"F1分数范围: {min(f1_scores):.4f} - {max(f1_scores):.4f}")
            print(f"平均F1分数: {np.mean(f1_scores):.4f}")
            print(f"中位数F1分数: {np.median(f1_scores):.4f}")
            
            # F1分数分布
            f1_ranges = [
                (0.0, 0.1, "0.0-0.1"),
                (0.1, 0.2, "0.1-0.2"),
                (0.2, 0.3, "0.2-0.3"),
                (0.3, 0.4, "0.3-0.4"),
                (0.4, 0.5, "0.4-0.5")
            ]
            
            print(f"\nF1分数分布:")
            for min_f1, max_f1, label in f1_ranges:
                count = len([s for s in f1_scores if min_f1 <= s < max_f1])
                percentage = count / len(f1_scores) * 100 if f1_scores else 0
                print(f"  {label}: {count} 个序列 ({percentage:.1f}%)")
        
        # 保存过滤后的序列数据
        filtered_save_path = os.path.join(self.save_dir, 'filtered_low_f1_sequences.json')
        with open(filtered_save_path, 'w', encoding='utf-8') as f:
            json.dump(self.low_f1_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n过滤后的序列数据已保存到: {filtered_save_path}")

def main():
    # 文件路径
    low_f1_file = './results/spotgeov2/WTNet/low_f1_sequences.json'
    predictions_file = './results/spotgeov2/WTNet/predictions_8807.json'
    processed_predictions_file = './results/spotgeov2/WTNet/angle_distance_processed_predictions_v3.json'
    gt_file = './datasets/spotgeov2/test_anno.json'
    
    # 检查文件是否存在
    for file_path in [low_f1_file, predictions_file, gt_file]:
        if not os.path.exists(file_path):
            print(f"错误：文件不存在: {file_path}")
            return
    
    # 创建可视化器
    visualizer = LowF1SequenceVisualizer(
        low_f1_file=low_f1_file,
        predictions_file=predictions_file,
        processed_predictions_file=processed_predictions_file,
        gt_file=gt_file,
        save_dir='./low_f1_visualizations'
    )
    
    # 创建过滤后的序列报告
    visualizer.create_filtered_sequences_report()
    
    # 可视化前5个F1分数最低的序列（过滤空序列后）
    print("\n=== 可视化F1分数最低的序列（过滤空序列后） ===")
    visualizer.visualize_multiple_sequences(max_sequences=71)
    
    # 创建对比可视化网格（过滤空序列后）
    print("\n=== 创建对比可视化网格（过滤空序列后） ===")
    visualizer.create_comparison_visualization(max_sequences=3)

if __name__ == '__main__':
    main() 