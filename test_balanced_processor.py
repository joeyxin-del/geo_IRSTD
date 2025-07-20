#!/usr/bin/env python3
"""
测试改进后的平衡序列后处理器
利用轨迹距离统计信息和先验知识进行漏检补全
"""

import json
import sys
import os
from processor.balanced_sequence_processor import BalancedSequenceProcessor

def test_balanced_processor():
    """测试平衡的序列后处理器"""
    
    # 检查文件是否存在
    pred_path = 'results/WTNet/predictions.json'
    gt_path = 'datasets/spotgeov2-IRSTD/test_anno.json'
    
    if not os.path.exists(pred_path):
        print(f"错误: 预测文件不存在: {pred_path}")
        return
    
    if not os.path.exists(gt_path):
        print(f"错误: 真实标注文件不存在: {gt_path}")
        return
    
    print("正在加载预测结果和真实标注...")
    with open(pred_path, 'r') as f:
        original_predictions = json.load(f)
    
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    print(f"加载了 {len(original_predictions)} 个预测结果")
    print(f"加载了 {len(ground_truth)} 个真实标注")
    
    # 创建改进后的平衡序列后处理器
    processor = BalancedSequenceProcessor(
        base_distance_threshold=1000,  # 适中的距离阈值
        temporal_window=3,
        confidence_threshold=0.05,     # 较低的置信度阈值
        min_track_length=3,            # 适中的最小轨迹长度
        max_frame_gap=3,               # 适中的最大帧间隔
        adaptive_threshold=True,        # 启用自适应阈值
        # 基于轨迹距离统计的新参数
        expected_sequence_length=5,    # 先验知识：目标在五帧中都出现
        trajectory_mean_distance=116.53,
        trajectory_std_distance=29.76,
        trajectory_max_distance=237.87,
        trajectory_min_distance=6.74,
        trajectory_median_distance=124.96
    )
    
    print("\n=== 处理器配置 ===")
    print(f"期望序列长度: {processor.expected_sequence_length}")
    print(f"轨迹平均距离: {processor.trajectory_mean_distance}")
    print(f"轨迹距离标准差: {processor.trajectory_std_distance}")
    print(f"统计距离阈值: {processor.statistical_distance_threshold:.2f}")
    
    # 进行序列后处理
    print("\n开始序列后处理...")
    processed_predictions = processor.process_sequence(original_predictions)
    
    # 评估改善效果
    print("\n评估处理效果...")
    try:
        improvement = processor.evaluate_improvement(original_predictions, processed_predictions, ground_truth)
        
        # 打印结果
        print("\n=== 平衡的序列后处理效果评估 ===")
        print(f"Precision 改善: {improvement['precision_improvement']:.4f}")
        print(f"Recall 改善: {improvement['recall_improvement']:.4f}")
        print(f"F1 Score 改善: {improvement['f1_improvement']:.4f}")
        print(f"MSE 改善: {improvement['mse_improvement']:.4f}")
        
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
        print(f"评估过程中出现错误: {e}")
        print("跳过评估，继续保存结果...")
    
    # 保存处理后的结果
    output_path = 'results/WTNet/improved_balanced_predictions.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(processed_predictions, f, indent=2)
    print(f"\n改进的处理后的预测结果已保存到: {output_path}")
    
    # 统计处理前后的变化
    print("\n=== 处理前后统计对比 ===")
    original_total_detections = sum(pred['num_objects'] for pred in original_predictions.values())
    processed_total_detections = sum(pred['num_objects'] for pred in processed_predictions.values())
    
    print(f"原始总检测数: {original_total_detections}")
    print(f"处理后总检测数: {processed_total_detections}")
    print(f"检测数变化: {processed_total_detections - original_total_detections}")
    
    # 统计序列完整性
    original_sequences = set()
    processed_sequences = set()
    
    for img_name in original_predictions.keys():
        parts = img_name.split('_')
        if len(parts) >= 2:
            original_sequences.add(int(parts[0]))
    
    for img_name in processed_predictions.keys():
        parts = img_name.split('_')
        if len(parts) >= 2:
            processed_sequences.add(int(parts[0]))
    
    print(f"原始序列数: {len(original_sequences)}")
    print(f"处理后序列数: {len(processed_sequences)}")
    print(f"新增序列数: {len(processed_sequences - original_sequences)}")

if __name__ == '__main__':
    test_balanced_processor() 