#!/usr/bin/env python3
"""
测试优化后的自适应角度距离处理器
"""

import json
import os
import sys
from processor.adaptive_angle_distance_processor import AdaptiveAngleDistanceProcessor

def test_adaptive_processor():
    """测试自适应处理器的效果"""
    
    # 文件路径
    pred_path = 'results/spotgeov2-IRSTD/WTNet/predictions_8807.json'
    gt_path = 'datasets/spotgeov2-IRSTD/test_anno.json'
    output_path = 'results/spotgeov2/WTNet/optimized_adaptive_predictions.json'
    
    # 检查文件是否存在
    if not os.path.exists(pred_path):
        print(f"错误：预测结果文件不存在: {pred_path}")
        return
    
    if not os.path.exists(gt_path):
        print(f"错误：真实标注文件不存在: {gt_path}")
        return
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("正在加载预测结果和真实标注...")
    try:
        with open(pred_path, 'r') as f:
            original_predictions = json.load(f)
        print(f"成功加载预测结果，包含 {len(original_predictions)} 张图像")
        
        with open(gt_path, 'r') as f:
            ground_truth = json.load(f)
        print(f"成功加载真实标注，包含 {len(ground_truth)} 个标注")
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return
    
    # 创建优化后的自适应角度距离处理器
    processor = AdaptiveAngleDistanceProcessor(
        base_distance_threshold=80.0,
        min_track_length=2,
        expected_sequence_length=5,
        min_cluster_size=2,  # 降低最小聚类大小
        angle_cluster_eps=15.0,  # 更严格的角度聚类
        step_cluster_eps=0.3,
        confidence_threshold=0.3,  # 适中的置信度阈值
        angle_tolerance=20.0,  # 适中的角度容差
        step_tolerance=0.4,
        point_distance_threshold=10.0,  # 更小的重合点阈值
        dominant_ratio_threshold=0.6,  # 适中的主导模式阈值
        secondary_ratio_threshold=0.15,
        max_dominant_patterns=2,  # 减少最大模式数量
        clustering_method='gmm',
        enable_f1_optimization=True,
        adaptive_threshold=True,
        enable_confidence_driven_filtering=True,
        multi_pattern_fusion=True,
        f1_optimization_threshold=0.4,
        min_confidence_for_keep=0.5,  # 提高保留阈值
        max_confidence_for_remove=0.15  # 降低移除阈值
    )
    
    print("开始优化后的自适应角度距离处理...")
    
    # 进行自适应角度距离处理
    processed_predictions = processor.process_dataset(
        original_predictions, save_visualization=False
    )
    
    # 评估改善效果
    try:
        improvement = processor.evaluate_improvement(original_predictions, processed_predictions, ground_truth)
        
        # 打印结果
        print("\n=== 优化后的自适应角度距离处理效果评估 ===")
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
        
        # 判断是否有效改善
        if improvement['f1_improvement'] > 0:
            print("\n✅ 处理有效果！F1分数有所提升")
        elif improvement['f1_improvement'] == 0:
            print("\n⚠️ 处理效果一般，F1分数没有变化")
        else:
            print("\n❌ 处理效果不好，F1分数下降")
            
    except Exception as e:
        print(f"评估过程中出错: {e}")
        print("跳过评估，继续保存结果...")
    
    # 保存处理后的结果
    try:
        with open(output_path, 'w') as f:
            json.dump(processed_predictions, f, indent=2)
        print(f"\n优化后的自适应角度距离处理结果已保存到: {output_path}")
    except Exception as e:
        print(f"保存结果时出错: {e}")

if __name__ == '__main__':
    test_adaptive_processor() 