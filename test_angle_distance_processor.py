#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试角度距离处理器的功能
"""

import json
import numpy as np
import sys
import os

# 添加处理器路径
sys.path.append('processor')
from angle_distance_processor import AngleDistanceProcessor

def create_test_data():
    """创建测试数据"""
    # 创建一个简单的测试序列
    test_predictions = {
        "1_1": {
            "coords": [[100, 100]],
            "num_objects": 1
        },
        "1_2": {
            "coords": [[120, 110]],
            "num_objects": 1
        },
        "1_3": {
            "coords": [[140, 120]],
            "num_objects": 1
        },
        "1_4": {
            "coords": [[160, 130]],
            "num_objects": 1
        },
        "1_5": {
            "coords": [[180, 140]],
            "num_objects": 1
        }
    }
    
    return test_predictions

def test_angle_distance_processor():
    """测试角度距离处理器"""
    print("=== 测试角度距离处理器 ===")
    
    # 创建测试数据
    test_predictions = create_test_data()
    print(f"创建测试数据，包含 {len(test_predictions)} 张图像")
    
    # 创建处理器
    processor = AngleDistanceProcessor(
        base_distance_threshold=100.0,
        angle_tolerance=5.0,
        min_angle_count=2,
        step_tolerance=0.2,
        min_step_count=2,
        point_distance_threshold=10.0
    )
    
    # 提取序列信息
    sequence_data = processor.extract_sequence_info(test_predictions)
    print(f"提取了 {len(sequence_data)} 个序列")
    
    # 处理每个序列
    for sequence_id, frames_data in sequence_data.items():
        print(f"\n处理序列 {sequence_id}:")
        print(f"  帧数: {len(frames_data)}")
        
        # 收集角度和步长统计
        pattern_stats = processor.collect_sequence_angles_and_steps(frames_data)
        
        print(f"  总角度数: {pattern_stats['total_angles']}")
        print(f"  总步长数: {pattern_stats['total_steps']}")
        print(f"  唯一角度数: {pattern_stats['unique_angles']}")
        print(f"  唯一步长数: {pattern_stats['unique_steps']}")
        
        print(f"  主导角度: {pattern_stats['dominant_angles']}")
        print(f"  主导步长: {pattern_stats['dominant_steps']}")
        
        # 处理序列
        completed_sequence, filtered_sequence = processor.process_sequence(sequence_id, frames_data)
        
        print(f"  补全后帧数: {len(completed_sequence)}")
        print(f"  过滤后帧数: {len(filtered_sequence)}")
        
        # 显示参与主导模式的点
        point_participation = pattern_stats['point_participation']
        participated_points = [k for k, v in point_participation.items() if v['participated']]
        print(f"  参与主导模式的点数: {len(participated_points)}")
        
        for point_key in participated_points:
            point_info = point_participation[point_key]
            print(f"    点 {point_key}: 角度分数={point_info['angle_score']:.3f}, 步长分数={point_info['step_score']:.3f}")
    
    # 处理整个数据集
    processed_predictions = processor.process_dataset(test_predictions)
    print(f"\n处理完成，输出 {len(processed_predictions)} 张图像")
    
    return processed_predictions

def test_with_real_data():
    """使用真实数据测试"""
    print("\n=== 使用真实数据测试 ===")
    
    # 检查是否存在真实数据文件
    pred_path = 'results/spotgeov2/WTNet/predictions.json'
    gt_path = 'datasets/spotgeov2-IRSTD/test_anno.json'
    
    if not os.path.exists(pred_path):
        print(f"预测文件不存在: {pred_path}")
        return
    
    if not os.path.exists(gt_path):
        print(f"真实标注文件不存在: {gt_path}")
        return
    
    try:
        # 加载数据
        with open(pred_path, 'r') as f:
            original_predictions = json.load(f)
        print(f"加载预测数据: {len(original_predictions)} 张图像")
        
        with open(gt_path, 'r') as f:
            ground_truth = json.load(f)
        print(f"加载真实标注: {len(ground_truth)} 个标注")
        
        # 创建处理器
        processor = AngleDistanceProcessor(
            base_distance_threshold=1000.0,
            angle_tolerance=2.5,
            min_angle_count=2,
            step_tolerance=0.2,
            min_step_count=2,
            point_distance_threshold=200.0
        )
        
        # 处理数据
        processed_predictions = processor.process_dataset(original_predictions)
        
        # 评估改善效果
        try:
            improvement = processor.evaluate_improvement(original_predictions, processed_predictions, ground_truth)
            
            print("\n=== 改善效果 ===")
            print(f"Precision 改善: {improvement['precision_improvement']:.4f}")
            print(f"Recall 改善: {improvement['recall_improvement']:.4f}")
            print(f"F1 Score 改善: {improvement['f1_improvement']:.4f}")
            print(f"MSE 改善: {improvement['mse_improvement']:.4f}")
            
        except Exception as e:
            print(f"评估过程中出错: {e}")
        
        # 保存结果
        output_path = 'results/spotgeov2/WTNet/angle_distance_test_results.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(processed_predictions, f, indent=2)
        print(f"结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"处理真实数据时出错: {e}")

if __name__ == '__main__':
    # 测试基本功能
    test_angle_distance_processor()
    
    # 测试真实数据
    test_with_real_data()