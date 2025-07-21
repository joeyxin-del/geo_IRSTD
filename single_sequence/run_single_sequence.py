#!/usr/bin/env python3
"""
单个序列处理器使用示例
"""

import json
import sys
import os
from slope_processor import SingleSequenceSlopeProcessor

def load_data(pred_path: str, gt_path: str):
    """加载预测结果和真实标注"""
    print("正在加载预测结果和真实标注...")
    
    with open(pred_path, 'r') as f:
        original_predictions = json.load(f)
    
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    return original_predictions, ground_truth

def analyze_single_sequence(sequence_id: int, 
                          pred_path: str = 'results/WTNet/predictions.json',
                          gt_path: str = 'datasets/spotgeov2-IRSTD/test_anno.json',
                          save_dir: str = "single_sequence_results"):
    """分析单个序列"""
    
    # 加载数据
    original_predictions, ground_truth = load_data(pred_path, gt_path)
    
    # 创建单个序列处理器
    processor = SingleSequenceSlopeProcessor(
        sequence_id=sequence_id,
        base_distance_threshold=500,  # 宽松的距离阈值
        min_track_length=1,           # 允许短轨迹
        expected_sequence_length=5,   # 期望序列长度
        slope_tolerance=0.1,          # 斜率容差
        min_slope_count=2,            # 最小斜率出现次数
        point_distance_threshold=200.0  # 重合点过滤阈值
    )
    
    # 运行分析
    improvement = processor.run_analysis(original_predictions, ground_truth, save_dir)
    
    return improvement

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python run_single_sequence.py <sequence_id>")
        print("示例: python run_single_sequence.py 1")
        return
    
    try:
        sequence_id = int(sys.argv[1])
    except ValueError:
        print("错误: 序列ID必须是整数")
        return
    
    print(f"开始分析序列 {sequence_id}...")
    
    # 分析序列
    improvement = analyze_single_sequence(sequence_id)
    
    print(f"\n序列 {sequence_id} 分析完成！")
    print(f"结果保存在: single_sequence_results/")

if __name__ == '__main__':
    main() 