#!/usr/bin/env python3
"""
测试自适应角度距离处理器
演示如何通过聚类自动发现主导角度和主导步长
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from adaptive_angle_distance_processor import AdaptiveAngleDistanceProcessor

def create_test_data():
    """创建测试数据，模拟一个具有明显运动模式的序列"""
    frames_data = {}
    
    # 模拟一个向右上方移动的轨迹，有一些噪声点
    base_x, base_y = 100, 100
    angle = 45  # 45度角
    step = 50   # 每帧移动50像素
    
    for frame in range(1, 11):
        coords = []
        
        # 主要轨迹点
        x = base_x + (frame - 1) * step * np.cos(np.radians(angle))
        y = base_y + (frame - 1) * step * np.sin(np.radians(angle))
        coords.append([x, y])
        
        # 添加一些噪声点（随机位置）
        if frame in [3, 6, 8]:  # 在某些帧添加噪声
            noise_x = base_x + np.random.randint(-100, 100)
            noise_y = base_y + np.random.randint(-100, 100)
            coords.append([noise_x, noise_y])
        
        frames_data[frame] = {
            'coords': coords,
            'num_objects': len(coords),
            'image_name': f'67_{frame:03d}.jpg'
        }
    
    return frames_data

def test_adaptive_processor():
    """测试自适应处理器"""
    print("=== 测试自适应角度距离处理器 ===")
    
    # 创建测试数据
    test_data = create_test_data()
    print(f"创建了包含 {len(test_data)} 帧的测试数据")
    
    # 创建自适应处理器
    processor = AdaptiveAngleDistanceProcessor(
        min_cluster_size=2,
        angle_cluster_eps=10.0,
        step_cluster_eps=0.5,
        confidence_threshold=0.5
    )
    
    # 计算角度和步长
    angles, steps = processor.calculate_angles_and_steps(test_data)
    print(f"\n计算得到 {len(angles)} 个角度-步长对")
    
    # 聚类分析
    dominant_angle, angle_confidence = processor.cluster_angles(angles)
    dominant_step, step_confidence = processor.cluster_steps(steps)
    
    print(f"\n聚类分析结果:")
    print(f"主导角度: {dominant_angle:.1f}° (置信度: {angle_confidence:.3f})")
    print(f"主导步长: {dominant_step:.1f} (置信度: {step_confidence:.3f})")
    
    # 过滤点
    filtered_data = processor.filter_points_by_pattern(test_data, dominant_angle, dominant_step)
    
    # 统计结果
    original_points = sum(len(frame['coords']) for frame in test_data.values())
    filtered_points = sum(len(frame['coords']) for frame in filtered_data.values())
    
    print(f"\n过滤结果:")
    print(f"原始点数: {original_points}")
    print(f"过滤后点数: {filtered_points}")
    print(f"移除点数: {original_points - filtered_points}")
    
    # 可视化结果
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制原始数据
    axes[0].set_title('Original Data')
    for frame, data in test_data.items():
        for coord in data['coords']:
            axes[0].scatter(coord[0], coord[1], c='red', s=100, marker='o')
            axes[0].annotate(f'F{frame}', (coord[0], coord[1]), xytext=(5, 5), textcoords='offset points')
    
    # 绘制过滤后数据
    axes[1].set_title('Filtered Data')
    for frame, data in filtered_data.items():
        for coord in data['coords']:
            axes[1].scatter(coord[0], coord[1], c='green', s=100, marker='o')
            axes[1].annotate(f'F{frame}', (coord[0], coord[1]), xytext=(5, 5), textcoords='offset points')
    
    # 设置坐标轴
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
    
    plt.tight_layout()
    plt.savefig('test_adaptive_processor.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: test_adaptive_processor.png")
    plt.close()

def test_with_real_data():
    """使用真实数据测试"""
    pred_path = 'results/spotgeov2-IRSTD/WTNet/predictions_8807.json'
    
    if not os.path.exists(pred_path):
        print(f"真实数据文件不存在: {pred_path}")
        return
    
    print("\n=== 使用真实数据测试 ===")
    
    processor = AdaptiveAngleDistanceProcessor()
    result = processor.analyze_sequence(pred_path, 67, save_visualization=True)
    
    if result:
        print("真实数据测试完成！")

if __name__ == '__main__':
    # 测试模拟数据
    test_adaptive_processor()
    
    # 测试真实数据（如果存在）
    test_with_real_data() 