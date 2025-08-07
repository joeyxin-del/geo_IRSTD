#!/usr/bin/env python3
"""
测试聚类方法和原始方法的效果比较
"""

import json
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from processor.angle_distance_processor_v3 import (
    PatternAnalyzer, 
    ClusteringPatternAnalyzer,
    GeometryUtils
)

def generate_test_angles(num_angles: int = 100, noise_level: float = 0.3) -> List[float]:
    """
    生成测试角度数据
    
    Args:
        num_angles: 角度数量
        noise_level: 噪声水平 (0-1)
    
    Returns:
        角度列表
    """
    angles = []
    
    # 生成主导角度模式
    dominant_angles = [45.0, 135.0, 225.0, 315.0]  # 四个主导方向
    weights = [0.4, 0.3, 0.2, 0.1]  # 各主导角度的权重
    
    for _ in range(num_angles):
        # 选择主导角度
        dominant_angle = np.random.choice(dominant_angles, p=weights)
        
        # 添加噪声
        if np.random.random() < noise_level:
            # 噪声角度
            angle = np.random.uniform(0, 360)
        else:
            # 主导角度 + 小偏差
            angle = dominant_angle + np.random.normal(0, 5)  # 5度标准差
            angle = angle % 360
        
        angles.append(angle)
    
    return angles

def compare_methods(angles: List[float], 
                   angle_tolerance: float = 10.0,
                   min_angle_count: int = 2,
                   angle_cluster_eps: float = 15.0,
                   min_cluster_size: int = 2,
                   confidence_threshold: float = 0.1) -> Dict:
    """
    比较原始方法和聚类方法的效果
    
    Args:
        angles: 角度列表
        angle_tolerance: 角度容差
        min_angle_count: 最小角度数量
        angle_cluster_eps: 聚类半径
        min_cluster_size: 最小聚类大小
        confidence_threshold: 置信度阈值
    
    Returns:
        比较结果字典
    """
    # 原始方法
    original_analyzer = PatternAnalyzer(
        angle_tolerance=angle_tolerance,
        min_angle_count=min_angle_count,
        min_step_count=1
    )
    
    # 聚类方法
    clustering_analyzer = ClusteringPatternAnalyzer(
        angle_tolerance=angle_tolerance,
        min_angle_count=min_angle_count,
        min_step_count=1,
        angle_cluster_eps=angle_cluster_eps,
        min_cluster_size=min_cluster_size,
        confidence_threshold=confidence_threshold
    )
    
    # 执行分析
    original_dominant = original_analyzer._find_dominant_angles(angles)
    clustering_dominant = clustering_analyzer._find_dominant_angles(angles)
    
    # 计算统计信息
    results = {
        'original_dominant_angles': original_dominant,
        'clustering_dominant_angles': clustering_dominant,
        'original_count': len(original_dominant),
        'clustering_count': len(clustering_dominant),
        'total_angles': len(angles),
        'angle_distribution': {}
    }
    
    # 分析角度分布
    angle_bins = np.arange(0, 361, 10)
    hist, _ = np.histogram(angles, bins=angle_bins)
    results['angle_distribution'] = {
        'bins': angle_bins[:-1].tolist(),
        'counts': hist.tolist()
    }
    
    return results

def visualize_comparison(results: Dict, save_path: str = None):
    """
    可视化比较结果
    
    Args:
        results: 比较结果
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 角度分布直方图
    bins = results['angle_distribution']['bins']
    counts = results['angle_distribution']['counts']
    ax1.bar(bins, counts, width=8, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('角度 (度)')
    ax1.set_ylabel('频次')
    ax1.set_title('角度分布直方图')
    ax1.grid(True, alpha=0.3)
    
    # 主导角度比较
    original_angles = [angle for angle, count in results['original_dominant_angles']]
    clustering_angles = [angle for angle, count in results['clustering_dominant_angles']]
    
    # 绘制原始方法的主导角度
    for angle in original_angles:
        ax2.axvline(x=angle, color='red', linestyle='--', linewidth=2, 
                   label='原始方法' if angle == original_angles[0] else "")
    
    # 绘制聚类方法的主导角度
    for angle in clustering_angles:
        ax2.axvline(x=angle, color='blue', linestyle='-', linewidth=2,
                   label='聚类方法' if angle == clustering_angles[0] else "")
    
    ax2.hist(results['angle_distribution']['bins'], 
             weights=results['angle_distribution']['counts'], 
             bins=36, alpha=0.5, color='gray', edgecolor='black')
    ax2.set_xlabel('角度 (度)')
    ax2.set_ylabel('频次')
    ax2.set_title('主导角度比较')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    plt.show()

def main():
    """主函数"""
    print("开始测试聚类方法和原始方法的效果比较...")
    
    # 生成测试数据
    test_angles = generate_test_angles(num_angles=200, noise_level=0.2)
    print(f"生成了 {len(test_angles)} 个测试角度")
    
    # 比较方法
    results = compare_methods(
        angles=test_angles,
        angle_tolerance=10.0,
        min_angle_count=2,
        angle_cluster_eps=15.0,
        min_cluster_size=2,
        confidence_threshold=0.1
    )
    
    # 输出结果
    print("\n=== 比较结果 ===")
    print(f"总角度数量: {results['total_angles']}")
    print(f"原始方法找到的主导角度数量: {results['original_count']}")
    print(f"聚类方法找到的主导角度数量: {results['clustering_count']}")
    
    print("\n原始方法主导角度:")
    for angle, count in results['original_dominant_angles']:
        print(f"  角度: {angle:.1f}°, 数量: {count}")
    
    print("\n聚类方法主导角度:")
    for angle, count in results['clustering_dominant_angles']:
        print(f"  角度: {angle:.1f}°, 数量: {count}")
    
    # 可视化结果
    visualize_comparison(results, 'clustering_comparison.png')
    
    # 分析差异原因
    print("\n=== 差异分析 ===")
    
    if results['original_count'] > results['clustering_count']:
        print("原始方法找到更多主导角度，可能原因:")
        print("1. 聚类方法的参数设置过于严格")
        print("2. 数据分布不适合DBSCAN聚类")
        print("3. 角度噪声影响了聚类效果")
    elif results['clustering_count'] > results['original_count']:
        print("聚类方法找到更多主导角度，可能原因:")
        print("1. 聚类方法能更好地处理复杂模式")
        print("2. 原始方法的分箱策略过于简单")
    else:
        print("两种方法找到的主导角度数量相同")
    
    # 建议改进
    print("\n=== 改进建议 ===")
    print("1. 调整聚类参数:")
    print("   - 降低 confidence_threshold (如 0.05)")
    print("   - 调整 angle_cluster_eps (如 20.0)")
    print("   - 减少 min_cluster_size (如 1)")
    print("2. 添加回退机制（已实现）")
    print("3. 动态调整聚类参数（已实现）")

if __name__ == '__main__':
    main() 