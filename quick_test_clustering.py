#!/usr/bin/env python3
"""
快速测试聚类方法的改进效果
"""

import numpy as np
from processor.angle_distance_processor_v3 import PatternAnalyzer, ClusteringPatternAnalyzer

def test_improved_clustering():
    """测试改进后的聚类方法"""
    
    # 生成测试角度数据（模拟真实场景）
    np.random.seed(42)
    
    # 生成主导角度模式
    dominant_angles = [45.0, 135.0, 225.0, 315.0]
    weights = [0.4, 0.3, 0.2, 0.1]
    
    angles = []
    for _ in range(100):
        if np.random.random() < 0.8:  # 80%的概率使用主导角度
            dominant_angle = np.random.choice(dominant_angles, p=weights)
            angle = dominant_angle + np.random.normal(0, 3)  # 3度标准差
            angle = angle % 360
        else:
            angle = np.random.uniform(0, 360)
        angles.append(angle)
    
    print(f"生成了 {len(angles)} 个测试角度")
    
    # 测试原始方法
    original_analyzer = PatternAnalyzer(angle_tolerance=10.0, min_angle_count=2)
    original_dominant = original_analyzer._find_dominant_angles(angles)
    
    # 测试改进后的聚类方法
    clustering_analyzer = ClusteringPatternAnalyzer(
        angle_tolerance=10.0, 
        min_angle_count=2,
        angle_cluster_eps=15.0,
        min_cluster_size=2,
        confidence_threshold=0.05  # 降低阈值
    )
    clustering_dominant = clustering_analyzer._find_dominant_angles(angles)
    
    print(f"\n原始方法找到 {len(original_dominant)} 个主导角度:")
    for angle, count in original_dominant:
        print(f"  角度: {angle:.1f}°, 数量: {count}")
    
    print(f"\n聚类方法找到 {len(clustering_dominant)} 个主导角度:")
    for angle, count in clustering_dominant:
        print(f"  角度: {angle:.1f}°, 数量: {count}")
    
    # 分析结果
    if len(clustering_dominant) >= len(original_dominant):
        print("\n✅ 聚类方法效果良好，找到的主导角度数量不少于原始方法")
    else:
        print("\n⚠️ 聚类方法效果仍需改进，找到的主导角度数量少于原始方法")
    
    return len(original_dominant), len(clustering_dominant)

if __name__ == '__main__':
    test_improved_clustering() 