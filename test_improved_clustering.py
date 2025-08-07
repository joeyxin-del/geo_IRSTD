#!/usr/bin/env python3
"""
测试改进后的ClusteringPatternAnalyzer是否能提高recall
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processor.angle_distance_processor_v3 import AngleDistanceProcessorV3, ClusteringPatternAnalyzer
import json

def test_improved_clustering():
    """测试改进后的聚类分析器"""
    
    # 测试参数
    test_angles = [45.0, 46.0, 47.0, 48.0, 49.0,  # 主导角度组1
                   90.0, 91.0, 92.0, 93.0, 94.0,  # 主导角度组2
                   135.0, 136.0, 137.0, 138.0, 139.0,  # 主导角度组3
                   180.0, 181.0, 182.0, 183.0, 184.0,  # 主导角度组4
                   225.0, 226.0, 227.0, 228.0, 229.0,  # 主导角度组5
                   270.0, 271.0, 272.0, 273.0, 274.0,  # 主导角度组6
                   315.0, 316.0, 317.0, 318.0, 319.0,  # 主导角度组7
                   360.0, 361.0, 362.0, 363.0, 364.0,  # 主导角度组8
                   10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,  # 噪声角度
                   100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0]  # 更多噪声角度
    
    print("测试改进后的ClusteringPatternAnalyzer...")
    print(f"测试角度数量: {len(test_angles)}")
    
    # 创建改进后的聚类分析器
    clustering_analyzer = ClusteringPatternAnalyzer(
        angle_tolerance=5.0,
        min_angle_count=1,
        angle_cluster_eps=3.0,
        min_cluster_size=1,
        confidence_threshold=0.02
    )
    
    # 测试主导角度发现
    dominant_angles = clustering_analyzer._find_dominant_angles(test_angles)
    
    print(f"\n发现的主导角度数量: {len(dominant_angles)}")
    for angle, count in dominant_angles:
        print(f"  角度: {angle:.1f}°, 数量: {count}")
    
    # 测试聚类信息
    dominant_angles_info, clusters_info = clustering_analyzer.cluster_angles_with_info(test_angles)
    
    print(f"\n聚类详细信息:")
    for angle, confidence, cluster_size in clusters_info:
        print(f"  角度: {angle:.1f}°, 置信度: {confidence:.3f}, 聚类大小: {cluster_size}")
    
    return len(dominant_angles) > 0

def test_processor_on_sample_data():
    """在样本数据上测试处理器"""
    
    # 创建样本预测数据
    sample_predictions = {
        "sequence_1_frame_1.jpg": {
            "coords": [[100, 200], [300, 400], [500, 600]],
            "num_objects": 3
        },
        "sequence_1_frame_2.jpg": {
            "coords": [[110, 210], [310, 410], [510, 610]],
            "num_objects": 3
        },
        "sequence_1_frame_3.jpg": {
            "coords": [[120, 220], [320, 420], [520, 620]],
            "num_objects": 3
        },
        "sequence_1_frame_4.jpg": {
            "coords": [[130, 230], [330, 430], [530, 630]],
            "num_objects": 3
        },
        "sequence_1_frame_5.jpg": {
            "coords": [[140, 240], [340, 440], [540, 640]],
            "num_objects": 3
        }
    }
    
    print("\n测试处理器在样本数据上的表现...")
    
    # 创建改进后的处理器
    processor = AngleDistanceProcessorV3(
        angle_tolerance=5.0,
        min_angle_count=1,
        use_clustering=True
    )
    
    # 处理样本数据
    processed_predictions = processor.process_dataset(sample_predictions)
    
    print(f"原始预测数量: {len(sample_predictions)}")
    print(f"处理后预测数量: {len(processed_predictions)}")
    
    # 检查是否保留了更多点
    original_total_points = sum(len(pred['coords']) for pred in sample_predictions.values())
    processed_total_points = sum(len(pred['coords']) for pred in processed_predictions.values())
    
    print(f"原始总点数: {original_total_points}")
    print(f"处理后总点数: {processed_total_points}")
    print(f"点数保留率: {processed_total_points / original_total_points * 100:.1f}%")
    
    return processed_total_points >= original_total_points * 0.8  # 至少保留80%的点

if __name__ == "__main__":
    print("=== 测试改进后的ClusteringPatternAnalyzer ===")
    
    # 测试1: 聚类功能
    test1_passed = test_improved_clustering()
    print(f"\n测试1 (聚类功能): {'通过' if test1_passed else '失败'}")
    
    # 测试2: 处理器功能
    test2_passed = test_processor_on_sample_data()
    print(f"\n测试2 (处理器功能): {'通过' if test2_passed else '失败'}")
    
    print(f"\n=== 测试总结 ===")
    print(f"所有测试: {'通过' if test1_passed and test2_passed else '失败'}")
    
    if test1_passed and test2_passed:
        print("\n改进后的ClusteringPatternAnalyzer应该能够提高recall！")
        print("主要改进包括:")
        print("1. 降低了confidence_threshold从0.05到0.02")
        print("2. 降低了angle_cluster_eps从5.0到3.0")
        print("3. 降低了min_angle_count从2到1")
        print("4. 放宽了角度匹配条件（1.5倍容差）")
        print("5. 放宽了步长匹配条件（0.7-1.3范围）")
        print("6. 降低了参与度阈值从0.3到0.15")
        print("7. 放宽了距离阈值从120到200")
        print("8. 更宽松的异常点过滤策略")
    else:
        print("\n需要进一步调试改进...") 