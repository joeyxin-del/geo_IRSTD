#!/usr/bin/env python3
"""
测试不同聚类算法在角度聚类上的效果
"""

import numpy as np
import matplotlib.pyplot as plt
from processor.adaptive_angle_distance_processor import AdaptiveAngleDistanceProcessor

def generate_test_angles():
    """生成测试用的角度数据"""
    # 模拟真实的角度分布：几个主导方向 + 噪声
    np.random.seed(42)
    
    # 主导角度1: 220度附近
    angles1 = np.random.normal(220, 10, 30)
    
    # 主导角度2: 140度附近  
    angles2 = np.random.normal(140, 8, 25)
    
    # 主导角度3: 320度附近
    angles3 = np.random.normal(320, 12, 20)
    
    # 噪声角度
    noise_angles = np.random.uniform(0, 360, 15)
    
    # 合并所有角度
    all_angles = np.concatenate([angles1, angles2, angles3, noise_angles])
    
    # 确保角度在[0, 360)范围内
    all_angles = all_angles % 360
    
    return all_angles.tolist()

def test_clustering_methods():
    """测试不同的聚类方法"""
    angles = generate_test_angles()
    
    # 定义要测试的聚类方法
    methods = ['dbscan', 'hdbscan', 'mean_shift', 'gmm', 'agglomerative', 'spectral']
    
    results = {}
    
    for method in methods:
        print(f"\n=== 测试聚类方法: {method} ===")
        
        try:
            # 创建处理器
            processor = AdaptiveAngleDistanceProcessor(
                clustering_method=method,
                min_cluster_size=3,
                angle_cluster_eps=15.0,
                max_dominant_patterns=5
            )
            
            # 进行聚类
            dominant_angles, confidences, clusters_info, _ = processor.cluster_angles(angles)
            
            results[method] = {
                'dominant_angles': dominant_angles,
                'confidences': confidences,
                'clusters_info': clusters_info,
                'success': True
            }
            
            print(f"发现 {len(dominant_angles)} 个主导角度:")
            for i, (angle, conf) in enumerate(dominant_angles):
                print(f"  角度 {i+1}: {angle:.1f}° (置信度: {conf:.3f})")
                
        except Exception as e:
            print(f"方法 {method} 失败: {e}")
            results[method] = {
                'dominant_angles': [],
                'confidences': [],
                'clusters_info': [],
                'success': False,
                'error': str(e)
            }
    
    return results, angles

def visualize_results(results, angles):
    """可视化聚类结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('不同聚类算法的角度聚类效果比较', fontsize=16, fontweight='bold')
    
    methods = list(results.keys())
    
    for i, method in enumerate(methods):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 绘制原始角度分布
        angles_rad = np.array(angles) * np.pi / 180.0
        x_coords = np.cos(angles_rad)
        y_coords = np.sin(angles_rad)
        
        # 绘制单位圆
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', alpha=0.5)
        ax.add_patch(circle)
        
        # 绘制所有角度点
        ax.scatter(x_coords, y_coords, alpha=0.6, s=30, color='lightblue', label='原始角度')
        
        # 绘制聚类结果
        if results[method]['success']:
            clusters_info = results[method]['clusters_info']
            colors = plt.cm.Set1(np.linspace(0, 1, len(clusters_info)))
            
            for j, (angle, conf, size) in enumerate(clusters_info):
                angle_rad = angle * np.pi / 180.0
                x = np.cos(angle_rad)
                y = np.sin(angle_rad)
                ax.scatter(x, y, s=size*5, c=[colors[j]], alpha=0.8, 
                         label=f'聚类{j+1}: {angle:.1f}°')
        
        ax.set_title(f'{method.upper()}')
        ax.set_xlabel('cos(angle)')
        ax.set_ylabel('sin(angle)')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('clustering_comparison.png', dpi=300, bbox_inches='tight')
    print("聚类比较图已保存为 clustering_comparison.png")
    plt.show()

def print_summary(results):
    """打印结果总结"""
    print("\n" + "="*60)
    print("聚类算法效果总结")
    print("="*60)
    
    for method, result in results.items():
        print(f"\n{method.upper()}:")
        if result['success']:
            dominant_angles = result['dominant_angles']
            confidences = result['confidences']
            print(f"  成功发现 {len(dominant_angles)} 个主导角度")
            print(f"  平均置信度: {np.mean(confidences):.3f}")
            print(f"  最高置信度: {max(confidences):.3f}")
        else:
            print(f"  失败: {result.get('error', '未知错误')}")

if __name__ == '__main__':
    print("开始测试不同聚类算法...")
    results, angles = test_clustering_methods()
    
    print_summary(results)
    
    try:
        visualize_results(results, angles)
    except Exception as e:
        print(f"可视化失败: {e}")
    
    print("\n测试完成！") 