#!/usr/bin/env python3
"""
角度聚类原理演示
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def demo_angle_clustering():
    """演示角度聚类原理"""
    
    # 模拟您的数据：集中在317°-320°的角度
    angles = [317.5, 318.0, 318.0, 318.5, 318.5, 319.0, 319.0, 319.5, 320.0, 320.0]
    
    print("原始角度数据:")
    print(f"角度范围: {min(angles):.1f}° - {max(angles):.1f}°")
    print(f"角度列表: {[f'{a:.1f}°' for a in angles]}")
    
    # 1. 角度到二维坐标转换
    angles_rad = np.array(angles) * np.pi / 180.0
    X = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])
    
    print(f"\n转换为二维坐标:")
    for i, (angle, coord) in enumerate(zip(angles, X)):
        print(f"  {angle:.1f}° → ({coord[0]:.3f}, {coord[1]:.3f})")
    
    # 2. 计算角度差异（传统方法 vs 正确方法）
    print(f"\n角度差异计算演示:")
    angle1, angle2 = 359.0, 1.0
    print(f"角度1: {angle1}°, 角度2: {angle2}°")
    
    # 错误方法：直接相减
    wrong_diff = abs(angle1 - angle2)
    print(f"错误方法: |{angle1} - {angle2}| = {wrong_diff}°")
    
    # 正确方法：考虑周期性
    correct_diff = min(abs(angle1 - angle2), 
                      abs(angle1 - angle2 + 360),
                      abs(angle1 - angle2 - 360))
    print(f"正确方法: min(|{angle1}-{angle2}|, |{angle1}-{angle2}+360|, |{angle1}-{angle2}-360|) = {correct_diff}°")
    
    # 3. 聚类中心计算演示
    print(f"\n聚类中心计算演示:")
    cluster_angles = [317.5, 318.0, 318.0, 318.5, 318.5]
    print(f"聚类角度: {cluster_angles}")
    
    # 错误方法：直接平均
    wrong_center = np.mean(cluster_angles)
    print(f"错误方法: 直接平均 = {wrong_center:.1f}°")
    
    # 正确方法：先平均cos/sin，再转回角度
    cluster_rad = np.array(cluster_angles) * np.pi / 180.0
    mean_cos = np.mean(np.cos(cluster_rad))
    mean_sin = np.mean(np.sin(cluster_rad))
    correct_center = math.degrees(math.atan2(mean_sin, mean_cos))
    if correct_center < 0:
        correct_center += 360
    print(f"正确方法: 平均cos/sin后转角度 = {correct_center:.1f}°")
    
    # 4. 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：角度分布
    axes[0].hist(angles, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_title('角度分布直方图')
    axes[0].set_xlabel('角度 (度)')
    axes[0].set_ylabel('频次')
    axes[0].grid(True, alpha=0.3)
    
    # 右图：单位圆上的分布
    # 绘制单位圆
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', alpha=0.5)
    axes[1].add_patch(circle)
    
    # 绘制角度点
    axes[1].scatter(X[:, 0], X[:, 1], alpha=0.7, s=100, color='red')
    
    # 标注角度
    for i, (angle, coord) in enumerate(zip(angles, X)):
        axes[1].annotate(f'{angle:.1f}°', (coord[0], coord[1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    axes[1].set_title('角度在单位圆上的分布')
    axes[1].set_xlabel('cos(angle)')
    axes[1].set_ylabel('sin(angle)')
    axes[1].set_xlim(-1.2, 1.2)
    axes[1].set_ylim(-1.2, 1.2)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('angle_clustering_demo.png', dpi=300, bbox_inches='tight')
    print(f"\n可视化已保存到: angle_clustering_demo.png")
    
    # 5. 不同eps参数的效果演示
    print(f"\n不同eps参数的效果演示:")
    
    eps_values = [1.0, 3.0, 5.0, 10.0]
    
    for eps in eps_values:
        # 计算在单位圆上的距离
        eps_rad = eps * np.pi / 180.0
        print(f"\neps = {eps}° (≈ {eps_rad:.3f} 弧度):")
        
        # 计算点对之间的距离
        distances = []
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                dist = np.linalg.norm(X[i] - X[j])
                distances.append(dist)
                if dist <= eps_rad:
                    print(f"  {angles[i]:.1f}° 和 {angles[j]:.1f}° 距离 {dist:.3f} ≤ {eps_rad:.3f} ✓")
                else:
                    print(f"  {angles[i]:.1f}° 和 {angles[j]:.1f}° 距离 {dist:.3f} > {eps_rad:.3f} ✗")
        
        # 估算聚类数
        close_pairs = sum(1 for d in distances if d <= eps_rad)
        total_pairs = len(distances)
        print(f"  总点对数: {total_pairs}, 近距离对数: {close_pairs}")
        
        if eps_rad >= max(distances):
            print(f"  → 所有点会被归为1个聚类")
        elif close_pairs == 0:
            print(f"  → 所有点都是噪声点")
        else:
            print(f"  → 可能形成多个聚类")

if __name__ == '__main__':
    demo_angle_clustering() 