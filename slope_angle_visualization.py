import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import math

def visualize_slope_threshold():
    """可视化斜率阈值0.1对应的角度，只画第一象限"""
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Slope Threshold 0.1 Angle Visualization (First Quadrant)', fontsize=16, fontweight='bold')
    
    # 左图：第一象限坐标系中的直线示意图
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 5)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_title('First Quadrant: Coordinate System with Slope Lines', fontsize=14, fontweight='bold')
    
    # 绘制坐标轴
    ax1.axhline(y=0, color='black', linewidth=1)
    ax1.axvline(x=0, color='black', linewidth=1)
    
    # 计算角度
    slope_threshold = 0.1
    angle_rad = math.atan(slope_threshold)
    angle_deg = math.degrees(angle_rad)
    
    print(f"斜率阈值 {slope_threshold} 对应的角度: {angle_deg:.2f}°")
    
    # 以斜率差1切分第一象限，绘制不同斜率的直线
    slopes = [0, 0.1, 1, 2, 3, 4, 5]  # 斜率差1的切分
    colors = ['green', 'blue', 'red', 'orange', 'purple', 'brown', 'pink']
    labels = ['0', '0.1', '1', '2', '3', '4', '5']
    
    for slope, color, label in zip(slopes, colors, labels):
        # 计算直线在第一象限的端点
        if slope <= 1:
            # 斜率小于等于1，从(0,0)到(5, 5*slope)
            x = np.linspace(0, 5, 100)
            y = slope * x
        else:
            # 斜率大于1，从(0,0)到(5/slope, 5)
            x = np.linspace(0, 5/slope, 100)
            y = slope * x
        
        # 判断是否在阈值范围内
        if abs(slope) <= slope_threshold:
            linestyle = '-'
            linewidth = 4
            alpha = 1.0
        else:
            linestyle = '-'
            linewidth = 2
            alpha = 0.7
        
        ax1.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, 
                alpha=alpha, label=f'slope={label}')
    
    # 添加阈值范围标注（第一象限）
    x_threshold = np.linspace(0, 5, 100)
    y_upper = slope_threshold * x_threshold
    y_lower = np.zeros_like(x_threshold)
    
    ax1.fill_between(x_threshold, y_lower, y_upper, 
                     alpha=0.3, color='gray', label=f'Threshold [0, {slope_threshold}]')
    
    # 添加角度标注
    ax1.text(1, 0.5, f'Threshold\n±{angle_deg:.1f}°', 
             fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # 右图：第一象限角度示意图
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_title(f'First Quadrant: Angle Visualization', fontsize=14, fontweight='bold')
    
    # 绘制第一象限的单位圆
    theta = np.linspace(0, np.pi/2, 100)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    ax2.plot(x_circle, y_circle, 'k-', linewidth=2, label='Unit Circle (Q1)')
    
    # 绘制坐标轴
    ax2.plot([0, 1], [0, 0], 'k-', linewidth=1)
    ax2.plot([0, 0], [0, 1], 'k-', linewidth=1)
    
    # 绘制不同角度对应的斜率线
    angles_to_show = [0, angle_deg, 45, 60, 75, 90]  # 对应斜率 0, 0.1, 1, 1.73, 3.73, ∞
    colors_angle = ['green', 'blue', 'red', 'orange', 'purple', 'brown']
    labels_angle = ['0°', f'{angle_deg:.1f}°', '45°', '60°', '75°', '90°']
    
    for angle_deg_val, color, label in zip(angles_to_show, colors_angle, labels_angle):
        angle_rad_val = math.radians(angle_deg_val)
        x_end = math.cos(angle_rad_val)
        y_end = math.sin(angle_rad_val)
        
        # 计算对应的斜率
        if angle_deg_val == 90:
            slope_text = "∞"
        else:
            slope_val = math.tan(angle_rad_val)
            slope_text = f"{slope_val:.1f}"
        
        ax2.plot([0, x_end], [0, y_end], color=color, linewidth=3, 
                label=f'{label} (slope={slope_text})', marker='o', markersize=8)
    
    # 添加阈值范围标注
    ax2.text(0.3, 0.1, f'Threshold\n0° to {angle_deg:.1f}°', 
             fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax2.legend(fontsize=8)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_path = 'slope_threshold_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"斜率阈值可视化已保存到: {output_path}")
    
    # 关闭图形
    plt.close()
    
    # 打印详细信息
    print(f"\n斜率阈值详细信息 (第一象限):")
    print(f"斜率阈值: {slope_threshold}")
    print(f"对应角度: {angle_deg:.2f}°")
    print(f"角度范围: 0° 到 {angle_deg:.2f}°")
    print(f"在第一象限中，斜率为 {slope_threshold} 的直线与x轴的夹角为 {angle_deg:.2f}°")
    print(f"这意味着所有斜率在 [0, {slope_threshold}] 范围内的直线")
    print(f"都被认为是相似的，角度范围约为 0° 到 {angle_deg:.2f}°")
    
    # 打印第一象限斜率切分信息
    print(f"\n第一象限斜率切分 (斜率差1):")
    for i, slope in enumerate(slopes):
        if slope == 0:
            angle = 0
        elif slope == float('inf'):
            angle = 90
        else:
            angle = math.degrees(math.atan(slope))
        print(f"斜率 {slope}: 角度 {angle:.1f}°")

if __name__ == '__main__':
    visualize_slope_threshold() 