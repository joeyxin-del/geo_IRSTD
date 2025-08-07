import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_3d_visualization():
    """创建序列67的三维示意图"""
    
    # 序列67的原始数据
    sequence_67_data = {
        1: [219.875, 381.125],
        2: [190.45454545454547, 403.8181818181818],
        3: None,  # 无检测点
        4: None,
        5: [103.66666666666667, 473.1111111111111]
    }
    
    # 创建图形
    fig = plt.figure(figsize=(15, 10))
    
    # 创建两个子图
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 提取有效数据点
    valid_frames = []
    valid_coords = []
    for frame, coord in sequence_67_data.items():
        if coord is not None:
            valid_frames.append(frame)
            valid_coords.append(coord)
    
    valid_coords = np.array(valid_coords)
    valid_frames = np.array(valid_frames)
    
    # 第一个子图：原始数据
    ax1.set_title('原始数据：存在假阴性和假阳性', fontsize=14, fontweight='bold')
    
    # 绘制有效检测点（绿色实心圆点）
    ax1.scatter(valid_coords[:, 0], valid_coords[:, 1], valid_frames, 
                c='green', s=100, marker='o', alpha=0.8, label='正确检测点')
    
    # 添加假阳性点（橙色圆点）
    false_positive_points = np.array([
        [280, 350, 2],
        [150, 250, 4]
    ])
    ax1.scatter(false_positive_points[:, 0], false_positive_points[:, 1], false_positive_points[:, 2],
                c='orange', s=100, marker='o', alpha=0.8, label='假阳性点')
    
    # 添加假阴性点（红色空心圆点）
    false_negative_point = np.array([250, 290, 3])
    ax1.scatter(false_negative_point[0], false_negative_point[1], false_negative_point[2],
                c='red', s=100, marker='o', facecolors='none', linewidth=2, label='假阴性点')
    
    # 绘制理想轨迹线
    ideal_trajectory_x = np.array([valid_coords[0, 0], 250, valid_coords[1, 0], valid_coords[2, 0], valid_coords[3, 0]])
    ideal_trajectory_y = np.array([valid_coords[0, 1], 290, valid_coords[1, 1], valid_coords[2, 1], valid_coords[3, 1]])
    ideal_trajectory_z = np.array([1, 2, 3, 4, 5])
    ax1.plot(ideal_trajectory_x, ideal_trajectory_y, ideal_trajectory_z, 
             'r--', linewidth=2, alpha=0.7, label='理想轨迹')
    
    # 设置坐标轴
    ax1.set_xlabel('X坐标', fontsize=12)
    ax1.set_ylabel('Y坐标', fontsize=12)
    ax1.set_zlabel('帧数', fontsize=12)
    ax1.set_zticks([1, 2, 3, 4, 5])
    ax1.set_zticklabels(['1', '2', '3', '4', '5'])
    ax1.legend()
    
    # 第二个子图：处理后数据
    ax2.set_title('处理后：删除假阳性并补全假阴性', fontsize=14, fontweight='bold')
    
    # 绘制校正后的检测点
    corrected_coords = np.array([
        [219.875, 381.125, 1],
        [190.45, 403.82, 2],
        [250, 290, 3],  # 补全的假阴性点
        [330.875, 206.125, 4],
        [103.67, 473.11, 5]
    ])
    ax2.scatter(corrected_coords[:, 0], corrected_coords[:, 1], corrected_coords[:, 2],
                c='green', s=100, marker='o', alpha=0.8, label='校正后检测点')
    
    # 绘制校正后的轨迹线
    ax2.plot(corrected_coords[:, 0], corrected_coords[:, 1], corrected_coords[:, 2],
             'r--', linewidth=2, alpha=0.7, label='校正后轨迹')
    
    # 设置坐标轴
    ax2.set_xlabel('X坐标', fontsize=12)
    ax2.set_ylabel('Y坐标', fontsize=12)
    ax2.set_zlabel('帧数', fontsize=12)
    ax2.set_zticks([1, 2, 3, 4, 5])
    ax2.set_zticklabels(['1', '2', '3', '4', '5'])
    ax2.legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('sequence_67_3d_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("三维示意图已生成并保存为 'sequence_67_3d_visualization.png'")

if __name__ == "__main__":
    create_3d_visualization()