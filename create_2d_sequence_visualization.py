import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_2d_sequence_visualization():
    """创建序列67的二维示意图，只显示预测点位置"""
    
    # 序列67的预测数据
    sequence_67_data = {
        1: [369.2916259765625, 218.64306640625],
        2: [348.15869140625, 190.6285400390625],
        3: [326.954833984375, 162.76812744140625],  # 无检测点
        4: [305.60205078125, 135.46051025390625],  # 无检测点
        5: [284.643798828125, 107.47576904296875]
    }
    
    # 创建图形，只画一个子图
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    
    # 提取有效数据点
    valid_frames = []
    valid_coords = []
    for frame, coord in sequence_67_data.items():
        if coord is not None:
            valid_frames.append(frame)
            valid_coords.append(coord)
    
    valid_coords = np.array(valid_coords)
    valid_frames = np.array(valid_frames)
    
    # 绘制预测点（蓝色x标记）
    ax1.scatter(valid_coords[:, 0], valid_coords[:, 1], 
                c='blue', s=150, marker='x', linewidth=2, alpha=0.8, label='Prediction')
    
    # 为每个点添加帧号标注
    for i, (x, y) in enumerate(valid_coords):
        ax1.annotate(f'F{valid_frames[i]}', (x, y), 
                     xytext=(5, 5), textcoords='offset points', 
                     fontsize=10, color='black', weight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # 设置坐标轴，固定为640×480
    ax1.set_xlim(0, 640)
    ax1.set_ylim(480, 0)  # 反转Y轴以匹配图像坐标系
    ax1.set_xlabel('X coordinate', fontsize=10)
    ax1.set_ylabel('Y coordinate', fontsize=10)
    
    # 添加网格线
    ax1.grid(True, alpha=0.3, color='gray')
    
    # 设置坐标轴颜色
    ax1.tick_params(colors='black')
    ax1.spines['bottom'].set_color('black')
    ax1.spines['top'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['right'].set_color('black')
    
    # 设置背景色
    ax1.set_facecolor('white')
    
    # 添加图例
    legend = ax1.legend(loc='upper right', frameon=True, 
                        facecolor='white', edgecolor='black',
                        labelcolor='black', fontsize=8)
    
    # 设置图形背景色
    fig.patch.set_facecolor('white')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('sequence_67_2d_visualization.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("二维示意图已生成并保存为 'sequence_67_2d_visualization.png'")
    print("\n序列67预测数据说明：")
    print("- 帧1: (219.875, 381.125)")
    print("- 帧2: (190.45, 403.82)")
    print("- 帧3: 无检测点")
    print("- 帧4: 无检测点")
    print("- 帧5: (103.67, 473.11)")

if __name__ == "__main__":
    create_2d_sequence_visualization() 