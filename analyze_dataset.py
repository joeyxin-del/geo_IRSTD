import json
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_distance(point1, point2):
    """计算两点之间的欧氏距离"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def analyze_trajectory_distances(predictions_file):
    """分析轨迹首尾距离
    
    Args:
        predictions_file: 预测文件路径
    """
    # 读取预测结果
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # 存储所有轨迹的距离信息
    trajectory_info = []
    
    # 遍历所有序列
    current_sequence = []
    current_seq_id = None
    
    # 按序列ID排序处理
    for img_id in sorted(predictions.keys()):
        seq_id = '_'.join(img_id.split('_')[:2])  # 例如: "3_1"
        frame_num = int(img_id.split('_')[1])  # 帧号
        
        # 如果是新序列
        if seq_id != current_seq_id:
            # 处理上一个序列
            if current_sequence:
                process_sequence(current_sequence, trajectory_info)
            # 开始新序列
            current_sequence = []
            current_seq_id = seq_id
        
        # 添加当前帧信息
        current_sequence.append({
            'frame_num': frame_num,
            'coords': predictions[img_id]['coords']
        })
        
        # 如果积累了5帧，处理这个序列
        if len(current_sequence) == 5:
            process_sequence(current_sequence, trajectory_info)
            current_sequence = []
    
    # 输出统计结果
    print_statistics(trajectory_info)
    
    # 绘制距离分布图
    plot_distance_distribution(trajectory_info)
    
    return trajectory_info

def process_sequence(sequence, trajectory_info):
    """处理单个5帧序列"""
    if len(sequence) != 5:
        return
        
    # 获取第一帧和最后一帧的坐标
    first_frame = sequence[0]['coords']
    last_frame = sequence[-1]['coords']
    
    # 如果首尾帧都有目标
    if first_frame and last_frame:
        seq_id = f"Sequence_{len(trajectory_info)//5 + 1}"
        
        # 计算所有可能的首尾点对的距离
        for start_idx, start_point in enumerate(first_frame):
            for end_idx, end_point in enumerate(last_frame):
                distance = calculate_distance(start_point, end_point)
                
                trajectory_info.append({
                    'sequence_id': seq_id,
                    'start_point': start_point,
                    'end_point': end_point,
                    'distance': distance,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })

def print_statistics(trajectory_info):
    """打印统计信息"""
    if not trajectory_info:
        print("没有找到有效的轨迹！")
        return
        
    distances = [info['distance'] for info in trajectory_info]
    
    print("\n=== 轨迹距离统计 ===")
    print(f"总轨迹数: {len(trajectory_info)}")
    print(f"平均距离: {np.mean(distances):.2f}")
    print(f"标准差: {np.std(distances):.2f}")
    print(f"最大距离: {max(distances):.2f}")
    print(f"最小距离: {min(distances):.2f}")
    print(f"中位数: {np.median(distances):.2f}")
    
    # 分位数统计
    percentiles = np.percentile(distances, [25, 50, 75])
    print(f"\n分位数统计:")
    print(f"25分位数: {percentiles[0]:.2f}")
    print(f"50分位数: {percentiles[1]:.2f}")
    print(f"75分位数: {percentiles[2]:.2f}")
    
    # 输出一些具体例子
    print("\n示例轨迹:")
    for info in trajectory_info[:5]:  # 显示前5个例子
        print(f"序列 {info['sequence_id']}: "
              f"起点{info['start_point']} -> 终点{info['end_point']}, "
              f"距离: {info['distance']:.2f}")

def plot_distance_distribution(trajectory_info):
    """绘制距离分布图"""
    distances = [info['distance'] for info in trajectory_info]
    
    plt.figure(figsize=(12, 6))
    
    # 绘制直方图
    plt.subplot(1, 2, 1)
    plt.hist(distances, bins=50, density=True, alpha=0.7)
    plt.title('轨迹距离分布')
    plt.xlabel('距离')
    plt.ylabel('密度')
    
    # 绘制箱线图
    plt.subplot(1, 2, 2)
    plt.boxplot(distances)
    plt.title('轨迹距离箱线图')
    plt.ylabel('距离')
    
    # 保存图片
    save_dir = Path('statistics')
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / 'trajectory_distances.png')
    plt.close()

def main():
    """主函数"""
    # 设置预测文件路径
    predictions_file = "results/spotgeov2/TADNet/predictions.json"
    
    # 分析轨迹距离
    trajectory_info = analyze_trajectory_distances(predictions_file)
    
    # 将结果保存到文件
    save_dir = Path('statistics')
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / 'trajectory_statistics.json', 'w') as f:
        json.dump({
            'trajectories': trajectory_info,
            'summary': {
                'total_trajectories': len(trajectory_info),
                'average_distance': float(np.mean([info['distance'] for info in trajectory_info])),
                'std_distance': float(np.std([info['distance'] for info in trajectory_info])),
                'max_distance': float(max(info['distance'] for info in trajectory_info)),
                'min_distance': float(min(info['distance'] for info in trajectory_info))
            }
        }, f, indent=2)

if __name__ == "__main__":
    main() 