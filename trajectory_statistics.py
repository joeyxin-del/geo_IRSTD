import json
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import cv2

def calculate_velocity(point1, point2):
    """计算两点间的速度向量"""
    return [point2[0] - point1[0], point2[1] - point1[1]]

def is_uniform_motion(points):
    """验证是否为匀速直线运动
    
    Args:
        points: 轨迹上的点列表 [(x1,y1), (x2,y2), ...]
    Returns:
        bool: 是否是匀速直线运动
    """
    if len(points) < 3:
        return True
        
    # 计算相邻点之间的速度向量
    velocities = []
    for i in range(len(points)-1):
        v = calculate_velocity(points[i], points[i+1])
        velocities.append(v)
    
    # 检查速度是否恒定（允许小误差）
    VELOCITY_THRESHOLD = 5.0  # 速度误差阈值
    for i in range(len(velocities)-1):
        diff_x = abs(velocities[i][0] - velocities[i+1][0])
        diff_y = abs(velocities[i][1] - velocities[i+1][1])
        if diff_x > VELOCITY_THRESHOLD or diff_y > VELOCITY_THRESHOLD:
            return False
    
    return True

def find_trajectories(sequence_predictions):
    """在5帧序列中找出匀速直线运动的轨迹
    
    Args:
        sequence_predictions: 预测结果列表，每个元素包含sequence_id、frame和object_coords
    Returns:
        list: 轨迹列表
    """
    trajectories = []
    
    # 将列表数据转换为按序列ID组织的字典
    sequences = {}
    for pred in sequence_predictions:
        seq_id = pred['sequence_id']
        frame_id = pred['frame']
        coords = pred['object_coords']
        
        if seq_id not in sequences:
            sequences[seq_id] = {}
        sequences[seq_id][frame_id] = coords
    
    # 处理每个序列
    for seq_id, frames in sequences.items():
        # 确保是5帧完整序列
        if len(frames) != 5 or not frames[1] or not frames[5]:
            continue
            
        print(f"\n处理序列 {seq_id}")
        
        # 从第一帧开始尝试构建轨迹
        for start_point in frames[1]:  # 第一帧的每个点
            # 尝试构建一条轨迹
            trajectory = [start_point]
            used_points = {1: [start_point]}
            
            # 向后逐帧匹配
            success = True
            for frame_idx in range(2, 6):
                if not frames[frame_idx]:  # 如果当前帧没有点
                    success = False
                    break
                    
                # 如果已经有两个或更多点，可以预测下一个位置
                if len(trajectory) >= 2:
                    # 计算速度向量
                    v = calculate_velocity(trajectory[-2], trajectory[-1])
                    # 预测下一个位置
                    predicted = [
                        trajectory[-1][0] + v[0],
                        trajectory[-1][1] + v[1]
                    ]
                    
                    # 在当前帧中找最接近预测位置的点
                    best_point = None
                    min_dist = float('inf')
                    for point in frames[frame_idx]:
                        dist = math.sqrt(
                            (point[0] - predicted[0])**2 + 
                            (point[1] - predicted[1])**2
                        )
                        if dist < min_dist and dist < 20:  # 设置匹配阈值
                            min_dist = dist
                            best_point = point
                    
                    if best_point is None:
                        success = False
                        break
                    
                    trajectory.append(best_point)
                    used_points[frame_idx] = [best_point]
                else:
                    # 前两帧，选择最近的点
                    best_point = None
                    min_dist = float('inf')
                    for point in frames[frame_idx]:
                        dist = math.sqrt(
                            (point[0] - trajectory[-1][0])**2 + 
                            (point[1] - trajectory[-1][1])**2
                        )
                        if dist < min_dist:
                            min_dist = dist
                            best_point = point
                    
                    trajectory.append(best_point)
                    used_points[frame_idx] = [best_point]
            
            # 验证是否是有效的匀速直线运动轨迹
            if success and is_uniform_motion(trajectory):
                trajectories.append({
                    'sequence_id': seq_id,
                    'points': trajectory,
                    'distance': math.sqrt(
                        (trajectory[-1][0] - trajectory[0][0])**2 +
                        (trajectory[-1][1] - trajectory[0][1])**2
                    )
                })
                print(f"找到轨迹: {trajectory}")
    
    return trajectories

def analyze_distances(trajectories):
    """分析轨迹距离"""
    if not trajectories:
        print("没有找到有效轨迹！")
        return
    
    distances = [t['distance'] for t in trajectories]
    
    print("\n=== 轨迹距离统计 ===")
    print(f"轨迹数量: {len(trajectories)}")
    print(f"平均距离: {np.mean(distances):.2f}")
    print(f"标准差: {np.std(distances):.2f}")
    print(f"最大距离: {max(distances):.2f}")
    print(f"最小距离: {min(distances):.2f}")
    print(f"中位数: {np.median(distances):.2f}")
    
    # 分析特定距离范围的序列
    print("\n=== 距离大于200的轨迹 ===")
    for t in trajectories:
        if t['distance'] > 200:
            print(f"序列 {t['sequence_id']}: 距离 = {t['distance']:.2f}")
    
    print("\n=== 距离小于50的轨迹 ===")
    for t in trajectories:
        if t['distance'] < 50:
            print(f"序列 {t['sequence_id']}: 距离 = {t['distance']:.2f}")
    
    # 绘制分布图
    plt.figure(figsize=(10, 5))
    plt.hist(distances, bins=20)
    plt.title('Trajectory Distances')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.savefig('statistics/distance_distribution.png')
    plt.close()

def analyze_intra_sequence_distance_diff(trajectories):
    """分析同一序列内不同轨迹组之间的距离差异
    
    Args:
        trajectories: 轨迹列表
    """
    # 按序列ID组织轨迹
    sequence_trajectories = {}
    for traj in trajectories:
        seq_id = traj['sequence_id']
        if seq_id not in sequence_trajectories:
            sequence_trajectories[seq_id] = []
        sequence_trajectories[seq_id].append(traj)
    
    # 分析每个序列内的距离差异
    max_diffs = []  # 存储每个序列内的最大距离差
    sequence_max_diffs = {}  # 存储每个序列的最大距离差详情
    large_diff_sequences = []  # 存储差距大于40的序列
    
    for seq_id, trajs in sequence_trajectories.items():
        if len(trajs) < 2:  # 跳过只有一条轨迹的序列
            continue
            
        # 获取该序列所有轨迹的距离
        distances = [t['distance'] for t in trajs]
        max_dist = max(distances)
        min_dist = min(distances)
        max_diff = max_dist - min_dist
        
        max_diffs.append(max_diff)
        sequence_max_diffs[seq_id] = {
            'max_diff': max_diff,
            'max_dist': max_dist,
            'min_dist': min_dist,
            'num_trajectories': len(trajs)
        }
        
        # 记录差距大于40的序列
        if max_diff > 40:
            large_diff_sequences.append({
                'sequence_id': seq_id,
                'max_diff': max_diff,
                'max_dist': max_dist,
                'min_dist': min_dist,
                'num_trajectories': len(trajs)
            })
    
    # 输出统计信息
    print("\n=== 序列内轨迹距离差异统计 ===")
    print(f"包含多条轨迹的序列数量: {len(max_diffs)}")
    print(f"平均最大差异: {np.mean(max_diffs):.2f}")
    print(f"最大差异的标准差: {np.std(max_diffs):.2f}")
    print(f"所有序列中的最大差异: {max(max_diffs):.2f}")
    print(f"所有序列中的最小差异: {min(max_diffs):.2f}")
    
    # 输出差异最大的前5个序列的详细信息
    print("\n=== 轨迹距离差异最大的序列 (Top 5) ===")
    sorted_sequences = sorted(sequence_max_diffs.items(), 
                            key=lambda x: x[1]['max_diff'], 
                            reverse=True)
    for seq_id, info in sorted_sequences[:5]:
        print(f"序列 {seq_id}:")
        print(f"  轨迹数量: {info['num_trajectories']}")
        print(f"  最大距离: {info['max_dist']:.2f}")
        print(f"  最小距离: {info['min_dist']:.2f}")
        print(f"  最大差异: {info['max_diff']:.2f}")
    
    # 输出所有差距大于40的序列
    print("\n=== 差距大于40的序列 ===")
    print(f"总计: {len(large_diff_sequences)}个序列")
    # 按差距大小排序
    large_diff_sequences.sort(key=lambda x: x['max_diff'], reverse=True)
    for seq in large_diff_sequences:
        print(f"序列 {seq['sequence_id']}:")
        print(f"  轨迹数量: {seq['num_trajectories']}")
        print(f"  最大距离: {seq['max_dist']:.2f}")
        print(f"  最小距离: {seq['min_dist']:.2f}")
        print(f"  最大差异: {seq['max_diff']:.2f}")
    
    # 绘制最大差异的分布直方图
    plt.figure(figsize=(10, 5))
    plt.hist(max_diffs, bins=20)
    plt.title('Distribution of Maximum Distance Differences within Sequences')
    plt.xlabel('Maximum Distance Difference')
    plt.ylabel('Count')
    plt.axvline(x=40, color='r', linestyle='--', label='Threshold (40)')
    plt.legend()
    plt.savefig('statistics/intra_sequence_distance_diff_distribution.png')
    plt.close()

def visualize_trajectories(trajectories, output_dir):
    """可视化轨迹
    
    Args:
        trajectories: 轨迹列表
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 按序列ID组织轨迹
    sequence_trajectories = {}
    for traj in trajectories:
        seq_id = traj['sequence_id']
        if seq_id not in sequence_trajectories:
            sequence_trajectories[seq_id] = []
        sequence_trajectories[seq_id].append(traj)
    
    # 为每个序列创建一张图
    for seq_id, trajs in sequence_trajectories.items():
        # 创建黑色背景图像
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 为每条轨迹随机生成一个颜色
        colors = np.random.randint(50, 255, (len(trajs), 3)).tolist()
        
        # 绘制每条轨迹
        for traj, color in zip(trajs, colors):
            points = traj['points']
            distance = traj['distance']
            
            # 绘制轨迹点
            for point in points:
                x, y = int(point[0]), int(point[1])
                cv2.circle(img, (x, y), 3, color, -1)
            
            # 绘制轨迹线
            points_array = np.array(points, dtype=np.int32)
            cv2.polylines(img, [points_array], False, color, 2)
            
            # 在轨迹起点添加距离标注
            start_x, start_y = int(points[0][0]), int(points[0][1])
            text = f"d={distance:.1f}"
            cv2.putText(img, text, (start_x, start_y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 保存图像
        output_path = os.path.join(output_dir, f'sequence_{seq_id}_trajectories.png')
        cv2.imwrite(output_path, img)
        # print(f"已保存序列 {seq_id} 的轨迹图到: {output_path}")

def main():
    predictions_file = os.path.join("datasets", "spotgeov2", "test_anno.json")
    
    # 读取预测结果
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # 创建输出目录
    Path('statistics').mkdir(exist_ok=True)
    
    # 找出轨迹
    trajectories = find_trajectories(predictions)
    
    # 分析距离
    analyze_distances(trajectories)
    
    # 分析序列内轨迹距离差异
    analyze_intra_sequence_distance_diff(trajectories)
    
    # 可视化轨迹
    visualize_trajectories(trajectories, 'statistics/trajectories')
    
    # 保存结果
    with open('statistics/trajectories.json', 'w') as f:
        json.dump(trajectories, f, indent=2)

if __name__ == "__main__":
    main()
