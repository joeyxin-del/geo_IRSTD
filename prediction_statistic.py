import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']  # Use common English fonts
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

class PredictionStatistic:
    """预测结果统计分析器"""
    
    def __init__(self, predictions_file: str):
        """
        初始化统计分析器
        
        Args:
            predictions_file: 预测结果文件路径
        """
        self.predictions_file = predictions_file
        self.predictions = None
        self.sequence_data = defaultdict(dict)
        self.trajectory_stats = {}
        
    def load_predictions(self):
        """加载预测结果"""
        print(f"正在加载预测结果: {self.predictions_file}")
        with open(self.predictions_file, 'r') as f:
            self.predictions = json.load(f)
        print(f"加载完成，共 {len(self.predictions)} 张图像")
        
    def extract_sequence_info(self):
        """提取序列信息"""
        print("正在提取序列信息...")
        
        for img_name, pred_info in self.predictions.items():
            parts = img_name.split('_')
            if len(parts) >= 2:
                try:
                    sequence_id = int(parts[0])
                    frame = int(parts[1])
                    self.sequence_data[sequence_id][frame] = {
                        'coords': pred_info['coords'],
                        'num_objects': pred_info['num_objects'],
                        'image_name': img_name
                    }
                except ValueError:
                    continue
        
        print(f"提取完成，共 {len(self.sequence_data)} 个序列")
        
    def group_targets_into_trajectories(self, sequence_frames, max_distance=20):
        """
        对单个序列下所有帧的检测点进行目标级轨迹结组（最近邻，最大距离max_distance）
        Args:
            sequence_frames: {frame_idx: {'coords': [[x, y], ...], ...}}
            max_distance: 最大允许关联距离
        Returns:
            List[List[Tuple[frame_idx, [x, y]]]]: 每条轨迹是一个列表，元素为(frame_idx, [x, y])
        """
        frames = sorted(sequence_frames.keys())
        trajectories = []  # 每条轨迹是[(frame_idx, [x, y]), ...]
        if not frames:
            return []
        # 初始化：第一帧的每个点新建一条轨迹
        for pt in sequence_frames[frames[0]]['coords']:
            trajectories.append([(frames[0], pt)])
        # 逐帧关联
        for i in range(1, len(frames)):
            cur_frame = frames[i]
            cur_points = sequence_frames[cur_frame]['coords']
            assigned = [False] * len(cur_points)
            # 记录每条轨迹是否已被当前帧的点延续
            traj_extended = [False] * len(trajectories)
            # 先尝试将当前帧的点关联到上一帧的轨迹末尾
            for t_idx, traj in enumerate(trajectories):
                last_frame, last_pt = traj[-1]
                if last_frame != frames[i-1]:
                    continue  # 只关联连续帧
                min_dist = float('inf')
                min_j = -1
                for j, pt in enumerate(cur_points):
                    if assigned[j]:
                        continue
                    dist = np.linalg.norm(np.array(pt) - np.array(last_pt))
                    if dist < min_dist:
                        min_dist = dist
                        min_j = j
                if min_j != -1 and min_dist <= max_distance:
                    # 关联成功
                    traj.append((cur_frame, cur_points[min_j]))
                    assigned[min_j] = True
                    traj_extended[t_idx] = True
            # 剩下未分配的点新建轨迹
            for j, pt in enumerate(cur_points):
                if not assigned[j]:
                    trajectories.append([(cur_frame, pt)])
        return trajectories

    def analyze_trajectory_completeness(self):
        """基于目标级轨迹结组的轨迹完整性统计"""
        print("正在分析目标级轨迹完整性...")
        complete_trajectories = 0  # 5帧
        incomplete_4_points = 0    # 4帧
        incomplete_3_points = 0    # 3帧
        incomplete_2_points = 0    # 2帧
        single_points = 0          # 1帧
        total_trajectories = 0
        trajectory_details = []
        for sequence_id, frames_data in self.sequence_data.items():
            trajs = self.group_targets_into_trajectories(frames_data, max_distance=20)
            for traj in trajs:
                length = len(traj)
                total_trajectories += 1
                if length == 5:
                    complete_trajectories += 1
                    traj_type = 'complete'
                elif length == 4:
                    incomplete_4_points += 1
                    traj_type = '4_points'
                elif length == 3:
                    incomplete_3_points += 1
                    traj_type = '3_points'
                elif length == 2:
                    incomplete_2_points += 1
                    traj_type = '2_points'
                elif length == 1:
                    single_points += 1
                    traj_type = 'single_point'
                else:
                    traj_type = 'other'
                trajectory_details.append({
                    'sequence_id': sequence_id,
                    'length': length,
                    'frames': [f for f, _ in traj],
                    'points': [pt for _, pt in traj],
                    'type': traj_type
                })
        self.trajectory_stats = {
            'summary': {
                'total_trajectories': total_trajectories,
                'complete_trajectories': complete_trajectories,
                'incomplete_4_points': incomplete_4_points,
                'incomplete_3_points': incomplete_3_points,
                'incomplete_2_points': incomplete_2_points,
                'single_points': single_points
            },
            'details': trajectory_details
        }
        return self.trajectory_stats
    
    def print_statistics(self):
        """打印统计结果 (基于目标轨迹)"""
        if not self.trajectory_stats:
            print("请先运行 analyze_trajectory_completeness()")
            return
        summary = self.trajectory_stats['summary']
        print("\n" + "="*60)
        print("目标级轨迹完整性统计")
        print("="*60)
        print(f"总轨迹数: {summary['total_trajectories']}")
        print(f"完整轨迹 (5帧): {summary['complete_trajectories']} ({summary['complete_trajectories']/summary['total_trajectories']*100:.2f}%)")
        print(f"4帧轨迹: {summary['incomplete_4_points']} ({summary['incomplete_4_points']/summary['total_trajectories']*100:.2f}%)")
        print(f"3帧轨迹: {summary['incomplete_3_points']} ({summary['incomplete_3_points']/summary['total_trajectories']*100:.2f}%)")
        print(f"2帧轨迹: {summary['incomplete_2_points']} ({summary['incomplete_2_points']/summary['total_trajectories']*100:.2f}%)")
        print(f"独立点 (1帧): {summary['single_points']} ({summary['single_points']/summary['total_trajectories']*100:.2f}%)")
        
        print("\n" + "-"*60)
        print("轨迹类型分析")
        print("-"*60)
        
        # 计算有效轨迹（2个点以上）
        valid_trajectories = (summary['complete_trajectories'] + summary['incomplete_4_points'] + 
                            summary['incomplete_3_points'] + summary['incomplete_2_points'])
        
        print(f"有效轨迹总数 (≥2个点): {valid_trajectories}")
        print(f"完整轨迹比例: {summary['complete_trajectories']/max(valid_trajectories, 1)*100:.2f}%")
        print(f"不完整轨迹比例: {(valid_trajectories - summary['complete_trajectories'])/max(valid_trajectories, 1)*100:.2f}%")
        
        # 分析2个点是否算轨迹
        print(f"\n关于2个点的讨论:")
        print(f"2个点的序列数: {summary['incomplete_2_points']}")
        if summary['incomplete_2_points'] > 0:
            print(f"占有效轨迹比例: {summary['incomplete_2_points']/max(valid_trajectories, 1)*100:.2f}%")
            print("从技术角度看，2个点可以构成一条直线，但信息量较少")
            print("建议：2个点可以作为轨迹的候选，但需要额外的验证")
        
        # 分析独立点
        print(f"\n独立点分析:")
        print(f"独立点序列数: {summary['single_points']}")
        if summary['single_points'] > 0:
            print(f"占总序列比例: {summary['single_points']/summary['total_trajectories']*100:.2f}%")
            print("独立点可能是：")
            print("1. 噪声检测")
            print("2. 单帧检测到的目标")
            print("3. 轨迹的起始或结束点")
    
    def analyze_detection_distribution(self):
        """分析目标轨迹的长度分布和帧分布"""
        if not self.trajectory_stats:
            print("请先运行 analyze_trajectory_completeness()")
            return

        print("\n" + "-"*60)
        print("Trajectory Length Distribution Analysis")
        print("-"*60)

        # 统计轨迹长度分布
        length_count = defaultdict(int)
        for traj in self.trajectory_stats['details']:
            length_count[traj['length']] += 1

        print("Trajectory length distribution:")
        for length in sorted(length_count.keys()):
            print(f"  {length} frames: {length_count[length]} trajectories")

        # 统计所有轨迹中每帧出现的检测点数分布
        frame_count = defaultdict(int)
        for traj in self.trajectory_stats['details']:
            for f in traj['frames']:
                frame_count[f] += 1

        print("\nDetection count per frame (across all trajectories):")
        for f in sorted(frame_count.keys()):
            print(f"  Frame {f}: {frame_count[f]} detections")
    
    def find_example_sequences(self):
        """查找目标轨迹示例"""
        if not self.trajectory_stats:
            print("请先运行 analyze_trajectory_completeness()")
            return

        print("\n" + "-"*60)
        print("Trajectory Example Analysis")
        print("-"*60)

        # 按轨迹类型分组
        trajectory_groups = defaultdict(list)
        for traj in self.trajectory_stats['details']:
            trajectory_groups[traj['type']].append(traj)

        # 显示每种类型的示例
        for traj_type, trajs in trajectory_groups.items():
            if len(trajs) > 0:
                print(f"\n{traj_type} examples:")
                for i, traj in enumerate(trajs[:5]):
                    print(f"  Sequence {traj['sequence_id']}: length={traj['length']}, frames={traj['frames']}")
                    print(f"    Points: {traj['points']}")
                if len(trajs) > 5:
                    print(f"    ... and {len(trajs)-5} more trajectories")
    
    def generate_visualization(self, save_path: str = "trajectory_statistics.png"):
        """Generate visualization charts in English"""
        if not self.trajectory_stats:
            print("Please run analyze_trajectory_completeness() first.")
            return
        
        summary = self.trajectory_stats['summary']
        
        # Create charts
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Trajectory type distribution pie chart
        labels = ['Complete (5 pts)', '4 pts', '3 pts', '2 pts', 'Single (1 pt)']
        sizes = [
            summary['complete_trajectories'],
            summary['incomplete_4_points'],
            summary['incomplete_3_points'],
            summary['incomplete_2_points'],
            summary['single_points']
        ]
        colors = ['#2E8B57', '#3CB371', '#20B2AA', '#4682B4', '#DDA0DD']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Trajectory Type Distribution')
        
        # 2. Valid trajectory bar chart
        valid_labels = ['Complete', 'Incomplete']
        valid_sizes = [summary['complete_trajectories'], 
                      summary['incomplete_4_points'] + summary['incomplete_3_points'] + summary['incomplete_2_points']]
        
        ax2.bar(valid_labels, valid_sizes, color=['#2E8B57', '#4682B4'])
        ax2.set_title('Valid Trajectory Distribution')
        ax2.set_ylabel('Number of Sequences')
        for i, v in enumerate(valid_sizes):
            ax2.text(i, v + 0.01, str(v), ha='center', va='bottom')
        
        # 3. Detection point count distribution
        detection_counts = [summary['complete_trajectories'] * 5, 
                          summary['incomplete_4_points'] * 4,
                          summary['incomplete_3_points'] * 3,
                          summary['incomplete_2_points'] * 2,
                          summary['single_points']]
        detection_labels = ['5 pts', '4 pts', '3 pts', '2 pts', '1 pt']
        
        ax3.bar(detection_labels, detection_counts, color=['#2E8B57', '#3CB371', '#20B2AA', '#4682B4', '#DDA0DD'])
        ax3.set_title('Detection Point Count Distribution')
        ax3.set_ylabel('Total Detection Points')
        for i, v in enumerate(detection_counts):
            ax3.text(i, v + 0.01, str(v), ha='center', va='bottom')
        
        # 4. Trajectory completeness analysis
        completeness_data = []
        completeness_labels = []
        
        if summary['complete_trajectories'] > 0:
            completeness_data.append(summary['complete_trajectories'])
            completeness_labels.append('Complete (100%)')
        
        if summary['incomplete_4_points'] > 0:
            completeness_data.append(summary['incomplete_4_points'])
            completeness_labels.append('4/5 (80%)')
        
        if summary['incomplete_3_points'] > 0:
            completeness_data.append(summary['incomplete_3_points'])
            completeness_labels.append('3/5 (60%)')
        
        if summary['incomplete_2_points'] > 0:
            completeness_data.append(summary['incomplete_2_points'])
            completeness_labels.append('2/5 (40%)')
        
        if summary['single_points'] > 0:
            completeness_data.append(summary['single_points'])
            completeness_labels.append('1/5 (20%)')
        
        ax4.bar(completeness_labels, completeness_data, color=['#2E8B57', '#3CB371', '#20B2AA', '#4682B4', '#DDA0DD'])
        ax4.set_title('Trajectory Completeness Analysis')
        ax4.set_ylabel('Number of Sequences')
        for i, v in enumerate(completeness_data):
            ax4.text(i, v + 0.01, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization chart saved to: {save_path}")
        plt.show()
    
    def save_detailed_report(self, save_path: str = "trajectory_analysis_report.json"):
        """保存详细分析报告"""
        if not self.trajectory_stats:
            print("请先运行 analyze_trajectory_completeness()")
            return

        summary = self.trajectory_stats['summary']
        analysis_conclusion = {
            'total_trajectories': summary['total_trajectories'],
            'complete_trajectories': summary['complete_trajectories'],
            'incomplete_trajectories': (
                summary['incomplete_4_points'] +
                summary['incomplete_3_points'] +
                summary['incomplete_2_points']
            ),
            'single_points': summary['single_points'],
            'completeness_rate': summary['complete_trajectories'] / summary['total_trajectories'] * 100,
            'valid_trajectory_rate': (
                summary['complete_trajectories'] +
                summary['incomplete_4_points'] +
                summary['incomplete_3_points'] +
                summary['incomplete_2_points']
            ) / summary['total_trajectories'] * 100,
            'analysis_notes': {
                'two_point_trajectories': f"{summary['incomplete_2_points']} trajectories with 2 points, {summary['incomplete_2_points']/summary['total_trajectories']*100:.2f}%",
                'single_point_analysis': f"{summary['single_points']} single-point trajectories, likely noise or single-frame detection",
                'completeness_analysis': f"Complete trajectory rate: {summary['complete_trajectories']/summary['total_trajectories']*100:.2f}%"
            }
        }

        report = {
            'summary': summary,
            'analysis_conclusion': analysis_conclusion,
            'details': self.trajectory_stats['details']
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"Detailed analysis report saved to: {save_path}")

def main():
    """主函数"""
    # 预测结果文件路径
    predictions_file = 'results/WTNet/predictions.json'
    
    # 创建统计分析器
    analyzer = PredictionStatistic(predictions_file)
    
    # 加载数据
    analyzer.load_predictions()
    analyzer.extract_sequence_info()
    
    # 分析轨迹完整性
    analyzer.analyze_trajectory_completeness()
    
    # 打印统计结果
    analyzer.print_statistics()
    
    # 分析检测点分布
    analyzer.analyze_detection_distribution()
    
    # 查找示例序列
    analyzer.find_example_sequences()
    
    # 生成可视化图表
    analyzer.generate_visualization()
    
    # 保存详细报告
    analyzer.save_detailed_report()

if __name__ == '__main__':
    main() 