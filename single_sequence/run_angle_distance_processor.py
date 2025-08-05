#!/usr/bin/env python3
"""
单帧角度距离处理器运行脚本
用于测试和分析单个序列的角度距离后处理效果
"""

import os
import sys
import argparse
from angle_distance_processor import SingleSequenceAngleDistanceProcessor

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='单帧角度距离处理器运行脚本')
    parser.add_argument('--pred_path', type=str, default='results/spotgeov2/WTNet/predictions.json',
                       help='预测结果文件路径')
    parser.add_argument('--sequence_id', type=int, default=69,
                       help='要分析的序列ID')
    parser.add_argument('--base_distance_threshold', type=float, default=1000.0,
                       help='基础距离阈值')
    parser.add_argument('--angle_tolerance', type=float, default=3,
                       help='角度容差（度）')
    parser.add_argument('--min_angle_count', type=int, default=2,
                       help='最小角度出现次数')
    parser.add_argument('--step_tolerance', type=float, default=0.2,
                       help='步长容差（比例）')
    parser.add_argument('--min_step_count', type=int, default=1,
                       help='最小步长出现次数')
    parser.add_argument('--point_distance_threshold', type=float, default=200.0,
                       help='重合点过滤阈值')
    parser.add_argument('--no_visualization', action='store_true',
                       help='不保存可视化结果')
    parser.add_argument('--batch_mode', action='store_true',
                       help='批量处理模式，处理多个序列')
    parser.add_argument('--start_sequence', type=int, default=1,
                       help='批量模式起始序列ID')
    parser.add_argument('--end_sequence', type=int, default=10,
                       help='批量模式结束序列ID')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.pred_path):
        print(f"错误：预测结果文件不存在: {args.pred_path}")
        return
    
    # 创建单序列角度距离处理器
    processor = SingleSequenceAngleDistanceProcessor(
        base_distance_threshold=args.base_distance_threshold,
        angle_tolerance=args.angle_tolerance,
        min_angle_count=args.min_angle_count,
        step_tolerance=args.step_tolerance,
        min_step_count=args.min_step_count,
        point_distance_threshold=args.point_distance_threshold
    )
    
    if args.batch_mode:
        # 批量处理模式
        print(f"=== 批量处理模式 ===")
        print(f"处理序列范围: {args.start_sequence} - {args.end_sequence}")
        
        successful_sequences = []
        failed_sequences = []
        
        for seq_id in range(args.start_sequence, args.end_sequence + 1):
            print(f"\n{'='*50}")
            print(f"处理序列 {seq_id}")
            print(f"{'='*50}")
            
            try:
                result = processor.analyze_sequence(
                    args.pred_path, 
                    seq_id,
                    save_visualization=not args.no_visualization
                )
                
                if result:
                    successful_sequences.append(seq_id)
                    print(f"✓ 序列 {seq_id} 处理成功")
                else:
                    failed_sequences.append(seq_id)
                    print(f"✗ 序列 {seq_id} 处理失败（无数据）")
                    
            except Exception as e:
                failed_sequences.append(seq_id)
                print(f"✗ 序列 {seq_id} 处理出错: {e}")
        
        # 打印批量处理结果
        print(f"\n{'='*60}")
        print(f"批量处理完成")
        print(f"{'='*60}")
        print(f"成功处理: {len(successful_sequences)} 个序列")
        print(f"失败序列: {len(failed_sequences)} 个序列")
        
        if successful_sequences:
            print(f"成功序列ID: {successful_sequences}")
        if failed_sequences:
            print(f"失败序列ID: {failed_sequences}")
            
    else:
        # 单序列处理模式
        print(f"=== 单序列处理模式 ===")
        print(f"处理序列: {args.sequence_id}")
        
        result = processor.analyze_sequence(
            args.pred_path, 
            args.sequence_id,
            save_visualization=not args.no_visualization
        )
        
        if result:
            print(f"\n✓ 序列 {args.sequence_id} 角度距离分析完成！")
        else:
            print(f"\n✗ 序列 {args.sequence_id} 处理失败（无数据）")

if __name__ == '__main__':
    main() 