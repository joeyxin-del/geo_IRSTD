#!/usr/bin/env python3
"""
提取spotgeov2-IRSTD数据集中的空序列，并保存为JSON文件
"""

import json
import os
from collections import defaultdict

def extract_empty_sequences(anno_file_path):
    """提取标注文件中的空序列"""
    print(f"正在分析文件: {anno_file_path}")
    
    # 读取标注文件
    with open(anno_file_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # 统计每个序列的目标数量
    sequence_stats = defaultdict(list)
    empty_sequences = []
    empty_sequences_data = []
    
    for anno in annotations:
        sequence_id = anno['sequence_id']
        frame = anno['frame']
        num_objects = anno['num_objects']
        object_coords = anno['object_coords']
        
        sequence_stats[sequence_id].append({
            'sequence_id': sequence_id,
            'frame': frame,
            'num_objects': num_objects,
            'object_coords': object_coords
        })
    
    # 检查每个序列是否完全没有目标
    for sequence_id, frames in sequence_stats.items():
        # 检查是否所有5帧都没有目标
        all_empty = all(frame['num_objects'] == 0 for frame in frames)
        
        if all_empty:
            empty_sequences.append(sequence_id)
            # 保存该序列的所有帧信息
            for frame_info in frames:
                empty_sequences_data.append(frame_info)
    
    return empty_sequences, empty_sequences_data

def main():
    """主函数"""
    # 数据集路径
    dataset_dir = "/autodl-fs/data/spotgeov2-IRSTD"
    train_anno_path = os.path.join(dataset_dir, "train_anno.json")
    test_anno_path = os.path.join(dataset_dir, "test_anno.json")
    
    print("=" * 60)
    print("提取 spotgeov2-IRSTD 数据集中的空序列")
    print("=" * 60)
    
    # 分析训练集
    print("\n【训练集分析】")
    if os.path.exists(train_anno_path):
        train_empty_sequences, train_empty_data = extract_empty_sequences(train_anno_path)
        print(f"训练集中完全没有目标的序列数量: {len(train_empty_sequences)}")
        print(f"训练集中完全没有目标的序列序号: {sorted(train_empty_sequences)}")
        
        # 保存训练集空序列
        train_empty_file = "train_empty_sequences.json"
        with open(train_empty_file, 'w', encoding='utf-8') as f:
            json.dump(train_empty_data, f, indent=2, ensure_ascii=False)
        print(f"训练集空序列已保存到: {train_empty_file}")
    else:
        print(f"训练集标注文件不存在: {train_anno_path}")
        train_empty_sequences = []
        train_empty_data = []
    
    # 分析测试集
    print("\n【测试集分析】")
    if os.path.exists(test_anno_path):
        test_empty_sequences, test_empty_data = extract_empty_sequences(test_anno_path)
        print(f"测试集中完全没有目标的序列数量: {len(test_empty_sequences)}")
        print(f"测试集中完全没有目标的序列序号: {sorted(test_empty_sequences)}")
        
        # 保存测试集空序列
        test_empty_file = "test_empty_sequences.json"
        with open(test_empty_file, 'w', encoding='utf-8') as f:
            json.dump(test_empty_data, f, indent=2, ensure_ascii=False)
        print(f"测试集空序列已保存到: {test_empty_file}")
    else:
        print(f"测试集标注文件不存在: {test_anno_path}")
        test_empty_sequences = []
        test_empty_data = []
    
    # 总结
    print("\n" + "=" * 60)
    print("【总结】")
    print(f"训练集空序列数: {len(train_empty_sequences)}")
    print(f"测试集空序列数: {len(test_empty_sequences)}")
    print(f"训练集空序列数据条数: {len(train_empty_data)}")
    print(f"测试集空序列数据条数: {len(test_empty_data)}")
    
    all_empty_sequences = sorted(set(train_empty_sequences + test_empty_sequences))
    print(f"所有空序列序号: {all_empty_sequences}")
    print("=" * 60)

if __name__ == "__main__":
    main() 