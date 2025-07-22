#!/usr/bin/env python3
"""
根据空序列信息过滤数据集索引文件，去掉没有目标的图片
"""

import json
import os
from collections import defaultdict

def load_empty_sequences(empty_file_path):
    """加载空序列信息"""
    print(f"正在加载空序列文件: {empty_file_path}")
    
    with open(empty_file_path, 'r', encoding='utf-8') as f:
        empty_data = json.load(f)
    
    # 提取空序列的sequence_id
    empty_sequences = set()
    for item in empty_data:
        empty_sequences.add(item['sequence_id'])
    
    print(f"找到 {len(empty_sequences)} 个空序列")
    return empty_sequences

def filter_index_file(index_file_path, empty_sequences, output_file_path):
    """过滤索引文件，去掉空序列的图片"""
    print(f"正在过滤索引文件: {index_file_path}")
    
    # 读取原始索引文件
    with open(index_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 过滤掉空序列的图片
    filtered_lines = []
    removed_count = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 解析图片路径，提取sequence_id
        # 格式为: sequence_id_frame_id_train/test
        parts = line.split('_')
        if len(parts) >= 2:
            try:
                sequence_id = int(parts[0])  # 第一个部分应该是sequence_id
                if sequence_id not in empty_sequences:
                    filtered_lines.append(line)
                else:
                    removed_count += 1
            except ValueError:
                # 如果无法解析sequence_id，保留该行
                filtered_lines.append(line)
        else:
            # 如果路径格式不符合预期，保留该行
            filtered_lines.append(line)
    
    # 保存过滤后的索引文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for line in filtered_lines:
            f.write(line + '\n')
    
    print(f"原始图片数量: {len(lines)}")
    print(f"过滤后图片数量: {len(filtered_lines)}")
    print(f"移除的图片数量: {removed_count}")
    print(f"过滤后的索引文件已保存到: {output_file_path}")

def main():
    """主函数"""
    # 文件路径
    dataset_dir = "/autodl-fs/data/spotgeov2-IRSTD/img_idx"
    train_empty_file = "train_empty_sequences.json"
    test_empty_file = "test_empty_sequences.json"
    
    train_index_file = os.path.join(dataset_dir, "train_spotgeov2-IRSTD.txt")
    test_index_file = os.path.join(dataset_dir, "test_spotgeov2-IRSTD.txt")
    
    train_filtered_file = "train_spotgeov2-IRSTD_filtered.txt"
    test_filtered_file = "test_spotgeov2-IRSTD_filtered.txt"
    
    print("=" * 60)
    print("过滤数据集索引文件，去掉空序列的图片")
    print("=" * 60)
    
    # 加载空序列信息
    if os.path.exists(train_empty_file):
        train_empty_sequences = load_empty_sequences(train_empty_file)
    else:
        print(f"训练集空序列文件不存在: {train_empty_file}")
        train_empty_sequences = set()
    
    if os.path.exists(test_empty_file):
        test_empty_sequences = load_empty_sequences(test_empty_file)
    else:
        print(f"测试集空序列文件不存在: {test_empty_file}")
        test_empty_sequences = set()
    
    # 过滤训练集索引文件
    print("\n【过滤训练集索引文件】")
    if os.path.exists(train_index_file):
        filter_index_file(train_index_file, train_empty_sequences, train_filtered_file)
    else:
        print(f"训练集索引文件不存在: {train_index_file}")
    
    # 过滤测试集索引文件
    print("\n【过滤测试集索引文件】")
    if os.path.exists(test_index_file):
        filter_index_file(test_index_file, test_empty_sequences, test_filtered_file)
    else:
        print(f"测试集索引文件不存在: {test_index_file}")
    
    # 总结
    print("\n" + "=" * 60)
    print("【总结】")
    print(f"训练集空序列数: {len(train_empty_sequences)}")
    print(f"测试集空序列数: {len(test_empty_sequences)}")
    print(f"训练集过滤后文件: {train_filtered_file}")
    print(f"测试集过滤后文件: {test_filtered_file}")
    print("=" * 60)

if __name__ == "__main__":
    main() 