#!/usr/bin/env python3
"""
测试过滤后的数据功能
"""

import argparse
import sys
import os

# 添加当前目录到Python路径
sys.path.append('.')

from dataset_spotgeo import TrainSetLoader, TestSetLoader

def test_filtered_data():
    """测试过滤后的数据加载"""
    print("=" * 60)
    print("测试过滤后的数据功能")
    print("=" * 60)
    
    # 测试参数
    dataset_dir = "/autodl-fs/data"
    dataset_name = "spotgeov2-IRSTD"
    patch_size = 512
    
    print("\n【测试原始数据】")
    try:
        # 测试原始训练数据
        train_dataset_original = TrainSetLoader(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            patch_size=patch_size,
            use_filtered_data=False
        )
        print(f"原始训练数据样本数: {len(train_dataset_original)}")
        
        # 测试原始测试数据
        test_dataset_original = TestSetLoader(
            dataset_dir=dataset_dir,
            train_dataset_name=dataset_name,
            test_dataset_name=dataset_name,
            patch_size=patch_size,
            use_filtered_data=False
        )
        print(f"原始测试数据样本数: {len(test_dataset_original)}")
        
    except Exception as e:
        print(f"原始数据加载失败: {e}")
    
    print("\n【测试过滤后的数据】")
    try:
        # 测试过滤后的训练数据
        train_dataset_filtered = TrainSetLoader(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            patch_size=patch_size,
            use_filtered_data=True
        )
        print(f"过滤后训练数据样本数: {len(train_dataset_filtered)}")
        
        # 测试过滤后的测试数据
        test_dataset_filtered = TestSetLoader(
            dataset_dir=dataset_dir,
            train_dataset_name=dataset_name,
            test_dataset_name=dataset_name,
            patch_size=patch_size,
            use_filtered_data=True
        )
        print(f"过滤后测试数据样本数: {len(test_dataset_filtered)}")
        
    except Exception as e:
        print(f"过滤后数据加载失败: {e}")
    
    print("\n【数据对比】")
    if 'train_dataset_original' in locals() and 'train_dataset_filtered' in locals():
        train_reduction = len(train_dataset_original) - len(train_dataset_filtered)
        train_reduction_pct = (train_reduction / len(train_dataset_original)) * 100
        print(f"训练数据减少: {train_reduction} 样本 ({train_reduction_pct:.1f}%)")
    
    if 'test_dataset_original' in locals() and 'test_dataset_filtered' in locals():
        test_reduction = len(test_dataset_original) - len(test_dataset_filtered)
        test_reduction_pct = (test_reduction / len(test_dataset_original)) * 100
        print(f"测试数据减少: {test_reduction} 样本 ({test_reduction_pct:.1f}%)")
    
    print("=" * 60)

if __name__ == "__main__":
    test_filtered_data() 