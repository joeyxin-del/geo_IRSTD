#!/usr/bin/env python3
"""
计算达到目标F1分数所需的recall值
"""

def calculate_required_recall(target_f1, precision):
    """
    根据目标F1分数和precision计算所需的recall
    
    Args:
        target_f1 (float): 目标F1分数
        precision (float): 当前precision值
    
    Returns:
        float: 所需的recall值
    """
    # F1 = 2 * (precision * recall) / (precision + recall)
    # 解方程: target_f1 = 2 * (precision * recall) / (precision + recall)
    # 2 * precision * recall = target_f1 * (precision + recall)
    # 2 * precision * recall = target_f1 * precision + target_f1 * recall
    # 2 * precision * recall - target_f1 * recall = target_f1 * precision
    # recall * (2 * precision - target_f1) = target_f1 * precision
    # recall = (target_f1 * precision) / (2 * precision - target_f1)
    
    required_recall = (target_f1 * precision) / (2 * precision - target_f1)
    return required_recall

def main():
    # 当前参数
    current_precision = 0.9578
    target_f1 = 0.94
    
    # 计算所需的recall
    required_recall = calculate_required_recall(target_f1, current_precision)
    
    print(f"当前参数:")
    print(f"  Precision: {current_precision:.4f}")
    print(f"  目标F1分数: {target_f1:.4f}")
    print(f"\n计算结果:")
    print(f"  所需Recall: {required_recall:.4f}")
    
    # 验证计算
    calculated_f1 = 2 * (current_precision * required_recall) / (current_precision + required_recall)
    print(f"\n验证:")
    print(f"  使用计算出的recall重新计算F1: {calculated_f1:.4f}")
    print(f"  与目标F1的差异: {abs(calculated_f1 - target_f1):.6f}")
    
    # 分析
    current_recall = 0.8972  # 从你的结果中获取
    recall_improvement_needed = required_recall - current_recall
    
    print(f"\n分析:")
    print(f"  当前Recall: {current_recall:.4f}")
    print(f"  需要提升的Recall: {recall_improvement_needed:.4f}")
    print(f"  相对提升幅度: {(recall_improvement_needed / current_recall * 100):.2f}%")

if __name__ == "__main__":
    main() 