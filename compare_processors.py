#!/usr/bin/env python3
"""
比较不同序列后处理器的效果
"""

import json
import time
from processor import ImprovedSequenceProcessor, SimpleSequenceProcessor, BalancedSequenceProcessor

def load_data():
    """加载预测结果和真实标注"""
    pred_path = 'results/WTNet/predictions.json'
    gt_path = 'datasets/spotgeov2-IRSTD/test_anno.json'
    
    print("正在加载数据...")
    with open(pred_path, 'r') as f:
        original_predictions = json.load(f)
    
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    return original_predictions, ground_truth

def run_processor(processor, name, original_predictions, ground_truth, eval_threshold=100.0):
    """运行单个处理器并记录结果"""
    print(f"\n{'='*50}")
    print(f"运行 {name}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    # 运行处理器
    processed_predictions = processor.process_sequence(original_predictions)
    
    # 评估结果 - 使用统一的评估阈值
    from eval_predictions import calculate_metrics
    
    # 计算原始预测的指标
    original_metrics = calculate_metrics(original_predictions, ground_truth, eval_threshold)
    
    # 计算处理后预测的指标
    processed_metrics = calculate_metrics(processed_predictions, ground_truth, eval_threshold)
    
    # 计算改善程度
    improvement = {
        'precision_improvement': processed_metrics['precision'] - original_metrics['precision'],
        'recall_improvement': processed_metrics['recall'] - original_metrics['recall'],
        'f1_improvement': processed_metrics['f1'] - original_metrics['f1'],
        'mse_improvement': original_metrics['mse'] - processed_metrics['mse'],
        'original_metrics': original_metrics,
        'processed_metrics': processed_metrics
    }
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 统计检测点数量
    original_count = sum(len(pred['coords']) for pred in original_predictions.values())
    processed_count = sum(len(pred['coords']) for pred in processed_predictions.values())
    
    return {
        'name': name,
        'processing_time': processing_time,
        'original_count': original_count,
        'processed_count': processed_count,
        'detection_change': processed_count - original_count,
        'improvement': improvement,
        'processed_predictions': processed_predictions
    }

def print_comparison_table(results):
    """打印比较表格"""
    print(f"\n{'='*80}")
    print("序列后处理器效果比较")
    print(f"{'='*80}")
    
    # 表头
    print(f"{'处理器':<15} {'处理时间(s)':<12} {'检测点变化':<12} {'Precision改善':<15} {'Recall改善':<12} {'F1改善':<10} {'MSE改善':<10}")
    print("-" * 80)
    
    # 数据行
    for result in results:
        name = result['name']
        time_taken = result['processing_time']
        detection_change = result['detection_change']
        precision_imp = result['improvement']['precision_improvement']
        recall_imp = result['improvement']['recall_improvement']
        f1_imp = result['improvement']['f1_improvement']
        mse_imp = result['improvement']['mse_improvement']
        
        print(f"{name:<15} {time_taken:<12.2f} {detection_change:<12} {precision_imp:<15.4f} {recall_imp:<12.4f} {f1_imp:<10.4f} {mse_imp:<10.2f}")
    
    print("-" * 80)

def print_detailed_results(results):
    """打印详细结果"""
    print(f"\n{'='*80}")
    print("详细结果对比")
    print(f"{'='*80}")
    
    for result in results:
        print(f"\n{result['name']}:")
        print(f"  处理时间: {result['processing_time']:.2f} 秒")
        print(f"  检测点变化: {result['detection_change']} ({result['original_count']} -> {result['processed_count']})")
        print(f"  原始指标:")
        print(f"    Precision: {result['improvement']['original_metrics']['precision']:.4f}")
        print(f"    Recall: {result['improvement']['original_metrics']['recall']:.4f}")
        print(f"    F1 Score: {result['improvement']['original_metrics']['f1']:.4f}")
        print(f"    MSE: {result['improvement']['original_metrics']['mse']:.2f}")
        print(f"  处理后指标:")
        print(f"    Precision: {result['improvement']['processed_metrics']['precision']:.4f}")
        print(f"    Recall: {result['improvement']['processed_metrics']['recall']:.4f}")
        print(f"    F1 Score: {result['improvement']['processed_metrics']['f1']:.4f}")
        print(f"    MSE: {result['improvement']['processed_metrics']['mse']:.2f}")
        print(f"  改善:")
        print(f"    Precision: {result['improvement']['precision_improvement']:+.4f}")
        print(f"    Recall: {result['improvement']['recall_improvement']:+.4f}")
        print(f"    F1 Score: {result['improvement']['f1_improvement']:+.4f}")
        print(f"    MSE: {result['improvement']['mse_improvement']:+.2f}")

def save_results(results):
    """保存所有结果"""
    for result in results:
        output_path = f"results/WTNet/{result['name'].lower().replace(' ', '_')}_predictions.json"
        with open(output_path, 'w') as f:
            json.dump(result['processed_predictions'], f, indent=2)
        print(f"{result['name']} 结果已保存到: {output_path}")

def find_best_processor(results):
    """找出最佳处理器"""
    print(f"\n{'='*50}")
    print("最佳处理器分析")
    print(f"{'='*50}")
    
    # 按F1改善排序
    sorted_by_f1 = sorted(results, key=lambda x: x['improvement']['f1_improvement'], reverse=True)
    best_f1 = sorted_by_f1[0]
    
    # 按MSE改善排序
    sorted_by_mse = sorted(results, key=lambda x: x['improvement']['mse_improvement'], reverse=True)
    best_mse = sorted_by_mse[0]
    
    # 按处理时间排序
    sorted_by_time = sorted(results, key=lambda x: x['processing_time'])
    fastest = sorted_by_time[0]
    
    print(f"F1改善最佳: {best_f1['name']} (+{best_f1['improvement']['f1_improvement']:.4f})")
    print(f"MSE改善最佳: {best_mse['name']} (+{best_mse['improvement']['mse_improvement']:.2f})")
    print(f"处理最快: {fastest['name']} ({fastest['processing_time']:.2f}s)")
    
    # 综合评分（F1改善 + MSE改善/1000 - 处理时间/10）
    for result in results:
        score = (result['improvement']['f1_improvement'] + 
                result['improvement']['mse_improvement']/1000 - 
                result['processing_time']/10)
        result['composite_score'] = score
    
    best_overall = max(results, key=lambda x: x['composite_score'])
    print(f"综合最佳: {best_overall['name']} (评分: {best_overall['composite_score']:.4f})")

def main():
    """主函数"""
    # 加载数据
    original_predictions, ground_truth = load_data()
    
    # 统一的评估阈值
    eval_threshold = 1000.0
    
    # 创建处理器
    processors = [
        (SimpleSequenceProcessor(distance_threshold=100.0), "简单处理器"),
        (ImprovedSequenceProcessor(
            distance_threshold=100.0,
            temporal_window=3,
            confidence_threshold=0.1,
            min_track_length=1,
            max_frame_gap=2
        ), "改进处理器"),
        (BalancedSequenceProcessor(
            base_distance_threshold=80.0,
            temporal_window=3,
            confidence_threshold=0.05,
            min_track_length=2,
            max_frame_gap=3,
            adaptive_threshold=True
        ), "平衡处理器")
    ]
    
    # 运行所有处理器
    results = []
    for processor, name in processors:
        result = run_processor(processor, name, original_predictions, ground_truth, eval_threshold)
        results.append(result)
    
    # 打印比较结果
    print_comparison_table(results)
    print_detailed_results(results)
    find_best_processor(results)
    
    # 保存结果
    save_results(results)
    
    print(f"\n{'='*50}")
    print("比较完成！")
    print(f"使用统一的评估阈值: {eval_threshold}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main() 