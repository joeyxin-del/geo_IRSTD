import json
import os
import sys
import numpy as np
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

# 导入validation模块中的函数
import validation

def flat_to_hierarchical(labels):
    """ Transforms a flat array of json-objects to a hierarchical python dict, indexed by
        sequence number and frame id. """
    seqs = dict()
    for label in labels:
        seq_id = label['sequence_id']
        frame_id = label['frame']
        coords = label['object_coords']
        
        if seq_id not in seqs.keys():
            seqs[seq_id] = defaultdict(dict)
        seqs[seq_id][frame_id] = np.array(coords)
    
    return seqs

def convert_numpy_types(obj):
    """将numpy类型转换为Python原生类型，以便JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def main():
    # 文件路径
    pred_path = 'results/spotgeov2/WTNet/angle_distance_processed_predictions_v3.json'
    # pred_path = 'results/spotgeov2-IRSTD/WTNet/predictions_8807.json'
    # pred_path = 'results/0808/improved_slope_processed_predictions.json'
    # pred_path = 'results/0808/simple_processed_predictions.json'
    pred_path = 'results/0808/aggressive_balanced_processed_predictions.json'
    gt_path = 'datasets/spotgeov2-IRSTD/test_anno.json'
    
    # 检查文件是否存在
    if not os.path.exists(pred_path):
        print(f"错误：预测结果文件不存在: {pred_path}")
        return
    
    if not os.path.exists(gt_path):
        print(f"错误：真实标注文件不存在: {gt_path}")
        return
    
    print("正在加载预测结果和真实标注...")
    
    # 加载预测结果
    try:
        with open(pred_path, 'r') as f:
            predictions_data = json.load(f)
            
    except json.JSONDecodeError as e:
        print(f"错误：解析预测结果文件失败: {str(e)}")
        return
    
    # 加载真实标注
    try:
        with open(gt_path, 'r') as f:
            ground_truth_data = json.load(f)
            if not isinstance(ground_truth_data, list):
                print("警告：真实标注文件格式不正确,应该是列表格式")
                ground_truth_data = [ground_truth_data]
    except json.JSONDecodeError as e:
        print(f"错误：解析真实标注文件失败: {str(e)}")
        return

    print(f"\n总共加载了 {len(predictions_data)} 张图片的预测结果")
    print(f"总共加载了 {len(ground_truth_data)} 帧的真实标注")

    # 将预测数据转换为validation.py期望的格式
    print("\n正在转换数据格式...")
    
    # 将预测数据转换为validation.py期望的格式
    predictions_formatted = []
    for img_name, pred_info in predictions_data.items():
        # 从图片名提取序列号和帧号
        parts = img_name.split('_')
        if len(parts) >= 2:
            try:
                sequence_id = int(parts[0])
                frame = int(parts[1])
                coords = pred_info['coords']
                predictions_formatted.append({
                    'sequence_id': sequence_id,
                    'frame': frame,
                    'num_objects': len(coords),
                    'object_coords': coords
                })
            except ValueError:
                print(f"警告：无法解析图片名 {img_name}")
                continue

    print(f"成功转换了 {len(predictions_formatted)} 个预测结果")

    # 使用validation.py中的函数计算指标
    print("\n正在使用validation.py中的函数计算评估指标...")
    
    # 转换为层次结构
    predictions_h = flat_to_hierarchical(predictions_formatted)
    labels_h = flat_to_hierarchical(ground_truth_data)
    
    # 直接调用score_sequences函数获取precision, recall, F1, mse
    precision, recall, F1, mse = validation.score_sequences(predictions_h, labels_h, tau=10, eps=10)
    
    # 为了获取TP、FN、FP，我们需要手动计算
    # 检查每个序列的匹配情况
    identifiers = set(predictions_h.keys()) - set([])  # 没有taboolist
    
    total_tp = 0
    total_fn = 0
    total_fp = 0
    
    for seq_id in identifiers:
        if seq_id in predictions_h and seq_id in labels_h:
            # 使用validation.py中的score_sequence函数
            tp, fn, fp, seq_mse = validation.score_sequence(predictions_h[seq_id], labels_h[seq_id], tau=1000, eps=10)
            total_tp += tp
            total_fn += fn
            total_fp += fp
    
    # 计算RMSE
    rmse = np.sqrt(mse) if mse > 0 else 0
    
    # 直接从数据中统计实际的总实例数
    total_pred_objects = sum(len(pred_info['coords']) for pred_info in predictions_data.values())
    total_gt_objects = sum(len(anno['object_coords']) for anno in ground_truth_data)
    
    print(f"\n数据统计:")
    print(f"预测总对象数: {total_pred_objects}")
    print(f"真实总对象数: {total_gt_objects}")
    
    total_instances = total_tp + total_fn + total_fp

    # 打印评估结果
    print("\n=== 评估结果 (使用validation.py函数) ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {F1:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    print(f"\n总实例数: {total_instances}")
    print(f"True Positives: {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")

    # 保存评估结果
    results = {
        'precision': precision,
        'recall': recall,
        'f1': F1,
        'mse': mse,
        'rmse': rmse,
        'total_instances': total_instances,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'method': 'validation.py_score_sequences',
        'parameters': {
            'tau': 10,
            'eps': 3
        }
    }
    
    # 转换numpy类型为Python原生类型
    results = convert_numpy_types(results)
    
    results_save_path = './results/spotgeov2/WTNet/validation_results_using_validation_py.json'
    with open(results_save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n评估结果已保存到: {results_save_path}")

if __name__ == '__main__':
    main()