import json
import numpy as np
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment

def calculate_distance(point1: List[float], point2: List[float]) -> float:
    """计算两点之间的欧氏距离"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def match_points(pred_points: List[List[float]], gt_points: List[List[float]], 
                distance_threshold: float = 5.0) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    使用匈牙利算法匹配预测点和真实点
    Args:
        pred_points: 预测的点坐标列表 [[x1,y1], [x2,y2], ...]
        gt_points: 真实的点坐标列表 [[x1,y1], [x2,y2], ...]
        distance_threshold: 匹配的距离阈值
    Returns:
        matches: 匹配的点对索引
        unmatched_preds: 未匹配的预测点索引
        unmatched_gts: 未匹配的真实点索引
    """
    if len(pred_points) == 0 or len(gt_points) == 0:
        return [], list(range(len(pred_points))), list(range(len(gt_points)))

    # 构建代价矩阵
    cost_matrix = np.zeros((len(pred_points), len(gt_points)))
    for i, pred in enumerate(pred_points):
        for j, gt in enumerate(gt_points):
            distance = calculate_distance(pred, gt)
            cost_matrix[i, j] = distance

    # 使用匈牙利算法进行匹配
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    # 根据距离阈值筛选有效匹配
    matches = []
    unmatched_preds = list(range(len(pred_points)))
    unmatched_gts = list(range(len(gt_points)))

    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if cost_matrix[pred_idx, gt_idx] <= distance_threshold:
            matches.append((pred_idx, gt_idx))
            if pred_idx in unmatched_preds:
                unmatched_preds.remove(pred_idx)
            if gt_idx in unmatched_gts:
                unmatched_gts.remove(gt_idx)

    return matches, unmatched_preds, unmatched_gts

def convert_filename_to_seq_frame(filename: str) -> Tuple[int, int]:
    """从文件名提取序列号和帧号"""
    # 假设文件名格式为 "sequence_frame_test"，如 "1_1_test"
    parts = filename.split('_')
    if len(parts) >= 2:
        try:
            sequence_id = int(parts[0])
            frame = int(parts[1])
            return sequence_id, frame
        except ValueError:
            return None, None
    return None, None

def organize_ground_truth(gt_list: List[Dict]) -> Dict[str, Dict]:
    """将真实标注组织成字典格式，以图片名为键"""
    organized = {}
    for anno in gt_list:
        # 从sequence_id和frame构建图片名
        sequence_id = anno.get('sequence_id', '')
        frame = anno.get('frame', '')
        if sequence_id != '' and frame != '':
            # 构建与预测结果相同格式的图片名 (例如: "1_1_test")
            image_name = f"{sequence_id}_{frame}_test"
            organized[image_name] = {
                'num_objects': anno.get('num_objects', 0),
                'object_coords': anno.get('object_coords', [])
            }
    
    if not organized:
        print("警告：未能成功组织任何标注数据，请检查数据格式")
        
    return organized

def calculate_metrics(predictions: Dict, ground_truth: List[Dict], distance_threshold: float = 1000.0) -> Dict:
    """
    计算预测结果的评估指标
    Args:
        predictions: 预测结果字典 {"image_name": {"coords": [[x,y], ...], "num_objects": n}}
        ground_truth: 真实标注列表 [{"image_name": "xxx", "num_objects": n, "object_coords": [[x,y], ...]}]
        distance_threshold: 匹配的距离阈值
    Returns:
        metrics: 包含各项评估指标的字典
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_mse = 0
    total_matched_points = 0
    
    processed_images = 0
    skipped_images = 0

    # 组织真实标注数据
    organized_gt = organize_ground_truth(ground_truth)

    # 打印一些样本数据以检查格式
    print("\n数据格式示例:")
    if predictions:
        sample_pred_key = list(predictions.keys())[0]
        print(f"\n预测数据示例 ({sample_pred_key}):")
        print(predictions[sample_pred_key])
    
    print("\n真实标注数据示例:")
    if ground_truth:
        print(ground_truth[0])

    for img_name, pred_info in predictions.items():
        # 直接使用图片名作为键
        if img_name not in organized_gt:
            print(f"警告：找不到图片 {img_name} 的真实标注")
            skipped_images += 1
            continue

        pred_points = pred_info['coords']
        gt_points = organized_gt[img_name]['object_coords']

        # 匹配点对
        matches, unmatched_preds, unmatched_gts = match_points(
            pred_points, gt_points, distance_threshold)

        # 计算TP, FP, FN
        tp = len(matches)
        fp = len(unmatched_preds)
        fn = len(unmatched_gts)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # 计算MSE
        for pred_idx, gt_idx in matches:
            distance = calculate_distance(pred_points[pred_idx], gt_points[gt_idx])
            total_mse += distance ** 2
            total_matched_points += 1

        processed_images += 1

    # 计算总体指标
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mse = total_mse / total_matched_points if total_matched_points > 0 else float('inf')
    rmse = np.sqrt(mse)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mse': mse,
        'rmse': rmse,
        'total_matched_points': total_matched_points,
        'processed_images': processed_images,
        'skipped_images': skipped_images,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn
    }

def main():
    # 加载预测结果和真实标注
    pred_path = './results/spotgeov2/TADNet/predictions.json'
    gt_path = './datasets/spotgeov2/test_anno.json'
    
    print("正在加载预测结果和真实标注...")
    with open(pred_path, 'r') as f:
        predictions = json.load(f)
    
    # 直接读取整个JSON文件
    with open(gt_path, 'r') as f:
        try:
            ground_truth = json.load(f)
            if not isinstance(ground_truth, list):
                print("警告：真实标注文件格式不正确,应该是列表格式")
                ground_truth = [ground_truth]  # 如果是单个对象,转换为列表
        except json.JSONDecodeError as e:
            print(f"错误：解析真实标注文件失败: {str(e)}")
            return

    print(f"\n总共加载了 {len(predictions)} 张图片的预测结果")
    print(f"总共加载了 {len(ground_truth)} 帧的真实标注")

    # 计算评估指标
    print("\n正在计算评估指标...")
    metrics = calculate_metrics(predictions, ground_truth, distance_threshold=5.0)

    # 打印评估结果
    print("\n=== 评估结果 ===")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"\n处理的图片数: {metrics['processed_images']}")
    print(f"跳过的图片数: {metrics['skipped_images']}")
    print(f"总匹配点对数: {metrics['total_matched_points']}")
    print(f"True Positives: {metrics['total_tp']}")
    print(f"False Positives: {metrics['total_fp']}")
    print(f"False Negatives: {metrics['total_fn']}")

    # 保存评估结果
    results_save_path = './results/spotgeov2/TADNet/evaluation_results.json'
    with open(results_save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n评估结果已保存到: {results_save_path}")

if __name__ == '__main__':
    main() 