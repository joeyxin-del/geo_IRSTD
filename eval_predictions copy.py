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

def calculate_metrics(predictions: Dict, ground_truth: List[Dict], distance_threshold: float = 5.0, exclude_empty_sequences: bool = False) -> Dict:
    """
    计算预测结果的评估指标
    Args:
        predictions: 预测结果字典 {"image_name": {"coords": [[x,y], ...], "num_objects": n}}
        ground_truth: 真实标注列表 [{"image_name": "xxx", "num_objects": n, "object_coords": [[x,y], ...]}]
        distance_threshold: 匹配的距离阈值
        exclude_empty_sequences: 是否排除真实标注为空的序列
    Returns:
        metrics: 包含各项评估指标的字典
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_mse = 0
    total_matched_points = 0
    tau_square = distance_threshold * distance_threshold  # 阈值的平方作为惩罚项
    
    processed_images = 0
    skipped_images = 0
    empty_gt_images = 0  # 新增：真实标注为空的图片数

    # 存储每张图片的指标
    per_image_metrics = {}
    # 存储每个序列的累积指标
    sequence_metrics = {}

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
        # 获取序列ID
        seq_id, _ = convert_filename_to_seq_frame(img_name)
        
        # 直接使用图片名作为键
        if img_name not in organized_gt:
            print(f"警告：找不到图片 {img_name} 的真实标注")
            skipped_images += 1
            continue

        pred_points = pred_info['coords']
        gt_points = organized_gt[img_name]['object_coords']

        # 检查是否为真实标注为空的图片
        is_empty_gt = len(gt_points) == 0
        
        if is_empty_gt:
            empty_gt_images += 1
            # 对于真实标注为空的图片，所有预测点都是假阳性
            tp = 0
            fp = len(pred_points)
            fn = 0
            
            # 计算MSE：所有预测点都加上惩罚项
            current_mse = fp * tau_square if fp > 0 else 0
            
            # 对于空GT的图片，precision=0, recall=1, f1=0
            img_precision = 0 if fp > 0 else 1  # 如果没有预测点，precision=1
            img_recall = 1  # 空GT的recall总是1
            img_f1 = 0 if fp > 0 else 1  # 如果有预测点，f1=0；否则f1=1
            img_mse = current_mse / max(1, len(pred_points)) if len(pred_points) > 0 else 0
            
        else:
            # 匹配点对
            matches, unmatched_preds, unmatched_gts = match_points(
                pred_points, gt_points, distance_threshold) #distance_threshold

            # 计算当前图片的指标
            tp = len(matches)
            fp = len(unmatched_preds)
            fn = len(unmatched_gts)
            
            # 计算当前图片的MSE
            current_mse = 0
            # 对匹配点计算实际距离的平方
            for pred_idx, gt_idx in matches:
                distance = calculate_distance(pred_points[pred_idx], gt_points[gt_idx])
                if distance <= distance_threshold:
                    current_mse += 0  # 距离小于阈值时为0
                else:
                    current_mse += distance * distance
            
            # 对未匹配点添加惩罚项
            current_mse += (fp + fn) * tau_square  # 每个未匹配点都加上τ²
            
            # 计算当前图片的指标
            img_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            img_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            img_f1 = 2 * (img_precision * img_recall) / (img_precision + img_recall) if (img_precision + img_recall) > 0 else 0
            img_mse = current_mse / max(len(gt_points), len(pred_points)) if max(len(gt_points), len(pred_points)) > 0 else 0
        
        # 存储每张图片的指标
        per_image_metrics[img_name] = {
            'precision': img_precision,
            'recall': img_recall,
            'f1': img_f1,
            'mse': img_mse,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'matched_points': len(matches) if not is_empty_gt else 0,
            'is_empty_gt': is_empty_gt  # 新增：标记是否为空GT
        }
        
        # 更新序列指标
        if seq_id not in sequence_metrics:
            sequence_metrics[seq_id] = {
                'tp': 0, 'fp': 0, 'fn': 0, 'total_mse': 0,
                'total_points': 0, 'processed_images': 0, 'empty_gt_frames': 0  # 新增：空GT帧数
            }
        sequence_metrics[seq_id]['tp'] += tp
        sequence_metrics[seq_id]['fp'] += fp
        sequence_metrics[seq_id]['fn'] += fn
        sequence_metrics[seq_id]['total_mse'] += current_mse
        sequence_metrics[seq_id]['total_points'] += max(len(gt_points), len(pred_points))
        sequence_metrics[seq_id]['processed_images'] += 1
        if is_empty_gt:
            sequence_metrics[seq_id]['empty_gt_frames'] += 1

        # 更新总体指标
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_mse += current_mse
        total_matched_points += max(len(gt_points), len(pred_points))
        processed_images += 1

    # 计算每个序列的最终指标
    for seq_id in sequence_metrics:
        seq = sequence_metrics[seq_id]
        
        # 检查序列是否所有帧都是空GT
        is_all_empty_gt = seq['empty_gt_frames'] == seq['processed_images']
        
        if is_all_empty_gt:
            # 如果序列所有帧都是空GT，则F1分数为0（如果有预测点）或1（如果没有预测点）
            seq['f1'] = 0 if seq['fp'] > 0 else 1
            seq['precision'] = 0 if seq['fp'] > 0 else 1
            seq['recall'] = 1  # 空GT的recall总是1
        else:
            # 正常计算序列指标
            seq_precision = seq['tp'] / (seq['tp'] + seq['fp']) if (seq['tp'] + seq['fp']) > 0 else 0
            seq_recall = seq['tp'] / (seq['tp'] + seq['fn']) if (seq['tp'] + seq['fn']) > 0 else 0
            seq['f1'] = 2 * (seq_precision * seq_recall) / (seq_precision + seq_recall) if (seq_precision + seq_recall) > 0 else 0
            seq['precision'] = seq_precision
            seq['recall'] = seq_recall
        
        seq['mse'] = seq['total_mse'] / seq['total_points'] if seq['total_points'] > 0 else 0
        seq['is_all_empty_gt'] = is_all_empty_gt  # 新增：标记是否所有帧都是空GT

    # 计算总体指标
    if exclude_empty_sequences:
        # 排除空序列的指标计算
        filtered_tp = 0
        filtered_fp = 0
        filtered_fn = 0
        filtered_mse = 0
        filtered_matched_points = 0
        filtered_processed_images = 0
        
        for seq_id, seq_metrics in sequence_metrics.items():
            if not seq_metrics['is_all_empty_gt']:
                filtered_tp += seq_metrics['tp']
                filtered_fp += seq_metrics['fp']
                filtered_fn += seq_metrics['fn']
                filtered_mse += seq_metrics['total_mse']
                filtered_matched_points += seq_metrics['total_points']
                filtered_processed_images += seq_metrics['processed_images']
        
        precision = filtered_tp / (filtered_tp + filtered_fp) if (filtered_tp + filtered_fp) > 0 else 0
        recall = filtered_tp / (filtered_tp + filtered_fn) if (filtered_tp + filtered_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mse = filtered_mse / filtered_matched_points if filtered_matched_points > 0 else 0
        rmse = np.sqrt(mse) if mse >= 0 else 0
    else:
        # 包含所有序列的指标计算
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mse = total_mse / total_matched_points if total_matched_points > 0 else 0
        rmse = np.sqrt(mse) if mse >= 0 else 0

    # 找出最佳性能的图片和序列
    best_f1_image = max(per_image_metrics.items(), key=lambda x: x[1]['f1'])
    best_mse_image = min(per_image_metrics.items(), key=lambda x: x[1]['mse'])
    best_f1_sequence = max(sequence_metrics.items(), key=lambda x: x[1]['f1'])
    best_mse_sequence = min(sequence_metrics.items(), key=lambda x: x[1]['mse'])

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mse': mse,
        'rmse': rmse,
        'total_matched_points': total_matched_points,
        'processed_images': processed_images,
        'skipped_images': skipped_images,
        'empty_gt_images': empty_gt_images,  # 新增：空GT图片数
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'per_image_metrics': per_image_metrics,
        'sequence_metrics': sequence_metrics,
        'best_performance': {
            'best_f1_image': {
                'image_name': best_f1_image[0],
                'f1_score': best_f1_image[1]['f1']
            },
            'best_mse_image': {
                'image_name': best_mse_image[0],
                'mse': best_mse_image[1]['mse']
            },
            'best_f1_sequence': {
                'sequence_id': best_f1_sequence[0],
                'f1_score': best_f1_sequence[1]['f1']
            },
            'best_mse_sequence': {
                'sequence_id': best_mse_sequence[0],
                'mse': best_mse_sequence[1]['mse']
            }
        }
    }

def main():
    # 加载预测结果和真实标注
    # pred_path = 'results/WTNet/predictions.json'
    # pred_path = 'results/WTNet/balanced_processed_predictions.json'
    # pred_path = 'results/WTNet/improved_kmeans_predictions.json'
    # pred_path = 'results/WTNet/aggressive_balanced_processed_predictions.json'
    # pred_path = 'results/WTNet/slope_based_processed_predictions.json'
    # pred_path = 'results/spotgeov2/WTNet/sequence_slope_processed_predictions.json'
    # pred_path = 'results/WTNet/improved_slope_processed_predictions.json'
    # pred_path = 'results/spotgeov2/WTNet/outlier_filtered_predictions.json'
    # pred_path = 'results/spotgeov2/WTNet/improved_slope_processed_predictions.json'
    # pred_path = 'results/spotgeov2/WTNet/angle_processed_predictions.json'
    # pred_path = 'results/spotgeov2/WTNet/8801angle_processed_predictions.json'
    pred_path = 'results/spotgeov2/WTNet/angle_distance_processed_predictions_v3.json'
    gt_path = 'datasets/spotgeov2-IRSTD/test_anno.json'
    
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

    # 计算评估指标（包含所有序列）
    print("\n正在计算评估指标（包含所有序列）...")
    metrics = calculate_metrics(predictions, ground_truth, distance_threshold=500, exclude_empty_sequences=False)

    # 打印评估结果
    print("\n=== 总体评估结果（包含所有序列） ===")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    if metrics['mse'] >= 0:
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
    else:
        print("MSE: 无有效匹配点")
        print("RMSE: 无有效匹配点")
    
    print(f"\n处理的图片数: {metrics['processed_images']}")
    print(f"跳过的图片数: {metrics['skipped_images']}")
    print(f"空GT图片数: {metrics['empty_gt_images']}")
    print(f"总匹配点对数: {metrics['total_matched_points']}")
    print(f"True Positives: {metrics['total_tp']}")
    print(f"False Positives: {metrics['total_fp']}")
    print(f"False Negatives: {metrics['total_fn']}")
    
    # 计算空GT图片的占比
    if metrics['processed_images'] > 0:
        empty_gt_percentage = (metrics['empty_gt_images'] / metrics['processed_images']) * 100
        print(f"空GT图片占比: {empty_gt_percentage:.2f}%")

    # 计算排除空序列后的评估指标
    print("\n正在计算评估指标（排除空序列）...")
    metrics_filtered = calculate_metrics(predictions, ground_truth, distance_threshold=500, exclude_empty_sequences=True)

    # 打印排除空序列后的评估结果
    print("\n=== 总体评估结果（排除空序列） ===")
    print(f"Precision: {metrics_filtered['precision']:.4f}")
    print(f"Recall: {metrics_filtered['recall']:.4f}")
    print(f"F1 Score: {metrics_filtered['f1']:.4f}")
    if metrics_filtered['mse'] >= 0:
        print(f"MSE: {metrics_filtered['mse']:.4f}")
        print(f"RMSE: {metrics_filtered['rmse']:.4f}")
    else:
        print("MSE: 无有效匹配点")
        print("RMSE: 无有效匹配点")

    # 计算指标改善情况
    print("\n=== 指标改善情况（排除空序列后） ===")
    precision_improvement = metrics_filtered['precision'] - metrics['precision']
    recall_improvement = metrics_filtered['recall'] - metrics['recall']
    f1_improvement = metrics_filtered['f1'] - metrics['f1']
    
    print(f"Precision改善: {precision_improvement:+.4f} ({precision_improvement/metrics['precision']*100:+.2f}%)")
    print(f"Recall改善: {recall_improvement:+.4f} ({recall_improvement/metrics['recall']*100:+.2f}%)")
    print(f"F1 Score改善: {f1_improvement:+.4f} ({f1_improvement/metrics['f1']*100:+.2f}%)")

    # 打印最佳性能结果
    print("\n=== 最佳性能 ===")
    print("\n单张图片最佳性能:")
    print(f"F1分数最高的图片: {metrics['best_performance']['best_f1_image']['image_name']}")
    print(f"最高F1分数: {metrics['best_performance']['best_f1_image']['f1_score']:.4f}")
    
    if metrics['best_performance']['best_mse_image']['mse'] >= 0:
        print(f"MSE最低的图片: {metrics['best_performance']['best_mse_image']['image_name']}")
        print(f"最低MSE: {metrics['best_performance']['best_mse_image']['mse']:.4f}")
    else:
        print("MSE: 所有图片都没有有效的匹配点")

    print("\n序列级别最佳性能:")
    print(f"F1分数最高的序列: {metrics['best_performance']['best_f1_sequence']['sequence_id']}")
    print(f"最高F1分数: {metrics['best_performance']['best_f1_sequence']['f1_score']:.4f}")
    
    if metrics['best_performance']['best_mse_sequence']['mse'] >= 0:
        print(f"MSE最低的序列: {metrics['best_performance']['best_mse_sequence']['sequence_id']}")
        print(f"最低MSE: {metrics['best_performance']['best_mse_sequence']['mse']:.4f}")
    else:
        print("MSE: 所有序列都没有有效的匹配点")

    # 列出F1分数小于0.5的序列号（排除真实标注为空的序列）
    print("\n=== F1分数小于0.5的序列（排除空GT序列） ===")
    low_f1_sequences = []
    empty_gt_sequences = []
    
    for seq_id, seq_metrics in metrics['sequence_metrics'].items():
        if seq_metrics['is_all_empty_gt']:
            # 统计空GT序列
            empty_gt_sequences.append({
                'sequence_id': seq_id,
                'f1_score': seq_metrics['f1'],
                'mse': seq_metrics['mse'],
                'fp': seq_metrics['fp'],
                'empty_gt_frames': seq_metrics['empty_gt_frames']
            })
        elif seq_metrics['f1'] < 0.9:
            # 只包含有真实标注点且F1分数小于0.5的序列
            low_f1_sequences.append({
                'sequence_id': seq_id,
                'f1_score': seq_metrics['f1'],
                'mse': seq_metrics['mse'],
                'precision': seq_metrics.get('precision', 0),
                'recall': seq_metrics.get('recall', 0),
                'tp': seq_metrics['tp'],
                'fp': seq_metrics['fp'],
                'fn': seq_metrics['fn'],
                'processed_images': seq_metrics['processed_images']
            })
    
    # 打印空GT序列统计
    print(f"\n=== 真实标注为空的序列统计 ===")
    print(f"空GT序列数: {len(empty_gt_sequences)}")
    if empty_gt_sequences:
        empty_gt_with_predictions = [s for s in empty_gt_sequences if s['fp'] > 0]
        print(f"空GT但有预测的序列数: {len(empty_gt_with_predictions)}")
        print(f"空GT且无预测的序列数: {len(empty_gt_sequences) - len(empty_gt_with_predictions)}")
    
    if low_f1_sequences:
        print(f"\n找到 {len(low_f1_sequences)} 个有真实标注点且F1分数小于0.9的序列:")
        # 按F1分数从低到高排序
        low_f1_sequences.sort(key=lambda x: x['f1_score'])
        
        # 保存低F1序列到JSON文件
        low_f1_save_path = './results/spotgeov2/WTNet/low_f1_sequences.json'
        low_f1_data = {
            'total_sequences_below_0_5': len(low_f1_sequences),
            'total_sequences': len(metrics['sequence_metrics']),
            'total_sequences_with_gt': len(metrics['sequence_metrics']) - len(empty_gt_sequences),
            'empty_gt_sequences': len(empty_gt_sequences),
            'percentage_below_0_5': (len(low_f1_sequences) / (len(metrics['sequence_metrics']) - len(empty_gt_sequences))) * 100 if (len(metrics['sequence_metrics']) - len(empty_gt_sequences)) > 0 else 0,
            'sequences': low_f1_sequences,
            'statistics': {
                'f1_0_0': len([s for s in low_f1_sequences if s['f1_score'] == 0.0]),
                'f1_0_0_to_0_1': len([s for s in low_f1_sequences if 0.0 < s['f1_score'] <= 0.1]),
                'f1_0_1_to_0_2': len([s for s in low_f1_sequences if 0.1 < s['f1_score'] <= 0.2]),
                'f1_0_2_to_0_3': len([s for s in low_f1_sequences if 0.2 < s['f1_score'] <= 0.3]),
                'f1_0_3_to_0_4': len([s for s in low_f1_sequences if 0.3 < s['f1_score'] <= 0.4]),
                'f1_0_4_to_0_5': len([s for s in low_f1_sequences if 0.4 < s['f1_score'] < 0.5]),
                'f1_0_5_to_0_6': len([s for s in low_f1_sequences if 0.5 < s['f1_score'] <= 0.6]),
                'f1_0_6_to_0_7': len([s for s in low_f1_sequences if 0.6 < s['f1_score'] <= 0.7]),
                'f1_0_7_to_0_8': len([s for s in low_f1_sequences if 0.7 < s['f1_score'] <= 0.8]),
                'f1_0_8_to_0_9': len([s for s in low_f1_sequences if 0.8 < s['f1_score'] <= 0.9]),
                'f1_0_9_to_1_0': len([s for s in low_f1_sequences if 0.9 < s['f1_score'] <= 1.0])
            }
        }
        
        with open(low_f1_save_path, 'w') as f:
            json.dump(low_f1_data, f, indent=2)
        print(f"\n低F1序列信息已保存到: {low_f1_save_path}")
        print(f"总序列数: {len(metrics['sequence_metrics'])}")
        print(f"有真实标注点的序列数: {len(metrics['sequence_metrics']) - len(empty_gt_sequences)}")
        print(f"F1 < 0.5的序列数: {len(low_f1_sequences)}")
        print(f"占比: {(len(low_f1_sequences) / (len(metrics['sequence_metrics']) - len(empty_gt_sequences))) * 100:.2f}%")
        
        # 打印统计信息
        print("\n=== F1分数分布统计（排除空GT序列） ===")
        stats = low_f1_data['statistics']
        print(f"F1 = 0.0: {stats['f1_0_0']} 个序列")
        print(f"F1 0.0-0.1: {stats['f1_0_0_to_0_1']} 个序列")
        print(f"F1 0.1-0.2: {stats['f1_0_1_to_0_2']} 个序列")
        print(f"F1 0.2-0.3: {stats['f1_0_2_to_0_3']} 个序列")
        print(f"F1 0.3-0.4: {stats['f1_0_3_to_0_4']} 个序列")
        print(f"F1 0.4-0.5: {stats['f1_0_4_to_0_5']} 个序列")
        print(f"F1 0.5-0.6: {stats['f1_0_5_to_0_6']} 个序列")
        print(f"F1 0.6-0.7: {stats['f1_0_6_to_0_7']} 个序列")
        print(f"F1 0.7-0.8: {stats['f1_0_7_to_0_8']} 个序列")
        print(f"F1 0.8-0.9: {stats['f1_0_8_to_0_9']} 个序列")
        print(f"F1 0.9-1.0: {stats['f1_0_9_to_1_0']} 个序列")
        
    else:
        print("没有找到有真实标注点且F1分数小于0.5的序列")

    # 保存评估结果
    results_save_path = './results/spotgeov2/WTNet/angle_distance_processed_evaluation_results_v3.json'
    with open(results_save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n评估结果已保存到: {results_save_path}")
    
    # 保存排除空序列后的评估结果
    results_filtered_save_path = './results/spotgeov2/WTNet/angle_distance_processed_evaluation_results_v3_filtered.json'
    with open(results_filtered_save_path, 'w') as f:
        json.dump(metrics_filtered, f, indent=2)
    print(f"排除空序列后的评估结果已保存到: {results_filtered_save_path}")

if __name__ == '__main__':
    main() 