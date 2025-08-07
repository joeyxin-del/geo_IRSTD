# 自适应角度距离处理器使用指南

## 概述

自适应角度距离处理器是一个灵活的后处理工具，可以通过聚类自动发现主导角度和主导步长，用于改善目标检测和跟踪的结果。该处理器支持多种配置模式，可以根据具体需求选择使用角度聚类、步长聚类或两者结合。

## 主要特性

- **可选的角度聚类**: 自动发现主导运动角度
- **可选的步长聚类**: 自动发现主导运动步长
- **灵活的权重配置**: 可调整角度和步长在综合评分中的权重
- **多种处理模式**: 支持单独使用角度、单独使用步长或两者结合
- **可视化支持**: 提供角度聚类过程的可视化
- **效果评估**: 自动评估处理前后的改善效果

## 处理模式

### 1. 同时使用角度和步长聚类（默认模式）
```bash
python processor/adaptive_angle_distance_processor.py \
    --pred_path results/spotgeov2-IRSTD/WTNet/predictions_8807.json \
    --gt_path datasets/spotgeov2-IRSTD/test_anno.json \
    --output_path results/adaptive_both_clustering.json \
    --use_angle_clustering \
    --use_step_clustering \
    --angle_weight 0.65 \
    --step_weight 0.35
```

### 2. 仅使用角度聚类
```bash
python processor/adaptive_angle_distance_processor.py \
    --pred_path results/spotgeov2-IRSTD/WTNet/predictions_8807.json \
    --gt_path datasets/spotgeov2-IRSTD/test_anno.json \
    --output_path results/adaptive_angle_only.json \
    --use_angle_clustering \
    --no_step_clustering
```

### 3. 仅使用步长聚类
```bash
python processor/adaptive_angle_distance_processor.py \
    --pred_path results/spotgeov2-IRSTD/WTNet/predictions_8807.json \
    --gt_path datasets/spotgeov2-IRSTD/test_anno.json \
    --output_path results/adaptive_step_only.json \
    --no_angle_clustering \
    --use_step_clustering
```

### 4. 不使用聚类（仅基础处理）
```bash
python processor/adaptive_angle_distance_processor.py \
    --pred_path results/spotgeov2-IRSTD/WTNet/predictions_8807.json \
    --gt_path datasets/spotgeov2-IRSTD/test_anno.json \
    --output_path results/adaptive_no_clustering.json \
    --no_angle_clustering \
    --no_step_clustering
```

## 命令行参数

### 基础参数
- `--pred_path`: 预测结果文件路径
- `--gt_path`: 真实标注文件路径
- `--output_path`: 输出文件路径

### 聚类控制参数
- `--use_angle_clustering`: 启用角度聚类
- `--no_angle_clustering`: 禁用角度聚类
- `--use_step_clustering`: 启用步长聚类
- `--no_step_clustering`: 禁用步长聚类

### 权重参数
- `--angle_weight`: 角度在综合评分中的权重（默认0.65）
- `--step_weight`: 步长在综合评分中的权重（默认0.35）

### 聚类参数
- `--min_cluster_size`: 最小聚类大小（默认1）
- `--angle_cluster_eps`: 角度聚类半径（度，默认12.0）
- `--step_cluster_eps`: 步长聚类半径（比例，默认0.35）
- `--confidence_threshold`: 主导模式置信度阈值（默认0.35）
- `--angle_tolerance`: 角度容差（度，默认12.0）
- `--step_tolerance`: 步长容差（比例，默认0.22）
- `--dominant_ratio_threshold`: 主导模式占比阈值（默认0.7）
- `--secondary_ratio_threshold`: 次主导模式占比阈值（默认0.15）
- `--max_dominant_patterns`: 最大主导模式数量（默认2）

### 其他参数
- `--base_distance_threshold`: 基础距离阈值（默认1000）
- `--point_distance_threshold`: 重合点过滤阈值（默认7.0）
- `--no_visualization`: 不保存可视化结果

## 使用示例

### 快速测试不同配置

运行示例脚本来测试所有配置：

```bash
python processor/run_adaptive_processor_examples.py
```

这将运行以下6种配置：
1. 同时使用角度和步长聚类（默认）
2. 仅使用角度聚类
3. 仅使用步长聚类
4. 不使用聚类
5. 角度权重更高（0.8:0.2）
6. 步长权重更高（0.3:0.7）

### 比较不同配置的结果

```bash
python processor/compare_configurations.py
```

这将生成一个比较报告，显示不同配置的处理结果统计。

## 配置建议

### 针对不同场景的推荐配置

1. **运动轨迹明显的数据集**（如车辆跟踪）
   - 推荐：同时使用角度和步长聚类
   - 权重：角度0.7，步长0.3

2. **运动方向相对固定的数据集**（如行人跟踪）
   - 推荐：仅使用角度聚类
   - 原因：步长变化较大，角度相对稳定

3. **运动速度相对稳定的数据集**（如匀速运动）
   - 推荐：仅使用步长聚类
   - 原因：角度变化较大，步长相对稳定

4. **复杂运动模式的数据集**
   - 推荐：同时使用角度和步长聚类
   - 权重：角度0.5，步长0.5（平衡配置）

### 参数调优建议

1. **角度容差**（`--angle_tolerance`）
   - 小值（5-10度）：适用于运动方向非常一致的数据
   - 大值（15-20度）：适用于运动方向变化较大的数据

2. **步长容差**（`--step_tolerance`）
   - 小值（0.1-0.2）：适用于运动速度非常稳定的数据
   - 大值（0.3-0.5）：适用于运动速度变化较大的数据

3. **置信度阈值**（`--confidence_threshold`）
   - 低值（0.2-0.4）：适用于噪声较多的数据
   - 高值（0.6-0.8）：适用于噪声较少的数据

## 输出说明

### 处理日志
处理器会输出详细的处理日志，包括：
- 每个序列的处理状态
- 主导角度和步长的发现结果
- 置信度信息
- 处理模式配置

### 评估结果
如果提供了真实标注，处理器会输出：
- Precision、Recall、F1 Score的改善情况
- MSE的改善情况
- TP、FP、FN的变化情况

### 可视化结果
如果启用了可视化，会生成角度聚类过程的可视化图像，保存在`clustering_visualizations/`目录下。

## 注意事项

1. **权重设置**: 角度权重和步长权重的总和应该等于1.0，如果不等于会自动归一化。

2. **聚类参数**: 聚类参数需要根据具体数据集进行调整，建议从默认值开始，逐步调优。

3. **性能考虑**: 禁用可视化可以显著提高处理速度，特别是在处理大量数据时。

4. **内存使用**: 处理大型数据集时，注意监控内存使用情况。

## 故障排除

### 常见问题

1. **"无法确定主导模式"**
   - 检查聚类参数是否合适
   - 降低`min_cluster_size`或`confidence_threshold`
   - 增加`angle_cluster_eps`或`step_cluster_eps`

2. **处理效果不佳**
   - 尝试不同的权重配置
   - 调整容差参数
   - 考虑只使用角度或只使用步长聚类

3. **处理速度慢**
   - 使用`--no_visualization`禁用可视化
   - 减少`max_dominant_patterns`
   - 调整聚类参数减少计算量

### 调试建议

1. 先在小数据集上测试配置
2. 使用可视化功能分析聚类结果
3. 逐步调整参数，观察效果变化
4. 记录最佳配置参数以便复现 