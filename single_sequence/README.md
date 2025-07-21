# 单个序列斜率处理器

这个模块提供了一个专门处理单个序列的斜率处理器，用于轨迹补全和异常点筛选。

## 功能特性

- **轨迹补全**: 基于序列内主导斜率进行轨迹插值和外推
- **异常点筛选**: 基于主导斜率过滤不符合轨迹模式的异常点
- **指标评估**: 计算处理前后的F1、MSE等指标变化
- **可视化对比**: 生成处理前后的可视化对比图

## 文件结构

```
single_sequence/
├── slope_processor.py      # 主要的处理器类
├── run_single_sequence.py  # 使用示例脚本
└── README.md              # 说明文档
```

## 使用方法

### 1. 基本使用

```python
from slope_processor import SingleSequenceSlopeProcessor

# 创建处理器
processor = SingleSequenceSlopeProcessor(
    sequence_id=1,                    # 要处理的序列ID
    base_distance_threshold=500,      # 距离阈值
    slope_tolerance=0.1,             # 斜率容差
    min_slope_count=2,               # 最小斜率出现次数
    point_distance_threshold=200.0   # 重合点过滤阈值
)

# 运行分析
improvement = processor.run_analysis(original_predictions, ground_truth)
```

### 2. 命令行使用

```bash
# 分析序列1
python run_single_sequence.py 1

# 分析序列5
python run_single_sequence.py 5
```

### 3. 自定义参数

```python
# 创建处理器时自定义参数
processor = SingleSequenceSlopeProcessor(
    sequence_id=1,
    base_distance_threshold=300,      # 更严格的距离阈值
    min_track_length=2,              # 最小轨迹长度
    expected_sequence_length=10,     # 期望序列长度
    slope_tolerance=0.05,            # 更严格的斜率容差
    min_slope_count=3,               # 更高的最小斜率出现次数
    point_distance_threshold=100.0   # 更严格的重合点过滤阈值
)
```

## 输出结果

### 1. 控制台输出

处理器会在控制台输出详细的处理信息：

```
开始分析序列 1...
序列 1 包含 10 帧
序列 1 找到 2 个主导斜率
  主导斜率 0.1234: 出现 15 次
  主导斜率 -0.5678: 出现 8 次

=== 序列 1 处理效果评估 ===
F1 Score 改善: 0.1234
Precision 改善: 0.0567
Recall 改善: 0.0890
MSE 改善: 45.6789
TP 改善: 5
FP 改善: -2
FN 改善: -3

=== 原始指标 ===
F1 Score: 0.7500
Precision: 0.8000
Recall: 0.7000
MSE: 123.4567

=== 处理后指标 ===
F1 Score: 0.8734
Precision: 0.8567
Recall: 0.7890
MSE: 77.7778
```

### 2. 可视化结果

处理器会生成可视化对比图，保存在 `single_sequence_results/` 目录下：

- `sequence_1_comparison.png`: 处理前后的对比图
  - 左图：处理前的预测结果
  - 右图：处理后的预测结果
  - 黄色圆点：真实标注（Ground Truth）
  - 红色叉号：预测点
  - 绿色圆圈：新增的点
  - 白色圆圈：删除的点

## 算法原理

### 1. 主导斜率提取

1. 计算序列内所有帧间点对的斜率
2. 使用容差聚类相似斜率
3. 统计每个斜率区间的出现次数
4. 选择出现次数达到阈值的斜率作为主导斜率

### 2. 轨迹补全

1. **插值补全**: 对符合主导斜率的点对生成中间点
2. **外推补全**: 基于主导斜率向前/向后外推轨迹
3. **间隙填补**: 填补序列中间的缺失帧

### 3. 异常点筛选

1. 计算每个点与其他帧中点的斜率
2. 检查斜率是否符合主导斜率模式
3. 保留斜率匹配率足够高的点
4. 过滤掉不符合轨迹模式的异常点

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sequence_id` | - | 要处理的序列ID |
| `base_distance_threshold` | 80.0 | 基础距离阈值，用于匹配点对 |
| `min_track_length` | 2 | 最小轨迹长度要求 |
| `expected_sequence_length` | 5 | 期望的序列长度 |
| `slope_tolerance` | 0.1 | 斜率容差，用于聚类相似斜率 |
| `min_slope_count` | 2 | 最小斜率出现次数，用于确定主导斜率 |
| `point_distance_threshold` | 5.0 | 重合点过滤阈值 |

## 注意事项

1. **数据格式**: 确保预测结果和真实标注的格式正确
2. **序列ID**: 序列ID必须存在于数据中
3. **内存使用**: 处理大型序列时注意内存使用
4. **参数调优**: 根据具体数据特点调整参数

## 示例数据

预测结果格式：
```json
{
  "1_1": {"coords": [[100, 200], [300, 400]], "num_objects": 2},
  "1_2": {"coords": [[150, 250]], "num_objects": 1},
  "1_3": {"coords": [[200, 300]], "num_objects": 1}
}
```

真实标注格式：
```json
[
  {"image_name": "1_1", "num_objects": 2, "object_coords": [[100, 200], [300, 400]]},
  {"image_name": "1_2", "num_objects": 1, "object_coords": [[150, 250]]},
  {"image_name": "1_3", "num_objects": 1, "object_coords": [[200, 300]]}
]
``` 