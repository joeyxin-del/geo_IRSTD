# 角度距离处理器 (AngleDistanceProcessor) 说明

## 概述

`AngleDistanceProcessor` 是在原有 `AngleProcessor` 基础上的改进版本，引入了**主导步长**概念，综合考虑角度和距离间隔的相似性进行轨迹补全和异常点筛选。

## 主要改进

### 1. 主导步长概念

在原有主导角度基础上，新增了主导步长概念：

- **步长定义**: 每帧的平均移动距离 = 两点间距离 / 帧间隔
- **步长容差**: 默认 0.2（比例），即 0.9-1.1 倍范围内认为相似
- **主导步长**: 在序列中出现次数达到阈值的步长模式

### 2. 综合评分机制

对每个点对进行综合评分：

```python
# 角度分数 (0-1)
angle_score = 1 - (angle_diff / angle_tolerance)

# 步长分数 (0-1)  
step_score = 1 - abs(step_ratio - 1) / 0.1

# 综合分数
combined_score = (angle_score + step_score) / 2
```

### 3. 双重筛选条件

点对必须同时满足：
- **角度匹配**: 属于主导角度（在角度容差范围内）
- **步长匹配**: 属于主导步长（在步长比例范围内）

## 核心参数

```python
class AngleDistanceProcessor:
    def __init__(self, 
                 base_distance_threshold: float = 80.0,      # 基础距离阈值
                 angle_tolerance: float = 10.0,             # 角度容差（度）
                 min_angle_count: int = 2,                  # 最小角度出现次数
                 step_tolerance: float = 0.2,               # 步长容差（比例）
                 min_step_count: int = 2,                   # 最小步长出现次数
                 point_distance_threshold: float = 5.0):    # 重合点过滤阈值
```

## 工作流程

### 1. 数据收集阶段

```python
def collect_sequence_angles_and_steps(self, frames_data):
    # 计算所有帧间配对的角度和步长
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            angle = calculate_angle(pos1, pos2)
            step_size = calculate_step_size(pos1, pos2, frame_gap)
```

### 2. 模式识别阶段

```python
# 统计角度分布
angle_counter = Counter()
for angle in all_angles:
    angle_key = round(angle / angle_tolerance) * angle_tolerance
    angle_counter[angle_key] += 1

# 统计步长分布  
step_counter = Counter()
for step in all_steps:
    if step > 0:
        step_key = round(step / (step * step_tolerance)) * (step * step_tolerance)
        step_counter[step_key] += 1
```

### 3. 点参与评估阶段

```python
for pair_info in angle_step_pairs:
    # 检查角度匹配
    angle_score = check_angle_match(angle, dominant_angles)
    
    # 检查步长匹配
    step_score = check_step_match(step_size, dominant_steps)
    
    # 综合评分
    combined_score = (angle_score + step_score) / 2
    
    if combined_score >= 0.5:  # 阈值可调整
        mark_point_participation(point1, point2)
```

### 4. 轨迹补全阶段

```python
def complete_trajectory_with_patterns(self, frames_data, dominant_angles, dominant_steps):
    # 对符合主导模式的点对生成缺失帧的点
    for pair_info in angle_step_pairs:
        if angle_match and step_match:
            generated_points = generate_points_from_pair_with_steps(...)
    
    # 外推轨迹填补间隙
    extrapolated_points = extrapolate_trajectory_with_steps(...)
```

### 5. 异常点筛选阶段

```python
def filter_outliers_by_dominant_patterns(self, frames_data, point_participation):
    for frame_id, frame_data in frames_data.items():
        for point_idx, point in enumerate(frame_data['coords']):
            if point_key in point_participation:
                if point_participation[point_key]['participated']:
                    keep_point(point)  # 参与主导模式，保留
                else:
                    remove_point(point)  # 未参与主导模式，可能是异常点
```

## 优势特点

### 1. 更精确的模式识别

- **双重验证**: 同时考虑角度和步长，减少误判
- **比例匹配**: 步长使用比例匹配，适应不同尺度的轨迹
- **综合评分**: 避免单一指标导致的偏差

### 2. 更稳健的轨迹补全

- **步长约束**: 生成的点符合主导步长模式
- **角度一致性**: 保持轨迹的角度一致性
- **自适应外推**: 根据实际步长模式进行外推

### 3. 更有效的异常点筛选

- **双重筛选**: 只有同时符合角度和步长模式的点才被认为是有效点
- **分数机制**: 通过综合分数量化点的参与程度
- **距离验证**: 结合距离阈值进行最终验证

## 使用示例

```python
# 创建处理器
processor = AngleDistanceProcessor(
    base_distance_threshold=1000.0,
    angle_tolerance=2.5,
    min_angle_count=2,
    step_tolerance=0.2,
    min_step_count=2,
    point_distance_threshold=200.0
)

# 处理数据集
processed_predictions = processor.process_dataset(original_predictions)

# 评估改善效果
improvement = processor.evaluate_improvement(
    original_predictions, processed_predictions, ground_truth
)
```

## 与原始处理器的对比

| 特性 | AngleProcessor | AngleDistanceProcessor |
|------|----------------|----------------------|
| 主导模式 | 仅角度 | 角度 + 步长 |
| 筛选条件 | 单一角度匹配 | 双重匹配（角度+步长） |
| 评分机制 | 二元参与/不参与 | 连续分数（0-1） |
| 轨迹补全 | 基于角度外推 | 基于角度和步长外推 |
| 异常点筛选 | 基于角度参与 | 基于综合模式参与 |

## 适用场景

1. **高精度轨迹补全**: 需要同时保持角度和距离一致性的场景
2. **复杂运动模式**: 轨迹具有明显步长规律的目标
3. **噪声环境**: 需要更强鲁棒性的异常点筛选
4. **多尺度目标**: 不同大小目标的轨迹处理

## 参数调优建议

### 角度相关参数
- `angle_tolerance`: 2.5-10.0 度，根据轨迹角度变化程度调整
- `min_angle_count`: 2-5，根据序列长度调整

### 步长相关参数  
- `step_tolerance`: 0.1-0.3，根据步长变化程度调整
- `min_step_count`: 2-5，根据序列长度调整

### 距离相关参数
- `base_distance_threshold`: 根据目标大小和图像分辨率调整
- `point_distance_threshold`: 避免生成过于接近的点 