# BalancedSequenceProcessor - 红外小目标检测序列后处理器

## 📖 项目简介

`BalancedSequenceProcessor` 是一个专门为红外小目标检测设计的智能序列后处理器。它通过自适应策略、智能插值和噪声移除技术，在提升检测精确率的同时尽可能保持召回率，为红外小目标检测任务提供了强大的后处理解决方案。

## ✨ 核心特性

- 🎯 **自适应策略**: 根据序列特征动态调整过滤策略
- 🔄 **智能插值**: 使用匈牙利算法进行最优匹配和插值
- 🧹 **噪声移除**: 有效去除孤立检测点，保留有意义轨迹
- ⚖️ **平衡优化**: 在精确率和召回率之间取得良好平衡
- 🚀 **高效处理**: 支持大规模序列数据的快速处理

## 📊 性能表现

| 指标 | 原始结果 | 处理后结果 | 改善 |
|------|----------|------------|------|
| **Precision** | 0.8833 | 0.9620 | **+0.0787** |
| **Recall** | 0.7736 | 0.7388 | -0.0349 |
| **F1 Score** | 0.8249 | 0.8358 | **+0.0109** |
| **MSE** | 3080.23 | 2849.97 | **+230.26** |

- **处理速度**: 1.26秒处理25600帧
- **检测点优化**: 39018 → 34212 (-4806个噪声点)
- **综合评分**: 0.1151 (三种处理器中最佳)

## 🏗️ 技术架构

```
BalancedSequenceProcessor
├── 自适应时序过滤 (Adaptive Temporal Filtering)
│   ├── 密度感知阈值调整
│   ├── 时序一致性计算
│   └── 自适应决策机制
├── 智能插值 (Smart Interpolation)
│   ├── 匈牙利算法匹配
│   ├── 距离约束筛选
│   └── 线性插值计算
└── 噪声检测移除 (Noise Detection Removal)
    ├── 时间窗口支持检查
    ├── 支持比例计算
    └── 孤立点移除
```

## 🚀 快速开始

### 安装依赖

```bash
pip install numpy scipy
```

### 基本使用

```python
from processor import BalancedSequenceProcessor

# 创建处理器
processor = BalancedSequenceProcessor(
    base_distance_threshold=80.0,
    temporal_window=3,
    adaptive_threshold=True
)

# 处理预测结果
processed_predictions = processor.process_sequence(original_predictions)

# 评估改善效果
improvement = processor.evaluate_improvement(
    original_predictions, 
    processed_predictions, 
    ground_truth
)
```

### 参数配置

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `base_distance_threshold` | 80.0 | 基础距离阈值 | 60-120 |
| `temporal_window` | 3 | 时间窗口大小 | 2-5 |
| `confidence_threshold` | 0.05 | 置信度阈值 | 0.02-0.1 |
| `min_track_length` | 2 | 最小轨迹长度 | 1-3 |
| `max_frame_gap` | 3 | 最大帧间隔 | 2-5 |
| `adaptive_threshold` | True | 自适应阈值 | 建议启用 |

## 📁 项目结构

```
geo_IRSTD/
├── processor/                           # 序列后处理器包
│   ├── __init__.py                      # 包初始化文件
│   ├── balanced_sequence_processor.py   # 平衡处理器实现
│   ├── improved_sequence_processor.py   # 改进处理器实现
│   └── simple_sequence_processor.py     # 简单处理器实现
├── compare_processors.py                # 处理器比较脚本
├── eval_predictions.py                  # 评估指标计算
├── BalancedSequenceProcessor_原理说明.md  # 详细原理文档
├── BalancedSequenceProcessor_流程图.md   # 流程图文档
└── README_BalancedSequenceProcessor.md  # 项目说明文档
```

## 🔬 算法原理

### 1. 自适应时序过滤

根据序列的检测密度特征，动态调整距离阈值：

```python
# 密度感知阈值调整
if avg_detections_per_frame > 3:
    distance_threshold = base_threshold * 0.8  # 高密度：严格阈值
elif avg_detections_per_frame < 1:
    distance_threshold = base_threshold * 1.5  # 低密度：宽松阈值
else:
    distance_threshold = base_threshold        # 标准阈值
```

### 2. 智能插值

使用匈牙利算法进行最优匹配：

```python
# 匈牙利算法匹配
cost_matrix = calculate_distance_matrix(coords1, coords2)
row_indices, col_indices = linear_sum_assignment(cost_matrix)

# 线性插值
for row_idx, col_idx in matches:
    if cost_matrix[row_idx, col_idx] <= threshold * 2:
        interpolated_pos = linear_interpolation(pos1, pos2, alpha)
```

### 3. 噪声移除

基于时间窗口支持比例进行决策：

```python
# 支持比例计算
support_ratio = support_frames / total_frames

# 决策条件
if support_ratio >= 0.2 or len(frames) <= 3:
    keep_detection()  # 保留检测点
else:
    remove_detection()  # 移除噪声
```

## 📈 实验结果

### 处理器对比

| 处理器 | F1改善 | MSE改善 | 处理时间 | 检测点变化 |
|--------|--------|---------|----------|------------|
| 简单处理器 | +0.0000 | +0.00 | 0.86s | 0 |
| 改进处理器 | +0.0077 | +159.64 | 1.05s | -3424 |
| **平衡处理器** | **+0.0109** | **+230.26** | 1.26s | **-4806** |

### 性能分析

- ✅ **F1改善最佳**: 平衡处理器 (+0.0109)
- ✅ **MSE改善最佳**: 平衡处理器 (+230.26)
- ✅ **综合评分最高**: 平衡处理器 (0.1151)
- ✅ **噪声移除最有效**: 平衡处理器 (-4806个检测点)

## 🎯 适用场景

- 🔥 **红外小目标检测**: 主要应用场景
- 📹 **序列数据后处理**: 视频序列优化
- 🎯 **目标跟踪优化**: 轨迹平滑和补全
- 🧹 **噪声检测清理**: 去除误检和噪声
- ⚖️ **精度召回平衡**: 需要平衡性能的场景

## 🔧 高级用法

### 批量处理

```python
# 处理多个模型的结果
models = ['WTNet', 'YOLO', 'FasterRCNN']
for model in models:
    pred_path = f'results/{model}/predictions.json'
    with open(pred_path, 'r') as f:
        predictions = json.load(f)
    
    processed = processor.process_sequence(predictions)
    
    # 保存结果
    output_path = f'results/{model}/balanced_processed_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(processed, f, indent=2)
```

### 参数调优

```python
# 高精度场景配置
high_precision_processor = BalancedSequenceProcessor(
    base_distance_threshold=60.0,  # 更严格的阈值
    temporal_window=5,             # 更大的时间窗口
    confidence_threshold=0.1,      # 更高的置信度要求
    adaptive_threshold=True
)

# 高召回场景配置
high_recall_processor = BalancedSequenceProcessor(
    base_distance_threshold=120.0, # 更宽松的阈值
    temporal_window=2,             # 更小的时间窗口
    confidence_threshold=0.02,     # 更低的置信度要求
    adaptive_threshold=True
)
```

## 📚 文档说明

- **[BalancedSequenceProcessor_原理说明.md](./BalancedSequenceProcessor_原理说明.md)**: 详细的技术原理和实现说明
- **[BalancedSequenceProcessor_流程图.md](./BalancedSequenceProcessor_流程图.md)**: 可视化的工作流程图
- **[compare_processors.py](./compare_processors.py)**: 三种处理器的对比脚本

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

### 开发环境

```bash
# 克隆项目
git clone <repository-url>
cd geo_IRSTD

# 安装依赖
pip install -r requirements.txt

# 运行测试
python compare_processors.py
```

### 代码规范

- 遵循PEP 8代码风格
- 添加详细的文档字符串
- 包含单元测试
- 提交前运行完整测试

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🙏 致谢

感谢IRSTD数据集提供者，以及所有为这个项目做出贡献的研究者和开发者。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 📧 Email: [your-email@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 📖 Wiki: [项目Wiki](https://github.com/your-repo/wiki)

---

**⭐ 如果这个项目对你有帮助，请给它一个星标！** 