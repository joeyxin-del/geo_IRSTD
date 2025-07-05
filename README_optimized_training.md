# 优化版训练脚本使用说明

## 概述

`train_geo_modified.py` 是对原始 `train_spotgeo.py` 的全面重构和优化版本，主要改进了代码的可读性、可维护性和健壮性，并新增了 SwanLab 监控、定期检查点保存和定期验证功能。

## 主要改进

### 1. 代码结构优化
- **面向对象设计**: 使用类来组织代码，提高模块化程度
- **配置管理**: 独立的 `TrainingConfig` 类管理所有配置参数
- **工厂模式**: `OptimizerFactory` 类统一管理优化器和调度器创建
- **职责分离**: 将训练、验证、日志等功能分离到不同方法中

### 2. 可读性提升
- **类型注解**: 添加完整的类型提示
- **文档字符串**: 为所有类和方法添加详细文档
- **命名规范**: 使用更清晰的变量和函数命名
- **代码注释**: 添加有意义的注释说明

### 3. 错误处理
- **异常处理**: 添加 try-catch 块处理可能的错误
- **日志系统**: 使用 Python logging 模块替代 print 语句
- **检查点恢复**: 改进的检查点加载机制

### 4. 功能增强
- **更好的日志记录**: 自动创建带时间戳的日志文件
- **配置验证**: 参数验证和错误提示
- **设备检测**: 自动检测并使用可用的计算设备
- **进度显示**: 改进的训练进度显示

### 5. 🆕 新增功能
- **SwanLab 监控**: 实时监控训练过程，可视化指标
- **定期检查点保存**: 每 `save_interval` 个 epoch 保存检查点
- **定期验证**: 每 `val_interval` 个 epoch 进行验证并保存最佳模型
- **智能恢复**: 支持从任意检查点恢复训练

## 使用方法

### 基本用法

```bash
# 使用默认参数训练
python train_geo_modified.py

# 指定模型和数据集
python train_geo_modified.py --model_names WTNet --dataset_names spotgeov2-IRSTD

# 自定义训练参数
python train_geo_modified.py \
    --model_names WTNet \
    --dataset_names spotgeov2-IRSTD \
    --batch_size 4 \
    --num_epochs 200 \
    --learning_rate 0.001 \
    --optimizer_name Adamw
```

### 高级用法

```bash
# 使用混合精度训练
python train_geo_modified.py --use_amp True

# 启用正交正则化
python train_geo_modified.py --use_orthogonal_reg True

# 从检查点恢复训练
python train_geo_modified.py --resume_paths ./log/spotgeov2-IRSTD/WTNet_epoch_50.pth

# 计算推理FPS
python train_geo_modified.py --calculate_fps True

# 自定义保存和验证间隔
python train_geo_modified.py \
    --save_interval 5 \
    --val_interval 2 \
    --use_swanlab True
```

### 🆕 SwanLab 监控

```bash
# 启用 SwanLab 监控（默认启用）
python train_geo_modified.py --use_swanlab True

# 禁用 SwanLab 监控
python train_geo_modified.py --use_swanlab False
```

**注意**: 需要先安装 SwanLab: `pip install swanlab`

## 参数说明

### 模型和数据集
- `--model_names`: 要训练的模型名称列表
- `--dataset_names`: 要使用的数据集名称列表
- `--dataset_dir`: 数据集目录路径
- `--img_norm_cfg`: 图像归一化配置

### 训练参数
- `--batch_size`: 训练批次大小
- `--patch_size`: 训练图像块大小
- `--num_epochs`: 训练轮数
- `--seed`: 随机种子

### 优化器设置
- `--optimizer_name`: 优化器名称 (Adam, Adamw, SGD, Adagrad)
- `--learning_rate`: 学习率
- `--scheduler_name`: 学习率调度器名称
- `--scheduler_settings`: 调度器设置

### 系统设置
- `--num_threads`: 数据加载器线程数
- `--save_dir`: 保存目录
- `--resume_paths`: 恢复训练的检查点路径
- `--threshold`: 预测阈值

### 高级功能
- `--use_amp`: 是否使用自动混合精度
- `--use_orthogonal_reg`: 是否使用正交正则化
- `--calculate_fps`: 是否计算推理FPS
- `--ema_decay`: EMA衰减率

### 🆕 监控和检查点
- `--save_interval`: 每 N 个 epoch 保存检查点（默认: 10）
- `--val_interval`: 每 N 个 epoch 进行验证（默认: 5）
- `--use_swanlab`: 是否启用 SwanLab 监控（默认: True）

## 输出文件

训练过程中会生成以下文件：

1. **检查点文件**: 保存在 `save_dir/dataset_name/` 目录下
   - `model_name_best.pth`: 最佳模型（基于验证 IoU）
   - `model_name_epoch_N.pth`: 第 N 轮的定期检查点
   - `model_name_latest.pth`: 最新检查点

2. **日志文件**: 保存在 `save_dir/dataset_name/` 目录下
   - `model_name_YYYYMMDD_HHMMSS.log`: 详细的训练日志

3. **🆕 SwanLab 实验**: 自动创建实验记录
   - 实时监控训练和验证指标
   - 可视化训练曲线
   - 实验配置记录

## 检查点管理

### 检查点类型
1. **定期检查点** (`model_name_epoch_N.pth`): 每 `save_interval` 个 epoch 保存
2. **最佳检查点** (`model_name_best.pth`): 验证 IoU 最高时保存
3. **最新检查点** (`model_name_latest.pth`): 每次验证时保存

### 恢复训练
```bash
# 从定期检查点恢复
python train_geo_modified.py --resume_paths ./log/spotgeov2-IRSTD/WTNet_epoch_50.pth

# 从最佳检查点恢复
python train_geo_modified.py --resume_paths ./log/spotgeov2-IRSTD/WTNet_best.pth

# 从最新检查点恢复
python train_geo_modified.py --resume_paths ./log/spotgeov2-IRSTD/WTNet_latest.pth
```

## SwanLab 监控

### 监控指标
- **训练指标**: 训练损失、学习率
- **验证指标**: mIoU、IoU、nIoU、PD、FA
- **系统指标**: 训练时间、推理时间

### 查看监控
1. 训练开始后，SwanLab 会自动启动本地服务器
2. 在浏览器中打开显示的 URL（通常是 `http://localhost:5678`）
3. 实时查看训练进度和指标变化

## 与原版本的主要区别

| 特性 | 原版本 | 优化版本 |
|------|--------|----------|
| 代码结构 | 单一大函数 | 面向对象设计 |
| 配置管理 | 全局变量 | 配置类 |
| 错误处理 | 无 | 完整的异常处理 |
| 日志记录 | print语句 | logging模块 |
| 类型安全 | 无类型注解 | 完整类型提示 |
| 文档 | 注释较少 | 详细文档字符串 |
| 可维护性 | 低 | 高 |
| **监控系统** | **无** | **SwanLab 实时监控** |
| **检查点管理** | **固定间隔** | **灵活配置** |
| **验证策略** | **固定间隔** | **可配置间隔** |

## 兼容性

优化版本保持了与原版本相同的功能，但提供了更好的代码结构和错误处理。所有原有的训练参数都得到保留，可以直接替换使用。

## 注意事项

1. 确保所有依赖包已正确安装
   ```bash
   pip install swanlab  # 用于监控
   ```
2. 检查数据集路径是否正确
3. 确保有足够的磁盘空间保存检查点和日志
4. 建议在首次使用时先用小数据集测试
5. SwanLab 监控需要网络连接（首次使用时会下载依赖）

## 故障排除

### 常见问题

1. **CUDA内存不足**: 减小 `batch_size` 或 `patch_size`
2. **数据集加载失败**: 检查 `dataset_dir` 路径和数据集格式
3. **检查点加载失败**: 确保检查点文件存在且格式正确
4. **优化器不支持**: 检查 `optimizer_name` 是否在支持列表中
5. **SwanLab 启动失败**: 检查网络连接或禁用 SwanLab 监控

### 获取帮助

如果遇到问题，请检查日志文件中的详细错误信息，或查看代码中的异常处理部分。 