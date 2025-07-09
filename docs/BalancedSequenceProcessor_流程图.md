# BalancedSequenceProcessor 工作流程图

## 整体流程

```mermaid
graph TD
    A[输入: 原始预测结果] --> B[提取序列信息]
    B --> C[自适应时序过滤]
    C --> D[智能插值]
    D --> E[噪声检测移除]
    E --> F[输出: 处理后结果]
    
    style A fill:#e1f5fe
    style F fill:#e8f5e8
    style C fill:#fff3e0
    style D fill:#f3e5f5
    style E fill:#fce4ec
```

## 详细流程图

### 1. 自适应时序过滤流程

```mermaid
graph TD
    A[开始处理序列] --> B[计算平均检测密度]
    B --> C{密度 > 3?}
    C -->|是| D[使用严格阈值<br/>base_threshold * 0.8]
    C -->|否| E{密度 < 1?}
    E -->|是| F[使用宽松阈值<br/>base_threshold * 1.5]
    E -->|否| G[使用标准阈值<br/>base_threshold]
    
    D --> H[计算时序一致性分数]
    F --> H
    G --> H
    
    H --> I{一致性比例 >= 0.3?}
    I -->|是| J[保留检测点]
    I -->|否| K{平均密度 < 1.5?}
    K -->|是| J
    K -->|否| L[移除检测点]
    
    J --> M[更新检测结果]
    L --> M
    M --> N[处理完成]
    
    style A fill:#e3f2fd
    style N fill:#e8f5e8
    style D fill:#fff3e0
    style F fill:#fff3e0
    style G fill:#fff3e0
```

### 2. 智能插值流程

```mermaid
graph TD
    A[开始插值处理] --> B[遍历连续帧对]
    B --> C{帧间隔 <= max_gap?}
    C -->|否| B
    C -->|是| D[构建代价矩阵]
    D --> E[匈牙利算法匹配]
    E --> F[遍历匹配对]
    F --> G{距离 <= threshold*2?}
    G -->|否| F
    G -->|是| H[线性插值计算]
    H --> I[添加插值检测点]
    I --> F
    F --> J[更新序列数据]
    J --> B
    B --> K[插值完成]
    
    style A fill:#e3f2fd
    style K fill:#e8f5e8
    style E fill:#f3e5f5
    style H fill:#f3e5f5
```

### 3. 噪声检测移除流程

```mermaid
graph TD
    A[开始噪声移除] --> B[遍历每个序列]
    B --> C{序列长度 >= 3?}
    C -->|否| B
    C -->|是| D[遍历每帧]
    D --> E[遍历检测点]
    E --> F[检查时间窗口支持]
    F --> G[计算支持比例]
    G --> H{支持比例 >= 0.2?}
    H -->|是| I[保留检测点]
    H -->|否| J{序列长度 <= 3?}
    J -->|是| I
    J -->|否| K[移除检测点]
    
    I --> L[更新检测结果]
    K --> L
    L --> E
    E --> M[处理下一帧]
    M --> D
    D --> N[处理下一个序列]
    N --> B
    B --> O[噪声移除完成]
    
    style A fill:#e3f2fd
    style O fill:#e8f5e8
    style F fill:#fce4ec
    style G fill:#fce4ec
```

## 数据流图

```mermaid
graph LR
    A[原始预测<br/>39018个检测点] --> B[自适应过滤]
    B --> C[过滤后<br/>35594个检测点]
    C --> D[智能插值]
    D --> E[插值后<br/>36500个检测点]
    E --> F[噪声移除]
    F --> G[最终结果<br/>34212个检测点]
    
    style A fill:#e1f5fe
    style G fill:#e8f5e8
    style B fill:#fff3e0
    style D fill:#f3e5f5
    style F fill:#fce4ec
```

## 参数影响图

```mermaid
graph TD
    A[参数配置] --> B[base_distance_threshold]
    A --> C[temporal_window]
    A --> D[adaptive_threshold]
    A --> E[max_frame_gap]
    
    B --> F[影响匹配精度]
    C --> G[影响时序一致性]
    D --> H[影响自适应能力]
    E --> I[影响插值范围]
    
    F --> J[Precision变化]
    G --> K[Recall变化]
    H --> L[平衡性]
    I --> M[完整性]
    
    J --> N[最终F1分数]
    K --> N
    L --> N
    M --> N
    
    style A fill:#e3f2fd
    style N fill:#e8f5e8
```

## 性能对比图

```mermaid
graph LR
    A[简单处理器<br/>F1: 0.8249<br/>MSE: 3080.23] --> B[改进处理器<br/>F1: 0.8326<br/>MSE: 2920.59]
    B --> C[平衡处理器<br/>F1: 0.8358<br/>MSE: 2849.97]
    
    style A fill:#ffebee
    style B fill:#fff3e0
    style C fill:#e8f5e8
```

## 关键决策点

### 1. 阈值选择决策
```
检测密度 > 3: 使用严格阈值 (0.8 * base_threshold)
检测密度 < 1: 使用宽松阈值 (1.5 * base_threshold)
其他情况: 使用标准阈值 (base_threshold)
```

### 2. 检测点保留决策
```
条件1: 时序一致性比例 >= 0.3
条件2: 平均检测密度 < 1.5
满足任一条件则保留检测点
```

### 3. 插值匹配决策
```
距离 <= 2 * base_threshold: 进行插值
距离 > 2 * base_threshold: 跳过插值
```

### 4. 噪声移除决策
```
支持比例 >= 0.2: 保留检测点
序列长度 <= 3: 保留检测点
其他情况: 移除检测点
```

## 算法复杂度分析

| 步骤 | 时间复杂度 | 空间复杂度 | 说明 |
|------|------------|------------|------|
| 序列信息提取 | O(n) | O(n) | n为图像数量 |
| 自适应过滤 | O(n * m * w) | O(n) | m为平均检测点数，w为时间窗口 |
| 智能插值 | O(n * m²) | O(m²) | 匈牙利算法复杂度 |
| 噪声移除 | O(n * m * w) | O(n) | 与过滤步骤类似 |

**总体复杂度**: O(n * m²)，其中n为图像数量，m为平均检测点数。 