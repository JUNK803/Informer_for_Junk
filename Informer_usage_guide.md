# Informer 气象预测完整使用指南

## 📌 概述

本指南将帮助你使用官方原版 Informer2020 进行气象预测，并与 LSTM 模型做严格对比。

---

## 🔧 环境准备

### 1. 安装依赖

```bash
pip install torch torchvision
pip install numpy pandas scikit-learn
pip install matplotlib seaborn
```

### 2. 下载官方代码

```bash
git clone https://github.com/zhouhaoyi/Informer2020.git
cd Informer2020
```

---

## 📂 文件部署

将以下自定义文件复制到 Informer2020 目录：

```
Informer2020/
├── data_loader_custom.py       # 新增：自定义数据加载器
├── exp_custom.py                # 新增：自定义实验流程
├── metrics_custom.py            # 新增：评估指标
├── run_weather_tasks.py         # 新增：主运行脚本
├── data/                        # 新增：数据目录
│   └── city_beijing.csv         # 你的气象数据
└── utils/
    ├── tools.py                 # 官方工具（保持不变）
    └── metrics.py               # 官方指标（我们用自定义的）
```

### 需要从官方保留的核心文件：

```python
# 确保以下文件存在且未修改
models/model.py
models/encoder.py
models/decoder.py
models/attn.py
models/embed.py
utils/tools.py
utils/masking.py
```

---

## 📊 数据格式要求

你的 CSV 数据应该符合以下格式：

```
date,feature_1,feature_2,...,feature_95,temperature
2020-01-01 00:00:00,0.123,0.456,...,0.789,0.234
2020-01-01 01:00:00,0.124,0.457,...,0.790,0.235
...
```

**重要约定：**
- 第一列：时间列（列名包含 'date' 或 'time'）
- 中间列：96 个气象特征（已归一化）
- 最后一列：`temperature`（目标变量，已归一化）
- 数据频率：小时级

---

## 🚀 运行步骤

### Step 1: 准备数据

将你的气象数据 CSV 文件放入 `data/` 目录：

```bash
mkdir -p data
cp /path/to/your/city_beijing.csv data/
```

### Step 2: 修改运行配置

编辑 `run_weather_tasks.py` 的配置区域：

```python
# ============== 配置区域 ==============
data_path = 'city_beijing.csv'  # 你的数据文件名
city_name = 'Beijing'            # 城市名称
# =====================================
```

### Step 3: 运行所有任务

```bash
python run_weather_tasks.py
```

这将自动运行以下 7 个任务：

| 任务 | Window | Horizon | Type |
|------|--------|---------|------|
| 1 | 24 | 1 | single_point |
| 2 | 96 | 1 | single_point |
| 3 | 24 | 6 | single_point |
| 4 | 96 | 6 | single_point |
| 5 | 96 | 6 | sequence |
| 6 | 96 | 24 | single_point |
| 7 | 96 | 24 | sequence |

---

## 📈 结果输出

### 1. 模型权重

保存在 `checkpoints/` 目录：

```
checkpoints/
└── Beijing_informer_w96_h24_sequence/
    ├── checkpoint.pth           # 最优模型
    ├── train_losses.npy         # 训练损失
    └── vali_losses.npy          # 验证损失
```

### 2. 预测结果

保存在 `results/` 目录：

```
results/
└── Beijing_informer_w96_h24_sequence/
    ├── metrics.npy              # [MAE, RMSE, R², Inference_Time]
    ├── pred.npy                 # 预测值
    └── true.npy                 # 真实值
```

### 3. 汇总报告

保存在 `results_summary/` 目录：

```
results_summary/
└── Beijing_informer_results.md  # Markdown 格式报告
```

报告示例：

```markdown
# Informer Results - Beijing

## Summary Table

| Task | Window | Horizon | Type | MAE | RMSE | R² | Train Time (s) | Inference Time (s) |
|------|--------|---------|------|-----|------|-------|----------------|-------------------|
| 1 | 24 | 1 | single_point | 0.0234 | 0.0345 | 0.9567 | 125.34 | 2.56 |
| 2 | 96 | 1 | single_point | 0.0198 | 0.0312 | 0.9678 | 178.92 | 3.21 |
...
```

---

## ⚙️ 关键对齐点（与 LSTM）

### 1. 数据处理
✅ 相同的数据划分比例（70% / 15% / 15%）
✅ 相同的归一化策略（StandardScaler on train）
✅ 相同的滑动窗口方式
✅ 相同的反归一化（仅 temperature）

### 2. 模型复杂度
✅ d_model=64（对齐 LSTM hidden_size=64）
✅ e_layers=2（对齐 LSTM num_layers=2）
✅ dropout=0.2（完全一致）

### 3. 训练策略
✅ 优化器：Adam
✅ 学习率：1e-4
✅ Batch Size：64
✅ Max Epochs：100
✅ Early Stopping Patience：10

### 4. 损失函数
✅ 单点预测：MSE
✅ 序列预测：加权 MSE（指数衰减）

### 5. 评估指标
✅ MAE（平均绝对误差）
✅ RMSE（均方根误差）
✅ R²（决定系数）
✅ 训练时间 & 推理时间

---

## 🔍 单独运行某个任务

如果你想单独运行某个特定任务，可以使用：

```python
from run_weather_tasks import run_single_task

# 运行 Task 5: Window=96, Horizon=6, Type=sequence
result = run_single_task(
    window_size=96,
    horizon=6,
    task_type='sequence',
    data_path='city_beijing.csv',
    city_name='Beijing'
)

print(result)
```

---

## 🐛 常见问题

### Q1: 数据列数不是 96 怎么办？

修改 `run_weather_tasks.py` 中的参数：

```python
parser.add_argument('--enc_in', type=int, default=你的特征数, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=你的特征数, help='decoder input size')
```

### Q2: 目标变量不叫 'temperature' 怎么办？

修改 `run_weather_tasks.py` 中的参数：

```python
parser.add_argument('--target', type=str, default='你的目标变量名', help='target feature')
```

### Q3: GPU 内存不足怎么办？

方法 1：减小 batch_size
```python
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
```

方法 2：减小模型维度
```python
parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
```

方法 3：使用 CPU
```python
parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
```

### Q4: 如何可视化预测结果？

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载结果
pred = np.load('results/Beijing_informer_w96_h24_sequence/pred.npy')
true = np.load('results/Beijing_informer_w96_h24_sequence/true.npy')

# 绘图
plt.figure(figsize=(15, 5))
plt.plot(true[:200], label='True', alpha=0.7)
plt.plot(pred[:200], label='Pred', alpha=0.7)
plt.legend()
plt.title('Temperature Prediction')
plt.xlabel('Time Step')
plt.ylabel('Temperature')
plt.savefig('prediction_plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 📊 与 LSTM 对比分析

运行完成后，你可以创建对比表格：

```markdown
| 模型 | Task | MAE | RMSE | R² | Train Time | Inference Time |
|------|------|-----|------|-----|------------|----------------|
| LSTM | 1 | 0.0234 | 0.0345 | 0.9567 | 98.23 | 1.45 |
| Informer | 1 | 0.0221 | 0.0332 | 0.9589 | 125.34 | 2.56 |
| LSTM | 2 | 0.0198 | 0.0312 | 0.9678 | 145.67 | 2.34 |
| Informer | 2 | 0.0185 | 0.0298 | 0.9701 | 178.92 | 3.21 |
...
```

**关键对比维度：**
1. 预测精度（MAE, RMSE, R²）
2. 计算效率（训练时间 & 推理时间）
3. 不同窗口长度的表现
4. 不同预测步长的表现
5. 单点 vs 序列预测的差异

---

## ✅ 检查清单

在运行之前，确保：

- [ ] 官方 Informer2020 代码已下载
- [ ] 所有自定义文件已放置在正确位置
- [ ] 数据文件格式正确（第一列时间，最后一列 temperature）
- [ ] 数据已归一化
- [ ] 配置参数已正确设置（data_path, city_name）
- [ ] Python 环境已安装所有依赖

---

## 🎯 预期输出

运行成功后，你会看到：

```
================================================================================
Running Task: Window=24, Horizon=1, Type=single_point
City: Beijing
================================================================================

Use GPU: cuda:0
>>>>>>>start training >>>>>>>>>>>>>>>>>>>>>>>>>>
Epoch: 1 cost time: 12.34s
Epoch: 1, Steps: 123 | Train Loss: 0.0234567 Vali Loss: 0.0198765
Epoch: 2 cost time: 11.98s
...
Early stopping
>>>>>>>testing >>>>>>>>>>>>>>>>>>>>>>>>>>
Test Results:
MAE: 0.0234, RMSE: 0.0345, R²: 0.9567
Inference Time: 2.56s

Task Completed!
Training Time: 125.34s
Inference Time: 2.56s
================================================================================
```

---

## 📞 进一步帮助

如果遇到问题：
1. 检查数据格式是否正确
2. 检查文件路径是否正确
3. 检查 GPU 可用性
4. 查看错误信息的详细 traceback

祝实验顺利！🎉