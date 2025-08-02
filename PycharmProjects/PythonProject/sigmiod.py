import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# 配置绘图样式
sns.set(style="whitegrid", palette="tab10", font_scale=1.2)
plt.rcParams['font.family'] = 'Times New Roman'

# 定义归一化函数
def minmax_normalize(x, x_min, x_max):
    """Min-Max归一化到[0,1]区间"""
    return (x - x_min) / (x_max - x_min)

# 定义统一Sigmoid函数
def unified_sigmoid(x_norm, k=5, b=0):
    """带全局参数的Sigmoid函数"""
    return 1 / (1 + np.exp(-(k * x_norm + b)))

# 生成原始参数数据
acc_raw = np.linspace(-3, 3, 500)  # 加速度范围[-3,3]
jerk_raw = np.linspace(-5, 5, 500) # jerk范围[-5,5]

# 归一化处理
acc_norm = minmax_normalize(acc_raw, -3, 3)
jerk_norm = minmax_normalize(jerk_raw, -5, 5)

# 计算Sigmoid输出
k, b = 5, 0  # 全局统一参数
acc_cost = unified_sigmoid(acc_norm, k, b)
jerk_cost = unified_sigmoid(jerk_norm, k, b)

# 创建画布布局
fig = plt.figure(figsize=(15, 6))
gs = GridSpec(1, 2, width_ratios=[1, 1])

# ----------------------------
# 子图1：原始参数与归一化对比
# ----------------------------
ax1 = fig.add_subplot(gs[0])
ax1.plot(acc_raw, acc_norm, label='Acceleration Normalized', lw=2.5)
ax1.plot(jerk_raw, jerk_norm, label='Jerk Normalized', lw=2.5, linestyle='--')
ax1.set_xlabel("Original Parameter Value", fontsize=12)
ax1.set_ylabel("Normalized Value [0-1]", fontsize=12)
ax1.set_title("Parameter Normalization Comparison", fontsize=14)
ax1.legend()

# ----------------------------
# 子图2：Sigmoid函数输出对比
# ----------------------------
ax2 = fig.add_subplot(gs[1])
ax2.plot(acc_raw, acc_cost, label='Acceleration Cost', lw=2.5)
ax2.plot(jerk_raw, jerk_cost, label='Jerk Cost', lw=2.5, linestyle='--')
ax2.set_xlabel("Original Parameter Value", fontsize=12)
ax2.set_ylabel("Unified Cost Output", fontsize=12)
ax2.set_title("Sigmoid Cost Function with Global Parameters (k=5, b=0)", fontsize=14)
ax2.legend()

# 高亮关键区域
for ax in [ax1, ax2]:
    ax.axvspan(-3, -2, color='red', alpha=0.1, label='High-Cost Region')
    ax.axvspan(2, 3, color='red', alpha=0.1)
    ax.axvspan(-0.5, 0.5, color='green', alpha=0.1, label='Low-Cost Region')

plt.tight_layout()
plt.show()