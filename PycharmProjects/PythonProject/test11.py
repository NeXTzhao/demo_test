import numpy as np
import matplotlib.pyplot as plt


def exponential_penalty(x, threshold=0.3, epsilon=1e-3, sharpness=1.0):
    k = sharpness * (-np.log(epsilon) / threshold)
    penalty = np.where(x <= threshold, np.exp(-k * x), 0.0)
    return penalty


# 生成距离样本
x = np.linspace(0.0, 0.5, 500)

# 设置不同sharpness的惩罚曲线
sharpness_values = [0.8, 1.0, 1.2, 1.8]
colors = ['blue', 'green', 'orange', 'red']

plt.figure(figsize=(8, 5))
for s, color in zip(sharpness_values, colors):
    y = exponential_penalty(x, threshold=0.3, epsilon=1e-3, sharpness=s)
    plt.plot(x, y, label=f'sharpness={s}', color=color)

plt.axvline(0.3, linestyle='--', color='gray', label='threshold')
plt.title("Exponential Penalty with Sharpness Variants")
plt.xlabel("err_dis (m)")
plt.ylabel("penalty")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
