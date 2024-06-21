import numpy as np

# 系统参数
h = 0.1
Tfinal = 5
N = int(Tfinal / h + 1)

A = np.array([[1, h], [0, 1]])
B = np.array([[0.5 * h ** 2], [h]])
n = A.shape[0]
m = B.shape[1]
x0 = np.array([10.0, -50])
xd = np.array([0, 0])
Q = np.eye(n) * 10
R = np.eye(m) * 1
Qn = np.eye(n) * 10

# 动态规划求解LQR
def dynamic_programming_lqr():
    P = [None] * N
    K = [None] * (N - 1)

    # 初始化终端状态的权重矩阵
    P[-1] = Qn

    # 反向递推求解P和K矩阵
    for i in range(N - 2, -1, -1):
        P_i = P[i + 1]
        K_i = np.linalg.inv(R + B.T @ P_i @ B) @ (B.T @ P_i @ A)
        P[i] = Q + A.T @ P_i @ (A - B @ K_i)
        K[i] = K_i

    x = np.zeros((n, N))
    u = np.zeros((m, N - 1))
    x[:, 0] = x0

    # 正向递推计算状态轨迹和控制输入序列
    for i in range(N - 1):
        u[:, i] = -K[i] @ x[:, i]  # 计算当前步的控制输入
        x[:, i + 1] = A @ x[:, i] + B @ u[:, i]  # 计算下一步的状态

    return u, x

# 运行动态规划方法
u_dp, x_dp = dynamic_programming_lqr()

# 可视化结果对比
import matplotlib.pyplot as plt

t = np.linspace(0, (N - 1) * h, N - 1)
t_full = np.linspace(0, N * h, N)

plt.figure(figsize=(14, 12))

# 控制输入对比图
plt.subplot(3, 1, 1)
plt.plot(t, u_dp.T, label='Dynamic Programming LQR Method', color='navy', marker='o', linestyle='-', linewidth=2, markersize=6)
plt.xlabel('Time (s)')
plt.ylabel('Control Input (u)')
plt.title('Control Input Comparison')
plt.legend(fontsize=12)
plt.grid(True)

# 状态轨迹对比图 - 位置 (q)
plt.subplot(3, 1, 2)
plt.plot(t_full, x_dp[0, :], label='Dynamic Programming LQR Method Position (q)', color='teal', marker='o', linestyle='-', linewidth=2, markersize=6)
plt.xlabel('Time (s)')
plt.ylabel('Position (q)')
plt.title('Position Trajectory Comparison')
plt.legend(fontsize=12)
plt.grid(True)

# 状态轨迹对比图 - 速度 (dq)
plt.subplot(3, 1, 3)
plt.plot(t_full, x_dp[1, :], label='Dynamic Programming LQR Method Velocity (dq)', color='forestgreen', marker='o', linestyle='-', linewidth=2, markersize=6)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (dq)')
plt.title('Velocity Trajectory Comparison')
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
