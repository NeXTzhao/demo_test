import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

# 系统参数
h = 0.1
Tfinal = 5
N = int(Tfinal / h + 1)

A = np.array([[1, h], [0, 1]])
B = np.array([[0.5 * h ** 2], [h]])
n = A.shape[0]
m = B.shape[1]
x0 = np.array([10.0, -50])
Q = np.eye(n) * 10
R = np.eye(m) * 1
Qn = np.eye(n) * 10

# 构建权重矩阵 H
H = block_diag(R, *[block_diag(Q, R) for _ in range(N - 2)], Qn)
# print(f'H = \n{H}')
# 构建Kronecker积部分
I_N_minus_1 = np.eye(N - 1)
block_B_I = np.hstack([B, -np.eye(n)])
C = np.kron(I_N_minus_1, block_B_I)

# 将A填入C的特定位置
for k in range(1, N - 1):
    C[k * n:(k + 1) * n, (k * (n + m) - n):(k * (n + m))] = A

# 构建向量d
d = np.concatenate([-A @ x0, np.zeros(C.shape[0] - n)])
# print(f'd = {d}')

# 构建KKT矩阵
KKT_matrix = np.block([[H, C.T], [C, np.zeros((C.shape[0], C.shape[0]))]])
# print(f'kkt_matrix = \n{KKT_matrix}')

# 构建KKT右端项
KKT_rhs = np.concatenate([np.zeros(H.shape[0]), d])
# print(f'KKT_rhs = {KKT_rhs}')

# 求解KKT系统
solution = np.linalg.solve(KKT_matrix, KKT_rhs)
# print(f'solution = {solution}')

# 初始化提取的控制输入和状态变量列表
u = []
x = []

# 遍历 solution，交替提取 u 和 x
for i in range(N - 1):
    u.append(solution[i * (n + m):(i * (n + m) + m)])
    x.append(solution[(i * (n + m) + m):(i + 1) * (n + m)])

# 转换为 numpy 数组并调整形状
u = np.array(u).reshape(N - 1, m)
x = np.array(x).reshape(N - 1, n)

# 还原初始状态
x = np.vstack([x0, x])
# print(f'u = \n{u}')
# print(f'x = \n{x}')

# 可视化初始猜测和优化后的控制输入
# t = np.linspace(0, (N - 1) * 0.1, N - 1)
control_t = np.linspace(0, (N - 1) * 0.1, N - 1)
state_t = np.linspace(0, N * h, N)

plt.figure(figsize=(12, 6))

# 控制输入对比图
plt.subplot(2, 1, 1)
plt.plot(control_t, u, label='Optimized Control Input (u_opt)', color='b', marker='o')
plt.xlabel('Time (s)')
plt.ylabel('Control Input (u)')
plt.title('QP Method')
plt.legend()
plt.grid(True)

# 状态轨迹图
plt.subplot(2, 1, 2)
plt.plot(state_t, x[:, 0], label='Position (q)', color='r', marker='o')
plt.plot(state_t, x[:, 1], label='Velocity (dq)', color='g', marker='o')
plt.xlabel('Time (s)')
plt.ylabel('State')
plt.title('State Trajectories')
plt.legend()
plt.grid(True)

# 显示图像
plt.tight_layout()
plt.show()
