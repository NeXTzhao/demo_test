import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

# 系统参数
A = np.array([[1, 0.1], [0, 1]])
B = np.array([[0.005], [0.1]])
Q = np.eye(2)
R = np.eye(1) * 0.1
Qn = np.eye(2) * 10
x0 = np.array([0, 0])
N = 50
n = A.shape[0]
m = B.shape[1]

# 构建权重矩阵 H
H = block_diag(R, *[block_diag(Q, R) for _ in range(N - 2)], Qn)

# 构建Kronecker积部分
I_N_minus_1 = np.eye(N-1)
block_B_I = np.hstack([B, -np.eye(n)])
C = np.kron(I_N_minus_1, block_B_I)

# 将A填入C的特定位置
for k in range(1, N-1):
    C[k*n:(k+1)*n, (k*(n+m)-n):(k*(n+m))] = A

# 构建向量d
d = np.concatenate([-A @ x0, np.zeros(C.shape[0] - n)])

# 构建KKT矩阵
KKT_matrix = np.block([
    [H, C.T],
    [C, np.zeros((C.shape[0], C.shape[0]))]
])

# 构建KKT右端项
KKT_rhs = np.concatenate([np.zeros(H.shape[0]), d])

# 求解KKT系统
solution = np.linalg.solve(KKT_matrix, KKT_rhs)
z = solution[:H.shape[0]]
print(f'z={z}')

# 提取状态和控制输入
u = z[:(N - 1) * m].reshape(N - 1, m)
print(f'u={u}')
x = z[(N - 1) * m:].reshape(N - 1, n)
print(f'x={x}')
# 还原初始状态
x = np.vstack([x0, x])

# 可视化初始猜测和优化后的控制输入
t = np.linspace(0, (N - 1) * 0.1, N - 1)

plt.figure(figsize=(12, 6))

# 控制输入对比图
plt.subplot(2, 1, 1)
plt.plot(t, u, label='Optimized Control Input (u)', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Control Input (u)')
plt.title('Optimized Control Input')
plt.legend()
plt.grid(True)

# 状态轨迹图
plt.subplot(2, 1, 2)
t_full = np.linspace(0, (N - 1) * 0.1, N)
plt.plot(t_full, x[:, 0], label='Position (q)', color='r')
plt.plot(t_full, x[:, 1], label='Velocity (dq)', color='g')
plt.xlabel('Time (s)')
plt.ylabel('State')
plt.title('State Trajectories')
plt.legend()
plt.grid(True)

# 显示图像
plt.tight_layout()
plt.show()
