import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import time

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

# 噪声参数
noise_mean = 0
noise_std_dev = 0.1


# QP方法
def qp_method():
    # 构建权重矩阵 H
    H = block_diag(R, *[block_diag(Q, R) for _ in range(N - 2)], Qn)

    # 构建Kronecker积部分
    I_N_minus_1 = np.eye(N - 1)
    block_B_I = np.hstack([B, -np.eye(n)])
    C = np.kron(I_N_minus_1, block_B_I)

    # 将A填入C的特定位置
    for k in range(1, N - 1):
        C[k * n:(k + 1) * n, (k * (n + m) - n):(k * (n + m))] = A

    # 构建向量d
    d = np.concatenate([-A @ x0, np.zeros(C.shape[0] - n)])

    # 构建KKT矩阵
    KKT_matrix = np.block([[H, C.T], [C, np.zeros((C.shape[0], C.shape[0]))]])

    # 构建KKT右端项
    KKT_rhs = np.concatenate([np.zeros(H.shape[0]), d])

    # 求解KKT系统
    solution = np.linalg.solve(KKT_matrix, KKT_rhs)

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

    return u, x


# Shooting Method
def shooting_method(max_iter=100, tol=1e-3, line_search_tol=1e-2):
    # 仿真系统函数
    def simulate_system(x0, u):
        x = np.zeros((2, N))
        x[:, 0] = x0
        for k in range(N - 1):
            x[:, k + 1] = A @ x[:, k] + B.flatten() * u[k]
        return x

    # 计算目标函数值
    def cost(x, u):
        terminal_cost = 0.5 * (x[:, -1] - xd).T @ Qn @ (x[:, -1] - xd)
        running_cost = 0.5 * np.sum([(x[:, k] - xd).T @ Q @ (x[:, k] - xd) + u[k] ** 2 * R for k in range(len(u))])
        return terminal_cost + running_cost

    # 反向步骤：计算协状态和控制输入增量
    def backward_pass(x, u):
        λ = np.zeros((2, N))
        Δu = np.zeros(N - 1)
        λ[:, N - 1] = Qn @ (x[:, N - 1] - xd)
        for k in range(N - 2, -1, -1):
            Δu[k] = -(u[k] + (B.T @ λ[:, k + 1])[0] / R)
            λ[:, k] = Q @ (x[:, k] - xd) + A.T @ λ[:, k + 1]
        return λ, Δu

    # 初始化控制输入和状态轨迹
    u = np.zeros(N - 1)  # 初始猜测
    x = simulate_system(x0, u)  # 初始状态轨迹

    for _ in range(max_iter):
        λ, Δu = backward_pass(x, u)
        α = 0.8
        while True:
            u_new = u + α * Δu
            x_new = simulate_system(x0, u_new)
            cost_new = cost(x_new, u_new)
            cost_current = cost(x, u)
            if cost_new <= cost_current - line_search_tol * α * np.dot(Δu, Δu):
                break
            α *= 0.5  # 回溯
        u = u_new
        x = x_new
        if np.max(np.abs(Δu)) < tol:
            break

    return u, x


# Riccati Recursion Method
# 有限时域
def finite_time_domain_riccati_method():
    P = [None] * N
    K = [None] * (N - 1)

    P[-1] = Qn

    # 反向递推计算P和K矩阵
    for i in range(N - 2, -1, -1):
        P_i = P[i + 1]
        K_i = np.linalg.inv(R + B.T @ P_i @ B) @ (B.T @ P_i @ A)
        P[i] = Q + A.T @ P_i @ (A - B @ K_i)
        K[i] = K_i
        # print(f'k = {K_i}')

    x = np.zeros((n, N))
    u = np.zeros((m, N - 1))
    x[:, 0] = x0

    # 正向递推计算状态轨迹和控制输入序列
    for i in range(N - 1):
        u[:, i] = -K[i] @ x[:, i]  # 计算当前步的控制输入
        x[:, i + 1] = A @ x[:, i] + B @ u[:, i]  # 计算下一步的状态

    return u, x


# 无限时域
def infinite_horizon_riccati_method(tolerance=1e-9, max_iterations=1000):
    def iterative_riccati(tolerance, max_iterations):
        # 初始化P矩阵
        P = Q
        P_old = np.zeros_like(Q)
        iteration = 0

        while np.max(np.abs(P - P_old)) > tolerance and iteration < max_iterations:
            P_old = P
            K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
            P = Q + A.T @ P @ (A - B @ K)
            iteration += 1

        return P

    # 使用迭代方法求解黎卡提方程
    P = iterative_riccati(tolerance, max_iterations)

    # 计算反馈增益矩阵K
    K = np.linalg.inv(R) @ B.T @ P

    # 初始化状态和控制输入序列
    n = A.shape[0]
    m = B.shape[1]
    x = np.zeros((n, N))
    u = np.zeros((m, N - 1))
    x[:, 0] = x0

    # 正向递推计算状态轨迹和控制输入序列
    for i in range(N - 1):
        u[:, i] = -K @ x[:, i]  # 计算当前步的控制输入
        # noise = np.random.normal(noise_mean, noise_std_dev, n)
        # x[:, i + 1] = A @ x[:, i] + B @ u[:, i] + noise  # 计算下一步的状态
        x[:, i + 1] = A @ x[:, i] + B @ u[:, i]
    return u, x


# 运行各个方法
u_qp, x_qp = qp_method()
u_shooting, x_shooting = shooting_method()
u_riccati_finite, x_riccati_finite = finite_time_domain_riccati_method()
u_riccati_infinite, x_riccati_infinite = infinite_horizon_riccati_method()

# 可视化结果对比
t = np.linspace(0, (N - 1) * 0.1, N - 1)
t_full = np.linspace(0, N * 0.1, N)

plt.figure(figsize=(14, 12))

# 控制输入对比图
plt.subplot(3, 1, 1)
plt.plot(t, u_qp, label='QP Method', color='navy', marker='o', linestyle='-', linewidth=2, markersize=6)
plt.plot(t, u_shooting, label='Shooting Method', color='darkorange', marker='s', linestyle='--', linewidth=2,
         markersize=6)
plt.plot(t, u_riccati_finite.T, label='Finite Horizon Riccati Method', color='green', marker='^', linestyle='-',
         linewidth=2, markersize=6)
plt.plot(t, u_riccati_infinite.T, label='Infinite Horizon Riccati Method', color='red', marker='x', linestyle='-',
         linewidth=2, markersize=6)
plt.xlabel('Time (s)')
plt.ylabel('Control Input (u)')
plt.title('Control Input Comparison')
plt.legend(fontsize=12)
plt.grid(True)

# 状态轨迹对比图 - 位置 (q)
plt.subplot(3, 1, 2)
plt.plot(t_full, x_qp[:, 0], label='QP Method Position (q)', color='teal', marker='o', linestyle='-', linewidth=2,
         markersize=6)
plt.plot(t_full, x_shooting[0, :], label='Shooting Method Position (q)', color='crimson', marker='s', linestyle='--',
         linewidth=2, markersize=6)
plt.plot(t_full, x_riccati_finite[0, :], label='Finite Horizon Riccati Method Position (q)', color='magenta',
         marker='*', linestyle='-',
         linewidth=2, markersize=6)
plt.plot(t_full, x_riccati_infinite[0, :], label='Infinite Horizon Riccati Method Position (q)', color='blue',
         marker='x', linestyle='-',
         linewidth=2, markersize=6)
plt.xlabel('Time (s)')
plt.ylabel('Position (q)')
plt.title('Position Trajectory Comparison')
plt.legend(fontsize=12)
plt.grid(True)

# 状态轨迹对比图 - 速度 (dq)
plt.subplot(3, 1, 3)
plt.plot(t_full, x_qp[:, 1], label='QP Method Velocity (dq)', color='forestgreen', marker='o', linestyle='-',
         linewidth=2, markersize=6)
plt.plot(t_full, x_shooting[1, :], label='Shooting Method Velocity (dq)', color='royalblue', marker='s', linestyle='--',
         linewidth=2, markersize=6)
plt.plot(t_full, x_riccati_finite[1, :], label='Finite Horizon Riccati Method Velocity (dq)', color='purple',
         marker='^', linestyle='-',
         linewidth=2, markersize=6)
plt.plot(t_full, x_riccati_infinite[1, :], label='Infinite Horizon Riccati Method Velocity (dq)', color='black',
         marker='x', linestyle='-',
         linewidth=2, markersize=6)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (dq)')
plt.title('Velocity Trajectory Comparison')
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
# plt.savefig('result.png')

# 运行多次以获取时间分布
qp_times = []
shooting_times = []
riccati_finite_times = []
riccati_infinite_times = []

for _ in range(100):
    start_time = time.time()
    u_qp, x_qp = qp_method()
    qp_times.append(time.time() - start_time)

    start_time = time.time()
    u_shooting, x_shooting = shooting_method()
    shooting_times.append(time.time() - start_time)

    start_time = time.time()
    u_riccati_finite, x_riccati_finite = finite_time_domain_riccati_method()
    riccati_finite_times.append(time.time() - start_time)

    start_time = time.time()
    u_riccati_infinite, x_riccati_infinite = infinite_horizon_riccati_method()
    riccati_infinite_times.append(time.time() - start_time)

plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
plt.boxplot([shooting_times, qp_times, riccati_finite_times, riccati_infinite_times],
            labels=['Shooting Method', 'QP Method', 'Finite Riccati Method', 'Infinite Riccati Method'],
            patch_artist=True,
            boxprops=dict(facecolor=colors[0], color=colors[0]),
            whiskerprops=dict(color=colors[0]),
            capprops=dict(color=colors[0]),
            medianprops=dict(color='black'))

plt.xlabel('Methods', fontsize=14)
plt.ylabel('Time (seconds)', fontsize=14)
plt.title('Method Time Comparison', fontsize=16)
plt.grid(True)

# 添加数据点
for i, method_times in enumerate([qp_times, shooting_times, riccati_finite_times, riccati_infinite_times]):
    x = np.random.normal(i + 1, 0.04, size=len(method_times))
    plt.plot(x, method_times, 'r.', alpha=0.8)

plt.tight_layout()
plt.show()
# plt.savefig('time.png')
