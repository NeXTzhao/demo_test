import autograd.numpy as np
import matplotlib.pyplot as plt


# 系统仿真函数
def simulate_system(x0, u, h, N):
    x = np.zeros((2, N + 1))
    x[:, 0] = x0
    for k in range(N):
        A = np.array([[1, h], [0, 1]])
        B = np.array([[0.5 * h ** 2], [h]])
        x[:, k + 1] = A @ x[:, k] + B.flatten() * u[k]
    return x


# 计算目标函数值
def cost(x, u, xd, Q, R, Qn):
    terminal_cost = 0.5 * (x[:, -1] - xd).T @ Qn @ (x[:, -1] - xd)
    running_cost = 0.5 * np.sum([(x[:, k] - xd).T @ Q @ (x[:, k] - xd) + u[k] ** 2 * R for k in range(len(u))])
    return terminal_cost + running_cost


# 向前步骤：模拟系统状态
def rollout(x0, u, h, N):
    return simulate_system(x0, u, h, N)


# 反向步骤：计算协状态和控制输入增量
def backward_pass(x, u, h, N, Q, R, Qn):
    λ = np.zeros((2, N + 1))
    Δu = np.zeros(N)
    λ[:, N] = Qn @ (x[:, N] - xd)
    for k in range(N - 1, -1, -1):
        A = np.array([[1, h], [0, 1]])
        B = np.array([[0.5 * h ** 2], [h]])
        Δu[k] = -(u[k] + (B.T @ λ[:, k + 1])[0] / R)  # 确保标量运算
        λ[:, k] = Q @ (x[:, k] - xd) + A.T @ λ[:, k + 1]
    return λ, Δu


# 牛顿法优化
def optimize_control_newton(x0, xd, h, N, Q, R, Qn, max_iter=100, tol=1e-3, line_search_tol=1e-2):
    u = np.zeros(N)  # 初始猜测
    x = rollout(x0, u, h, N)  # 初始状态轨迹
    for _ in range(max_iter):
        # 反向步骤
        λ, Δu = backward_pass(x, u, h, N, Q, R, Qn)
        # 线搜索与前向步骤
        α = 0.8
        while True:
            u_new = u + α * Δu
            x_new = rollout(x0, u_new, h, N)
            cost_new = cost(x_new, u_new, xd, Q, R, Qn)
            cost_current = cost(x, u, xd, Q, R, Qn)
            if cost_new <= cost_current - line_search_tol * α * np.dot(Δu, Δu):
                break
            α *= 0.5  # 回溯
        # 更新
        u = u_new
        x = x_new
        if np.max(np.abs(Δu)) < tol:
            break
    return u, x


# 参数
n = 2
m = 1
h = 0.1
Tfinal = 10.0
N = int(Tfinal / h + 1)
x0 = np.array([10.0, -100])
xd = np.array([0, 0])
Q = np.eye(n) * 10
R = np.eye(m) * 0.1
Qn = np.eye(n) * 10

# 求解最优控制输入
u_opt, x_opt = optimize_control_newton(x0, xd, h, N, Q, R, Qn)

u_initial = np.zeros(N)  # 初始猜测

t = np.linspace(0, N * h, N)

plt.figure(figsize=(12, 6))

# 控制输入对比图
plt.subplot(1, 2, 1)
plt.plot(t, u_initial, label='Initial Control Input (u_initial)', linestyle='--', color='gray')
plt.plot(t, u_opt, label='Optimized Control Input (u_opt)', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Control Input (u)')
plt.title('Initial vs Optimized Control Input')
plt.legend()
plt.grid(True)

# 状态轨迹图
plt.subplot(1, 2, 2)
t_full = np.linspace(0, N * h, N + 1)
plt.plot(t_full, x_opt[0, :], label='Position (q)', color='r')
plt.plot(t_full, x_opt[1, :], label='Velocity (dq)', color='g')
plt.xlabel('Time (s)')
plt.ylabel('State')
plt.title('State Trajectories')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
# plt.savefig('result.png')
