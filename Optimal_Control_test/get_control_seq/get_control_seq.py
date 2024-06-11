import numpy as np
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt

# 定义圆弧轨迹 (x, y, theta)
r = 5  # 半径
theta_max = np.pi / 2  # 最大角度 (90度)
num_points = 50  # 点的数量

angles = np.linspace(0, theta_max, num_points)
trajectory = np.zeros((num_points, 3))
trajectory[:, 0] = r * np.cos(angles)  # x坐标
trajectory[:, 1] = r * np.sin(angles)  # y坐标
trajectory[:, 2] = angles  # theta

# 给定的参数
v = 10.0  # 速度
L = 2.8  # 前后轴之间的距离
L_front_axle_to_front_bumper = 0.95  # 前轴到前保险杠的距离

# 状态空间模型矩阵
A = np.array([
    [0, v, -(L + L_front_axle_to_front_bumper) / L, v],
    [0, 0, 0, -v / L],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])

B = np.array([
    [0],
    [0],
    [0],
    [1]
])

# 代价函数的权重矩阵
Q = np.diag([10, 10, 1, 1])  # 状态代价矩阵
R = np.array([[1]])  # 控制输入代价矩阵

# 求解离散代数Riccati方程
P = solve_discrete_are(A, B, Q, R)

# 计算LQR增益矩阵
K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

# 初始化状态
state = np.array([5, 0, 0, 0])  # 初始状态 [E_lat, E_theta, steering, steering']

# 存储实际轨迹
actual_x = [0]
actual_y = [0]
actual_theta = [0]


# 定义Runge-Kutta方法
def runge_kutta_update(x,theta, u, v, L, delta_time):
    def f(x, u):
        dxdt = np.zeros_like(x)
        dxdt[0] = v * np.cos(theta)
        dxdt[1] = v * np.sin(theta)
        dxdt[2] = (v / L) * x[3]
        dxdt[3] = u
        return dxdt

    k1 = f(x, u)
    k2 = f(x + 0.5 * delta_time * k1, u)
    k3 = f(x + 0.5 * delta_time * k2, u)
    k4 = f(x + delta_time * k3, u)
    new_state = x + (delta_time / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return new_state


# 轨迹跟踪
for point in trajectory:
    # 计算误差
    E_lat = (point[0] - actual_x[-1]) * np.cos(actual_theta[-1]) + (point[1] - actual_y[-1]) * np.sin(actual_theta[-1])
    E_theta = point[2] - actual_theta[-1]
    error = np.array([E_lat, E_theta, state[2], state[3]])

    u = -K @ error  # 计算控制输入

    # 使用Runge-Kutta方法更新实际状态
    delta_time = 0.1  # 时间步长
    theta = point[2]
    state = runge_kutta_update(state,theta, u[0], v, L, delta_time)
    print(f'state = {state}')

    # 更新实际位置和角度
    new_x = actual_x[-1] + v * np.cos(actual_theta[-1]) * delta_time
    new_y = actual_y[-1] + v * np.sin(actual_theta[-1]) * delta_time
    new_theta = actual_theta[-1] + (v / L) * state[2] * delta_time

    actual_x.append(new_x)
    actual_y.append(new_y)
    actual_theta.append(new_theta)

# 可视化轨迹和跟踪结果
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(trajectory[:, 0], trajectory[:, 1], 'r.', label='Reference Trajectory')
plt.plot(actual_x, actual_y, 'b-', label='Tracked Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Trajectory Tracking with LQR')

plt.subplot(1, 2, 2)
plt.plot(range(len(trajectory[:, 2])), trajectory[:, 2], 'r--', label='Reference Theta')
plt.plot(range(len(actual_theta)), actual_theta, 'b-', label='Tracked Theta')
plt.xlabel('Time Step')
plt.ylabel('Theta')
plt.legend()
plt.title('Theta Tracking with LQR')

plt.show()
