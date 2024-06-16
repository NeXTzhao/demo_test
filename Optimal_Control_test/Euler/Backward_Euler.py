import numpy as np
import matplotlib.pyplot as plt

# 参数设置
g = 9.81  # 重力加速度
L = 1.0   # 摆长
dt = 0.1  # 时间步长
t_max = 10 # 模拟时间
n_steps = int(t_max / dt)

# 初始化数组
theta_forward = np.zeros(n_steps)
omega_forward = np.zeros(n_steps)
theta_backward = np.zeros(n_steps)
omega_backward = np.zeros(n_steps)
t = np.linspace(0, t_max, n_steps)

# 初始条件
theta_forward[0] = 0.1  # 初始角度（弧度）
omega_forward[0] = 0.0  # 初始角速度
theta_backward[0] = 0.1  # 初始角度（弧度）
omega_backward[0] = 0.0  # 初始角速度

# 前向欧拉法数值积分
for n in range(1, n_steps):
    omega_forward[n] = omega_forward[n-1] - dt * g / L * np.sin(theta_forward[n-1])
    theta_forward[n] = theta_forward[n-1] + dt * omega_forward[n-1]

# 后向欧拉法数值积分
for n in range(1, n_steps):
    # 迭代求解隐式方程
    theta_new = theta_backward[n-1]
    for _ in range(10):  # 迭代次数，可以增加以提高精度
        theta_new = theta_backward[n-1] + dt * (omega_backward[n-1] - dt * g / L * np.sin(theta_new))
    omega_backward[n] = omega_backward[n-1] - dt * g / L * np.sin(theta_new)
    theta_backward[n] = theta_backward[n-1] + dt * omega_backward[n]

# 计算能量（近似）
energy_forward = 0.5 * (L**2) * omega_forward**2 + g * L * (1 - np.cos(theta_forward))
energy_backward = 0.5 * (L**2) * omega_backward**2 + g * L * (1 - np.cos(theta_backward))

# 绘制结果
plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(t, theta_forward, label='θ (Forward Euler)')
plt.plot(t, theta_backward, label='θ (Backward Euler)')
plt.xlabel('Time')
plt.ylabel('θ (angle)')
plt.title('Angle with Forward and Backward Euler Methods')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, omega_forward, label='ω (Forward Euler)')
plt.plot(t, omega_backward, label='ω (Backward Euler)')
plt.xlabel('Time')
plt.ylabel('ω (angular velocity)')
plt.title('Angular Velocity with Forward and Backward Euler Methods')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, energy_forward, label='Energy (Forward Euler)')
plt.plot(t, energy_backward, label='Energy (Backward Euler)')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Energy with Forward and Backward Euler Methods')
plt.legend()

plt.tight_layout()
plt.show()
