import autograd.numpy as np  # 使用 autograd.numpy 替代 numpy
import matplotlib.pyplot as plt
from autograd import grad

# 参数设置
g = 9.81  # 重力加速度
L = 1.0  # 摆长
dt = 0.1  # 时间步长
t_max = 10  # 模拟时间
n_steps = int(t_max / dt)

# 初始化数组
theta_fixed = np.zeros(n_steps)
omega_fixed = np.zeros(n_steps)
theta_newton = np.zeros(n_steps)
omega_newton = np.zeros(n_steps)
t = np.linspace(0, t_max, n_steps)

# 初始条件
theta_fixed[0] = 0.1  # 初始角度（弧度）
omega_fixed[0] = 0.0  # 初始角速度
theta_newton[0] = 0.1  # 初始角度（弧度）
omega_newton[0] = 0.0  # 初始角速度


# 定义隐式方程
def implicit_eq(theta_new, theta_old, omega_old):
    return theta_new - theta_old - dt * (omega_old - dt * g / L * np.sin(theta_new))


# 自动求导
implicit_eq_grad = grad(implicit_eq, 0)  # 对 theta_new 求导

# 初始化迭代次数记录
fixed_point_iterations = []
newton_iterations = []

for n in range(1, n_steps):
    print(f"\nTime Step {n}:")

    # 固定点迭代法
    theta_new_fixed = theta_fixed[n - 1]
    iteration_count_fixed = 1
    print(f"  Fixed Point Iterations:")
    while True:
        theta_next_fixed = theta_fixed[n - 1] + dt * (omega_fixed[n - 1] - dt * g / L * np.sin(theta_new_fixed))
        error_fixed_point = abs(theta_next_fixed - theta_new_fixed)
        print(f"    Iteration {iteration_count_fixed}: θ_new = {theta_new_fixed}, error = {error_fixed_point}")
        if error_fixed_point < 1e-8:
            break
        theta_new_fixed = theta_next_fixed
        iteration_count_fixed += 1
    fixed_point_iterations.append(iteration_count_fixed)

    omega_fixed[n] = omega_fixed[n - 1] - dt * g / L * np.sin(theta_new_fixed)
    theta_fixed[n] = theta_fixed[n - 1] + dt * omega_fixed[n]

    # 牛顿法迭代
    theta_new_newton = theta_newton[n - 1]
    error_newton = 1.0  # 初始误差设置为一个较大的值
    iteration_count_newton = 1
    print(f"  Newton Iterations:")
    while error_newton > 1e-8:
        f = implicit_eq(theta_new_newton, theta_newton[n - 1], omega_newton[n - 1])
        f_prime = implicit_eq_grad(theta_new_newton, theta_newton[n - 1], omega_newton[n - 1])
        theta_next_newton = theta_new_newton - f / f_prime
        error_newton = abs(theta_next_newton - theta_new_newton)
        print(
            f"    Iteration {iteration_count_newton}: θ_new = {theta_new_newton}, f = {f}, f_prime = {f_prime}, error = {error_newton}")
        theta_new_newton = theta_next_newton
        iteration_count_newton += 1
    newton_iterations.append(iteration_count_newton)

    omega_newton[n] = omega_newton[n - 1] - dt * g / L * np.sin(theta_new_newton)
    theta_newton[n] = theta_newton[n - 1] + dt * omega_newton[n]

# 绘制迭代次数的对比
plt.figure(figsize=(10, 6))
plt.plot(t[1:], fixed_point_iterations, marker='o', linestyle='-', label='Fixed Point Iteration Count')
plt.plot(t[1:], newton_iterations, marker='o', linestyle='-', label='Newton Iteration Count')
plt.xlabel('Time')
plt.ylabel('Iteration Count')
plt.title('Iteration Count Comparison Between Fixed Point and Newton Methods')
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig('fix_point_newton.png')