import numpy as np
import matplotlib.pyplot as plt
from bsplineFun import BSplineUtilities
from plot import Plotter, color_dict
from frenet_ode import FrenetODE

# B-spline曲线的参数
k = 3
t = [0, 0, 0, 0, 1, 1, 1, 1]
c = [[-10, 0], [-10, 15], [10, 15], [10, 0]]

# 创建BSplineUtilities类的实例
bspline = BSplineUtilities(k, t, c)

# 初始化车辆状态和控制输入
CartesianState = np.asarray([8, 10, 0.4, 5])
Control = np.asarray([1.5, 1.5])
v_init = 10
dt = 0.2

# 创建绘图器实例
fig, ax = plt.subplots()
plotter = Plotter(ax, bspline)
plotter.plot_curve()

# 将当前笛卡尔状态转换为Frenet状态
init_frenet_state = bspline.cartesian_to_frenet(CartesianState)

# 初始化FrenetODE实例
frenet_ode = FrenetODE(lambda s: bspline.kappaFromS(s), v_init, dt)
print('total s= ', bspline.total_s())
current_frenet_state = init_frenet_state

# frenet ode计算出来的下一步状态
frenet_ode_next_frenet_state = frenet_ode.rk4_step_frenet(current_frenet_state, Control)
plotter.draw_frenet_state_vehicle(current_frenet_state, 'step0', color_dict.get('light_green'))
print('Status calculated by frenet ode: ', frenet_ode_next_frenet_state)
# 将frenet ode计算出来的下一步状态重新转化为笛卡尔坐标系重新投影
project_next_cartesian_state = bspline.frenet_to_cartesian(frenet_ode_next_frenet_state)
plotter.draw_cartesian_state_vehicle(project_next_cartesian_state, 'step1', color_dict.get('deep_green'))
project_next_frenet_state = bspline.cartesian_to_frenet(project_next_cartesian_state)
print('Status calculated by projection: ', project_next_frenet_state)

plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()
