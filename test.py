import numpy as np
import matplotlib.pyplot as plt

from bsplineFun import BSplineUtilities
from plot import Plotter, color_dict
from frenet_ode import FrenetODE


# test bspline
def test_bspline():
    # B-spline曲线的参数
    k = 3
    t = [0, 0, 0, 0, 1, 1, 1, 1]
    c = [[-10, 0], [-10, 15], [10, 15], [10, 0]]

    # 创建BSplineUtilities类的实例
    bspline = BSplineUtilities(k, t, c)

    # 测试笛卡尔到Frenet的转换
    CartesianState = np.asarray([8, 13, 0.1])
    print("Row CartesianState: ", CartesianState)
    s, n, alpha = bspline.cartesian_to_frenet(CartesianState)
    print(f"Frenet Coordinates: s = {s}, n = {n}, alpha = {alpha}")
    # 测试Frenet到笛卡尔的转换
    x, y, phi = bspline.frenet_to_cartesian([s, n, alpha])
    print(f"Cartesian Coordinates: x = {x}, y = {y}, phi = {phi}")


def test_frenetODE():
    # B-spline曲线的参数
    k = 3
    t = [0, 0, 0, 0, 1, 1, 1, 1]
    c = [[-10, 0], [-10, 15], [10, 15], [10, 0]]

    # 创建BSplineUtilities类的实例
    bspline = BSplineUtilities(k, t, c)

    CartesianState = np.asarray([8, 13, 0.8])
    Control = np.asarray([5, 0.1])

    v_init = 10
    dt = 0.1
    fig, ax = plt.subplots()
    plotter = Plotter(ax, bspline)

    # 将当前笛卡尔状态转换为Frenet状态
    init_frenet_state = bspline.cartesian_to_frenet(CartesianState)

    # 使用动力学模型计算下一个Frenet状态
    frenet_ode = FrenetODE(lambda s: bspline.kappaFromS(s), v_init)
    next_frenet_state = frenet_ode.rk4_step_frenet(init_frenet_state, Control, dt)

    # 绘制B-spline曲线
    plotter.plot_curve()

    # 绘制当前状态的车辆
    plotter.draw_frenet_state_vehicle(init_frenet_state, 'init_frenet_state', color_dict.get('light_blue'))
    plotter.draw_frenet_state_vehicle(next_frenet_state, 'next_frenet_state', color_dict.get('deep_blue'))

    plt.axis('equal')
    plt.legend()
    plt.grid('true')
    plt.show()


test_bspline()
test_frenetODE()
