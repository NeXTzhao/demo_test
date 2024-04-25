import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

color_dict = {
    'light_blue': (0.678, 0.847, 0.902, 1.0),  # 浅蓝色
    'deep_blue': (0.0, 0.0, 0.545, 1.0),  # 深蓝色
    'light_green': (0.564, 0.933, 0.564, 0.5),  # 浅绿色，透明度调至0.5
    'deep_green': (0.0, 0.392, 0.0, 1.0),  # 深绿色
    'light_red': (1.0, 0.714, 0.757, 0.5),  # 浅红色，透明度调至0.5
    'deep_red': (0.545, 0.0, 0.0, 1.0),  # 深红色
}


class Plotter:
    def __init__(self, ax, bspline_util, length=4.5, width=1.8):
        """
        参数:
        - ax: matplotlib的坐标轴。
        - bspline_util: BSplineUtilities的实例，用于计算和绘制B-spline曲线。
        - length: 车辆的长度，默认值为4.5米。
        - width: 车辆的宽度，默认值为1.8米。
        """
        self._ax = ax
        self.bspline_util = bspline_util
        self._length = length
        self._width = width

    def plot_curve(self):
        """
        绘制B-spline曲线和表示车辆的矩形。

        参数:
        - cartesian_state: (x, y, phi)。
        """
        # 绘制曲线
        t_values = np.linspace(self.bspline_util.t[0], self.bspline_util.t[-1], 400)
        curve_points = self.bspline_util.get_curve_points(t_values)
        self._ax.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='B-spline Curve')

    def plot_point_to_curve(self, cartesian_state):
        """
        绘制车辆中心到曲线最近点的连接线

        参数:
        - cartesian_state: (x, y, phi)。
        """
        x, y, phi, v = cartesian_state
        optimized_t, closest_point = self.bspline_util.closest_point_and_t(np.asarray([x, y]))
        self._ax.plot(x, y, 'ro')
        self._ax.plot(closest_point[0], closest_point[1], 'rx')
        self._ax.plot([x, closest_point[0]], [y, closest_point[1]], 'g--')

    def draw_cartesian_state_vehicle(self, carPose, name, color):
        self.draw_vehicle(carPose, name, color)
        self.plot_point_to_curve(carPose)

    def draw_frenet_state_vehicle(self, carPose, name, color):
        carPose = self.bspline_util.frenet_to_cartesian(carPose)
        self.draw_vehicle(carPose, name, color)
        self.plot_point_to_curve(carPose)

    def draw_vehicle(self, carPose, name, color):
        """
        在指定的坐标轴上绘制一个矩形表示车辆。
        """
        center = np.asarray([carPose[0], carPose[1]])
        heading = carPose[2]
        corners = np.array([
            [-self._length / 2, self._width / 2],  # 左上角
            [self._length / 2, self._width / 2],  # 右上角
            [self._length / 2, -self._width / 2],  # 右下角
            [-self._length / 2, -self._width / 2]  # 左下角
        ])
        # 旋转角点
        rotation_matrix = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)]
        ])
        # 平移到正确的位置
        rotated_corners = np.dot(corners, rotation_matrix)
        translated_corners = rotated_corners + center
        # 绘制矩形
        polygon = patches.Polygon(translated_corners, closed=True, fill=False, color=color, label=name)
        self._ax.add_patch(polygon)
