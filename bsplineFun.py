from matplotlib import pyplot as plt
from scipy.interpolate import BSpline
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.optimize import minimize
import numpy as np


class BSplineUtilities:
    def __init__(self, k, t, c):
        """
        初始化B-spline曲线。

        参数:
        - k: B-spline的次数。
        - t: 节点向量，numpy数组形式。
        - c: 控制点，numpy数组形式，每个控制点是一个二维坐标。
        """
        self.k = k  # B-spline的次数
        self.t = np.array(t)  # 节点向量
        self.c = np.array(c)  # 控制点
        self.bspline = BSpline(t, c, k)  # 创建B-spline曲线对象

    def total_s(self):
        """

        返回:
        - 总弧长，标量。
        """
        return self.arc_length(1)

    def speed(self, t):
        """
        计算曲线在给定参数t处的速度。

        参数:
        - t: 曲线参数，标量。

        返回:
        - 曲线在t处的速度，标量。
        """
        der = self.bspline.derivative()(t)  # 计算导数
        return np.sqrt((der ** 2).sum())

    def arc_length(self, t1):
        """
        通过数值积分计算曲线从起点到给定参数t1处的弧长。

        参数:
        - t1: 曲线参数，标量。

        返回:
        - 弧长，标量。
        """
        return quad(self.speed, 0, t1)[0]

    def find_t_for_given_s(self, s, t_bounds=(0, 1)):
        """
        给定弧长s，找到对应的曲线参数t。

        参数:
        - s: 弧长，标量。
        - t_bounds: 参数t的搜索范围，元组形式。

        返回:
        - 对应的曲线参数t，标量。
        """
        return brentq(lambda t: self.arc_length(t) - s, *t_bounds)

    def tangle(self, s):
        """
        计算给定弧长s处曲线的切线角。

        参数:
        - s: 弧长，标量。

        返回:
        - 切线角，单位为弧度。
        """
        tValue = self.find_t_for_given_s(s)
        derivative = self.bspline.derivative()(tValue)
        return np.arctan2(derivative[1], derivative[0])

    def curvature(self, t):
        """
        计算曲线在给定参数t处的曲率。

        参数:
        - t: 曲线参数，标量。

        返回:
        - 曲率，标量。
        """
        first_der = self.bspline.derivative()(t)
        second_der = self.bspline.derivative(2)(t)
        return np.abs(np.cross(first_der, second_der)) / np.linalg.norm(first_der) ** 3

    def kappaFromS(self, s):
        """
        给定弧长s，计算对应点的曲率。

        参数:
        - s: 弧长，标量。

        返回:
        - 曲率，标量。
        """
        if s <= 0:
            print('弧长s错误，s = ', s)
            s = 0
        total_s = self.total_s()
        if s >= total_s:
            s = total_s
        return self.curvature(self.find_t_for_given_s(s))

    def closest_point_and_t(self, p0):
        """
        找到曲线上最接近给定点p0的点及其对应的曲线参数t。

        参数:
        - p0: 给定的点，numpy数组形式。

        返回:
        - 最优化的曲线参数t和最接近的点。
        """

        def objective_function(t, P0):
            # 目标函数：最小化点到曲线上一点的投影长度的平方
            R = self.bspline(t).flatten()  # 将结果转换为一维数组
            T = self.bspline.derivative()(t).flatten()  # 将结果转换为一维数组
            V = R - P0.flatten()  # 确保P0也是一维数组，向量差
            return np.dot(V, T) ** 2

        result = minimize(lambda t: objective_function(t, p0), x0=0.5, method='L-BFGS-B', bounds=[(0, 1)])
        if result.success:
            optimized_t = result.x[0]
            print("找到的参数t值:", optimized_t)
        else:
            print("优化失败:", result.message)
        optimized_t = result.x[0]
        closest_point = self.bspline(optimized_t)
        return optimized_t, closest_point

    def get_curve_points(self, t_values):
        """
        获取一系列t值对应的曲线上的点。

        参数:
        - t_values: 一个数组，包含要计算的曲线参数t的值。

        返回:
        - 一个数组，包含曲线上对应t值的点。
        """
        return np.array([self.bspline(t) for t in t_values])

    def cartesian_to_frenet(self, CartesianState):
        """
        将笛卡尔坐标转换为Frenet坐标，包括航向角的计算。

        参数:
        - point: 笛卡尔坐标系中的点 (x, y)，表示车辆当前的位置。
        - vehicle_heading: 车辆的航向角，单位为弧度，从x轴正方向到车辆当前方向的角度，顺时针为正。

        返回:
        - (s, n, theta): Frenet坐标系中的纵向坐标s、横向坐标n和航向角theta。
        """
        point = np.array([CartesianState[0], CartesianState[1]])
        vehicle_heading = CartesianState[2]
        velocity = CartesianState[3]
        # 找到曲线上最接近的点及其参数t
        optimized_t, closest_point = self.closest_point_and_t(point)

        # 计算弧长s，即从曲线起点到最接近点的距离
        s = self.arc_length(optimized_t)

        # 计算横向偏移n，即车辆位置到曲线最近点的垂直距离
        vector_to_curve = np.array(point) - np.array(closest_point)
        n = np.linalg.norm(vector_to_curve)

        # 计算曲线在t处的切线向量，并将其单位化
        tangent_vector = self.bspline.derivative()(optimized_t).flatten()
        tangent_vector /= np.linalg.norm(tangent_vector)

        # 计算切线角alpha，即曲线切线与x轴正方向之间的角度
        tangle = np.arctan2(tangent_vector[1], tangent_vector[0])
        alpha = vehicle_heading - tangle

        # 计算航向角alpha,即车辆航向与曲线切线方向之间的角度差
        # 这反映了车辆相对于路径的方向

        # 调整n的符号，以表示车辆相对于曲线的左侧或右侧
        # 通过计算法向量与车辆到曲线最近点向量的点积来判断
        normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])  # 通过旋转90度获得法向量
        if np.dot(vector_to_curve, normal_vector) < 0:
            n = -n  # 如果点积小于0，表示车辆在曲线的右侧，n取负值

        return s, n, alpha, velocity

    def frenet_to_cartesian(self, FrenetState):
        """
        将Frenet坐标转换回笛卡尔坐标，并计算曲线在该点的切线角。

        参数:
        - s: Frenet坐标系中的纵向坐标s，表示沿曲线的弧长。
        - n: Frenet坐标系中的横向坐标n，表示相对于曲线的垂直偏移。

        返回:
        - 笛卡尔坐标系中的点 (x, y) 和 切线角 tangle（弧度）。
        """
        s, n, alpha, velocity = FrenetState
        # 通过弧长s找到对应的曲线参数t
        t = self.find_t_for_given_s(s)

        # 计算曲线上的点，即Frenet坐标原点
        curve_point = self.bspline(t)

        # 计算曲线在t处的切线向量并单位化
        tangent_vector = self.bspline.derivative()(t).flatten()
        tangent_norm = np.linalg.norm(tangent_vector)
        tangent_vector /= tangent_norm

        # 计算法线向量（垂直于切线）
        normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])

        # 沿法线方向移动n单位得到笛卡尔坐标
        cartesian_point = curve_point + n * normal_vector

        # cartesian下的航向角
        tangle = np.arctan2(tangent_vector[1], tangent_vector[0])
        phi = tangle + alpha

        return cartesian_point[0], cartesian_point[1], phi, velocity

    def plot_point_to_curve(self, P0):
        """
        可视化点到曲线的最短距离

        参数:
        - point: 给定位置点
        """
        P0 = np.asarray(P0)
        optimized_t, closest_point = self.closest_point_and_t(P0)
        print("t: ", optimized_t)
        print("closest point: ", closest_point)
        t_values = np.linspace(0, 1, 50)
        curve_points = self.get_curve_points(t_values)  # 使用BSpline对象获取曲线点
        plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='B-spline Curve')
        plt.plot(self.c[:, 0], self.c[:, 1], 'ko', label='Control Points')  # 控制点
        plt.plot(self.c[:, 0], self.c[:, 1], 'k--', label='Control Polygon')  # 控制多边形
        plt.plot(P0[0], P0[1], 'ro', label='Given Point')
        plt.plot(closest_point[0], closest_point[1], 'rx', label='Closest Point')
        plt.plot([P0[0], closest_point[0]], [P0[1], closest_point[1]], 'g--')

        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('B-spline Curve with Closest Point to Given Point')
        plt.grid(True)
        plt.show()
