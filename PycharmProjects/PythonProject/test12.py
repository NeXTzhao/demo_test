import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


# ================== 非凸测试函数（Rosenbrock）==================
def F(x):
    """Rosenbrock函数，n=2时是非凸的经典测试函数"""
    return np.array([100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,  # f1: 标准Rosenbrock
                     0.1 * (x[0] - 2) ** 4 + 0.1 * (x[1] - 1) ** 2])  # f2: 自定义非凸函数


def grad_F(x):
    grad_f1 = np.array([
        -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0] ** 2)
    ])
    grad_f2 = np.array([
        0.4 * (x[0] - 2) ** 3,
        0.2 * (x[1] - 1)
    ])
    return np.vstack([grad_f1, grad_f2])


# ================== 优化算法核心 ==================
def compute_direction(x, grad_F):
    """带非凸约束的方向计算"""
    d = cp.Variable(len(x))
    constraints = [cp.norm(d) <= 5]  # 添加方向约束防止发散

    # 构建多目标优化问题
    objectives = [g @ d + 0.5 * cp.sum_squares(d) for g in grad_F]
    obj = cp.max(cp.hstack(objectives))  # 修正后的max表达式

    problem = cp.Problem(cp.Minimize(obj), constraints)
    problem.solve(solver=cp.ECOS)
    return d.value


class LineSearchBase:
    """线搜索基类"""

    def __init__(self, delta=1e-4, rho=0.7, mu=1.0):
        self.delta = delta
        self.rho = rho
        self.mu = mu

    def _armijo_condition(self, F_new, C, alpha, grad, d):
        return np.all(F_new <= C + self.delta * alpha * (grad @ d))


class AverageNonmonotoneLS(LineSearchBase):
    """非单调average-type线搜索"""

    def __init__(self, eta=0.85, **kwargs):
        super().__init__(**kwargs)
        self.eta = eta

    def initialize(self, F0):
        self.C = F0.copy()
        self.q = 1.0

    def update(self, F_new):
        self.q = self.eta * self.q + 1
        self.C = (self.eta * (self.q - 1) * self.C + F_new) / self.q


class MonotoneLS(LineSearchBase):
    """单调线搜索"""

    def initialize(self, F0):
        self.C = F0.copy()

    def update(self, F_new):
        self.C = F_new.copy()


# ================== 优化流程 ==================
def optimize(ls_type='nonmonotone', max_iter=100):
    np.random.seed(42)
    x = np.array([-1.5, 2.0])  # Rosenbrock的挑战性初始点

    # 初始化线搜索
    ls = AverageNonmonotoneLS() if ls_type == 'nonmonotone' else MonotoneLS()
    ls.initialize(F(x))

    # 记录优化轨迹
    history = {'x': [x.copy()], 'F': [F(x)]}

    for _ in range(max_iter):
        grad = grad_F(x)
        d = compute_direction(x, grad)

        if np.linalg.norm(d) < 1e-6:
            break

        # 回溯线搜索
        alpha = ls.mu
        for _ in range(30):
            x_new = x + alpha * d
            F_new = F(x_new)
            if ls._armijo_condition(F_new, ls.C, alpha, grad, d):
                break
            alpha *= ls.rho
        else:
            break

        # 更新状态
        x = x_new
        ls.update(F_new)
        history['x'].append(x.copy())
        history['F'].append(F_new)

    return history


# 修改可视化部分，增加原函数可视化
def plot_comparison(mono_hist, nmono_hist):
    plt.figure(figsize=(15, 6), dpi=100)

    # ================= 原函数3D可视化 =================
    ax1 = plt.subplot(131, projection='3d')

    # 生成网格数据
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z1 = 100 * (Y - X ** 2) ** 2 + (1 - X) ** 2  # f1
    Z2 = 0.1 * (X - 2) ** 4 + 0.1 * (Y - 1) ** 2  # f2

    # 绘制双目标函数曲面
    surf1 = ax1.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.6, label='f1')
    surf2 = ax1.plot_surface(X, Y, Z2, cmap='plasma', alpha=0.6, label='f2')

    # 添加颜色条
    plt.colorbar(surf1, ax=ax1, pad=0.1).set_label('f1 Value')
    plt.colorbar(surf2, ax=ax1, pad=0.2).set_label('f2 Value')

    ax1.set_title('Objective Functions 3D View', fontsize=12)
    ax1.set_xlabel('x1', fontsize=9)
    ax1.set_ylabel('x2', fontsize=9)
    ax1.set_zlabel('f(x)', fontsize=9)
    ax1.view_init(elev=30, azim=-45)

    # ================= 优化路径投影 =================
    ax2 = plt.subplot(132)

    # 绘制双目标函数等高线
    levels = np.logspace(0, 5, 20)
    CS1 = ax2.contour(X, Y, Z1, levels=levels, cmap='viridis', alpha=0.6)
    CS2 = ax2.contour(X, Y, Z2, levels=20, cmap='plasma', alpha=0.6)

    # 绘制优化路径
    for label, hist, color in zip(['Monotone', 'Nonmonotone'],
                                  [mono_hist, nmono_hist],
                                  ['blue', 'red']):
        path = np.array(hist['x'])
        ax2.plot(path[:, 0], path[:, 1], 'o-', markersize=5, linewidth=1.5,
                 color=color, label=label, markevery=3)
        ax2.scatter(path[-1, 0], path[-1, 1], s=80, edgecolor=color,
                    facecolor='white', zorder=3)

    ax2.set_title('Optimization Path Projection', fontsize=12)
    ax2.set_xlabel('x1', fontsize=10)
    ax2.set_ylabel('x2', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(fontsize=9)

    # ================= 收敛曲线 =================
    ax3 = plt.subplot(133)

    for label, hist, color in zip(['Monotone', 'Nonmonotone'],
                                  [mono_hist, nmono_hist],
                                  ['#1f77b4', '#d62728']):
        F_values = np.array(hist['F'])
        ax3.plot(F_values[:, 0], 's-', color=color, markevery=5,
                 markersize=6, linewidth=1.5, label=f'{label} F1')
        ax3.plot(F_values[:, 1], 'o--', color=color, markevery=5,
                 markersize=6, linewidth=1.5, label=f'{label} F2')

    ax3.set_title('Objective Convergence', fontsize=12)
    ax3.set_xlabel('Iteration', fontsize=10)
    ax3.set_ylabel('Function Value', fontsize=10)
    ax3.grid(True, linestyle=':', alpha=0.7)
    ax3.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


# ================== 执行比较 ==================
if __name__ == "__main__":
    mono_hist = optimize(ls_type='monotone', max_iter=50)
    nmono_hist = optimize(ls_type='nonmonotone', max_iter=50)
    plot_comparison(mono_hist, nmono_hist)
