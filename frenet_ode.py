import numpy as np


class FrenetODE:
    def __init__(self, kappa_func, v_init, dt):
        """
        初始化Frenet模拟器。

        参数:
        - kappa_func: 函数，根据路径的弧长s计算曲率kappa的函数。
        - v_init: 初始速度。
        """
        self.kappa_func = kappa_func
        self.v_init = v_init
        self.dt = dt

    def frenet_ode(self, state, control):
        """
        Frenet坐标系下的微分方程。

        参数:
        - t: 时间，标量。
        - state: 当前状态，包含弧长s、法向偏移n和航向角alpha。
        - control: 控制输入，包含速度v和方向盘转角phidot。

        返回:
        - [ds_dt, dn_dt, dalpha_dt]：状态的时间导数。
        """
        s, n, alpha, v = state
        a, phidot = control
        kappa = self.kappa_func(s)  # 动态获取曲率 kappa
        print('kappa=', kappa)
        print('s=', s)

        # Frenet 微分方程
        ds_dt = v * np.cos(alpha) / (1 - n * kappa)
        dn_dt = v * np.sin(alpha)
        dalpha_dt = phidot - v * kappa * np.cos(alpha) / (1 - n * kappa)
        dv_dt = a
        return [ds_dt, dn_dt, dalpha_dt, dv_dt]

    def rk4_step_frenet(self, state, control):
        """
        使用RK4方法进行一步时间更新。

        参数:
        - state: 当前状态，包含弧长s、法向偏移n和航向角alpha
        - control: 控制输入，包含速度v和方向盘转角phidot
        - dt: 时间步长

        返回:
        - state_next: 下一状态。
        """
        state = np.asarray(state)

        k1 = np.asarray(self.frenet_ode(state, control))
        k2 = np.asarray(self.frenet_ode(state + k1 * self.dt / 2, control))
        k3 = np.asarray(self.frenet_ode(state + k2 * self.dt / 2, control))
        k4 = np.asarray(self.frenet_ode(state + k3 * self.dt, control))

        # 结合 k1, k2, k3, k4 来更新状态
        state_next = state + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6
        return state_next

    def trajectory(self, current_frenet_state, Control, dt, total_time):
        num_steps = int(total_time / dt)
        for step in range(num_steps):
            # 计算下一个Frenet状态
            next_frenet_state = self.rk4_step_frenet(current_frenet_state, Control)

            # alpha = (step + 1) / num_steps
            # color = (0.0, 0.392, 0.0, alpha)
            # plotter.draw_frenet_state_vehicle(current_frenet_state, f'step{step}', color=color)

            # 更新状态
            current_frenet_state = next_frenet_state
