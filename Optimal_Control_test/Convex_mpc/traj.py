import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import comb


def bezier_curve(control_points, num_points=35):
	n = len(control_points) - 1
	t_values = np.linspace(0, 1, num_points)
	curve = np.zeros((num_points, 2))
	for i in range(num_points):
		t = t_values[i]
		point = np.zeros(2)
		for k in range(n + 1):
			bernstein_poly = comb(n, k) * (t ** k) * ((1 - t) ** (n - k))
			point += bernstein_poly * control_points[k]
		curve[i] = point
	return curve


def generate_extreme_trajectory(num_points=35, deceleration_type='linear'):
	control_points = np.array([
		# [0, 0], [50, 50], [100, 0]]
		[0, 0], [10, 0], [20, 0], [20, 5], [30, 5], [40, 5]]
	)

	trajectory = bezier_curve(control_points, num_points=num_points)

	if deceleration_type == 'linear':
		velocities = np.linspace(5, 20, num_points)
	elif deceleration_type == 'exponential':
		velocities = 5 + 15 * np.exp(-0.1 * np.arange(num_points))
	elif deceleration_type == 'position_based':
		distances = np.sqrt(np.sum(np.diff(trajectory, axis=0) ** 2, axis=1)).cumsum()
		max_distance = distances[-1]
		velocities = 20 - 15 * (distances / max_distance)
		velocities = np.insert(velocities, 0, 20)
	else:
		raise ValueError("Unsupported deceleration type")

	headings = np.zeros(num_points)
	delta_v = np.zeros(num_points)  # 方向盘角度
	omega = np.zeros(num_points)  # 方向盘角速度
	acceleration = np.zeros(num_points)  # 加速度
	odom = np.zeros(num_points)  # 累计里程

	for i in range(1, num_points):
		dx = trajectory[i, 0] - trajectory[i - 1, 0]
		dy = trajectory[i, 1] - trajectory[i - 1, 1]
		headings[i] = np.arctan2(dy, dx)

		# 假设简单的加速度模型
		acceleration[i] = (velocities[i] - velocities[i - 1]) * 10  # 简化处理，假设dt=0.1s
		odom[i] = odom[i - 1] + np.sqrt(dx ** 2 + dy ** 2)

		# 计算转向角和角速度
		if i > 1:
			delta_v[i] = headings[i] - headings[i - 1]
			omega[i] = delta_v[i] - delta_v[i - 1]

	trajectory_data = pd.DataFrame({
		'X_POS': trajectory[:, 0],
		'Y_POS': trajectory[:, 1],
		'SPEED': velocities,
		'THETA': headings,
		'DELTAV': delta_v,
		'OMEGA': omega,
		'ODOM': odom,
		'ACCEL': acceleration
	})

	return trajectory_data


def save_trajectory_to_txt(trajectory_data, filename='trajectory.csv'):
	trajectory_data.to_csv(filename, header=True, index=False, sep=',')
	print(f'Trajectory saved to {filename}')


# 生成并保存轨迹
extreme_trajectory = generate_extreme_trajectory(deceleration_type='linear')
save_trajectory_to_txt(extreme_trajectory)

# 可视化轨迹路径
plt.figure(figsize=(10, 6))
plt.plot(extreme_trajectory['X_POS'], extreme_trajectory['Y_POS'], label='Bezier Trajectory Path', color='blue')
plt.scatter(extreme_trajectory['X_POS'], extreme_trajectory['Y_POS'], color='red', marker='o',
            label='Trajectory Points')
plt.title('Extreme Trajectory Path with Lane Change')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.grid()
plt.legend()
plt.show()
