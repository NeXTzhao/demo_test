import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 读取CSV文件
data_df = pd.read_csv('data/data.csv')
debug_df = pd.read_csv('data/debug.csv')

# 可视化轨迹
st.title('Vehicle Trajectory')

# 创建按钮来控制显示2D和3D视图
view = st.radio('Select View', ('2D', '3D'), index=0)
fig = go.Figure()

# 车辆尺寸
vehicle_length = 2.5
vehicle_width = 1.8


# 计算车辆的四个角点
def get_vehicle_corners(x, y, theta):
	corners_x = [
		x + (vehicle_length / 2) * np.cos(theta) - (vehicle_width / 2) * np.sin(theta),
		x + (vehicle_length / 2) * np.cos(theta) + (vehicle_width / 2) * np.sin(theta),
		x - (vehicle_length / 2) * np.cos(theta) + (vehicle_width / 2) * np.sin(theta),
		x - (vehicle_length / 2) * np.cos(theta) - (vehicle_width / 2) * np.sin(theta),
		x + (vehicle_length / 2) * np.cos(theta) - (vehicle_width / 2) * np.sin(theta)
	]
	corners_y = [
		y + (vehicle_length / 2) * np.sin(theta) + (vehicle_width / 2) * np.cos(theta),
		y + (vehicle_length / 2) * np.sin(theta) - (vehicle_width / 2) * np.cos(theta),
		y - (vehicle_length / 2) * np.sin(theta) - (vehicle_width / 2) * np.cos(theta),
		y - (vehicle_length / 2) * np.sin(theta) + (vehicle_width / 2) * np.cos(theta),
		y + (vehicle_length / 2) * np.sin(theta) + (vehicle_width / 2) * np.cos(theta)
	]
	return corners_x, corners_y


# 绘制2D视图
def plot_2d():
	# 绘制参考轨迹
	fig.add_trace(go.Scatter(
		x=data_df['x_ref'],
		y=data_df['y_ref'],
		mode='lines+markers',
		name='Reference 2D',
		marker=dict(color='cyan'),  # 改为亮蓝色
		hovertext=[
			f'Index: {i}<br>'
			f'x: {data_df["x_ref"][i]:.2f} m<br>'
			f'y: {data_df["y_ref"][i]:.2f} m<br>'
			# f'Speed: {data_df["speed"][i]:.2f} m/s<br>'
			# f'Theta: {data_df["theta"][i]:.2f} rad<br>'
			# f'Acceleration: {data_df["accel"][i]:.2f} m/s²'  # 如果没有加速度数据，记得处理该字段
			for i in range(len(data_df))
		],  # 悬浮窗中显示索引和其他信息
		hoverinfo='text'  # 只显示文本（索引、速度、theta等）
	))

	# 绘制优化轨迹
	fig.add_trace(go.Scatter(
		x=data_df['x_coords'],
		y=data_df['y_coords'],
		mode='lines+markers',
		name='Optimized 2D',
		marker=dict(color='green'),  # 绿色
		hovertext=[
			f'Index: {i}<br>'
			f'x: {data_df["x_coords"][i]:.2f} m<br>'
			f'y: {data_df["y_coords"][i]:.2f} m<br>'
			# f'Speed: {data_df["speed"][i]:.2f} m/s<br>'
			# f'Theta: {data_df["theta"][i]:.2f} rad<br>'
			# f'Acceleration: {data_df["accel"][i]:.2f} m/s²'
			for i in range(len(data_df))
		],  # 悬浮窗中显示索引和其他信息
		hoverinfo='text'  # 只显示文本（索引、速度、theta等）
	))

	# 绘制动力学推演轨迹
	fig.add_trace(go.Scatter(
		x=data_df['dynamic_traj_x'],
		y=data_df['dynamic_traj_y'],
		mode='lines+markers',
		name='dynamic_traj 2D',
		marker=dict(color='orange'),  # 改为亮橙色
		hovertext=[
			f'Index: {i}<br>'
			f'x: {data_df["dynamic_traj_x"][i]:.2f} m<br>'
			f'y: {data_df["dynamic_traj_y"][i]:.2f} m<br>'
			# f'Speed: {data_df["speed"][i]:.2f} m/s<br>'
			# f'Theta: {data_df["theta"][i]:.2f} rad<br>'
			# f'Acceleration: {data_df["accel"][i]:.2f} m/s²'
			for i in range(len(data_df))
		],  # 悬浮窗中显示索引和其他信息
		hoverinfo='text'  # 只显示文本（索引、速度、theta等）
	))

	for i in range(len(data_df)):
		corners_x, corners_y = get_vehicle_corners(data_df['x_coords'][i], data_df['y_coords'][i], data_df['theta'][i])
		fig.add_trace(go.Scatter(
			x=corners_x,
			y=corners_y,
			mode='lines',
			line=dict(color='rgba(200, 200, 200, 0.8)', width=1.5),  # 浅灰色，带透明度
			showlegend=False
		))

	fig.update_xaxes(scaleanchor="y", scaleratio=1)
	fig.update_yaxes(scaleanchor="x", scaleratio=1)
	fig.update_layout(xaxis_title='X (m)', yaxis_title='Y (m)', autosize=True)


# 绘制3D视图
def plot_3d():
	fig.add_trace(
		go.Scatter3d(x=data_df['x_ref'], y=data_df['y_ref'], z=data_df.index, mode='lines+markers', name='Reference 3D',
		             marker=dict(color='blue', opacity=1.0)))
	fig.add_trace(go.Scatter3d(x=data_df['x_coords'], y=data_df['y_coords'], z=data_df.index, mode='lines+markers',
	                           name='Optimized 3D', marker=dict(color='red', opacity=1.0)))
	fig.add_trace(go.Scatter3d(x=data_df['x_ref'], y=data_df['y_ref'], z=[0] * len(data_df), mode='lines+markers',
	                           name='Reference 2D', marker=dict(color='blue', opacity=0.5)))
	fig.add_trace(go.Scatter3d(x=data_df['x_coords'], y=data_df['y_coords'], z=[0] * len(data_df), mode='lines+markers',
	                           name='Optimized 2D', marker=dict(color='red', opacity=0.5)))
	for i in range(len(data_df)):
		corners_x, corners_y = get_vehicle_corners(data_df['x_coords'][i], data_df['y_coords'][i], data_df['theta'][i])
		fig.add_trace(
			go.Scatter3d(x=corners_x, y=corners_y, z=[0] * 5, mode='lines', line=dict(color='black'), showlegend=False))
	fig.update_layout(width=960, height=960,
	                  scene=dict(xaxis=dict(title='X (m)'), yaxis=dict(title='Y (m)'), zaxis=dict(title='Index')))


if view == '2D':
	plot_2d()
else:
	plot_3d()

st.plotly_chart(fig)

# 统一显示状态和控制信息
st.title('State and Control Information')
fig = make_subplots(rows=3, cols=2, shared_xaxes=False,
                    subplot_titles=('velocity', 'acc', 'theta', 'steering', 'd(aa)_jerk', 'dd(steering)_alpha'))
fig.add_trace(
	go.Scatter(y=data_df['velocity'], mode='lines+markers', name='velocity', marker=dict(color='red')), row=1,
	col=1)
fig.add_trace(
	go.Scatter(y=data_df['acc'], mode='lines+markers', name='acc', marker=dict(color='blue')),
	row=1, col=2)
fig.add_trace(go.Scatter(y=data_df['theta'], mode='lines+markers', name='theta', marker=dict(color='green')),
              row=2, col=1)
fig.add_trace(go.Scatter(y=data_df['steering'], mode='lines+markers', name='steering', marker=dict(color='magenta')),
              row=2,
              col=2)
fig.add_trace(go.Scatter(y=data_df['d(aa)_jerk'], mode='lines+markers', name='d(aa)_jerk', marker=dict(color='purple')),
              row=3, col=1)
fig.add_trace(go.Scatter(y=data_df['dd(steering)_alpha'], mode='lines+markers', name='dd(steering)_alpha',
                         marker=dict(color='orange')), row=3,
              col=2)
fig.update_layout(autosize=True)
st.plotly_chart(fig)

# 可视化调试信息
st.title('Debug Information')
fig = make_subplots(rows=3, cols=1, shared_xaxes=False,
                    subplot_titles=('Objective Value', 'Primal Residual', 'Dual Residual'))

fig.add_trace(
	go.Scatter(y=debug_df['objective_value'], mode='lines+markers', name='Objective Value', marker=dict(color='blue')),
	row=1, col=1)
fig.add_trace(
	go.Scatter(y=debug_df['primal_residual'], mode='lines+markers', name='Primal Residual', marker=dict(color='red')),
	row=2, col=1)
fig.add_trace(
	go.Scatter(y=debug_df['dual_residual'], mode='lines+markers', name='Dual Residual', marker=dict(color='green')),
	row=3, col=1)

fig.update_layout(autosize=True)
st.plotly_chart(fig)


# 读取矩阵数据
def read_matrix_from_csv(filename):
	df = pd.read_csv(filename, header=None)
	return df.values


# 读取并解析矩阵数据
H = read_matrix_from_csv('data/H.csv')
g = read_matrix_from_csv('data/g.csv')
A = read_matrix_from_csv('data/A.csv')
b = read_matrix_from_csv('data/b.csv')
C = read_matrix_from_csv('data/C.csv')
l = read_matrix_from_csv('data/l.csv')
u = read_matrix_from_csv('data/u.csv')


# 可视化矩阵数据的函数
def plot_matrix(matrix, title):
	# 根据矩阵的大小动态设置画面的尺寸
	height = max(300, min(800, matrix.shape[0] * 20))  # 每行20像素，高度在300到800之间
	width = max(400, min(800, matrix.shape[1] * 20))  # 每列20像素，宽度在300到800之间

	fig = go.Figure(data=go.Heatmap(
		z=matrix,
		x=[f'{i}' for i in range(matrix.shape[1])],
		y=[f'{i}' for i in range(matrix.shape[0])][::-1],  # 反转 y 轴顺序
		colorscale='Viridis',  # 使用 Viridis 颜色映射
		text=matrix,
		texttemplate="%{text:.2f}",
		hoverinfo='x+y+z'  # 显示 x, y, z 值
	))

	# 更新布局
	fig.update_layout(
		title=f'{title} (Shape: {matrix.shape[0]}x{matrix.shape[1]})',
		xaxis_title='Columns',
		yaxis_title='Rows',
		autosize=False,
		width=width,  # 设置合适的宽度
		height=height  # 设置合适的高度
	)
	return fig


# 组合相关矩阵进行可视化的函数
def plot_combined_matrices(matrices, titles):
	col_num = len(matrices)
	fig = make_subplots(rows=1, cols=col_num,
	                    subplot_titles=[f'{title} (Shape: {matrix.shape[0]}x{matrix.shape[1]})' for matrix, title in
	                                    zip(matrices, titles)])

	for i, (matrix, title) in enumerate(zip(matrices, titles)):
		# 根据矩阵的大小动态设置画面的尺寸
		height = max(300, min(800, matrix.shape[0] * 20))  # 每行20像素，高度在300到800之间
		# width = max(300, min(800, matrix.shape[1] * 20))  # 每列20像素，宽度在300到800之间

		heatmap = go.Heatmap(
			z=matrix,
			x=[f'{j}' for j in range(matrix.shape[1])],
			y=[f'{j}' for j in range(matrix.shape[0])][::-1],  # 反转 y 轴顺序
			colorscale='YlGnBu',  # 使用指定的颜色映射
			text=matrix,
			texttemplate="%{text:.2f}",
			hoverinfo='x+y+z'
		)
		fig.add_trace(heatmap, row=1, col=i + 1)

		# 更新布局
		fig.update_layout(
			# title=title,
			xaxis_title='Columns',
			yaxis_title='Rows',
			autosize=False,
			width=800,  # 设置合适的宽度
			height=height  # 设置合适的高度
		)
	return fig


# 3D 可视化函数
def plot_3d_surface(matrix, title):
	flipped_matrix = matrix[::-1]
	fig = go.Figure(data=[go.Surface(z=flipped_matrix)])
	fig.update_layout(title=title, autosize=True,
	                  scene=dict(
		                  xaxis_title='Columns',
		                  yaxis_title='Rows',
		                  zaxis_title='Values'
	                  ))
	return fig


# 目标函数相关矩阵
st.markdown('## Objective Function Matrices')
st.markdown(r'''
$$
\begin{aligned}
\text{min: }& \frac{1}{2} X^T H X + g^T X \\
\text{s.t. : }
& AX = b \\
& L \leq CX \leq U \\
\\
& u^T = [a, \delta] \\
& x^T = [x, y, \theta, v] \\
& X^T = [u_1, x_2, u_2, x_3, \dots, u_{n-1}, x_n]
\end{aligned}
$$
''')

st.plotly_chart(plot_matrix(H, 'H Matrix'))
st.plotly_chart(plot_matrix(g.reshape(-1, 1), 'g Vector'))

# 等式约束相关矩阵
st.markdown('### Equality Constraints Matrices')
st.markdown(r'''
$$
\text{Equality Constraints: } A X = b
$$
''')
st.plotly_chart(plot_matrix(A, 'A Matrix'))
st.plotly_chart(plot_matrix(b.reshape(-1, 1), 'b Vector'))

# 不等式约束相关矩阵
st.markdown('### Inequality Constraints Matrices')
st.markdown(r'''
$$
\text{Inequality Constraints: } L \leq C X \leq U
$$
''')
# 单独显示 C 矩阵
st.plotly_chart(plot_matrix(C, 'C Matrix'))
# 将 l 和 u 矩阵并排显示
ineq_matrices = [l.reshape(-1, 1), u.reshape(-1, 1)]
ineq_titles = ['l Vector', 'u Vector']
fig_ineq = plot_combined_matrices(ineq_matrices, ineq_titles)
st.plotly_chart(fig_ineq)
