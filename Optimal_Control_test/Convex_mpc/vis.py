import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 读取CSV文件
data_df = pd.read_csv('data/data.csv')
debug_df = pd.read_csv('data/debug.csv')

# 设置页面配置，调整页面宽度
st.set_page_config(layout="wide")

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

# 绘制车辆方向
def plot_vehicle_direction(x, y, theta):
	# 箭头的长度和方向
	arrow_length = 0.6  # 箭头长度
	arrow_x = [x, x + arrow_length * np.cos(theta)]
	arrow_y = [y, y + arrow_length * np.sin(theta)]

	# 箭头的尖端
	arrowhead_x = x + arrow_length * np.cos(theta) * 0.8
	arrowhead_y = y + arrow_length * np.sin(theta) * 0.8

	# 创建箭头的两条边，构成箭头尖
	arrowhead_x2 = arrowhead_x + 0.1 * np.cos(theta + np.pi / 4)
	arrowhead_y2 = arrowhead_y + 0.1 * np.sin(theta + np.pi / 4)
	arrowhead_x3 = arrowhead_x + 0.1 * np.cos(theta - np.pi / 4)
	arrowhead_y3 = arrowhead_y + 0.1 * np.sin(theta - np.pi / 4)

	return arrow_x, arrow_y, [arrowhead_x, arrowhead_y, arrowhead_x2, arrowhead_y2, arrowhead_x3, arrowhead_y3]


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
		# 获取车辆四角坐标
		corners_x, corners_y = get_vehicle_corners(data_df['x_coords'][i], data_df['y_coords'][i], data_df['theta_coords'][i])
		# 绘制车辆
		fig.add_trace(go.Scatter(
			x=corners_x,
			y=corners_y,
			mode='lines',
			line=dict(color='rgba(200, 200, 200, 0.8)', width=1.5),  # 浅灰色，带透明度
			showlegend=False
		))

		# 车辆方向箭头
		arrow_length = 0.6  # 箭头长度
		arrow_x = [data_df['x_coords'][i], data_df['x_coords'][i] + arrow_length * np.cos(data_df['theta_coords'][i])]
		arrow_y = [data_df['y_coords'][i], data_df['y_coords'][i] + arrow_length * np.sin(data_df['theta_coords'][i])]

		# 绘制箭头线
		fig.add_trace(go.Scatter(
			x=arrow_x,
			y=arrow_y,
			mode='lines',
			line=dict(color='red', width=4, dash='solid'),  # 红色箭头，宽度稍微加粗，实线
			showlegend=False
		))

		# 绘制箭头的尖端
		arrowhead_length = 0.2  # 尖端长度
		arrowhead_angle = np.pi / 6  # 尖端角度

		# 箭头尖端两个点
		head_x1 = arrow_x[1] + arrowhead_length * np.cos(data_df['theta_coords'][i] + arrowhead_angle)
		head_y1 = arrow_y[1] + arrowhead_length * np.sin(data_df['theta_coords'][i] + arrowhead_angle)

		head_x2 = arrow_x[1] + arrowhead_length * np.cos(data_df['theta_coords'][i] - arrowhead_angle)
		head_y2 = arrow_y[1] + arrowhead_length * np.sin(data_df['theta_coords'][i] - arrowhead_angle)

		# 绘制箭头尖端
		fig.add_trace(go.Scatter(
			x=[arrow_x[1], head_x1, head_x2],
			y=[arrow_y[1], head_y1, head_y2],
			mode='lines',
			line=dict(color='red', width=4),
			showlegend=False
		))

	fig.update_xaxes(scaleanchor="y", scaleratio=1)
	fig.update_yaxes(scaleanchor="x", scaleratio=1)
	fig.update_layout(
		xaxis_title='X (m)',
		yaxis_title='Y (m)',
		autosize=True,
		width=800,  # 设置宽度
		height=600,  # 设置高度
	)


# 绘制3D视图
def plot_3d():
	# 绘制参考轨迹 3D（使用不同的颜色和透明度）
	fig.add_trace(
		go.Scatter3d(x=data_df['x_ref'], y=data_df['y_ref'], z=data_df.index, mode='lines+markers', name='Reference 3D',
		             marker=dict(color='deepskyblue', size=5, opacity=1.0, line=dict(color='black', width=0.5)))
	)

	# 绘制优化轨迹 3D
	fig.add_trace(
		go.Scatter3d(x=data_df['x_coords'], y=data_df['y_coords'], z=data_df.index, mode='lines+markers',
		             name='Optimized 3D',
		             marker=dict(color='tomato', size=5, opacity=1.0, line=dict(color='black', width=0.5)))
	)

	# 绘制动态轨迹 3D
	# fig.add_trace(
	# 	go.Scatter3d(x=data_df['dynamic_traj_x'], y=data_df['dynamic_traj_y'], z=data_df.index, mode='lines+markers',
	# 	             name='Dynamic Trajectory 3D',
	# 	             marker=dict(color='orange', size=5, opacity=0.5, line=dict(color='black', width=0.5)))
	# )

	# # 绘制参考轨迹 2D（使用透明度以区分）
	# fig.add_trace(
	# 	go.Scatter3d(x=data_df['x_ref'], y=data_df['y_ref'], z=[0] * len(data_df), mode='lines+markers',
	# 	             name='Reference 2D', marker=dict(color='deepskyblue', opacity=0.5, size=5))
	# )
	#
	# # 绘制优化轨迹 2D
	# fig.add_trace(
	# 	go.Scatter3d(x=data_df['x_coords'], y=data_df['y_coords'], z=[0] * len(data_df), mode='lines+markers',
	# 	             name='Optimized 2D', marker=dict(color='tomato', opacity=0.5, size=5))
	# )
	#
	# # 绘制动态轨迹 2D
	# fig.add_trace(
	# 	go.Scatter3d(x=data_df['dynamic_traj_x'], y=data_df['dynamic_traj_y'], z=[0] * len(data_df),
	# 	             mode='lines+markers',
	# 	             name='Dynamic Trajectory 2D', marker=dict(color='orange', opacity=0.5, size=5))
	# )
	#
	# 绘制车辆轮廓，减少重复渲染，提高性能
	# for i in range(0, len(data_df), max(1, len(data_df) // 50)):  # 只绘制每50个数据点的车辆轮廓，避免重复渲染
	# 	corners_x, corners_y = get_vehicle_corners(data_df['x_coords'][i], data_df['y_coords'][i],
	# 	                                           data_df['theta_coords'][i])
	# 	fig.add_trace(
	# 		go.Scatter3d(x=corners_x, y=corners_y, z=[0] * 5, mode='lines', line=dict(color='darkgrey', width=2),
	# 		             showlegend=False)
	# 	)

	# 更新布局，设置图表尺寸、背景颜色、交互体验和轴标签
	fig.update_layout(
		width=1400,  # 增加图表宽度
		height=1000,  # 增加图表高度
		title='3D Trajectory Visualization',
		title_x=0.5,  # 居中标题
		# scene=dict(
		# 	xaxis=dict(
		# 		title='X (m)',
		# 		backgroundcolor='rgba(0, 0, 0, 0.1)',
		# 		gridcolor='white',
		# 		range=[min(data_df['x_coords']) - 20, max(data_df['x_coords']) + 20]  # 设置x轴范围
		# 	),
		# 	yaxis=dict(
		# 		title='Y (m)',
		# 		backgroundcolor='rgba(0, 0, 0, 0.1)',
		# 		gridcolor='white',
		# 		range=[min(data_df['y_coords']) - 20, max(data_df['y_coords']) + 20]  # 设置y轴范围
		# 	),
		# 	zaxis=dict(
		# 		title='Index',
		# 		backgroundcolor='rgba(0, 0, 0, 0.1)',
		# 		gridcolor='white',
		# 		range=[0, len(data_df) + 20]  # 设置z轴范围，使得轨迹不占据整个空间
		# 	),
		# 	aspectmode='cube',  # 设置x、y、z轴比例一致
		# ),
		scene_camera=dict(
			eye=dict(x=1.5, y=1.5, z=1.5)  # 改变视角，使得初始视角适应数据
		),
		margin=dict(l=0, r=0, b=0, t=50)  # 调整图表边距
	)


if view == '2D':
	plot_2d()
else:
	plot_3d()

st.plotly_chart(fig)

# # # 统一显示状态和控制信息
st.title('State and Control Information')
fig = make_subplots(
	rows=4, cols=2, shared_xaxes=False,
	subplot_titles=(
		'velocity', 'theta', 'steering', 'd(steering)_omega',
		'odom', 'accel', 'd(a)_jerk', 'dd(steering)_alpha'
	),
	vertical_spacing=0.1,  # 设置垂直间距
	horizontal_spacing=0.1  # 设置水平间距
)
#
# # # 添加数据
fig.add_trace(go.Scatter(y=data_df['v_coords'], mode='lines+markers', name='velocity', marker=dict(color='#1f77b4')),
              row=1, col=1)  # 深蓝色
fig.add_trace(go.Scatter(y=data_df['theta_coords'], mode='lines+markers', name='theta', marker=dict(color='#ff7f0e')),
              row=1, col=2)  # 橙色
fig.add_trace(
	go.Scatter(y=data_df['steering_coords'], mode='lines+markers', name='steering', marker=dict(color='#2ca02c')),
	row=2, col=1)  # 绿色
fig.add_trace(
	go.Scatter(y=data_df['omega_coords'], mode='lines+markers', name='d(steering)_omega', marker=dict(color='#d62728')),
	row=2, col=2)  # 红色
fig.add_trace(go.Scatter(y=data_df['odom_coords'], mode='lines+markers', name='odom', marker=dict(color='#9467bd')),
              row=3, col=1)  # 紫色
fig.add_trace(go.Scatter(y=data_df['acc_coords'], mode='lines+markers', name='accel', marker=dict(color='#8c564b')),
              row=3, col=2)  # 棕色
fig.add_trace(
	go.Scatter(y=data_df['jerk_values'], mode='lines+markers', name='d(a)_jerk', marker=dict(color='#e377c2')), row=4,
	col=1)  # 粉红色
fig.add_trace(go.Scatter(y=data_df['alpha_values'], mode='lines+markers', name='dd(steering)_alpha',
                         marker=dict(color='#7f7f7f')), row=4, col=2)  # 灰色

# # 更新布局设置
fig.update_layout(
	# autosize=True,
	font=dict(color='white'),  # 设置字体颜色为白色
	height=1000,  # 增加图表高度
	width=1000,  # 增加图表宽度

	# 统一设置所有子图的网格线
	xaxis=dict(showgrid=True, gridcolor='gray', zeroline=False),
	yaxis=dict(showgrid=True, gridcolor='gray', zeroline=False),
	xaxis2=dict(showgrid=True, gridcolor='gray', zeroline=False),
	yaxis2=dict(showgrid=True, gridcolor='gray', zeroline=False),
	xaxis3=dict(showgrid=True, gridcolor='gray', zeroline=False),
	yaxis3=dict(showgrid=True, gridcolor='gray', zeroline=False),
	xaxis4=dict(showgrid=True, gridcolor='gray', zeroline=False),
	yaxis4=dict(showgrid=True, gridcolor='gray', zeroline=False),
	xaxis5=dict(showgrid=True, gridcolor='gray', zeroline=False),
	yaxis5=dict(showgrid=True, gridcolor='gray', zeroline=False),
	xaxis6=dict(showgrid=True, gridcolor='gray', zeroline=False),
	yaxis6=dict(showgrid=True, gridcolor='gray', zeroline=False),
	xaxis7=dict(showgrid=True, gridcolor='gray', zeroline=False),
	yaxis7=dict(showgrid=True, gridcolor='gray', zeroline=False),
	xaxis8=dict(showgrid=True, gridcolor='gray', zeroline=False),
	yaxis8=dict(showgrid=True, gridcolor='gray', zeroline=False)
)

# # 显示图表
st.plotly_chart(fig)
#
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
