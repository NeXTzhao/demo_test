import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 读取CSV文件
# data_df = pd.read_csv('data/data.csv')
data_ref = pd.read_csv('data/trajectory_ref.csv')
data_sample = pd.read_csv('data/trajectory_sample.csv')
# debug_df = pd.read_csv('data/debug.csv')

# 设置 Streamlit 页面为宽模式
st.set_page_config(layout="wide")

# 可视化轨迹
st.title('Vehicle Trajectory')

# 绘制2D视图
def plot_2d():
	# 绘制原始参考轨迹
	fig.add_trace(go.Scatter(
		x=data_ref['X_POS'],
		y=data_ref['Y_POS'],
		mode='lines+markers',
		name='Reference 2D',
		marker=dict(color='green'),
		hovertext=[
			f'Index: {i}<br>'
			f'x: {data_ref["X_POS"][i]:.2f} m<br>'
			f'y: {data_ref["Y_POS"][i]:.2f} m<br>'
			# f'Speed: {data_df["speed"][i]:.2f} m/s<br>'
			# f'Theta: {data_df["theta"][i]:.2f} rad<br>'
			# f'Acceleration: {data_df["accel"][i]:.2f} m/s²'  # 如果没有加速度数据，记得处理该字段
			for i in range(len(data_ref))
		],  # 悬浮窗中显示索引和其他信息
		hoverinfo='text'  # 只显示文本（索引、速度、theta等）
	))

	# 绘制原始采样参考轨迹
	fig.add_trace(go.Scatter(
		x=data_sample['X_POS'],
		y=data_sample['Y_POS'],
		mode='lines+markers',
		name='sample 2D',
		marker=dict(color='cyan'),  # 改为亮蓝色
		hovertext=[
			f'Index: {i}<br>'
			f'x: {data_sample["X_POS"][i]:.2f} m<br>'
			f'y: {data_sample["Y_POS"][i]:.2f} m<br>'
			# f'Speed: {data_df["speed"][i]:.2f} m/s<br>'
			# f'Theta: {data_df["theta"][i]:.2f} rad<br>'
			# f'Acceleration: {data_df["accel"][i]:.2f} m/s²'  # 如果没有加速度数据，记得处理该字段
			for i in range(len(data_sample))
		],  # 悬浮窗中显示索引和其他信息
		hoverinfo='text'  # 只显示文本（索引、速度、theta等）
	))

	fig.update_xaxes(scaleanchor="y", scaleratio=1)
	fig.update_yaxes(scaleanchor="x", scaleratio=1)
	fig.update_layout(xaxis_title='X (m)', yaxis_title='Y (m)', autosize=True)



plot_2d()
st.plotly_chart(fig)