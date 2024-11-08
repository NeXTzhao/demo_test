import numpy as np
import streamlit as st
import plotly.graph_objects as go

# 设置直线轨迹
num_points = 100
x_data = np.linspace(-10, 10, num_points)  # X 位置
y_data = 2 * x_data + 1  # Y 位置，直线方程 y = 2x + 1

# 计算 cost 函数（位置和速度分量的平方和）
SPEED = np.sqrt(x_data ** 2 + y_data ** 2)  # 示例：SPEED 为点到原点的距离
THETA = np.arctan2(y_data, x_data)  # 示例：THETA 为角度
OMEGA = np.gradient(SPEED)  # 示例：OMEGA 为速度的梯度

# 计算 cost 值（位置和速度分量的平方和）
z_cost = x_data ** 2 + y_data ** 2 + SPEED ** 2 + THETA ** 2 + OMEGA ** 2

# 创建 Streamlit 应用
st.title('非线性函数迭代搜索可视化')

# 选择迭代步骤
iter_step = st.slider('选择迭代步骤', min_value=0, max_value=num_points - 1, value=0)

# 根据选择的迭代步骤更新数据
x_data_step = x_data[:iter_step + 1]
y_data_step = y_data[:iter_step + 1]
z_cost_step = z_cost[:iter_step + 1]


# 创建等高线图函数
def create_contour_plot(x, y, z_data, title):
    x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
    z_grid = np.zeros_like(x_grid)

    # 将数据点的值插入到网格中
    for i in range(len(x)):
        xi = np.searchsorted(np.linspace(x.min(), x.max(), 100), x[i])
        yi = np.searchsorted(np.linspace(y.min(), y.max(), 100), y[i])
        z_grid[yi, xi] = z_data[i]

    fig = go.Figure(data=[go.Contour(
        z=z_grid,
        x=np.linspace(x.min(), x.max(), 100),
        y=np.linspace(y.min(), y.max(), 100),
        colorscale='Viridis'
    )])
    fig.update_layout(
        title=title,
        xaxis_title='X_POS',
        yaxis_title='Y_POS',
        coloraxis_colorbar=dict(title='Value')
    )
    return fig


# 创建并显示等高线图
st.subheader(f'迭代步 {iter_step} 的 Cost 等高线图（位置和速度分量的平方和）')
fig_cost = create_contour_plot(x_data_step, y_data_step, z_cost_step,
                               f'迭代步 {iter_step} 的 Cost 等高线图（位置和速度分量的平方和）')
st.plotly_chart(fig_cost)
