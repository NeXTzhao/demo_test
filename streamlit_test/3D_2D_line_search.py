import numpy as np
import plotly.graph_objects as go


# 示例目标函数
def objective_function(x, y):
    return x ** 2 + y ** 2


# 生成目标函数的值（网格数据）
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = objective_function(X, Y)

# 示例线搜索路径数据
path_x = [0, 1, 2, 3, 4, 5]
path_y = [0, 0.5, 1, 1.5, 2, 2.5]
path_z = [objective_function(x, y) for x, y in zip(path_x, path_y)]

# 绘制3D图
fig = go.Figure(data=[
    go.Surface(z=Z, x=X, y=Y, opacity=0.7, colorscale='Viridis'),
    go.Scatter3d(x=path_x, y=path_y, z=path_z, mode='lines+markers', line=dict(color='red', width=5),
                 marker=dict(size=5))
])

fig.update_layout(title='3D Surface with Line Search Path', scene=dict(
    xaxis_title='X',
    yaxis_title='Y',
    zaxis_title='Z'
))

fig.show()

# 绘制2D等高线图
fig2 = go.Figure(data=[
    go.Contour(z=Z, x=x, y=y, colorscale='Viridis'),
    go.Scatter(x=path_x, y=path_y, mode='lines+markers', line=dict(color='red', width=5), marker=dict(size=5))
])

fig2.update_layout(title='2D Contour with Line Search Path', xaxis_title='X', yaxis_title='Y')

fig2.show()
