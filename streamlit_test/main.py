import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

st.title("矩形小车沿车道线移动动画")

# 初始化状态
if 'paused' not in st.session_state:
    st.session_state.paused = False
if 'current_position' not in st.session_state:
    st.session_state.current_position = 0
if 'car_x' not in st.session_state:
    st.session_state.car_x = 0
if 'car_y' not in st.session_state:
    st.session_state.car_y = 0
if 'car_z' not in st.session_state:
    st.session_state.car_z = 0
if 'camera_eye_x' not in st.session_state:
    st.session_state.camera_eye_x = 1.25
if 'camera_eye_y' not in st.session_state:
    st.session_state.camera_eye_y = 1.25
if 'camera_eye_z' not in st.session_state:
    st.session_state.camera_eye_z = 1.25

def toggle_pause():
    st.session_state.paused = not st.session_state.paused

st.button("暂停/继续", on_click=toggle_pause)

# 创建一个容器来放置滑动条
sliders_placeholder = st.container()

with sliders_placeholder:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.camera_eye_x = st.slider("视角 X", -10.0, 10.0, st.session_state.camera_eye_x, 0.01)
    with col2:
        st.session_state.camera_eye_y = st.slider("视角 Y", -10.0, 10.0, st.session_state.camera_eye_y, 0.01)
    with col3:
        st.session_state.camera_eye_z = st.slider("视角 Z", 0.1, 10.0, st.session_state.camera_eye_z, 0.01)

placeholder = st.empty()
info_placeholder = st.empty()

# 定义车道线，每条车道宽3.75米，取中间为参考车道线
length = 50  # 车道长度
x = np.linspace(-length / 2, length / 2, 100)
lane_width = 3.75
y1 = np.zeros_like(x)
y2 = np.ones_like(x) * lane_width

# 定义小车形状
def create_car_edges(x , y, z, length=4.0, width=2.0, height=1.5):
    # 定义立方体的8个顶点
    vertices = np.array([
        [x - length / 2, y - width / 2, z],
        [x + length / 2, y - width / 2, z],
        [x + length / 2, y + width / 2, z],
        [x - length / 2, y + width / 2, z],
        [x - length / 2, y - width / 2, z + height],
        [x + length / 2, y - width / 2, z + height],
        [x + length / 2, y + width / 2, z + height],
        [x - length / 2, y + width / 2, z + height]
    ])

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom edges
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top edges
        [0, 4], [1, 5], [2, 6], [3, 7]   # Side edges
    ]

    edge_x = []
    edge_y = []
    edge_z = []

    for edge in edges:
        for vertex in edge:
            edge_x.append(vertices[vertex][0])
            edge_y.append(vertices[vertex][1])
            edge_z.append(vertices[vertex][2])
        edge_x.append(None)  # 添加None来断开边
        edge_y.append(None)
        edge_z.append(None)

    return go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='red', width=2.5), name='小车')

def create_figure():
    fig = go.Figure(data=[
        go.Scatter3d(x=x, y=y1, z=[0] * len(x), mode='lines',  line=dict(color='blue', width=1.0),name='车道线1'),
        go.Scatter3d(x=x, y=y2, z=[0] * len(x), mode='lines', line=dict(color='blue', width=1.0), name='车道线2'),
        create_car_edges(st.session_state.car_x, st.session_state.car_y, 0)  # 创建小车
    ])

    # 确保XYZ轴比例相同
    fig.update_layout(
        scene=dict(
            aspectmode='cube',  # 使用cube模式以确保比例一致
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(nticks=4, range=[-25,25], title='X'),
            yaxis=dict(nticks=4, range=[-25,25], title='Y'),
            zaxis=dict(nticks=4, range=[0, 50], title='Z'),
        ),
        scene_camera=dict(
            eye=dict(x=st.session_state.camera_eye_x, y=st.session_state.camera_eye_y, z=st.session_state.camera_eye_z)
        ),
        # width=800, height=600,
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=True,
        legend=dict(
            x=1,
            y=0.5,
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        )
    )

    return fig

# 初次渲染图表
fig = create_figure()
plot = placeholder.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

while True:
    if not st.session_state.paused:
        st.session_state.current_position += 1
        current_position = st.session_state.current_position % len(x)
        st.session_state.car_x = x[current_position]
        st.session_state.car_y = (y1[current_position] + y2[current_position]) / 2  # 取两条车道线的中间

        fig = create_figure()
        plot = placeholder.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

        info_placeholder.write(
            f"<div style='border:1px solid #ccc;padding:10px;'>"
            f"<strong>当前时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            f"<strong>, 当前坐标:</strong> (x: {st.session_state.car_x:.2f}, y: {st.session_state.car_y:.2f}, z: {st.session_state.car_z:.2f})"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        info_placeholder.write(
            f"<div style='border:1px solid #ccc;padding:10px;'>"
            f"<strong>当前时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            f"<strong>, 当前坐标:</strong> (x: {st.session_state.car_x:.2f}, y: {st.session_state.car_y:.2f}, z: {st.session_state.car_z:.2f})"
            f"</div>",
            unsafe_allow_html=True
        )
        st.stop()  # 如果暂停，停止脚本运行
    time.sleep(0.1)  # 增加sleep时间，以减少数据刷新频率
