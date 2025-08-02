import math


def calculate_relative_components_with_heading(A, B):
    """
    计算点B相对于点A的局部坐标系的位置分量和朝向差
    :param A: 元组 (x, y, theta_rad)，表示参考点A的坐标和朝向（弧度）
    :param B: 元组 (x, y, theta_rad)，表示目标点B的坐标和朝向
    :return: 字典，包含纵向、横向分量和朝向差（弧度）
    """
    x1, y1, theta1 = A
    x2, y2, theta2 = B

    # 计算坐标差（世界坐标系）
    dx_world = x2 - x1
    dy_world = y2 - y1

    # 将坐标差转换到A的局部坐标系（x轴为A的朝向）
    cos_theta = math.cos(theta1)
    sin_theta = math.sin(theta1)
    dx_local = dx_world * cos_theta + dy_world * sin_theta  # 纵向分量（平行方向）
    dy_local = -dx_world * sin_theta + dy_world * cos_theta  # 横向分量（垂直方向）

    # 计算相对朝向角度（B的朝向与A的朝向的差）
    relative_heading = theta2 - theta1
    # 规范化到 [-π, π] 范围
    relative_heading = (relative_heading + math.pi) % (2 * math.pi) - math.pi

    return {
        "longitudinal": dx_local,
        "lateral": dy_local,
        "relative_heading_rad": relative_heading
    }


# 示例用法
if __name__ == "__main__":
    # 示例点A：坐标(0, 0)，朝向0弧度（朝x轴正方向）
    A = (0, 0, 0)
    # 示例点B：坐标(2, 1)，朝向90度（π/2弧度，朝y轴正方向）
    B = (2, 1, math.pi / 2)

    # 计算相对分量
    result = calculate_relative_components_with_heading(A, B)

    # 输出结果
    print(f"纵向分量（平行方向）: {result['longitudinal']:.2f}")
    print(f"横向分量（垂直方向）: {result['lateral']:.2f}")
    print(f"相对朝向角度: {math.degrees(result['relative_heading_rad']):.1f}°")

    # 解释结果
    if result['lateral'] > 0:
        print("B位于A的左侧")
    elif result['lateral'] < 0:
        print("B位于A的右侧")

    if abs(result['relative_heading_rad'] - math.pi / 2) < 0.1:
        print("B的朝向与A的朝向垂直（左侧）")
    elif abs(result['relative_heading_rad'] + math.pi / 2) < 0.1:
        print("B的朝向与A的朝向垂直（右侧）")