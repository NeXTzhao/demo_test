import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
file_path = "data.csv"  # 替换成你的 CSV 文件路径
df = pd.read_csv(file_path, delimiter=" ", skipinitialspace=True)  # 处理可能的额外空格

# 预处理数据，去掉冒号并提取数值
def clean_value(value):
    if isinstance(value, str) and ":" in value:
        try:
            return float(value.split(":")[-1])  # 取 `:` 后面的数值
        except ValueError:
            return None  # 转换失败的设为 None
    return value

df = df.applymap(clean_value)

# 删除包含 NaN（空值）的行
df.dropna(inplace=True)

# 确保 'x' 和 'y' 列存在
if 'x' in df.columns and 'y' in df.columns:
    plt.figure(figsize=(8, 6))
    plt.plot(df['x'], df['y'], marker='o', linestyle='-', markersize=2, label="Trajectory")

    # 标注起点和终点
    plt.scatter(df['x'].iloc[0], df['y'].iloc[0], color='green', label="Start", zorder=3)
    plt.scatter(df['x'].iloc[-1], df['y'].iloc[-1], color='red', label="End", zorder=3)

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("XY Trajectory Plot")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("CSV 文件中缺少 'x' 或 'y' 列，请检查数据格式。")
