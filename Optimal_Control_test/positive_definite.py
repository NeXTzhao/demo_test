import numpy as np
import matplotlib.pyplot as plt

# 生成正定矩阵，通过构造方式确保正定性
np.random.seed(0)
dim = 50
A_positive_definite = np.random.randn(dim, dim)
A_positive_definite = np.dot(A_positive_definite, A_positive_definite.T)

# 生成不定矩阵，通过在对角线上减去一个足够大的值
shift_value = 200  # 这个值应该足够大，以确保矩阵变为不定
A_indefinite = np.copy(A_positive_definite)

# 修改矩阵以确保其为不定矩阵
A_indefinite[dim - 2, dim - 2] -= (A_indefinite[dim - 2, dim - 2] + shift_value * 1.5)
A_indefinite[dim - 1, dim - 1] -= (A_indefinite[dim - 1, dim - 1] + shift_value)

# 生成正定矩阵，通过构造方式确保正定性
np.random.seed(10)
dim = 50
A_positive_definite = np.random.randn(dim, dim)
A_positive_definite = np.dot(A_positive_definite, A_positive_definite.T)

# 生成不定矩阵，通过在对角线上减去一个足够大的值
shift_value = 200  # 这个值应该足够大，以确保矩阵变为不定
A_indefinite = np.copy(A_positive_definite)

# 修改矩阵以确保其为不定矩阵
A_indefinite[dim - 2, dim - 2] -= (A_indefinite[dim - 2, dim - 2] + shift_value * 1.5)
A_indefinite[dim - 1, dim - 1] -= (A_indefinite[dim - 1, dim - 1] + shift_value)


# 定义二次型函数，使用矩阵运算
def quadratic_form_matrix(X1, X2, A):
    shape = X1.shape
    X = np.zeros((shape[0], shape[1], A.shape[0]))
    X[:, :, 0] = X1
    X[:, :, 1] = X2
    X = X.reshape(-1, A.shape[0])  # 变形为2D数组，每行是一个点
    Z = np.einsum('...i,ij,...j', X, A, X)
    return Z.reshape(shape)


# 创建网格点
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(x1, x2)

# 计算不定矩阵的二次型
Z1 = quadratic_form_matrix(X1, X2, A_indefinite)

# 计算正定矩阵的二次型
Z2 = quadratic_form_matrix(X1, X2, A_positive_definite)

# 计算矩阵的条件数
condition_number_indefinite = np.linalg.cond(A_indefinite, 2)
condition_number_positive_definite = np.linalg.cond(A_positive_definite, 2)

# 绘制等高线图和条件数条形图
fig, axs = plt.subplots(1, 3, figsize=(24, 12))

# 不定矩阵的等高线图
contour1 = axs[0].contour(X1, X2, Z1, levels=20)
axs[0].set_xlabel('$x_1$')
axs[0].set_ylabel('$x_2$')
axs[0].set_title(f'Indefinite Matrix\nCondition Number: {condition_number_indefinite:.2e}')
fig.colorbar(contour1, ax=axs[0])
axs[0].grid(True)
# axs[0].axis('equal')

# 正定矩阵的等高线图
contour2 = axs[1].contour(X1, X2, Z2, levels=20)
axs[1].set_xlabel('$x_1$')
axs[1].set_ylabel('$x_2$')
axs[1].set_title(f'Positive Definite Matrix\nCondition Number: {condition_number_positive_definite:.2e}')
fig.colorbar(contour2, ax=axs[1])
axs[1].grid(True)
# axs[1].axis('equal')

# 绘制条件数的条形图
labels = ['Indefinite Matrix', 'Positive Definite Matrix']
condition_numbers = [condition_number_indefinite, condition_number_positive_definite]

bars = axs[2].bar(labels, condition_numbers, color=['purple', 'b'])

# 添加数据标签
for bar in bars:
    height = bar.get_height()
    axs[2].annotate(f'{height:.2e}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

axs[2].set_title('Condition Numbers of Matrices')
axs[2].set_ylabel('Condition Number')
axs[2].grid(True)
plt.tight_layout()
# axs[2].axis('equal')

# 计算矩阵的特征值
eigvals_positive_definite = np.linalg.eigvalsh(A_positive_definite)
eigvals_indefinite = np.linalg.eigvalsh(A_indefinite)

# 绘制特征值分布
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].plot(eigvals_positive_definite, 'bo-', label='Positive Definite Matrix')
axs[0].set_title('Eigenvalues of Positive Definite Matrix')
axs[0].set_xlabel('Index')
axs[0].set_ylabel('Eigenvalue')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(eigvals_indefinite, 'ro-', label='Indefinite Matrix')
axs[1].set_title('Eigenvalues of Indefinite Matrix')
axs[1].set_xlabel('Index')
axs[1].set_ylabel('Eigenvalue')
axs[1].grid(True)
axs[1].legend()

plt.show()
