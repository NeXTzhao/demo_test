import numpy as np
import matplotlib.pyplot as plt

# 生成正定矩阵，通过构造方式确保正定性
np.random.seed(0)
A_positive_definite = np.random.randn(100, 100)
A_positive_definite = np.dot(A_positive_definite, A_positive_definite.T)

# 生成不定矩阵，通过在对角线上减去一个足够大的值
shift_value = 80  # 这个值应该足够大，以确保矩阵变为不定
A_indefinite = np.copy(A_positive_definite)

index = 0
A_indefinite[index, index] -= shift_value
A_indefinite[index + 1, index + 1] -= shift_value


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
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)

# 计算不定矩阵的二次型
Z1 = quadratic_form_matrix(X1, X2, A_indefinite)

# 计算正定矩阵的二次型
Z2 = quadratic_form_matrix(X1, X2, A_positive_definite)

# 绘制等高线图
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# 不定矩阵的等高线图
contour1 = axs[0].contour(X1, X2, Z1, levels=20)
axs[0].set_xlabel('$x_1$')
axs[0].set_ylabel('$x_2$')
axs[0].set_title('Contour plot of the quadratic form (Indefinite)')
fig.colorbar(contour1, ax=axs[0])
axs[0].grid(True)

# 正定矩阵的等高线图
contour2 = axs[1].contour(X1, X2, Z2, levels=20)
axs[1].set_xlabel('$x_1$')
axs[1].set_ylabel('$x_2$')
axs[1].set_title('Contour plot of the quadratic form (Positive Definite)')
fig.colorbar(contour2, ax=axs[1])
axs[1].grid(True)

plt.show()
