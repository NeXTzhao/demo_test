import matplotlib.pyplot as plt
from autograd import jacobian, hessian
import autograd.numpy as np
import scipy.linalg


class Optimizer:
    def __init__(self, objective, constraint, x_init, y_init, epsilon=1e-6, iters=100, c1=1e-4, c2=0.9):
        self.objective = objective
        self.constraint = constraint
        self.x_init = x_init
        self.y_init = y_init
        self.epsilon = epsilon
        self.iters = iters
        self.c1 = c1
        self.c2 = c2

    def armijo_condition(self, combined_objective, jac_combined, x, p, alpha):
        return combined_objective(x + alpha * p) <= combined_objective(x) + self.c1 * alpha * np.dot(jac_combined(x), p)

    def strong_wolfe_condition(self, combined_objective, jac_combined, x, p, alpha):
        return (self.armijo_condition(combined_objective, jac_combined, x, p, alpha) and
                abs(np.dot(jac_combined(x + alpha * p), p)) <= self.c2 * abs(np.dot(jac_combined(x), p)))

    def weak_wolfe_condition(self, combined_objective, jac_combined, x, p, alpha):
        return (self.armijo_condition(combined_objective, jac_combined, x, p, alpha) and
                np.dot(jac_combined(x + alpha * p), p) >= self.c2 * np.dot(jac_combined(x), p))

    # 矩阵分解求逆函数

    @staticmethod
    def matrix_inverse(hess_matrix):
        if np.all(np.linalg.eigvals(hess_matrix) > 0):  # 检查 Hessian 矩阵是否正定
            # 使用 Cholesky 分解
            L = np.linalg.cholesky(hess_matrix)
            L_T = L.T
            hess_inv = np.linalg.solve(L_T, np.linalg.solve(L, np.eye(hess_matrix.shape[0])))
        else:
            # 检查 Hessian 矩阵是否是非奇异方阵
            if np.linalg.matrix_rank(hess_matrix) == hess_matrix.shape[0]:
                # 使用 LU 分解
                lu, piv = scipy.linalg.lu_factor(hess_matrix)
                hess_inv = scipy.linalg.lu_solve((lu, piv), np.eye(hess_matrix.shape[0]))
            else:
                # 使用 QR 分解
                Q, R = np.linalg.qr(hess_matrix)
                hess_inv = np.linalg.solve(R, Q.T)

        return hess_inv

    def newton_step(self, jac_combined, hess_combined, x, y):
        jac_vector = jac_combined(np.array([x, y]))
        hess_matrix = hess_combined(np.array([x, y]))
        # hess_inv = np.linalg.inv(hess_matrix)
        hess_inv = self.matrix_inverse(hess_matrix)
        delta = -hess_inv @ jac_vector
        return delta

    @staticmethod
    def damping_newton_step(jac_combined, hess_combined, x, y, damping=1e-4):
        jac_vector = jac_combined(np.array([x, y]))
        hess_matrix = hess_combined(np.array([x, y]))
        hess_matrix += damping * np.eye(hess_matrix.shape[0])  # 添加阻尼因子
        hess_inv = np.linalg.inv(hess_matrix)
        delta = -hess_inv @ jac_vector
        return delta

    def penalty_method(self, mu=100, t=1):
        print('Penalty Method')
        x, y = self.x_init, self.y_init
        path = [(x, y)]

        def penalty_fun(x1):
            return 0.5 * t * self.constraint(x1) ** 2

        def combined_objective(x1):
            return self.objective(x1) + penalty_fun(x1)

        jac_vector = jacobian(combined_objective)
        hess_matrix = hessian(combined_objective)

        index = 0
        for _ in range(self.iters):
            index += 1
            if abs(penalty_fun(np.array([x, y]))) <= self.epsilon:
                break

            delta = self.newton_step(jac_vector, hess_matrix, x, y)

            alpha = 1.0
            while not self.armijo_condition(combined_objective, jac_vector, np.array([x, y]), delta, alpha):
                alpha *= 0.5

            x += alpha * delta[0]
            y += alpha * delta[1]
            path.append((x, y))

            t *= mu  # Increase the barrier parameter

            print(f"Iteration {index + 1}:")
            print(f"x = {x}, y = {y}")
            print(f"Objective value = {self.objective(np.array([x, y]))}")
            print(f"Constraint value = {self.constraint(np.array([x, y]))}")
            print(f"Step size = {alpha}\n")
        return path

    def barrier_method(self, mu=0.1, t=10.0):
        print('Barrier Method')
        x, y = self.x_init, self.y_init
        path = [(x, y)]

        def barrier_function(x1):
            constraint_value = self.constraint(x1)
            if constraint_value >= 0:
                return np.inf  # Return a large number if constraint is violated
            return -t * np.log(-constraint_value)

        def combined_objective(x1):
            return self.objective(x1) + barrier_function(x1)

        jac_vector = jacobian(combined_objective)
        hess_matrix = hessian(combined_objective)

        index = 0
        for _ in range(self.iters):
            index += 1
            if abs(barrier_function(np.array([x, y]))) <= self.epsilon:
                break

            delta = self.newton_step(jac_vector, hess_matrix, x, y)

            alpha = 1.0
            while not self.armijo_condition(combined_objective, jac_vector, np.array([x, y]), delta, alpha):
                alpha *= 0.5

            x += alpha * delta[0]
            y += alpha * delta[1]
            path.append((x, y))

            t *= mu  # Increase the barrier parameter

            print(f"Iteration {index + 1}:")
            print(f"x = {x}, y = {y}")
            print(f"Objective value = {self.objective(np.array([x, y]))}")
            print(f"Constraint value = {self.constraint(np.array([x, y]))}")
            print(f"Step size = {alpha}\n")

        return path

    def augmented_lagrangian_method(self, mu=10.0, lambda_=1.0, rho=2):
        print('Augmented Lagrangian Method')
        x, y = self.x_init, self.y_init
        path = [(x, y)]

        def lagrangian(x1):
            return self.objective(x1) + lambda_ * self.constraint(x1) + (mu / 2) * self.constraint(x1) ** 2

        jac_vector = jacobian(lagrangian)
        hess_matrix = hessian(lagrangian)
        index = 0
        for _ in range(self.iters):
            index += 1
            # delta = self.damping_newton_step(jac_vector, hess_matrix, x, y)
            delta = self.newton_step(jac_vector, hess_matrix, x, y)

            alpha = 1.0
            while not self.armijo_condition(lagrangian, jac_vector, np.array([x, y]), delta, alpha):
                alpha *= 0.5

            x += alpha * delta[0]
            y += alpha * delta[1]

            path.append((x, y))

            lambda_ += mu * self.constraint(np.array([x, y]))

            if abs(self.constraint(np.array([x, y]))) > self.epsilon:
                mu *= rho

            if abs(self.constraint(np.array([x, y]))) < self.epsilon:
                break

            print(f"Iteration {index + 1}:")
            print(f"x = {x}, y = {y}")
            print(f"Objective value = {self.objective(np.array([x, y]))}")
            print(f"Constraint value = {self.constraint(np.array([x, y]))}")
            print(f"Step size = {alpha}\n")
        return path


# Define the objective function
def objective(x):
    return x[0] ** 2 + 2 * x[1] ** 2


# Define the constraint
def constraint(x):
    return x[0] - x[1] + 1


# Initialize optimizer
optimizer = Optimizer(objective, constraint, x_init=-3.0, y_init=2.0)

# Run methods
path_barrier = optimizer.barrier_method()
path_penalty = optimizer.penalty_method()
path_augmented_lagrangian = optimizer.augmented_lagrangian_method()


# Plot the objective function contours
def plot_combined(paths, method_names):
    # 2D plot
    fig1, ax1 = plt.subplots()
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    Z = objective([X, Y])

    ax1.contour(X, Y, Z)
    ax1.plot(x, x + 1, 'y', label='Constraint: x - y + 1 = 0')

    colors = ['r', 'g', 'b', 'm', 'c', 'y']
    markers = ['o', 'x', 's', 'd', '^', 'v']

    for path, method_name, color, marker in zip(paths, method_names, colors, markers):
        path = np.array(path)
        ax1.plot(path[:, 0], path[:, 1], color=color, marker=marker, linestyle='-',
                 label=(method_name + ' (iterator:' + str(path[:, 0].size) + ' )'))

    ax1.set_title('Optimization Paths (2D)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    fig1.tight_layout()

    # 3D plot
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis')

    x_constraint = np.linspace(-3, 3, 100)
    y_constraint = x_constraint + 1
    z_constraint = x_constraint ** 2 + 2 * y_constraint ** 2
    ax2.plot(x_constraint, y_constraint, z_constraint, label='Constraint: x - y + 1 = 0', color='b')

    for path, method_name, color, marker in zip(paths, method_names, colors, markers):
        path = np.array(path)
        X_path = path[:, 0]
        Y_path = path[:, 1]
        Z_path = X_path ** 2 + 2 * Y_path ** 2
        ax2.plot(X_path, Y_path, Z_path, marker=marker, color=color, linestyle='-', label=method_name)

    ax2.set_title('Optimization Paths (3D)')
    ax2.legend()
    fig2.tight_layout()

    plt.show()


plot_combined([path_barrier, path_penalty, path_augmented_lagrangian],
              ['Barrier Method', 'Penalty Method', 'Augmented Lagrangian Method'])
