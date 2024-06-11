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

    def newton_step(self, jac_combined, hess_combined, x, y):
        jac_vector = jac_combined(np.array([x, y]))
        hess_matrix = hess_combined(np.array([x, y]))
        hess_inv = self.matrix_inverse(hess_matrix)
        delta = -hess_inv @ jac_vector
        return delta, np.linalg.cond(hess_matrix)  # Calculate condition number

    @staticmethod
    def matrix_inverse(hess_matrix):
        if np.all(np.linalg.eigvals(hess_matrix) > 0):
            L = np.linalg.cholesky(hess_matrix)
            L_T = L.T
            hess_inv = np.linalg.solve(L_T, np.linalg.solve(L, np.eye(hess_matrix.shape[0])))
        else:
            if np.linalg.matrix_rank(hess_matrix) == hess_matrix.shape[0]:
                lu, piv = scipy.linalg.lu_factor(hess_matrix)
                hess_inv = scipy.linalg.lu_solve((lu, piv), np.eye(hess_matrix.shape[0]))
            else:
                Q, R = np.linalg.qr(hess_matrix)
                hess_inv = np.linalg.solve(R, Q.T)
        return hess_inv

    def penalty_method(self, mu=10, t=1):
        print('Penalty Method')
        x, y = self.x_init, self.y_init
        path = [(x, y)]
        condition_numbers = []
        t_values = [t]

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

            delta, cond_num = self.newton_step(jac_vector, hess_matrix, x, y)
            condition_numbers.append(cond_num)
            t_values.append(t)

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
        return path, condition_numbers, t_values

    def plot_combined_path_and_condition_numbers(self, path, condition_numbers):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot the contour and path
        x_vals = np.linspace(-4, 4, 100)
        y_vals = np.linspace(-4, 4, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = self.objective([X, Y])
        # plt.contourf
        ax1.contourf(X, Y, Z, levels=15)
        ax1.plot(x_vals, x_vals + 1, 'y', label='Constraint: x - y + 1 = 0')

        path = np.array(path)
        ax1.plot(path[:, 0], path[:, 1], color='r', marker='o', linestyle='-', label='Penalty Method')

        ax1.set_title('Optimization Path and Contour')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.legend()

        # Plot the condition numbers as a line plot with annotations
        ax2.plot(np.arange(len(condition_numbers)), condition_numbers, color='r', marker='o', linestyle='-', label='Condition Number')
        for i, cond in enumerate(condition_numbers):
            ax2.annotate(f'{cond:.2e}', (i, cond), textcoords="offset points", xytext=(0,10), ha='center')

        ax2.set_title('Condition Numbers')
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Condition Number')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_contours_for_different_t(self, t_values):
        unique_t_values = sorted(set(t_values))
        n_plots = len(unique_t_values)-3
        fig, axes = plt.subplots(1, 4, figsize=(5 * n_plots, 5))

        x_vals = np.linspace(-4, 4, 100)
        y_vals = np.linspace(-4, 4, 100)
        X, Y = np.meshgrid(x_vals, y_vals)

        for i, (t, ax) in enumerate(zip(unique_t_values, axes)):
            Z = self.objective([X, Y]) + 0.5 * t * (self.constraint([X, Y]) ** 2)

            ax.contour(X, Y, Z, levels=200)
            # ax.plot(x_vals, x_vals + 1, 'y', label='Constraint: x - y + 1 = 0')

            ax.set_title(f'Contour at Sigma={t:.2e}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            # ax.legend()

        plt.tight_layout()
        plt.show()


# Define the objective function
def objective(x):
    return x[0] ** 2 + 2 * x[1] ** 2


# Define the constraint
def constraint(x):
    return x[0] - x[1] + 1


# Initialize optimizer
optimizer = Optimizer(objective, constraint, x_init=-3.0, y_init=2.0)

# Run penalty method
path_penalty, cond_penalty, t_values_penalty = optimizer.penalty_method()

# Plot the combined path and condition numbers
optimizer.plot_combined_path_and_condition_numbers(path_penalty, cond_penalty)

# Plot contours for different t values
optimizer.plot_contours_for_different_t(t_values_penalty)
