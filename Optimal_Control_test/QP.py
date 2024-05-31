# import numpy as np
import matplotlib.pyplot as plt
from autograd import jacobian, hessian
import autograd.numpy as np
from mpl_toolkits.mplot3d import Axes3D


# Define the objective function
def objective(x):
    return x[0] ** 2 + 2 * x[1] ** 2


# Define the constraint
def constraint(x):
    return x[0] - x[1] + 1


# Barrier method
def barrier_method():
    mu = 0.1  # Scaling factor for the barrier parameter
    t = 10.0  # Initial barrier parameter
    x, y = -1.5, 2.0  # Initialize x and y as floats
    epsilon = 1e-3
    iters = 100
    path = [(x, y)]
    # Armijo condition parameter
    c1 = 1e-4
    # Wolfe conditions parameters
    c2 = 0.9

    def barrier_function(x1):
        constraint_value = constraint(x1)
        if constraint_value >= 0:
            return np.inf  # Return a large number if constraint is violated
        return -np.log(-constraint_value)

    def combined_objective(x1):
        return objective(x1) + t * barrier_function(x1)

    jac_combined = jacobian(combined_objective)
    hess_combined = hessian(combined_objective)

    def wolfe_condition(x, p, alpha):
        f_x = combined_objective(x)
        f_x_alpha_p = combined_objective(x + alpha * p)
        grad_f_x = jac_combined(x)
        grad_f_x_alpha_p = jac_combined(x + alpha * p)

        return (f_x_alpha_p <= f_x + c1 * alpha * np.dot(grad_f_x, p) and
                np.dot(grad_f_x_alpha_p, p) >= c2 * np.dot(grad_f_x, p))

    def armijo_condition(x, p, alpha):
        return combined_objective(x + alpha * p) <= combined_objective(x) + c1 * alpha * np.dot(jac_combined(x), p)

    for _ in range(iters):
        print(f'Iteration {_}:')
        print(f'barrier_fun = {-barrier_function(np.array([x, y]))}')
        print(f'constraint = {constraint(np.array([x, y]))}')

        if abs(t * barrier_function(np.array([x, y]))) <= epsilon:
            break

        jac_vector = jac_combined(np.array([x, y]))
        print(f'jac_vector = {jac_vector}')
        hess_matrix = hess_combined(np.array([x, y]))
        print(f'hess_matrix = {hess_matrix}')

        hess_inv = np.linalg.inv(hess_matrix)
        print(f'hess_inv = {hess_inv}')

        delta = -hess_inv @ jac_vector
        print(f'delta = {delta}')

        # Wolfe condition line search
        alpha = 1.0
        # while not wolfe_condition(np.array([x, y]), delta, alpha):
        #     alpha *= 0.5

        # Armijo condition line search
        while not armijo_condition(np.array([x, y]), delta, alpha):
            alpha *= 0.8

        x += alpha * delta[0]
        y += alpha * delta[1]
        print(f'x = {x}, y = {y}')
        path.append((x, y))

        t *= mu  # Increase the barrier parameter
        print(f't = {t}\n')

    return path


# Interior point method
def interior_point_method():
    mu = 1
    x, y = -1, -1
    alpha = 0.01
    epsilon = 1e-6
    iters = 100
    path = [(x, y)]

    for _ in range(iters):
        if abs(constraint(x, y)) < epsilon:
            mu *= 0.9
        grad_x = 2 * x - mu * (1) / (x - y + 1)
        grad_y = 4 * y + mu * (1) / (x - y + 1)
        x -= alpha * grad_x
        y -= alpha * grad_y
        path.append((x, y))

    return path


# Augmented Lagrangian method
def augmented_lagrangian_method():
    mu = 1
    lambda_ = 1
    x, y = -1, -1
    alpha = 0.01
    epsilon = 1e-6
    iters = 100
    path = [(x, y)]
    for _ in range(iters):
        if abs(constraint(x, y)) < epsilon:
            break
        grad_x = 2 * x + lambda_ + mu * (x - y + 1)
        grad_y = 4 * y - lambda_ - mu * (x - y + 1)
        x -= alpha * grad_x
        y -= alpha * grad_y
        lambda_ += mu * constraint(x, y)
        path.append((x, y))
    return path


# Plot the objective function contours
def plot_contour():
    x = np.linspace(-4, 4, 400)
    y = np.linspace(-4, 4, 400)
    X, Y = np.meshgrid(x, y)
    Z = objective([X, Y])
    plt.contourf(X, Y, Z)


# Plot the constraint
def plot_constraint_2D():
    x = np.linspace(-4, 4, 400)
    y = x + 1
    plt.plot(x, y, 'y', label='Constraint')


def plot_constraint_3D():
    x_constraint = np.linspace(-2, 2, 400)
    y_constraint = x_constraint + 1
    z_constraint = x_constraint ** 2 + 2 * y_constraint ** 2
    plt.plot(x_constraint, y_constraint, z_constraint, label='Constraint: x - y + 1 = 0', color='b')


# Plot 2D
def plot_2D(paths, method_names):
    plot_contour()
    plot_constraint_2D()

    markers = ['o', 'x', 's']
    colors = ['r', 'g', 'b']
    for path, method_name, marker, color in zip(paths, method_names, markers, colors):
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], color=color, marker=marker, linestyle='-', label=method_name)

    plt.title('Optimization Paths')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.legend()
    plt.show()


def plot_meshes_3D():
    # Define the grid for X and Y
    X = np.linspace(-10, 10, 400)
    Y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(X, Y)
    Z = X ** 2 + 2 * Y ** 2

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis')


def plot_3D(paths, method_names):
    plot_meshes_3D()
    plot_constraint_3D()

    markers = ['o', 'x', 's']
    colors = ['r', 'g', 'b']
    for path, method_name, marker, color in zip(paths, method_names, markers, colors):
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], color=color, marker=marker, linestyle='-', label=method_name)
        path = np.array(path)
        X_path = path[:, 0]
        Y_path = path[:, 1]
        Z_path = X_path ** 2 + 2 * Y_path ** 2

        # Plot the optimization path
        plt.plot(X_path, Y_path, Z_path, marker='o', color=color, linestyle='-', label=method_name)

    plt.show()


# Visualize the iteration process for each method
path_barrier = barrier_method()
# path_interior = interior_point_method()
# path_augmented = augmented_lagrangian_method()

plot_2D([path_barrier], ['Barrier Method'])
plot_3D([path_barrier], ['Barrier Method'])
