#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

// 通用范数计算函数模板
template<int Order, int StateDim, int ControlDim>
typename std::enable_if<Order != 0, double>::type
computeNorm(const std::vector<Eigen::Matrix<double, ControlDim, 1>> &delta_u) {
  double norm = 0.0;
  for (const auto &control : delta_u) {
    double control_norm = 0.0;
    for (int i = 0; i < ControlDim; ++i) {
      control_norm += std::pow(std::abs(control(i)), Order);
    }
    norm += control_norm;
  }
  return std::pow(norm, 1.0 / Order);
}

// 特化无穷范数
template<int Order, int StateDim, int ControlDim>
typename std::enable_if<Order == 0, double>::type
computeNorm(const std::vector<Eigen::Matrix<double, ControlDim, 1>> &delta_u) {
  double max_norm = 0.0;
  for (const auto &control : delta_u) {
    for (int i = 0; i < ControlDim; ++i) {
      double current_norm = std::abs(control(i));
      if (current_norm > max_norm) {
        max_norm = current_norm;
      }
    }
  }
  return max_norm;
}

template<int StateDim, int ControlDim>
class Vehicle {
 public:
  using State = Eigen::Matrix<double, StateDim, 1>;
  using Control = Eigen::Matrix<double, ControlDim, 1>;

  enum StateIndex { X = 0, Y = 1, THETA = 2, V = 3 };
  enum ControlIndex { A = 0, DELTA = 1 };

  explicit Vehicle(double wheelbase) : L(wheelbase) {}

  State dynamics(const State &state, const Control &control) const {
    State dState;
    dState(X) = state(V) * cos(state(THETA));           // dx
    dState(Y) = state(V) * sin(state(THETA));           // dy
    dState(THETA) = state(V) / L * tan(control(DELTA)); // dtheta
    dState(V) = control(A);                             // dv
    return dState;
  }

  State updateState(const State &state, const Control &control, double dt) const {
    State k1 = dynamics(state, control);
    State k2 = dynamics(state + 0.5 * dt * k1, control);
    State k3 = dynamics(state + 0.5 * dt * k2, control);
    State k4 = dynamics(state + dt * k3, control);

    State newState = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
    return newState;
  }

  void linearize(const State &state, const Control &control, double dt, Eigen::MatrixXd &A, Eigen::MatrixXd &B) const {
    A = Eigen::MatrixXd::Identity(StateDim, StateDim);
    B = Eigen::MatrixXd::Zero(StateDim, ControlDim);

    double v = state(V);
    double theta = state(THETA);
    double delta = control(DELTA);
    double sin_theta = sin(theta);
    double cos_theta = cos(theta);
    double cos_delta = cos(delta);
    double tan_delta = tan(delta);

    // clang-format off
    A << 1, 0, -v * sin_theta * dt, cos_theta * dt,
        0, 1, v * cos_theta * dt, sin_theta * dt,
        0, 0, 1, tan_delta * dt / L,
        0, 0, 0, 1;

    B << 0, 0,
        0, 0,
        0, v * dt / (L * cos_delta * cos_delta),
        dt, 0;
    // clang-format on
  }

 private:
  double L;  // 车辆轴距
};

template<int StateDim, int ControlDim>
struct ShootingMethodConfig {
  double dt;
  int max_iter;
  double tol;
  double line_search_tol;

  explicit ShootingMethodConfig(double dt, int max_iter = 100, double tol = 1e-2, double line_search_tol = 1e-2)
      : dt(dt), max_iter(max_iter), tol(tol), line_search_tol(line_search_tol) {}
};

template<int StateDim, int ControlDim>
class ShootingMethod {
 public:
  using State = Eigen::Matrix<double, StateDim, 1>;
  using Control = Eigen::Matrix<double, ControlDim, 1>;

  ShootingMethod(const Vehicle<StateDim, ControlDim> &vehicle,
                 const ShootingMethodConfig<StateDim, ControlDim> &config)
      : vehicle(vehicle), config(config) {}

  std::pair<Eigen::MatrixXd, std::vector<Control>> solve(State &initialState,
                                                         const Eigen::MatrixXd &targetTrajectory,
                                                         std::vector<Control> &controls,
                                                         const Eigen::MatrixXd &Q,
                                                         const Eigen::MatrixXd &R,
                                                         const Eigen::MatrixXd &Qn) {
    size_t N = targetTrajectory.cols();

    Eigen::MatrixXd x = simulate_system(initialState, controls, config.dt); // 初始状态轨迹

    for (int iter = 0; iter < config.max_iter; ++iter) {
      std::cout << "Iteration " << iter << std::endl;
      std::cout << "Initial state:\n" << initialState.transpose() << std::endl;
      std::cout << "State trajectory:\n" << x << std::endl;

      Eigen::MatrixXd lambda = Eigen::MatrixXd::Zero(StateDim, N);
      std::vector<Control> delta_u(controls.size());

      std::vector<Eigen::MatrixXd> A_list(N - 1), B_list(N - 1);
      for (size_t k = 0; k < N - 1; ++k) {
        vehicle.linearize(x.col(k), controls[k], config.dt, A_list[k], B_list[k]);
        std::cout << "A[" << k << "]:\n" << A_list[k] << "\n";
        std::cout << "B[" << k << "]:\n" << B_list[k] << "\n";
      }

      backward_pass(x, targetTrajectory, lambda, controls, delta_u, A_list, B_list, Q, R, Qn);

      std::cout << "Lambda (co-state) values after backward pass:\n" << lambda << std::endl;
      std::cout << "Control update values (delta_u):\n";
      for (size_t k = 0; k < delta_u.size(); ++k) {
        std::cout << "delta_u[" << k << "]: a = " << delta_u[k](Vehicle<StateDim, ControlDim>::A)
                  << ", delta = " << delta_u[k](Vehicle<StateDim, ControlDim>::DELTA) << std::endl;
      }

      double alpha = 1.0;
      while (true) {
        std::vector<Control> u_new = controls;
        for (size_t k = 0; k < u_new.size(); ++k) {
          u_new[k](Vehicle<StateDim, ControlDim>::A) += alpha * delta_u[k](Vehicle<StateDim, ControlDim>::A);
          u_new[k](Vehicle<StateDim, ControlDim>::DELTA) += alpha * delta_u[k](Vehicle<StateDim, ControlDim>::DELTA);
        }

        Eigen::MatrixXd x_new = simulate_system(initialState, u_new, config.dt);
        double cost_new = cost(x_new, u_new, targetTrajectory, Q, R, Qn);
        double cost_current = cost(x, controls, targetTrajectory, Q, R, Qn);

        std::cout << "Alpha: " << alpha << ", Cost new: " << cost_new << ", Cost current: " << cost_current
                  << std::endl;
        std::cout << "Updated state trajectory (x_new):\n" << x_new << std::endl;

        if (cost_new <= cost_current - config.line_search_tol * computeNorm<2, StateDim, ControlDim>(delta_u)) {
          controls = u_new;
          x = x_new;
          std::cout << "Updated controls accepted with alpha = " << alpha << std::endl;
          break;
        }
        alpha *= 0.5;

        if (alpha < 1e-8) {
          std::cout << "Line search failed to find a valid step size" << std::endl;
          break;
        }
      }

      std::cout << "Control norm: " << computeNorm<0, StateDim, ControlDim>(delta_u) << std::endl;

      if (computeNorm<0, StateDim, ControlDim>(delta_u) < config.tol) {
        std::cout << "Convergence achieved." << std::endl;
        break;
      }
    }

    return {x, controls}; // Return optimized trajectory and controls
  }

 private:
  const Vehicle<StateDim, ControlDim> &vehicle;
  ShootingMethodConfig<StateDim, ControlDim> config;

  // 仿真系统函数
  Eigen::MatrixXd simulate_system(const State &x0, const std::vector<Control> &u, double dt) {
    size_t N = u.size() + 1;
    Eigen::MatrixXd x(StateDim, N);
    State state = x0;
    x.col(0) = state;

    for (size_t k = 0; k < u.size(); ++k) {
      state = vehicle.updateState(state, u[k], dt);
      x.col(k + 1) = state;
    }
    return x;
  }

  // 计算目标函数值
  double cost(const Eigen::MatrixXd &x,
              const std::vector<Control> &u,
              const Eigen::MatrixXd &targetTrajectory,
              const Eigen::MatrixXd &Q,
              const Eigen::MatrixXd &R,
              const Eigen::MatrixXd &Qn) {
    // 计算终端成本
    Eigen::VectorXd terminal_error = x.col(x.cols() - 1) - targetTrajectory.col(targetTrajectory.cols() - 1);
    double terminal_cost = 0.5 * (terminal_error.transpose() * Qn * terminal_error).value();

    // 计算运行成本
    double running_cost = 0.0;
    for (size_t k = 0; k < u.size(); ++k) {
      Eigen::Matrix<double, StateDim, 1> state_error = x.col(k) - targetTrajectory.col(k);
      running_cost +=
          0.5 * (state_error.transpose() * Q * state_error).value() + 0.5 * (u[k].transpose() * R * u[k]).value();
    }
    return terminal_cost + running_cost;
  }

  // 反向步骤：计算协状态和控制输入增量
  void backward_pass(const Eigen::MatrixXd &x,
                     const Eigen::MatrixXd &targetTrajectory,
                     Eigen::MatrixXd &lambda,
                     std::vector<Control> &controls,
                     std::vector<Control> &delta_u,
                     const std::vector<Eigen::MatrixXd> &A_list,
                     const std::vector<Eigen::MatrixXd> &B_list,
                     const Eigen::MatrixXd &Q,
                     const Eigen::MatrixXd &R,
                     const Eigen::MatrixXd &Qn) {
    size_t N = x.cols();

    lambda.col(N - 1) = Qn * (x.col(N - 1) - targetTrajectory.col(N - 1));
    for (int k = N - 2; k >= 0; --k) {
      delta_u[k](Vehicle<StateDim, ControlDim>::A) = -(controls[k](Vehicle<StateDim, ControlDim>::A) +
                                                       (B_list[k].transpose() * lambda.col(k + 1))(0,0) / R(0, 0));
      delta_u[k](Vehicle<StateDim, ControlDim>::DELTA) = -(controls[k](Vehicle<StateDim, ControlDim>::DELTA) +
                                                           (B_list[k].transpose() * lambda.col(k + 1))(1,0) / R(1, 1));
      lambda.col(k) = Q * (x.col(k) - targetTrajectory.col(k)) + A_list[k].transpose() * lambda.col(k + 1);
    }
  }
};

int main() {
  double wheelbase = 2.5;
  constexpr int StateDim = 4;
  constexpr int ControlDim = 2;

  Vehicle<StateDim, ControlDim> vehicle(wheelbase);

  // 定义目标轨迹
  Eigen::MatrixXd targetTrajectory(StateDim, 10);
  for (int i = 0; i < 10; ++i) {
    targetTrajectory(Vehicle<StateDim, ControlDim>::X, i) = i * 1.0;
    targetTrajectory(Vehicle<StateDim, ControlDim>::Y, i) = i * 1.0;
    targetTrajectory(Vehicle<StateDim, ControlDim>::THETA, i) = 0;
    targetTrajectory(Vehicle<StateDim, ControlDim>::V, i) = 1;
  }

  Vehicle<StateDim, ControlDim>::State initialState;
  initialState << 0, 0, 0, 1;
  double dt = 0.2;

  ShootingMethodConfig<StateDim, ControlDim> config(dt);

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(StateDim, StateDim);
  Q(0, 0) = 10;
  Q(1, 1) = 10;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(ControlDim, ControlDim);
//  R(0, 0) = 10;
  R(1, 1) = 1;
  Eigen::MatrixXd Qn = Eigen::MatrixXd::Identity(StateDim, StateDim);

  std::vector<Vehicle<StateDim, ControlDim>::Control> controls(
      targetTrajectory.cols() - 1); // Initial guess for controls

  ShootingMethod<StateDim, ControlDim> shooting(vehicle, config);
  auto [optimal_trajectory, optimal_controls] = shooting.solve(initialState, targetTrajectory, controls, Q, R, Qn);

  for (const auto &control : optimal_controls) {
    std::cout << "a: " << control(Vehicle<StateDim, ControlDim>::A) << ", delta: "
              << control(Vehicle<StateDim, ControlDim>::DELTA) << std::endl;
  }

  std::vector<double> trajectory_x, trajectory_y, opt_x, opt_y, control_a, control_delta;
  for (int i = 0; i < targetTrajectory.cols(); ++i) {
    trajectory_x.push_back(targetTrajectory(Vehicle<StateDim, ControlDim>::X, i));
    trajectory_y.push_back(targetTrajectory(Vehicle<StateDim, ControlDim>::Y, i));
  }
  for (int i = 0; i < optimal_trajectory.cols(); ++i) {
    opt_x.push_back(optimal_trajectory(Vehicle<StateDim, ControlDim>::X, i));
    opt_y.push_back(optimal_trajectory(Vehicle<StateDim, ControlDim>::Y, i));
  }
  for (const auto &control : optimal_controls) {
    control_a.push_back(control(Vehicle<StateDim, ControlDim>::A));
    control_delta.push_back(control(Vehicle<StateDim, ControlDim>::DELTA));
  }

  // Plotting the trajectory
  plt::figure();
  plt::named_plot("Ref Trajectory", trajectory_x, trajectory_y, "b.");
  plt::named_plot("Opt Trajectory", opt_x, opt_y, "r.");

  plt::xlabel("X");
  plt::ylabel("Y");
  plt::title("Vehicle Trajectory");
  plt::legend();
//  plt::show();

  // Plotting the controls
  plt::figure();
  plt::subplot(2, 1, 1);
  plt::named_plot("Acceleration", control_a, "ro-");
  plt::xlabel("Time step");
  plt::ylabel("Acceleration (a)");
  plt::title("Control Inputs");
  plt::legend();

  plt::subplot(2, 1, 2);
  plt::named_plot("Steering angle", control_delta, "g*-");
  plt::xlabel("Time step");
  plt::ylabel("Steering angle (delta)");
  plt::legend();
  plt::show();

  return 0;
}
