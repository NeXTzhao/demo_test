#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>

#include "common/log.h"
#include "common/matplotlibcpp.h"

namespace plt = matplotlibcpp;

// 通用范数计算函数模板
template <int Order, int StateDim, int ControlDim>
typename std::enable_if<Order != 0, double>::type computeNorm(
    const std::vector<Eigen::Matrix<double, ControlDim, 1>>& delta_u) {
  double norm = 0.0;
  for (const auto& control : delta_u) {
    norm += std::pow(control.template lpNorm<Order>(), Order);
  }
  return std::pow(norm, 1.0 / Order);
}

// 特化无穷范数
template <int Order, int StateDim, int ControlDim>
typename std::enable_if<Order == 0, double>::type computeNorm(
    const std::vector<Eigen::Matrix<double, ControlDim, 1>>& delta_u) {
  double max_norm = 0.0;
  for (const auto& control : delta_u) {
    max_norm = std::max(max_norm, control.template lpNorm<Eigen::Infinity>());
  }
  return max_norm;
}

template <int StateDim, int ControlDim>
class Vehicle {
 public:
  using State = Eigen::Matrix<double, StateDim, 1>;
  using Control = Eigen::Matrix<double, ControlDim, 1>;

  enum StateIndex { X = 0, Y = 1, THETA = 2, V = 3 };
  enum ControlIndex { A = 0, DELTA = 1 };

  explicit Vehicle(double wheelbase, double max_acceleration,
                   double min_acceleration, double max_steering_angle,
                   double min_steering_angle)
      : L(wheelbase),
        max_a(max_acceleration),
        min_a(min_acceleration),
        max_delta(max_steering_angle),
        min_delta(min_steering_angle) {}

  State dynamics(const State& state, const Control& control) const {
    State dState;
    const double v = state(V);
    const double theta = state(THETA);
    const double delta = control(DELTA);
    dState(X) = v * cos(theta);          // dx
    dState(Y) = v * sin(theta);          // dy
    dState(THETA) = v / L * tan(delta);  // dtheta
    dState(V) = control(A);              // dv
    return dState;
  }

  State updateState(const State& state, Control& control, double dt) const {
    // Apply control limits
    control = applyControlLimits(control);

    State k1 = dynamics(state, control);
    State k2 = dynamics(state + 0.5 * dt * k1, control);
    State k3 = dynamics(state + 0.5 * dt * k2, control);
    State k4 = dynamics(state + dt * k3, control);

    State newState = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
    return newState;
  }

  void linearize(const State& state, const Control& control, double dt,
                 Eigen::MatrixXd& A, Eigen::MatrixXd& B) const {
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

  Control applyControlLimits(const Control& control) const {
    Control limited_control = control;
    limited_control(A) = std::max(min_a, std::min(max_a, control(A)));
    limited_control(DELTA) =
        std::max(min_delta, std::min(max_delta, control(DELTA)));
    return limited_control;
  }

 private:
  double L;                                   // 车辆轴距
  double max_a, min_a, max_delta, min_delta;  // 控制输入的限制
};

template <int StateDim, int ControlDim>
struct ShootingMethodConfig {
  double dt;
  int max_iter;
  double tol;
  double line_search_alpha;
  double line_search_tol;
  double line_search_beta;
  double line_search_alpha_min;

  explicit ShootingMethodConfig(double _dt, int _max_iter = 1000,
                                double _tol = 1e-2,
                                double _line_search_alpha = 1.0,
                                double _line_search_tol = 1e-3,
                                double _line_search_beta = 0.85,
                                double _line_search_alpha_min = 1e-6)
      : dt(_dt),
        max_iter(_max_iter),
        tol(_tol),
        line_search_alpha(_line_search_alpha),
        line_search_tol(_line_search_tol),
        line_search_beta(_line_search_beta),
        line_search_alpha_min(_line_search_alpha_min) {}
};

template <int StateDim, int ControlDim>
class ShootingMethod {
 public:
  using State = Eigen::Matrix<double, StateDim, 1>;
  using Control = Eigen::Matrix<double, ControlDim, 1>;

  ShootingMethod(const Vehicle<StateDim, ControlDim>& vehicle,
                 const ShootingMethodConfig<StateDim, ControlDim>& config)
      : vehicle(vehicle), config(config) {}

  std::pair<Eigen::MatrixXd, std::vector<Control>> solve(
      State& initialState, const Eigen::MatrixXd& targetTrajectory,
      std::vector<Control>& controls, const Eigen::MatrixXd& Q,
      const Eigen::MatrixXd& R, const Eigen::MatrixXd& Qn) {
    size_t N = targetTrajectory.cols();

    Eigen::MatrixXd states =
        simulate_system(initialState, controls, config.dt);  // 初始状态轨迹

    Eigen::MatrixXd initial_trajectory = states;
    Eigen::MatrixXd mid_trajectory;
    Eigen::MatrixXd final_trajectory;
    std::vector<Eigen::MatrixXd> process_trajectory;

    ADEBUG << "Target Trajectory Dimensions: " << targetTrajectory.rows() << "x"
           << targetTrajectory.cols();
    ADEBUG << "Initial State Dimensions: " << initialState.rows() << "x"
           << initialState.cols();
    ADEBUG << "State Trajectory Dimensions: " << states.rows() << "x"
           << states.cols();
    ADEBUG << "Q Matrix Dimensions: " << Q.rows() << "x" << Q.cols();
    ADEBUG << "R Matrix Dimensions: " << R.rows() << "x" << R.cols();
    ADEBUG << "Qn Matrix Dimensions: " << Qn.rows() << "x" << Qn.cols();

    for (int iter = 0; iter <= config.max_iter; ++iter) {
      AINFO << "===================" << " Iteration: " << iter
            << " ==========================";
      ADEBUG << "Initial state:\n" << initialState.transpose();
      ADEBUG << "State trajectory:\n" << states;

      Eigen::MatrixXd lambda = Eigen::MatrixXd::Zero(StateDim, N);
      std::vector<Control> delta_u(controls.size());

      std::vector<Eigen::MatrixXd> A_list(N - 1), B_list(N - 1);
      for (size_t k = 0; k < N - 1; ++k) {
        vehicle.linearize(states.col(k), controls[k], config.dt, A_list[k],
                          B_list[k]);
      }

      backward_pass(states, targetTrajectory, lambda, controls, delta_u, A_list,
                    B_list, Q, R, Qn);

      double cost_current = cost(states, controls, targetTrajectory, Q, R, Qn);
      bool success = line_search(initialState, states, controls, delta_u,
                                 targetTrajectory, Q, R, Qn, cost_current);

      double delta_u_norm = computeNorm<0, StateDim, ControlDim>(delta_u);

      if (delta_u_norm < config.tol) {
        AINFO << "Convergence achieved";
        final_trajectory = states;
        break;
      } else {
        AINFO << "Iteration " << iter
              << " times, still no convergence, Delta_U Infinite Norm: "
              << delta_u_norm;
      }

      if (iter == config.max_iter / 5) {
        mid_trajectory = states;
      }

      if (!success) {
        break;
      }
    }

    if (final_trajectory.cols() == 0) {
      final_trajectory = states;
    }
    // Visualize the trajectories
    visualize_trajectories(initial_trajectory, mid_trajectory, final_trajectory,
                           targetTrajectory);

    return {states, controls};  // Return optimized trajectory and controls
  }

 private:
  const Vehicle<StateDim, ControlDim>& vehicle;
  ShootingMethodConfig<StateDim, ControlDim> config;

  // 仿真系统函数
  Eigen::MatrixXd simulate_system(const State& x0, std::vector<Control>& u,
                                  double dt) {
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
  double cost(const Eigen::MatrixXd& x, const std::vector<Control>& u,
              const Eigen::MatrixXd& targetTrajectory, const Eigen::MatrixXd& Q,
              const Eigen::MatrixXd& R, const Eigen::MatrixXd& Qn) {
    // 计算终端成本
    Eigen::VectorXd terminal_error =
        x.col(x.cols() - 1) - targetTrajectory.col(targetTrajectory.cols() - 1);
    double terminal_cost =
        0.5 * (terminal_error.transpose() * Qn * terminal_error).value();

    // 计算运行成本
    double running_cost = 0.0;
    for (size_t k = 0; k < u.size(); ++k) {
      Eigen::Matrix<double, StateDim, 1> state_error =
          x.col(k) - targetTrajectory.col(k);
      running_cost +=
          0.5 * (state_error.transpose() * Q * state_error).value() +
          0.5 * (u[k].transpose() * R * u[k]).value();
    }
    //    AINFO << "-------------- Calculate Cost Process --------------------";
    //    ADEBUG << "Terminal Cost: " << terminal_cost
    //           << ", Running Cost: " << running_cost
    //           << ", Total Cost: " << terminal_cost + running_cost;
    return terminal_cost + running_cost;
  }

  // 反向步骤：计算协状态和控制输入增量
  void backward_pass(
      const Eigen::MatrixXd& x, const Eigen::MatrixXd& targetTrajectory,
      Eigen::MatrixXd& lambda, const std::vector<Control>& controls,
      std::vector<Control>& delta_u, const std::vector<Eigen::MatrixXd>& A_list,
      const std::vector<Eigen::MatrixXd>& B_list, const Eigen::MatrixXd& Q,
      const Eigen::MatrixXd& R, const Eigen::MatrixXd& Qn) {
    AINFO << "-------------- Backward pass Process --------------------";

    int N = (int)x.cols();

    // 终端协态
    lambda.col(N - 1) = Qn * (x.col(N - 1) - targetTrajectory.col(N - 1));

    for (int k = N - 2; k >= 0; --k) {
      // 计算协态
      lambda.col(k) = Q * (x.col(k) - targetTrajectory.col(k)) +
                      A_list[k].transpose() * lambda.col(k + 1);

      // 计算控制输入增量
      delta_u[k] = -R.inverse() * B_list[k].transpose() * lambda.col(k + 1);
      ADEBUG << "Backward pass k = " << k;
      ADEBUG << "A[" << k << "]:\n" << A_list[k];
      ADEBUG << "B[" << k << "]:\n" << B_list[k];
      ADEBUG << "Lambda[" << k << "]:\n" << lambda.col(k);
      ADEBUG << "Controls[" << k
             << "]: a = " << controls[k](Vehicle<StateDim, ControlDim>::A)
             << ", delta = "
             << controls[k](Vehicle<StateDim, ControlDim>::DELTA);
      ADEBUG << "Delta_u[" << k
             << "]: a = " << delta_u[k](Vehicle<StateDim, ControlDim>::A)
             << ", delta = "
             << delta_u[k](Vehicle<StateDim, ControlDim>::DELTA);
    }
  }

  std::vector<Control> updateControls(const std::vector<Control>& controls,
                                      const std::vector<Control>& delta_u,
                                      double alpha) const {
    std::vector<Control> u_new(controls.size());
    for (size_t k = 0; k < controls.size(); ++k) {
      u_new[k] = controls[k] + alpha * delta_u[k];
      u_new[k] = vehicle.applyControlLimits(u_new[k]);
    }
    return u_new;
  }

  // 线搜索过程
  bool line_search(State& initialState, Eigen::MatrixXd& states,
                   std::vector<Control>& controls,
                   const std::vector<Control>& delta_u,
                   const Eigen::MatrixXd& targetTrajectory,
                   const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                   const Eigen::MatrixXd& Qn, double cost_current) {
    AINFO << "-------------- Line Search Process --------------------";

    double alpha = config.line_search_alpha;
    double norm_delta_u = computeNorm<2, StateDim, ControlDim>(delta_u);
    int line_search_iterations = 0;

    while (alpha > config.line_search_alpha_min) {
      std::vector<Control> u_new = updateControls(controls, delta_u, alpha);
      Eigen::MatrixXd x_new = simulate_system(initialState, u_new, config.dt);

      double cost_new = cost(x_new, u_new, targetTrajectory, Q, R, Qn);

      ADEBUG << "Line search alpha: " << alpha
             << ", Current cost: " << cost_current << ", New cost: " << cost_new
             << ", Cost gradient: " << norm_delta_u << ", Goal cost: "
             << cost_current - config.line_search_tol * alpha * norm_delta_u
             << ", Actual reduction cost: "
             << cost_new - (cost_current -
                            config.line_search_tol * alpha * norm_delta_u);

      if (cost_new <=
          cost_current - config.line_search_tol * alpha * norm_delta_u) {
        controls = u_new;
        states = x_new;
        AINFO << "Updated controls accepted with alpha = " << alpha
              << ", step size after " << line_search_iterations
              << " iterations";
        return true;
      }

      alpha *= config.line_search_beta;
      line_search_iterations++;
    }

    AERROR << "Line search failed to find a valid step size after "
           << line_search_iterations << " iterations";
    return false;
  }

  // 可视化轨迹
  void visualize_trajectories(const Eigen::MatrixXd& initial_trajectory,
                              const Eigen::MatrixXd& mid_trajectory,
                              const Eigen::MatrixXd& final_trajectory,
                              const Eigen::MatrixXd& targetTrajectory) {
    std::vector<double> target_x, target_y, init_x, init_y, mid_x, mid_y,
        final_x, final_y;
    for (int i = 0; i < targetTrajectory.cols(); ++i) {
      target_x.push_back(targetTrajectory(Vehicle<StateDim, ControlDim>::X, i));
      target_y.push_back(targetTrajectory(Vehicle<StateDim, ControlDim>::Y, i));
    }
    for (int i = 0; i < initial_trajectory.cols(); ++i) {
      init_x.push_back(initial_trajectory(Vehicle<StateDim, ControlDim>::X, i));
      init_y.push_back(initial_trajectory(Vehicle<StateDim, ControlDim>::Y, i));
    }
    for (int i = 0; i < mid_trajectory.cols(); ++i) {
      mid_x.push_back(mid_trajectory(Vehicle<StateDim, ControlDim>::X, i));
      mid_y.push_back(mid_trajectory(Vehicle<StateDim, ControlDim>::Y, i));
    }
    for (int i = 0; i < final_trajectory.cols(); ++i) {
      final_x.push_back(final_trajectory(Vehicle<StateDim, ControlDim>::X, i));
      final_y.push_back(final_trajectory(Vehicle<StateDim, ControlDim>::Y, i));
    }

    plt::figure();
    plt::named_plot("Target Trajectory", target_x, target_y, "go-");
    plt::named_plot("Initial Trajectory", init_x, init_y, "ro-");
    plt::named_plot("Mid Trajectory", mid_x, mid_y, "bo-");
    plt::named_plot("Final Trajectory", final_x, final_y, "ko-");
    plt::xlabel("X");
    plt::ylabel("Y");
    plt::title("Trajectory Optimization");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
  }
};

int main() {
  double wheelbase = 2.5;
  double max_acceleration = 6.0;
  double min_acceleration = -6.0;
  double max_steering_angle = 0.85;
  double min_steering_angle = -0.85;
  double times = 5;
  double dt = 0.2;
  int steps = static_cast<int>(times / dt + 1.0);

  constexpr int StateDim = 4;
  constexpr int ControlDim = 2;

  Vehicle<StateDim, ControlDim> vehicle(wheelbase, max_acceleration,
                                        min_acceleration, max_steering_angle,
                                        min_steering_angle);

  // 定义目标轨迹
  Eigen::MatrixXd targetTrajectory(StateDim, steps);
  for (int i = 0; i < steps; ++i) {
    targetTrajectory(Vehicle<StateDim, ControlDim>::X, i) = i * 1.0;
    targetTrajectory(Vehicle<StateDim, ControlDim>::Y, i) = i * 1.0;
    targetTrajectory(Vehicle<StateDim, ControlDim>::THETA, i) = 0;
    targetTrajectory(Vehicle<StateDim, ControlDim>::V, i) = 1;
  }

  Vehicle<StateDim, ControlDim>::State initialState;
  initialState << 0, 0, 0.55, 0.50;

  ShootingMethodConfig<StateDim, ControlDim> config(dt);

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(StateDim, StateDim);
  Q.diagonal() << 1.0, 1.0, 6.5, 0.95;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(ControlDim, ControlDim);
  R.diagonal() << 5, 10;
  Eigen::MatrixXd Qn = Eigen::MatrixXd::Identity(StateDim, StateDim);
  Qn.diagonal() << 8, 8, 2, 2;

  std::vector<Vehicle<StateDim, ControlDim>::Control> controls(
      targetTrajectory.cols() - 1);  // Initial guess for controls

  // 初始化控制输入
  for (auto& control : controls) {
    control << 0.5, 0.1;  // 这里选择了较小的初始控制输入
  }

  ShootingMethod<StateDim, ControlDim> shooting(vehicle, config);
  auto [optimal_trajectory, optimal_controls] =
      shooting.solve(initialState, targetTrajectory, controls, Q, R, Qn);

  bool vis_fig = 1;
  if (vis_fig) {
    // 画图
    std::vector<double> trajectory_x, trajectory_y, opt_x, opt_y, control_a,
        control_delta;
    for (int i = 0; i < targetTrajectory.cols(); ++i) {
      trajectory_x.push_back(
          targetTrajectory(Vehicle<StateDim, ControlDim>::X, i));
      trajectory_y.push_back(
          targetTrajectory(Vehicle<StateDim, ControlDim>::Y, i));
    }
    for (int i = 0; i < optimal_trajectory.cols(); ++i) {
      opt_x.push_back(optimal_trajectory(Vehicle<StateDim, ControlDim>::X, i));
      opt_y.push_back(optimal_trajectory(Vehicle<StateDim, ControlDim>::Y, i));
    }
    for (const auto& control : optimal_controls) {
      control_a.push_back(control(Vehicle<StateDim, ControlDim>::A));
      control_delta.push_back(control(Vehicle<StateDim, ControlDim>::DELTA));
    }

    plt::figure();

    plt::subplot(2, 1, 1);
    plt::named_plot("Acceleration", control_a, "ro-");
    plt::xlabel("Time step");
    plt::ylabel("Acceleration (a)");
    plt::title("Control Inputs");
    plt::grid(true);
    plt::legend();

    plt::subplot(2, 1, 2);
    plt::named_plot("Steering angle", control_delta, "go-");
    plt::xlabel("Time step");
    plt::ylabel("Steering angle (delta)");
    plt::legend();
    plt::tight_layout();
    plt::grid(true);
    plt::show();
  }
  return 0;
}
