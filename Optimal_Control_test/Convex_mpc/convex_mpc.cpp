#include <Eigen/Dense>
#include <cmath>
#include <utility>
#include <vector>

#include "common/log.h"
#include "common/matplotlibcpp.h"
#include "proxsuite/proxqp/dense/dense.hpp"

namespace plt = matplotlibcpp;
using namespace proxsuite::proxqp;

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

  explicit Vehicle(double wheelbase, double _max_speed, double _min_speed,
                   double _max_acceleration, double _min_acceleration,
                   double _max_steering_angle, double _min_steering_angle)
      : L(wheelbase),
        max_speed(_max_speed),
        min_speed(_min_speed),
        max_a(_max_acceleration),
        min_a(_min_acceleration),
        max_delta(_max_steering_angle),
        min_delta(_min_steering_angle) {}

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
  double L;  // 车辆轴距
 public:
  double max_speed, min_speed, max_a, min_a, max_delta,
      min_delta;  // 控制输入的限制
};

template <int StateDim, int ControlDim>
class MPC {
 public:
  using State = typename Vehicle<StateDim, ControlDim>::State;
  using Control = typename Vehicle<StateDim, ControlDim>::Control;

  MPC(const Vehicle<StateDim, ControlDim>& vehicle, int horizon, double dt,
      Eigen::MatrixXd Q, Eigen::MatrixXd R, Eigen::MatrixXd Q_N)
      : vehicle_(vehicle),
        horizon_(horizon),
        dt_(dt),
        Q_(std::move(Q)),
        R_(std::move(R)),
        Q_N_(std::move(Q_N)) {}

  std::vector<Control> solve(const State& initial_state,
                             const std::vector<State>& reference_trajectory) {
    // 初始化优化变量
    std::vector<State> state(horizon_, State::Zero());
    std::vector<Control> control(horizon_ - 1, Control::Zero());

    // 初始化状态
    state[0] = initial_state;

    const int num_vars = (StateDim + ControlDim) * (horizon_ - 1);
    const int num_eq_constraints = StateDim * (horizon_ - 1);
    const int num_in_constraints = (StateDim + ControlDim) * (horizon_ - 1);

    Eigen::MatrixXd H;
    Eigen::VectorXd g;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    Eigen::MatrixXd C;
    Eigen::VectorXd l;
    Eigen::VectorXd u;

    // 设置初始目标函数
    SetCostFunction(H, g, reference_trajectory);

    // 设置初始等式约束
    Eigen::MatrixXd A_t(StateDim, StateDim);
    Eigen::MatrixXd B_t(StateDim, ControlDim);
    vehicle_.linearize(initial_state, control[0], dt_, A_t, B_t);
    SetEqualityConstraints(A, b, A_t, B_t, initial_state);

    // 设置初始不等式约束
    SetInequalityConstraints(C, l, u);

    // cost function
    AINFO << "H dimensions: (" << C.rows() << ", " << C.cols() << ")";
    AINFO << "g dimensions: (" << g.rows() << ", " << g.cols() << ")";
    // dynamic function
    AINFO << "A_t dimensions: (" << A_t.rows() << ", " << A_t.cols() << ")";
    AINFO << "B_t dimensions: (" << B_t.rows() << ", " << B_t.cols() << ")";
    // equality constraints
    AINFO << "A dimensions: (" << A.rows() << ", " << A.cols() << ")";
    AINFO << "b dimensions: (" << b.rows() << ", " << b.cols() << ")";
    // inequality constraints
    AINFO << "C dimensions: (" << C.rows() << ", " << C.cols() << ")";
    AINFO << "l dimensions: (" << l.rows() << ", " << l.cols() << ")";
    AINFO << "u dimensions: (" << u.rows() << ", " << u.cols() << ")";

    AINFO << "Initializing and solving QP problem";
    dense::QP<double> qp_solver(num_vars, num_eq_constraints,
                                num_in_constraints);
    qp_solver.init(H, g, A, b, C, l, u);

    for (int k = 0; k < horizon_ - 1; ++k) {
      // 更新不等式约束
      UpdateInequalityConstraints(l, u, state[k], control[k]);

      // 重新求解优化问题
      qp_solver.update(H, g, A, b, C, l, u);
      qp_solver.solve();

      // 提取当前时间步的控制输入增量 Δu
      Eigen::VectorXd solution = qp_solver.results.x;
      Eigen::VectorXd delta_u = solution.segment(0, ControlDim);

      // 更新控制输入
      control[k] += delta_u;

      // 使用系统动力学模型更新状态
      state[k + 1] = vehicle_.updateState(state[k], control[k], dt_);

      std::cout << "Control input increment delta_u: " << delta_u.transpose()
                << std::endl;
      std::cout << "Ref state: " << reference_trajectory[k + 1].transpose() << std::endl;
      std::cout << "New state: " << state[k + 1].transpose() << std::endl;

      // 更新下一步的线性化和约束
      vehicle_.linearize(state[k + 1], control[k], dt_, A_t, B_t);
      SetEqualityConstraints(A, b, A_t, B_t, state[k + 1]);
    }

    trajectory_ = state;

    AINFO << "MPC solve complete";
    return control;
  }

  const std::vector<State>& getTrajectory() const { return trajectory_; }

 private:
  const Vehicle<StateDim, ControlDim>& vehicle_;
  int horizon_;
  double dt_;
  std::vector<State> trajectory_;
  Eigen::MatrixXd Q_, R_, Q_N_;

  void SetCostFunction(Eigen::MatrixXd& H, Eigen::VectorXd& g,
                       const std::vector<State>& reference_trajectory) {
    int num_rows = (StateDim + ControlDim) * (horizon_ - 1);
    int num_cols = (StateDim + ControlDim) * (horizon_ - 1);

    H = Eigen::MatrixXd::Zero(num_rows, num_cols);
    g = Eigen::VectorXd::Zero(num_rows);

    Eigen::MatrixXd Q_R =
        Eigen::MatrixXd::Zero(StateDim + ControlDim, StateDim + ControlDim);
    Q_R.block(0, 0, StateDim, StateDim) = Q_;
    Q_R.block(StateDim, StateDim, ControlDim, ControlDim) = R_;

    H.block(0, 0, ControlDim, ControlDim) = R_;

    int blockSize = StateDim + ControlDim;
    for (int i = 1; i < horizon_ - 1; ++i) {
      H.block((i - 1) * blockSize + ControlDim,
              (i - 1) * blockSize + ControlDim, blockSize, blockSize) = Q_R;
    }

    H.block(num_rows - StateDim, num_cols - StateDim, StateDim, StateDim) =
        Q_N_;

    for (int i = 1; i < horizon_; ++i) {
      const State& xref = reference_trajectory[i];
      g.segment(ControlDim + (i - 1) * blockSize, StateDim) = -Q_ * xref;
    }
    const State& x_n_ref = reference_trajectory.back();
    std::cout << "x_back: \n" << x_n_ref <<std::endl;
    g.tail(StateDim) = -Q_N_ * x_n_ref;

    ADEBUG << "Quadratic function matrix H:\n" << H;
    ADEBUG << "Linear function matrix g:\n" << g;
  }

  void SetEqualityConstraints(Eigen::MatrixXd& A, Eigen::VectorXd& b,
                              const Eigen::MatrixXd& A_t,
                              const Eigen::MatrixXd& B_t,
                              const Eigen::VectorXd& x0) {
    int num_eq_constraints = StateDim * (horizon_ - 1);
    int a_cols = (ControlDim + StateDim) * (horizon_ - 1);
    A = Eigen::MatrixXd::Zero(num_eq_constraints, a_cols);
    b = Eigen::VectorXd::Zero(num_eq_constraints);

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(StateDim, StateDim);

    Eigen::MatrixXd B_I =
        Eigen::MatrixXd::Zero(StateDim, StateDim + ControlDim);
    B_I.block(0, 0, StateDim, ControlDim) = B_t;
    B_I.block(0, ControlDim, StateDim, StateDim) = -I;

    for (int i = 0; i < horizon_ - 1; ++i) {
      if (i == 0) {
        A.block(0, 0, StateDim, StateDim + ControlDim) = B_I;
      } else {
        A.block(i * StateDim, i * (StateDim + ControlDim) - StateDim, StateDim,
                StateDim) = A_t;
        A.block(i * StateDim, i * (StateDim + ControlDim), StateDim,
                StateDim + ControlDim) = B_I;
      }
    }

    b.segment(0, StateDim) = -A_t * x0;
    ADEBUG << "Equality constraint matrix A:\n" << A;
    ADEBUG << "Equality constraint vector b:\n" << b;
  }

  void SetInequalityConstraints(Eigen::MatrixXd& C, Eigen::VectorXd& l,
                                Eigen::VectorXd& u) {
    int num_in_constraints = (StateDim + ControlDim) * (horizon_ - 1);

    C = Eigen::MatrixXd::Identity(num_in_constraints, num_in_constraints);
    l = Eigen::VectorXd::Constant(num_in_constraints, -1e10);
    u = Eigen::VectorXd::Constant(num_in_constraints, 1e10);

    double min_a = vehicle_.min_a;
    double max_a = vehicle_.max_a;
    double min_delta = vehicle_.min_delta;
    double max_delta = vehicle_.max_delta;
    double min_v = vehicle_.min_speed;
    double max_v = vehicle_.max_speed;
    double inf = std::numeric_limits<double>::infinity();

    // State bounds
    Eigen::VectorXd MIN_X(StateDim);
    MIN_X << -inf, -inf, -inf, min_v;

    Eigen::VectorXd MAX_X(StateDim);
    MAX_X << inf, inf, inf, max_v;

    // Control bounds
    Eigen::VectorXd MIN_U(ControlDim);
    MIN_U << min_a, min_delta;

    Eigen::VectorXd MAX_U(ControlDim);
    MAX_U << max_a, max_delta;

    // Combine MIN_X and MIN_U into l_block
    Eigen::VectorXd l_block(StateDim + ControlDim);
    l_block.head(StateDim) = MIN_X;
    l_block.tail(ControlDim) = MIN_U;

    // Combine MAX_X and MAX_U into u_block
    Eigen::VectorXd u_block(StateDim + ControlDim);
    u_block.head(StateDim) = MAX_X;
    u_block.tail(ControlDim) = MAX_U;

    // u1
    l.segment(0, ControlDim) = MIN_U;
    u.segment(0, ControlDim) = MAX_U;

    // x_n
    l.segment(num_in_constraints - StateDim, StateDim) = MIN_X;
    u.segment(num_in_constraints - StateDim, StateDim) = MAX_X;

    for (int t = 1; t < horizon_ - 1; ++t) {
      int start_idx = (t - 1) * (StateDim + ControlDim) + ControlDim;
      l.segment(start_idx, StateDim + ControlDim) = l_block;
      u.segment(start_idx, StateDim + ControlDim) = u_block;
    }

    ADEBUG << "Inequality constraint matrix C:\n" << C;
    ADEBUG << "Inequality lower bound vector l:\n" << l;
    ADEBUG << "Inequality upper bound vector u:\n" << u;
  }

  void UpdateInequalityConstraints(Eigen::VectorXd& l, Eigen::VectorXd& u,
                                   const State& state, const Control& control) {
    double min_a = vehicle_.min_a;
    double max_a = vehicle_.max_a;
    double min_delta = vehicle_.min_delta;
    double max_delta = vehicle_.max_delta;
    double min_v = vehicle_.min_speed;
    double max_v = vehicle_.max_speed;
    double inf = std::numeric_limits<double>::infinity();

    Eigen::VectorXd MIN_X(StateDim);
    MIN_X << -inf, -inf, -inf, min_v;

    Eigen::VectorXd MAX_X(StateDim);
    MAX_X << inf, inf, inf, max_v;

    Eigen::VectorXd MIN_U(ControlDim);
    MIN_U << min_a, min_delta;

    Eigen::VectorXd MAX_U(ControlDim);
    MAX_U << max_a, max_delta;

    Eigen::VectorXd l_block(StateDim + ControlDim);
    l_block.head(StateDim) = MIN_X;
    l_block.tail(ControlDim) = MIN_U;

    Eigen::VectorXd u_block(StateDim + ControlDim);
    u_block.head(StateDim) = MAX_X;
    u_block.tail(ControlDim) = MAX_U;

    l.segment(0, ControlDim) = MIN_U;
    u.segment(0, ControlDim) = MAX_U;

    l.segment(l.size() - StateDim, StateDim) = MIN_X;
    u.segment(u.size() - StateDim, StateDim) = MAX_X;

    for (int t = 1; t < horizon_ - 1; ++t) {
      int start_idx = (t - 1) * (StateDim + ControlDim) + ControlDim;
      l.segment(start_idx, StateDim + ControlDim) = l_block;
      u.segment(start_idx, StateDim + ControlDim) = u_block;
    }
  }
};

int main() {
  const int StateDim = 4;
  const int ControlDim = 2;

  double wheelbase = 2.5;
  double max_speed = 120 / 3.6;
  double min_speed = 0.0;
  double max_acceleration = 10.0;
  double min_acceleration = -10.0;
  double max_steering_angle = 2;
  double min_steering_angle = -2;

  int horizon = 10;
  double dt = 0.2;

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(StateDim, StateDim);
  Q.diagonal() << 50.0, 50.0, 1.0, 0.5;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(ControlDim, ControlDim);
  R.diagonal() << 1.0, 1.0;
  Eigen::MatrixXd Q_N = Eigen::MatrixXd::Identity(StateDim, StateDim);
  Q_N.diagonal() << 10, 10, 10, 0.1;

  Vehicle<StateDim, ControlDim> vehicle(wheelbase, max_speed, min_speed,
                                        max_acceleration, min_acceleration,
                                        max_steering_angle, min_steering_angle);


  MPC<StateDim, ControlDim> mpc(vehicle, horizon, dt, Q, R, Q_N);

  Vehicle<StateDim, ControlDim>::State initial_state;
  initial_state << 0.0, 0.0, 0.4, 5;

  std::vector<Vehicle<StateDim, ControlDim>::State> targetTrajectory(
      horizon, initial_state);
  // 设置参考轨迹为简单的直线
  for (int i = 0; i < horizon; ++i) {
    targetTrajectory[i](Vehicle<StateDim, ControlDim>::X) = i;
    targetTrajectory[i](Vehicle<StateDim, ControlDim>::Y) = i;
    targetTrajectory[i](Vehicle<StateDim, ControlDim>::THETA) = 0.79;
    targetTrajectory[i](Vehicle<StateDim, ControlDim>::V) = 5;
  }

  auto control_inputs = mpc.solve(initial_state, targetTrajectory);

  // 提取轨迹
  const auto& trajectory = mpc.getTrajectory();

  // 提取轨迹中的X和Y坐标
  std::vector<double> x_ref, y_ref, x_coords, y_coords;
  for (const auto& state : targetTrajectory) {
    x_ref.push_back(state(Vehicle<StateDim, ControlDim>::X));
    y_ref.push_back(state(Vehicle<StateDim, ControlDim>::Y));
  }
  for (const auto& state : trajectory) {
    x_coords.push_back(state(Vehicle<StateDim, ControlDim>::X));
    y_coords.push_back(state(Vehicle<StateDim, ControlDim>::Y));
  }

  // 可视化轨迹
  plt::named_plot("ref", x_ref, y_ref, "bo-");
  plt::named_plot("opt", x_coords, y_coords, "ro-");
  plt::xlabel("X");
  plt::ylabel("Y");
  plt::grid(true);
  plt::legend();
  plt::title("Vehicle Trajectory");
  plt::show();

  return 0;
}