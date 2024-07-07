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
    const std::vector<Eigen::Matrix<double, ControlDim, 1>> &delta_u) {
  double norm = 0.0;
  for (const auto &control : delta_u) {
    norm += std::pow(control.template lpNorm<Order>(), Order);
  }
  return std::pow(norm, 1.0 / Order);
}

// 特化无穷范数
template <int Order, int StateDim, int ControlDim>
typename std::enable_if<Order == 0, double>::type computeNorm(
    const std::vector<Eigen::Matrix<double, ControlDim, 1>> &delta_u) {
  double max_norm = 0.0;
  for (const auto &control : delta_u) {
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

  State dynamics(const State &state, const Control &control) const {
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

  State updateState(const State &state, Control &control, double dt) const {
    // Apply control limits
    control = applyControlLimits(control);

    State k1 = dynamics(state, control);
    State k2 = dynamics(state + 0.5 * dt * k1, control);
    State k3 = dynamics(state + 0.5 * dt * k2, control);
    State k4 = dynamics(state + dt * k3, control);

    State newState = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
    return newState;
  }

  void linearize(const State &state, const Control &control, double dt,
                 Eigen::MatrixXd &A, Eigen::MatrixXd &B) const {
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

  Control applyControlLimits(const Control &control) const {
    Control limited_control = control;
    limited_control(A) = std::max(min_a, std::min(max_a, control(A)));
    limited_control(DELTA) =
        std::max(min_delta, std::min(max_delta, control(DELTA)));
    return limited_control;
  }

 private:
  double L;  // 车辆轴距
 public:
  double max_a, min_a, max_delta, min_delta;  // 控制输入的限制
};

template <int StateDim, int ControlDim>
class MPC {
 public:
  using State = typename Vehicle<StateDim, ControlDim>::State;
  using Control = typename Vehicle<StateDim, ControlDim>::Control;

  MPC(const Vehicle<StateDim, ControlDim> &vehicle, int horizon, double dt,
      Eigen::MatrixXd Q, Eigen::MatrixXd R, Eigen::MatrixXd Q_N)
      : vehicle_(vehicle),
        horizon_(horizon),
        dt_(dt),
        Q_(std::move(Q)),
        R_(std::move(R)),
        Q_N_(std::move(Q_N)) {}

  std::vector<Control> solve(const State &initial_state,
                             const std::vector<State> &reference_trajectory) {
    // 初始化优化变量
    std::vector<State> state(horizon_ , State::Zero());
    std::vector<Control> control(horizon_-1, Control::Zero());

    // 初始化状态
    state[0] = initial_state;

    const int num_vars = (StateDim + ControlDim) * (horizon_ - 1);
    const int num_eq_constraints = StateDim * (horizon_ - 1);
    const int num_in_constraints = (StateDim + ControlDim) * (horizon_ - 1);

    Eigen::MatrixXd H;
    Eigen::MatrixXd A_t(StateDim, StateDim);
    Eigen::MatrixXd B_t(StateDim, ControlDim);
    vehicle_.linearize(initial_state, control[0], dt_, A_t, B_t);

    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    Eigen::MatrixXd C;
    Eigen::VectorXd l;
    Eigen::VectorXd u;

    SetCostFunction(H, reference_trajectory);
    SetEqualityConstraints(A, b, A_t, B_t, initial_state);
    SetInequalityConstraints(C, l, u);

    AINFO << "H dimensions: (" << C.rows() << ", " << C.cols() << ")";
    AINFO << "A dimensions: (" << A.rows() << ", " << A.cols() << ")";
    AINFO << "b dimensions: (" << b.rows() << ", " << b.cols() << ")";
    AINFO << "A_t dimensions: (" << A_t.rows() << ", " << A_t.cols() << ")";
    AINFO << "B_t dimensions: (" << B_t.rows() << ", " << B_t.cols() << ")";
    AINFO << "C dimensions: (" << C.rows() << ", " << C.cols() << ")";
    AINFO << "l dimensions: (" << l.rows() << ", " << l.cols() << ")";
    AINFO << "u dimensions: (" << u.rows() << ", " << u.cols() << ")";

    AINFO << "Initializing and solving QP problem";
    dense::QP<double> qp_solver(num_vars, num_eq_constraints,
                                num_in_constraints);
    qp_solver.init(H, Eigen::VectorXd::Zero(num_vars), A, b, C, l, u);
    qp_solver.solve();

    Eigen::VectorXd solution = qp_solver.results.x;

    // 初始化提取的控制输入和状态变量列表
    std::vector<State> opt_x;
    std::vector<Control> opt_u;

    // 提取初始状态
    opt_x.push_back(initial_state);

    // 遍历 solution，交替提取 u 和 x
    for (int i = 0; i < horizon_ - 1; ++i) {
      Eigen::VectorXd u_i =
          solution.segment(i * (StateDim + ControlDim), ControlDim);
      Eigen::VectorXd x_i =
          solution.segment(i * (StateDim + ControlDim) + ControlDim, StateDim);
      opt_u.push_back(u_i);
      opt_x.push_back(x_i);
    }

    // 提取最后一个控制输入和状态变量
    Eigen::VectorXd u_last = solution.segment(
        (horizon_ - 1) * (StateDim + ControlDim) - ControlDim, ControlDim);
    opt_u.push_back(u_last);

    // 将轨迹数据保存到成员变量
    trajectory_ = state;

    AINFO << "MPC solve complete";
    return control;
  }

  const std::vector<State> &getTrajectory() const { return trajectory_; }

 private:
  const Vehicle<StateDim, ControlDim> &vehicle_;
  int horizon_;
  double dt_;
  std::vector<State> trajectory_;
  Eigen::MatrixXd Q_, R_, Q_N_;

  /*  void SetCostFunction(Eigen::MatrixXd &H,
                         const std::vector<State> &reference_trajectory) {
      Eigen::MatrixXd Q =
          Eigen::MatrixXd::Identity(StateDim, StateDim);  // 状态代价矩阵
      Eigen::MatrixXd R =
          Eigen::MatrixXd::Identity(ControlDim, ControlDim);  //
    控制输入代价矩阵 Eigen::MatrixXd Q_N = Eigen::MatrixXd::Identity(StateDim,
    StateDim);  // 终端状态代价矩阵

      // 计算总变量数
      int num_rows = (StateDim + ControlDim) * (horizon_ - 1);
      int num_cols = (StateDim + ControlDim) * (horizon_ - 1);

      // 初始化H矩阵
      H = Eigen::MatrixXd::Zero(num_rows, num_cols);
      Eigen::MatrixXd Q_R =
          Eigen::MatrixXd::Zero(StateDim + ControlDim, StateDim + ControlDim);
      Q_R.block(0, 0, StateDim, StateDim) = Q;
      Q_R.block(StateDim, StateDim, ControlDim, ControlDim) = R;

      int blockSize = StateDim + ControlDim;
      for (int i = 1; i < horizon_ - 1; ++i) {
        if (i == 0) {
          H.block(0, 0, ControlDim, ControlDim) = R;
        } else {
          H.block((i - 1) * blockSize + ControlDim,
                  (i - 1) * blockSize + ControlDim, blockSize, blockSize) = Q_R;
        }
      }

      // 填充最后一行的终端状态代价矩阵
      H.block(num_rows - StateDim, num_cols - StateDim, StateDim, StateDim) =
    Q_N;

      ADEBUG << "Cost function matrix H:\n" << H;
    }
    */
  void SetCostFunction(Eigen::MatrixXd &H,
                       const std::vector<State> &reference_trajectory) {
    int num_rows = (StateDim + ControlDim) * (horizon_ - 1);
    int num_cols = (StateDim + ControlDim) * (horizon_ - 1);

    H = Eigen::MatrixXd::Zero(num_rows, num_cols);
    Eigen::MatrixXd Q_R =
        Eigen::MatrixXd::Zero(StateDim + ControlDim, StateDim + ControlDim);
    Q_R.block(0, 0, StateDim, StateDim) = Q_;
    Q_R.block(StateDim, StateDim, ControlDim, ControlDim) = R_;

    int blockSize = StateDim + ControlDim;
    for (int i = 1; i < horizon_ - 1; ++i) {
      H.block((i - 1) * blockSize + ControlDim,
              (i - 1) * blockSize + ControlDim, blockSize, blockSize) = Q_R;
    }

    H.block(num_rows - StateDim, num_cols - StateDim, StateDim, StateDim) =
        Q_N_;

    for (int i = 0; i < horizon_ - 1; ++i) {
      Eigen::MatrixXd Q_ref = Q_;
      for (int j = 0; j < StateDim; ++j) {
        Q_ref(j, j) *=
            (reference_trajectory[i + 1](j) - reference_trajectory[i](j));
      }
      H.block(i * blockSize, i * blockSize, StateDim, StateDim) += Q_ref;
    }

    ADEBUG << "Cost function matrix H:\n" << H;
  }

  void SetEqualityConstraints(Eigen::MatrixXd &A, Eigen::VectorXd &b,
                              const Eigen::MatrixXd &A_t,
                              const Eigen::MatrixXd &B_t,
                              const Eigen::VectorXd &x0) {
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

  void SetInequalityConstraints(Eigen::MatrixXd &C, Eigen::VectorXd &l,
                                Eigen::VectorXd &u) {
    int num_in_constraints = (StateDim + ControlDim) * (horizon_ - 1);

    C = Eigen::MatrixXd::Identity(num_in_constraints, num_in_constraints);
    l = Eigen::VectorXd::Constant(num_in_constraints, -1e10);
    u = Eigen::VectorXd::Constant(num_in_constraints, 1e10);

    double min_a = vehicle_.min_a;
    double max_a = vehicle_.max_a;
    double min_delta = vehicle_.min_delta;
    double max_delta = vehicle_.max_delta;
    double min_v = 0.0;
    double max_v = 120 / 3.6;
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
};

int main() {
  const int StateDim = 4;
  const int ControlDim = 2;

  double wheelbase = 2.5;
  double max_acceleration = 3.0;
  double min_acceleration = -3.0;
  double max_steering_angle = 0.5;
  double min_steering_angle = -0.5;

  Eigen::MatrixXd Q =
      Eigen::MatrixXd::Identity(StateDim, StateDim);  // 状态代价矩阵
  Eigen::MatrixXd R =
      Eigen::MatrixXd::Identity(ControlDim, ControlDim);  // 控制输入代价矩阵
  Eigen::MatrixXd Q_N =
      Eigen::MatrixXd::Identity(StateDim, StateDim);  // 终端状态代价矩阵

  Vehicle<StateDim, ControlDim> vehicle(wheelbase, max_acceleration,
                                        min_acceleration, max_steering_angle,
                                        min_steering_angle);

  int horizon = 10;
  double dt = 0.1;
  MPC<StateDim, ControlDim> mpc(vehicle, horizon, dt, Q, R, Q_N);

  Vehicle<StateDim, ControlDim>::State initial_state;
  initial_state << 1, 1, 1, 1;

  std::vector<Vehicle<StateDim, ControlDim>::State> reference_trajectory(
      horizon + 1, initial_state);
  // 设置参考轨迹为简单的直线
  for (int i = 0; i <= horizon; ++i) {
    reference_trajectory[i](Vehicle<StateDim, ControlDim>::X) = i * dt;
  }

  auto control_inputs = mpc.solve(initial_state, reference_trajectory);

  // 提取轨迹
  const auto &trajectory = mpc.getTrajectory();

  // 提取轨迹中的X和Y坐标
  std::vector<double> x_coords;
  std::vector<double> y_coords;
  for (const auto &state : trajectory) {
    x_coords.push_back(state(Vehicle<StateDim, ControlDim>::X));
    y_coords.push_back(state(Vehicle<StateDim, ControlDim>::Y));
  }

  // 可视化轨迹
  plt::plot(x_coords, y_coords, "ro-");
  plt::xlabel("X");
  plt::ylabel("Y");
  plt::title("Vehicle Trajectory");
  plt::show();

  return 0;
}