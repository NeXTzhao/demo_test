#include <Eigen/Dense>
#include <cmath>
#include <utility>
#include <vector>

#include "common/log.h"
#include "common/matplotlibcpp.h"
#include "nlohmann/json.hpp"
#include "proxsuite/proxqp/dense/dense.hpp"

namespace plt = matplotlibcpp;
using json = nlohmann::json;
using namespace proxsuite::proxqp;

const std::string root_path =
    "/home/next/demo_test/Optimal_Control_test/Convex_mpc/data/";

// 写入数据到CSV文件的函数
void writeDataToCSV(const std::string& filename,
                    const std::vector<std::vector<double>>& data,
                    const std::vector<std::string>& headers) {
  std::ofstream file(filename);
  for (const auto& header : headers) {
    file << header << ",";
  }
  file << "\n";

  for (size_t i = 0; i < data[0].size(); ++i) {
    for (const auto& column : data) {
      file << column[i] << ",";
    }
    file << "\n";
  }

  file.close();
}

// 写入Eigen矩阵到CSV文件的函数
void writeEigenToCSV(const std::string& filename,
                     const Eigen::MatrixXd& matrix) {
  std::ofstream file(filename);
  if (file.is_open()) {
    for (int i = 0; i < matrix.rows(); ++i) {
      for (int j = 0; j < matrix.cols(); ++j) {
        file << matrix(i, j);
        if (j < matrix.cols() - 1) {
          file << ",";
        }
      }
      file << "\n";
    }
    file.close();
  } else {
    std::cerr << "Unable to open file " << filename << std::endl;
  }
}

// 读取参数文件
json readParams(const std::string& filename) {
  std::ifstream file(filename);
  json params;
  file >> params;
  return params;
}
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
    std::vector<Control> control(horizon_, Control::Zero());

    // 初始化状态
    state[0] = initial_state;
    control[0] << 0.1, 0.1;

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
    qp_solver.solve();

    Eigen::VectorXd solution = qp_solver.results.x;

    AINFO << "X: \n" << solution;
    for (int i = 0; i < horizon_ - 1; ++i) {
      if (i == 0) {
        control[i + 1] = solution.segment(0, ControlDim);
      } else {
        int index = (i - 1) * (StateDim + ControlDim) + ControlDim;
        state[i] = solution.segment(index, StateDim);
        control[i] = solution.segment(index + StateDim, ControlDim);
      }
    }
    state.back() = solution.tail(StateDim);
    control.back() = control[control.size() - 2];

    // 调试信息
    std::vector<double> objective_values, primal_residuals, dual_residuals;
    objective_values.push_back(qp_solver.results.info.objValue);
    primal_residuals.push_back(qp_solver.results.info.pri_res);
    dual_residuals.push_back(qp_solver.results.info.dua_res);

    trajectory_ = state;
    writeDataToCSV(root_path + "debug.csv",
                   {objective_values, primal_residuals, dual_residuals},
                   {"objective_value", "primal_residual", "dual_residual"});

    // 保存矩阵 H, g, A, b, C, l, u
    writeEigenToCSV(root_path + "H.csv", H);
    writeEigenToCSV(root_path + "g.csv", g);
    writeEigenToCSV(root_path + "A.csv", A);
    writeEigenToCSV(root_path + "b.csv", b);
    writeEigenToCSV(root_path + "C.csv", C);
    writeEigenToCSV(root_path + "l.csv", l);
    writeEigenToCSV(root_path + "u.csv", u);

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

    for (int i = 1; i < horizon_ - 1; ++i) {
      const State& xref = reference_trajectory[i];
      g.segment(ControlDim + (i - 1) * blockSize, StateDim) = -Q_ * xref;
    }
    const State& x_n_ref = reference_trajectory.back();
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

    b.head(StateDim) = -A_t * x0;
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
    //    double inf = std::numeric_limits<double>::infinity();`
    double inf = 1e10;

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

  void UpdateEqualityConstraints(const Eigen::MatrixXd& A_t,
                                 const Eigen::VectorXd& x0,
                                 Eigen::VectorXd& b) {
    b.head(StateDim) = -A_t * x0;
    ADEBUG << "Equality constraint vector b:\n" << b;
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
  // 读取JSON文件
  std::ifstream i(root_path + "params.json");
  json j;
  i >> j;

  const int StateDim = 4;
  const int ControlDim = 2;

  double wheelbase = j["wheelbase"];
  double max_speed = j["max_speed"];
  double min_speed = j["min_speed"];
  double max_acceleration = j["max_acceleration"];
  double min_acceleration = j["min_acceleration"];
  double max_steering_angle = j["max_steering_angle"];
  double min_steering_angle = j["min_steering_angle"];
  int horizon = j["horizon"];
  double dt = j["dt"];

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(StateDim, StateDim);
  Q.diagonal() << j["Q"][0], j["Q"][1], j["Q"][2], j["Q"][3];
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(ControlDim, ControlDim);
  R.diagonal() << j["R"][0], j["R"][1];
  Eigen::MatrixXd Q_N = Eigen::MatrixXd::Identity(StateDim, StateDim);
  Q_N.diagonal() << j["Q_N"][0], j["Q_N"][1], j["Q_N"][2], j["Q_N"][3];

  Vehicle<StateDim, ControlDim> vehicle(wheelbase, max_speed, min_speed,
                                        max_acceleration, min_acceleration,
                                        max_steering_angle, min_steering_angle);

  MPC<StateDim, ControlDim> mpc(vehicle, horizon, dt, Q, R, Q_N);

  Vehicle<StateDim, ControlDim>::State initial_state;
  initial_state << j["init_state"][0], j["init_state"][1], j["init_state"][2],
      j["init_state"][3];

  std::vector<Vehicle<StateDim, ControlDim>::State> targetTrajectory(
      horizon, initial_state);
  // 设置参考轨迹为简单的直线
  for (int i = 0; i < horizon; ++i) {
    double t = i;
    targetTrajectory[i](Vehicle<StateDim, ControlDim>::X) = t;
    targetTrajectory[i](Vehicle<StateDim, ControlDim>::Y) = t;
    targetTrajectory[i](Vehicle<StateDim, ControlDim>::THETA) = 0.79;
    targetTrajectory[i](Vehicle<StateDim, ControlDim>::V) = j["ref"][3];
  }

  auto control_inputs = mpc.solve(initial_state, targetTrajectory);

  // 提取轨迹
  const auto& trajectory = mpc.getTrajectory();

  // 提取轨迹中的X和Y坐标
  std::vector<double> x_ref, y_ref, x_coords, y_coords, v_coords, theta_coords;
  for (const auto& state : targetTrajectory) {
    x_ref.push_back(state(Vehicle<StateDim, ControlDim>::X));
    y_ref.push_back(state(Vehicle<StateDim, ControlDim>::Y));
  }
  for (const auto& state : trajectory) {
    x_coords.push_back(state(Vehicle<StateDim, ControlDim>::X));
    y_coords.push_back(state(Vehicle<StateDim, ControlDim>::Y));
    theta_coords.push_back(state(Vehicle<StateDim, ControlDim>::THETA));
    v_coords.push_back(state(Vehicle<StateDim, ControlDim>::V));
  }
  // 提取控制量中的A和DELTA
  std::vector<double> a_values, delta_values;
  for (const auto& control : control_inputs) {
    a_values.push_back(control(Vehicle<StateDim, ControlDim>::A));
    delta_values.push_back(control(Vehicle<StateDim, ControlDim>::DELTA));
  }

  // 将所有数据写入一个CSV文件
  writeDataToCSV(root_path + "data.csv",
                 {x_ref, y_ref, x_coords, y_coords, v_coords, theta_coords,
                  a_values, delta_values},
                 {"x_ref", "y_ref", "x_coords", "y_coords", "velocity", "theta",
                  "acceleration", "steering_angle"});

  bool flag = false;
  if (flag) {
    // 设置画面大小并可视化轨迹
    plt::figure_size(1200, 800);
    plt::named_plot("Reference", x_ref, y_ref, "bo-");
    plt::named_plot("Optimized", x_coords, y_coords, "ro-");
    plt::xlabel("X");
    plt::ylabel("Y");
    plt::grid(true);
    plt::legend();
    plt::title("Vehicle Trajectory");

    // 设置画面大小并可视化控制量
    plt::figure_size(1200, 1200);
    plt::subplot(4, 1, 1);
    plt::named_plot("Acceleration", a_values, "ro-");
    plt::ylabel("Acceleration");
    plt::grid(true);
    plt::legend();
    plt::title("Control Inputs");

    plt::subplot(4, 1, 2);
    plt::named_plot("Steering Angle", delta_values, "bo-");
    plt::xlabel("Time step");
    plt::ylabel("Steering Angle");
    plt::grid(true);
    plt::legend();

    plt::subplot(4, 1, 3);
    plt::named_plot("Velocity", v_coords, "go-");
    plt::ylabel("Velocity");
    plt::grid(true);
    plt::title("Vehicle States");

    plt::subplot(4, 1, 4);
    plt::named_plot("Theta", theta_coords, "mo-");
    plt::xlabel("Time step");
    plt::ylabel("Theta");
    plt::grid(true);

    plt::tight_layout();
    plt::show();
  }
  return 0;
}