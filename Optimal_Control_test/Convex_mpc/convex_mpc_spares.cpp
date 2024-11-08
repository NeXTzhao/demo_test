#include <Eigen/Dense>
#include <cmath>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>  // used for generating a random convex qp
#include <utility>
#include <vector>

#include "common/log.h"
#include "common/matplotlibcpp.h"
#include "nlohmann/json.hpp"
#include "proxsuite/proxqp/dense/dense.hpp"
#include "proxsuite/proxqp/sparse/sparse.hpp"

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

  enum StateIndex {
    X_POS = 0,
    Y_POS = 1,
    SPEED = 2,
    THETA = 3,  /* heading */
    DELTAV = 4, /* steering */
    OMEGA = 5,  /* d(steering) */
    ODOM = 6,
    ACCEL = 7,
    X_DIM = 8
  };
  enum ControlIndex {
    JERK = 0,   /* d(a) */
    ALPHAV = 1, /* dd(steering) */
    U_DIM = 2
  };

  explicit Vehicle(double wheelbase, double _max_speed, double _min_speed,
                   double _max_acceleration, double _min_acceleration,
                   double _max_steering_angle, double _min_steering_angle,
                   double _max_alpha, double _min_alpha, double _max_jerk,
                   double _min_jerk)
      : L(wheelbase),
        max_speed(_max_speed),
        min_speed(_min_speed),
        max_a(_max_acceleration),
        min_a(_min_acceleration),
        max_delta(_max_steering_angle),
        min_delta(_min_steering_angle),
        max_alpha(_max_alpha),
        min_alpha(_min_alpha),
        max_jerk(_max_jerk),
        min_jerk(_min_jerk) {}

  State EvalOneStep(const State& x, const Control& u, double increment,
                    int USE_RK = 4) const {
    const State& limited_x = x;
    const Control& limited_u = u;
    State next_state;

    // RK-2 method
    if (USE_RK == 2) {
      const State k1 = dynamics(limited_x, limited_u);
      const State k2 = dynamics(limited_x + 0.5 * increment * k1, limited_u);
      next_state = limited_x + increment * k2;
    }

    // RK-4 method
    if (USE_RK == 4) {
      const State k1 = dynamics(limited_x, limited_u);
      const State k2 = dynamics(limited_x + 0.5 * increment * k1, limited_u);
      const State k3 = dynamics(limited_x + 0.5 * increment * k2, limited_u);
      const State k4 = dynamics(limited_x + increment * k3, limited_u);
      next_state = limited_x + (increment / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
      //   std::cout << "k1: " << k1.transpose() << std::endl;
      //   std::cout << "k2: " << k2.transpose() << std::endl;
      //   std::cout << "k3: " << k3.transpose() << std::endl;
      //   std::cout << "k4: " << k4.transpose() << std::endl;
      //   std::cout << "next_state: " << next_state.transpose() <<
      //        std::endl;
    }
    return next_state;
  }

  State dynamics(const State& x, const Control& u) const {
    State diff;
    // clang-format off
    diff << x(StateIndex::SPEED) * std::cos(x(StateIndex::THETA)),                                      // NOLINT
            x(StateIndex::SPEED) * std::sin(x(StateIndex::THETA)),                                      // NOLINT
            x(StateIndex::ACCEL),                                                                     // NOLINT
            x(StateIndex::SPEED) * std::tan(x(StateIndex::DELTAV)) / L,  // NOLINT
            x(StateIndex::OMEGA),                                                                        // NOLINT
            u(ControlIndex::ALPHAV),                                                                      // NOLINT
            x(StateIndex::SPEED),
            u(ControlIndex::JERK);                                                                     // NOLINT
    // clang-format on
    return diff;
  }

  void linearize(const State& x, double increment, Eigen::MatrixXd& A,
                 Eigen::MatrixXd& B) const {
    Eigen::Matrix<double, X_DIM, X_DIM> f_x;
    Eigen::Matrix<double, X_DIM, U_DIM> f_u;
    const double h = increment;
    const double v = x(StateIndex::SPEED);
    const double theta = x(StateIndex::THETA);
    const double delta = x(StateIndex::DELTAV);
    const double cos_theta = std::cos(theta); /*unused variable*/
    const double sin_theta = std::sin(theta);
    const double tan_delta = std::tan(delta);
    const double cos_delta = std::cos(delta);
    const double wheel_base = L;

    // clang-format off
    f_x<< 1 , 0, h * cos_theta, -v * h * cos_theta, 0, 0, 0, 0, //x
          0 , 1, h * sin_theta, v * h * cos_theta, 0, 0, 0, 0,    //y
          0 , 0, 1, 0, 0, 0, 0, h, //speed
          0 , 0, h * tan_delta / wheel_base, 1, h * (1 + tan_delta * tan_delta ) * v / wheel_base, 0, 0, 0, //theta
          0 , 0, 0, 0, 1, h, 0, 0, //delta
          0 , 0, 0, 0, 0, 1, 0, 0, //omega
          0 , 0, h, 0, 0, 0, 1, 0,  //odom
          0 , 0, 0, 0, 0, 0, 0, 1;  //accel
    f_u<< 0 , 0,
          0 , 0,
          0 , 0,
          0 , 0,
          0 , 0,
          0 , h,
          0 , 0,
          h , 0;
    // clang-format on
    A = f_x;
    B = f_u;
    //    AINFO << "A:" << A;
    //    AINFO << "B:" << B;
  }

 private:
  double L;  // 车辆轴距
 public:
  double max_speed, min_speed, max_a, min_a, max_delta, min_delta, max_alpha,
      min_alpha, max_jerk, min_jerk;
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
    //    control[0] << 0.0, 0.1;

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
    vehicle_.linearize(initial_state, dt_, A_t, B_t);
    SetEqualityConstraints(A, b, A_t, B_t, initial_state);

    // 设置初始不等式约束
    SetInequalityConstraints(C, l, u);

    //    // cost function
    //        AINFO << "H dimensions: (" << C.rows() << ", " << C.cols() << ")";
    //        AINFO << "g dimensions: (" << g.rows() << ", " << g.cols() << ")";
    //        // dynamic function
    //        AINFO << "A_t dimensions: (" << A_t.rows() << ", " << A_t.cols()
    //        <<
    //        ")"; AINFO << "B_t dimensions: (" << B_t.rows() << ", " <<
    //        B_t.cols()
    //        << ")";
    //        // equality constraints
    //        AINFO << "A dimensions: (" << A.rows() << ", " << A.cols() << ")";
    //        AINFO << "b dimensions: (" << b.rows() << ", " << b.cols() << ")";
    //        // inequality constraints
    //        AINFO << "C dimensions: (" << C.rows() << ", " << C.cols() << ")";
    //        AINFO << "l dimensions: (" << l.rows() << ", " << l.cols() << ")";
    //        AINFO << "u dimensions: (" << u.rows() << ", " << u.cols() << ")";
    //
    //    AINFO << "Initializing and solving QP problem";

    Eigen::SparseMatrix<double> H_sparse = H.sparseView();
    Eigen::SparseMatrix<double> A_sparse = A.sparseView();
    Eigen::SparseMatrix<double> C_sparse = C.sparseView();

    sparse::QP<double, int> qp_solver(num_vars, num_eq_constraints,
                                      num_in_constraints);
    qp_solver.init(H_sparse, g, A_sparse, b, C_sparse, l, u);

    //    dense::QP<double> qp_solver(num_vars, num_eq_constraints,
    //                                num_in_constraints);
    //    qp_solver.init(H, g, A, b, C, l, u);

    // 调整求解器参数
    qp_solver.settings.max_iter = 3;
    qp_solver.settings.max_iter_in = 3;
    //    qp_solver.settings.eps_abs = 1.E-3;
    //    qp_solver.settings.verbose = true;
    //    qp_solver.settings.check_duality_gap = false;
    //    qp_solver.settings.mu_update_factor = 0.7;
    //    qp_solver.settings.alpha_bcl = 0.5;
    //    qp_solver.settings.beta_bcl = 0.5;
    //    qp_solver.settings.eps_refact = 1.E-3;
    //    qp_solver.settings.initial_guess =
    //        InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
    auto start = std::chrono::high_resolution_clock::now();
    qp_solver.solve();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << std::fixed << std::setprecision(6)
              << "MPC solve time: " << duration.count() << " ms" << std::endl;

    Eigen::VectorXd solution = qp_solver.results.x;

    //    AINFO << "X: \n" << solution;
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

    state_ = state;
    control_ = control;
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

    //    AINFO << "MPC solve complete";
    return control;
  }

  const std::vector<State>& getState() const { return state_; }
  const std::vector<Control>& getControl() const { return control_; }

 private:
  const Vehicle<StateDim, ControlDim>& vehicle_;
  int horizon_;
  double dt_;
  std::vector<State> state_;
  std::vector<Control> control_;

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

    double max_alpha = vehicle_.max_alpha;
    double min_alpha = vehicle_.min_alpha;
    double max_jerk = vehicle_.max_jerk;
    double min_jerk = vehicle_.min_jerk;

    //    double inf = std::numeric_limits<double>::infinity();`
    double inf = 1e10;

    // State bounds
    Eigen::VectorXd MIN_X(StateDim);
    MIN_X << -inf, -inf, min_v, -inf, min_delta, -inf, -inf, min_a;

    Eigen::VectorXd MAX_X(StateDim);
    MAX_X << inf, inf, max_v, inf, max_delta, inf, inf, max_a;

    // Control bounds
    Eigen::VectorXd MIN_U(ControlDim);
    MIN_U << min_jerk, min_alpha;

    Eigen::VectorXd MAX_U(ControlDim);
    MAX_U << max_jerk, max_alpha;

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

    double max_alpha = vehicle_.max_alpha;
    double min_alpha = vehicle_.min_alpha;
    double max_jerk = vehicle_.max_jerk;
    double min_jerk = vehicle_.min_jerk;

    double inf = std::numeric_limits<double>::infinity();

    Eigen::VectorXd MIN_X(StateDim);
    MIN_X << -inf, -inf, min_v, -inf, min_delta, -inf, -inf, min_a;

    Eigen::VectorXd MAX_X(StateDim);
    MAX_X << inf, inf, max_v, inf, max_delta, inf, inf, max_a;

    Eigen::VectorXd MIN_U(ControlDim);
    MIN_U << min_jerk, min_alpha;

    Eigen::VectorXd MAX_U(ControlDim);
    MAX_U << max_jerk, max_alpha;

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

template <size_t StateDim, size_t ControlDim>
void loadTrajectoryFromCSV(
    std::vector<typename Vehicle<StateDim, ControlDim>::State>&
        targetTrajectory,
    const std::string& filename) {
  std::ifstream file(filename);
  std::string line;

  // 跳过表头
  std::getline(file, line);

  size_t i = 0;  // 索引
  while (std::getline(file, line) && i < targetTrajectory.size()) {
    std::stringstream ss(line);
    std::string item;

    // 读取并解析每一列的数据
    std::getline(ss, item, ',');  // X_POS
    targetTrajectory[i][Vehicle<StateDim, ControlDim>::X_POS] = std::stod(item);

    std::getline(ss, item, ',');  // Y_POS
    targetTrajectory[i][Vehicle<StateDim, ControlDim>::Y_POS] = std::stod(item);

    std::getline(ss, item, ',');  // SPEED
    targetTrajectory[i][Vehicle<StateDim, ControlDim>::SPEED] = std::stod(item);

    std::getline(ss, item, ',');  // THETA
    targetTrajectory[i][Vehicle<StateDim, ControlDim>::THETA] = std::stod(item);

    std::getline(ss, item, ',');  // DELTAV
    targetTrajectory[i][Vehicle<StateDim, ControlDim>::DELTAV] =
        std::stod(item);

    std::getline(ss, item, ',');  // OMEGA
    targetTrajectory[i][Vehicle<StateDim, ControlDim>::OMEGA] = std::stod(item);

    std::getline(ss, item, ',');  // ODOM
    targetTrajectory[i][Vehicle<StateDim, ControlDim>::ODOM] = std::stod(item);

    std::getline(ss, item, ',');  // ACCEL
    targetTrajectory[i][Vehicle<StateDim, ControlDim>::ACCEL] = std::stod(item);

    ++i;  // 更新索引
  }
}

int main() {
  // 读取JSON文件
  std::ifstream i(root_path + "params.json");
  json j;
  i >> j;

  const int StateDim = 8;
  const int ControlDim = 2;

  double wheelbase = j["wheelbase"];
  double max_speed = j["max_speed"];
  double min_speed = j["min_speed"];
  double max_acceleration = j["max_acceleration"];
  double min_acceleration = j["min_acceleration"];
  double max_steering_angle = j["max_steering_angle"];
  double min_steering_angle = j["min_steering_angle"];
  double max_alpha = j["max_alpha"];
  double min_alpha = j["min_alpha"];
  double max_jerk = j["max_jerk"];
  double min_jerk = j["min_jerk"];

  int horizon = j["horizon"];
  double dt = j["dt"];

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(StateDim, StateDim);
  Q.diagonal() << j["Q"][0], j["Q"][1], j["Q"][2], j["Q"][3], j["Q"][4],
      j["Q"][5], j["Q"][6], j["Q"][7];
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(ControlDim, ControlDim);
  R.diagonal() << j["R"][0], j["R"][1];
  Eigen::MatrixXd Q_N = Eigen::MatrixXd::Identity(StateDim, StateDim);
  Q_N.diagonal() << j["Q_N"][0], j["Q_N"][1], j["Q_N"][2], j["Q_N"][3],
      j["Q_N"][4], j["Q_N"][5], j["Q_N"][6], j["Q_N"][7];

  Vehicle<StateDim, ControlDim> vehicle(
      wheelbase, max_speed, min_speed, max_acceleration, min_acceleration,
      max_steering_angle, min_steering_angle, max_alpha, min_alpha, max_jerk,
      min_jerk);

  MPC<StateDim, ControlDim> mpc(vehicle, horizon, dt, Q, R, Q_N);

  Vehicle<StateDim, ControlDim>::State initial_state;
  initial_state << j["init_state"][0], j["init_state"][1], j["init_state"][2],
      j["init_state"][3], j["init_state"][4], j["init_state"][5],
      j["init_state"][6], j["init_state"][7];
  AINFO << "init state: " << initial_state.transpose() << "\n";

  std::vector<Vehicle<StateDim, ControlDim>::State> targetTrajectory(
      horizon, initial_state);

  loadTrajectoryFromCSV<StateDim, ControlDim>(
      targetTrajectory,
      "/home/next/demo_test/Optimal_Control_test/Convex_mpc/trajectory.csv");

  //  for (const auto& item: targetTrajectory) {
  //    std::cout << "x: " <<item[0] << ", y:" << item[1] << std::endl;
  //  }

  auto start = std::chrono::high_resolution_clock::now();

  mpc.solve(targetTrajectory[0], targetTrajectory);

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << std::fixed << std::setprecision(6)
            << "All MPC solve time: " << duration.count() << " ms" << std::endl;
  // 提取轨迹
  const auto& trajectory = mpc.getState();
  const auto& control_inputs = mpc.getControl();

  // 提取轨迹中的X和Y坐标
  std::vector<double> x_ref, y_ref, x_coords, y_coords, v_coords, theta_coords,
      acc_coords, steering_coords;
  for (const auto& state : targetTrajectory) {
    x_ref.push_back(state(Vehicle<StateDim, ControlDim>::X_POS));
    y_ref.push_back(state(Vehicle<StateDim, ControlDim>::Y_POS));
  }
  for (const auto& state : trajectory) {
    x_coords.push_back(state(Vehicle<StateDim, ControlDim>::X_POS));
    y_coords.push_back(state(Vehicle<StateDim, ControlDim>::Y_POS));
    theta_coords.push_back(state(Vehicle<StateDim, ControlDim>::THETA));
    v_coords.push_back(state(Vehicle<StateDim, ControlDim>::SPEED));
    acc_coords.push_back(state(Vehicle<StateDim, ControlDim>::ACCEL));
    steering_coords.push_back(state(Vehicle<StateDim, ControlDim>::DELTAV));
  }
  // 提取控制量中的A和DELTA
  std::vector<double> jerk_values, alpha_values;
  for (const auto& control : control_inputs) {
    jerk_values.push_back(control(Vehicle<StateDim, ControlDim>::JERK));
    alpha_values.push_back(control(Vehicle<StateDim, ControlDim>::ALPHAV));
  }

  // 控制量反推轨迹
  std::vector<double> dynamic_traj_x, dynamic_traj_y;
  std::vector<Vehicle<StateDim, ControlDim>::State> dynamic_traj;
  dynamic_traj.push_back(targetTrajectory.front());
  dynamic_traj_x.push_back(
      dynamic_traj.back()[Vehicle<StateDim, ControlDim>::X_POS]);
  dynamic_traj_y.push_back(
      dynamic_traj.back()[Vehicle<StateDim, ControlDim>::Y_POS]);
  for (int i = 0; i < control_inputs.size(); ++i) {
    dynamic_traj.push_back(
        vehicle.EvalOneStep(dynamic_traj.back(), control_inputs[i], dt));

    dynamic_traj_x.push_back(
        dynamic_traj.back()[Vehicle<StateDim, ControlDim>::X_POS]);

    dynamic_traj_y.push_back(
        dynamic_traj.back()[Vehicle<StateDim, ControlDim>::Y_POS]);

    std::cout << "i: " << i << ", opt_x: "
              << trajectory[i][Vehicle<StateDim, ControlDim>::X_POS]
              << ", opt_y: "
              << trajectory[i][Vehicle<StateDim, ControlDim>::Y_POS]
              << ", opt_theta: "
              << trajectory[i][Vehicle<StateDim, ControlDim>::THETA]
              << ", x_dyn: " << dynamic_traj_x[i]
              << ", y_dyn: " << dynamic_traj_y[i] << " | jerk: "
              << control_inputs[i][Vehicle<StateDim, ControlDim>::JERK]
              << ", alpha: "
              << control_inputs[i][Vehicle<StateDim, ControlDim>::ALPHAV]
              << std::endl;
  }

  // 将所有数据写入一个CSV文件
  writeDataToCSV(root_path + "data.csv",
                 {x_ref, y_ref, x_coords, y_coords, dynamic_traj_x,
                  dynamic_traj_y, v_coords, acc_coords, theta_coords,
                  steering_coords, jerk_values, alpha_values},
                 {"x_ref", "y_ref", "x_coords", "y_coords", "dynamic_traj_x",
                  "dynamic_traj_y", "velocity", "acc", "theta", "steering",
                  "d(aa)_jerk", "dd(steering)_alpha"});
  return 0;
}

//		[0, 0], [10, 0], [20, 0], [20, 5], [30, 5], [40, 5]
