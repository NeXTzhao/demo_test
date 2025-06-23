#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#include "common/log.h"
#include "nlohmann/json.hpp"
#include "third_party/tinympc/tiny_api.hpp"
#define NHORIZON 35

using json = nlohmann::json;

const std::string root_path =
    "/home/next/要备份的/demo_test/Optimal_Control_test/Convex_mpc/data/";
const std::string row_traj_path =
    "/home/next/要备份的/demo_test/Optimal_Control_test/Convex_mpc/"
    "trajectory.csv";

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

  std::vector<State> simulate(const State& initial_state,
                              const std::vector<Control>& controls,
                              double increment, int USE_RK = 2) const {
    std::vector<State> states;
    states.reserve(controls.size() + 1);  // 预留空间以优化性能
    states.push_back(initial_state);      // 第一个状态是初始状态

    State current_state = initial_state;

    for (const auto& control : controls) {
      // 计算下一个状态
      State next_state = EvalOneStep(current_state, control, increment, USE_RK);
      states.push_back(next_state);
      current_state = next_state;
    }

    return states;
  }

  State EvalOneStep(const State& x, const Control& u, double increment,
                    int USE_RK = 2) const {
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
    const double wheel_base = L;

    // clang-format off
    f_x<< 1 , 0, h * cos_theta, -v * h * sin_theta, 0, 0, 0, 0, //x
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
    //    AINFO << "A: " << A.rows() << "x" << A.cols() << "\n" << f_x;
    //    AINFO << "B: " << B.rows() << "x" << B.cols() << "\n" << f_u;
  }

  // 计算状态变量范围
  void calculateWeightRange(const State& initial_state, double dt, int horizon,
                            Eigen::MatrixXd& q_weights,
                            Eigen::MatrixXd& r_weights) {
    // 定义极限控制输入组合（最大、最小值）
    std::vector<Control> control_extremes = {{max_jerk, max_alpha},
                                             {max_jerk, min_alpha},
                                             {min_jerk, max_alpha},
                                             {min_jerk, min_alpha}};

    State min_state = initial_state;
    State max_state = initial_state;

    // 遍历每种极限控制输入组合
    for (const auto& control : control_extremes) {
      State current_state = initial_state;

      for (int i = 0; i < horizon; ++i) {
        current_state = dynamics(current_state, control);

        // 更新每个状态量的最小值和最大值
        for (int j = 0; j < X_DIM; ++j) {
          min_state[j] = std::min(min_state[j], current_state[j]);
          max_state[j] = std::max(max_state[j], current_state[j]);
        }
      }
    }

    autoTuneQWeights(min_state, max_state, q_weights, r_weights);
  }

  void autoTuneQWeights(const State& min_state, const State& max_state,
                        Eigen::MatrixXd& q_weights,
                        Eigen::MatrixXd& r_weights) {
    double x_range =
        max_state[StateIndex::X_POS] - min_state[StateIndex::X_POS];
    //    double x_range = 10;
    //    double y_range = 10;

    double y_range =
        max_state[StateIndex::Y_POS] - min_state[StateIndex::Y_POS];
    //    double speed_range = max_speed - min_speed;
    double speed_range =
        max_state[StateIndex::SPEED] - min_state[StateIndex::SPEED];
    // 对 theta 进行角度范围的归一化处理
    double theta_min =
        std::fmod(min_state[StateIndex::THETA] + M_PI, 2 * M_PI) - M_PI;
    double theta_max =
        std::fmod(max_state[StateIndex::THETA] + M_PI, 2 * M_PI) - M_PI;
    double theta_range = theta_max - theta_min;
    if (theta_range < 0) theta_range += 2 * M_PI;
    double delta_range =
        max_state[StateIndex::DELTAV] - min_state[StateIndex::DELTAV];

    double omega_range =
        max_state[StateIndex::OMEGA] - min_state[StateIndex::OMEGA];
    double odom_range =
        max_state[StateIndex::ODOM] - min_state[StateIndex::ODOM];
    double accel_range =
        max_state[StateIndex::ACCEL] - min_state[StateIndex::ACCEL];

    double jerk_range = max_jerk - min_jerk;
    double alpha_range = max_alpha - min_alpha;

    AINFO << "x_range: " << x_range
          << ", max_x: " << max_state[StateIndex::X_POS]
          << ", min_x: " << min_state[StateIndex::X_POS];
    AINFO << "y_range: " << y_range
          << ", max_y: " << max_state[StateIndex::Y_POS]
          << ", min_y: " << min_state[StateIndex::Y_POS];
    AINFO << "speed_range: " << speed_range
          << ", max_speed: " << max_state[StateIndex::SPEED]
          << ", min_speed: " << min_state[StateIndex::SPEED];
    AINFO << "theta_range: " << theta_range
          << ", max_theta: " << max_state[StateIndex::THETA]
          << ", min_theta: " << min_state[StateIndex::THETA];
    AINFO << "delta_range: " << delta_range
          << ", max_delta: " << max_state[StateIndex::DELTAV]
          << ", min_delta: " << min_state[StateIndex::DELTAV];
    AINFO << "omega_range: " << omega_range
          << ", max_omega: " << max_state[StateIndex::OMEGA]
          << ", min_omega: " << min_state[StateIndex::OMEGA];
    AINFO << "odom_range: " << odom_range
          << ", max_odom: " << max_state[StateIndex::ODOM]
          << ", min_odom: " << min_state[StateIndex::ODOM];
    AINFO << "accel_range: " << accel_range
          << ", max_accel: " << max_state[StateIndex::ACCEL]
          << ", min_accel: " << min_state[StateIndex::ACCEL];
    AINFO << "jerk_range: " << jerk_range << ", max_jerk: " << max_jerk
          << ", min_jerk: " << min_jerk;
    AINFO << "alpha_range: " << alpha_range << ", max_alpha: " << max_alpha
          << ", min_alpha: " << min_alpha << '\n';

    // 计算每个状态变量的倒数比例因子，避免除零
    Eigen::VectorXd q_inv_factors(X_DIM);
    q_inv_factors[StateIndex::X_POS] = normalize(x_range);
    q_inv_factors[StateIndex::Y_POS] = normalize(y_range);
    q_inv_factors[StateIndex::SPEED] = normalize(speed_range);
    q_inv_factors[StateIndex::THETA] = normalize(theta_range);
    q_inv_factors[StateIndex::DELTAV] = normalize(delta_range);
    q_inv_factors[StateIndex::OMEGA] = normalize(omega_range);
    q_inv_factors[StateIndex::ODOM] = normalize(odom_range);
    q_inv_factors[StateIndex::ACCEL] = normalize(accel_range);
    //    std::cout << "q_inv_factors: \n" << q_inv_factors << std::endl;

    Eigen::VectorXd r_inv_factors(U_DIM);
    r_inv_factors[ControlIndex::JERK] = normalize(jerk_range);
    r_inv_factors[ControlIndex::ALPHAV] = normalize(alpha_range);
    //    std::cout << "r_inv_factors: \n" << r_inv_factors << std::endl;

    // 可选：对 Q 权重进行缩放，以便控制整体权重范围
    for (int i = 0; i < StateIndex::X_DIM; ++i) {
      q_weights(i, i) *= q_inv_factors[i];
    }
    for (int i = 0; i < ControlIndex::U_DIM; ++i) {
      r_weights(i, i) *= r_inv_factors[i];
    }
    std::cout << "q_weights: \n" << q_weights << std::endl;
    std::cout << "r_weights: \n" << r_weights << std::endl;
  }

  double normalize(double value, double c = 1.0) {
    /*1*/
    // 避免对数0的问题
    //    if (value <= 0) {
    //      return 1.0;  // 对数的最小值，可以根据需要调整
    //    }
    //    return std::log(value + c);

    /*2*/
    return value > 0 ? 1.0 / value : 1.0;
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

  MPC(Vehicle<StateDim, ControlDim>& vehicle, int horizon, double dt,
      const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
      const Eigen::MatrixXd& Q_N)
      : vehicle_(vehicle),
        horizon_(horizon),
        dt_(dt),
        Q_(Q),
        R_(R),
        Q_N_(Q_N),
        solver_(nullptr) {}

  void solve(const State& initial_state,
             const std::vector<State>& reference_trajectory) {
    state_.clear();
    control_.clear();

    SetBoundConstraints(x_min_, x_max_, u_min_, u_max_, horizon_);

    // 线性化系统
    vehicle_.linearize(initial_state, dt_, A_, B_);
    tinytype rho_value = 1;
    tinytype verbose = 0;
    int status =
        tiny_setup(&solver_, A_, B_, Q_, R_, rho_value, StateDim, ControlDim,
                   horizon_, x_min_, x_max_, u_min_, u_max_, verbose);
    if (status != 0 || solver_ == nullptr) {
      std::cout << "TinyMPC setup failed." << std::endl;
    }

    solver_->settings->max_iter = 5;
    solver_->settings->abs_pri_tol = 1;
    solver_->settings->abs_dua_tol = 1;
    solver_->settings->check_termination = 1;

    const int point_num = reference_trajectory.size();
    Eigen::MatrixXd Xref_total(StateDim, point_num);
    for (int i = 0; i < point_num; ++i) {
      Xref_total.col(i) = reference_trajectory[i];
    }

    State x0 = initial_state;
    // state_.push_back(x0);

    TinyWorkspace* work = solver_->work;

    for (int k = 0; k < point_num; ++k) {
      // 更新参考轨迹
      int rem = std::min(horizon_, point_num - k);
      work->Xref.leftCols(rem) = Xref_total.block(0, k, StateDim, rem);
      if (rem < horizon_) {
        work->Xref.rightCols(horizon_ - rem) =
            Xref_total.col(point_num - 1).replicate(1, horizon_ - rem);
      }
      tiny_set_x0(solver_, x0);
      tiny_solve(solver_);

      const auto& x_step = work->x.col(0);
      const auto& u_step = work->u.col(0);

      x0 = vehicle_.EvalOneStep(x0, u_step, dt_);

      state_.push_back(x_step);
      if (k < point_num - 1) {
        control_.push_back(u_step);
      }
    }
    AINFO << "X size: " << state_.size() << "u size: " << control_.size();
  }

  const std::vector<State>& getState() const { return state_; }
  const std::vector<Control>& getControl() const { return control_; }

 private:
  void SetBoundConstraints(Eigen::MatrixXd& x_min, Eigen::MatrixXd& x_max,
                           Eigen::MatrixXd& u_min, Eigen::MatrixXd& u_max,
                           int horizon_size) {
    Eigen::VectorXd state_min(StateDim), state_max(StateDim);
    Eigen::VectorXd control_min(ControlDim), control_max(ControlDim);

    const double inf = 1e12;

    state_min << -inf, -inf, vehicle_.min_speed, -inf, vehicle_.min_delta, -inf,
        -inf, vehicle_.min_a;
    state_max << inf, inf, vehicle_.max_speed, inf, vehicle_.max_delta, inf,
        inf, vehicle_.max_a;

    control_min << vehicle_.min_jerk, vehicle_.min_alpha;
    control_max << vehicle_.max_jerk, vehicle_.max_alpha;

    x_min = state_min.replicate(1, horizon_size);
    x_max = state_max.replicate(1, horizon_size);
    u_min = control_min.replicate(1, horizon_size - 1);
    u_max = control_max.replicate(1, horizon_size - 1);
  }

 private:
  Vehicle<StateDim, ControlDim>& vehicle_;
  const int horizon_;
  const double dt_;
  Eigen::MatrixXd Q_, R_, Q_N_;

  std::vector<State> state_;
  std::vector<Control> control_;

  TinySolver* solver_;
  Eigen::MatrixXd A_, B_;
  Eigen::MatrixXd x_min_, x_max_, u_min_, u_max_;
};

// template <int StateDim, int ControlDim>
// class MPC {
//  public:
//   using State = typename Vehicle<StateDim, ControlDim>::State;
//   using Control = typename Vehicle<StateDim, ControlDim>::Control;
//
//   MPC(Vehicle<StateDim, ControlDim>& vehicle, int horizon, double dt,
//       Eigen::MatrixXd Q, Eigen::MatrixXd R, Eigen::MatrixXd Q_N)
//       : vehicle_(vehicle),
//         horizon_(horizon),
//         dt_(dt),
//         Q_(std::move(Q)),
//         R_(std::move(R)),
//         Q_N_(std::move(Q_N)) {}
//
//   void solve(const State& initial_state,
//              const std::vector<State>& reference_trajectory) {
//     state_.clear();
//     control_.clear();
//
//     TinySolver* solver;
//
//     Eigen::MatrixXd A_t(StateDim, StateDim);
//     Eigen::MatrixXd B_t(StateDim, ControlDim);
//     vehicle_.linearize(initial_state, dt_, A_t, B_t);
//
//     // 权重归一化
//     //    vehicle_.calculateWeightRange(initial_state, dt_, NHORIZON, Q_,
//     R_);
//
//     Eigen::MatrixXd x_min, x_max, u_min, u_max;
//     SetBoundConstraints(x_min, x_max, u_min, u_max, NHORIZON);
//
//     tinytype rho_value = 1;
//     tinytype verbose = 1;
//
//     int status =
//         tiny_setup(&solver, A_t, B_t, Q_, R_, rho_value, StateDim,
//         ControlDim,
//                    NHORIZON, x_min, x_max, u_min, u_max, verbose);
//
//     // Update whichever settings we'd like
//     solver->settings->max_iter = 100;
//
//     // Alias solver->work for brevity
//     TinyWorkspace* work = solver->work;
//
//     // reference trajectory
//     int point_num = reference_trajectory.size();
//     int state_num = reference_trajectory[0].size();
//     Eigen::MatrixXd Xref_total(state_num, point_num);
//
//     for (size_t i = 0; i < point_num; ++i) {
//       //      std::cout << "ref traj " << i << ": "
//       //                << reference_trajectory[i].transpose() << std::endl;
//       Xref_total.col(i) = reference_trajectory[i];
//     }
//
//     // 将 reference_matrix 中的数据映射到 work->Xref
//     work->Xref = Xref_total.block(0, 0, StateDim, NHORIZON);
//     //    std::cout << "Xref_total: \n" << work->Xref << std::endl;
//
//     State x0;
//     x0 = work->Xref.col(0);
//     state_.push_back(x0);
//
//     for (int k = 0; k < point_num; ++k) {
//       // 1. 更新测量值
//       tiny_set_x0(solver, x0);
//
//       // 2. 更新参考轨迹
//       if (k + NHORIZON <= point_num) {
//         work->Xref = Xref_total.block(0, k, StateDim, NHORIZON);
//       } else {
//         int remaining = point_num - k;
//         work->Xref.leftCols(remaining) =
//             Xref_total.block(0, k, StateDim, remaining);
//         work->Xref.rightCols(NHORIZON - remaining) =
//             Xref_total.col(point_num - 1).replicate(1, NHORIZON - remaining);
//       }
//
//       // 3. 重置对偶变量
//       work->y = Eigen::Matrix<tinytype, ControlDim, NHORIZON - 1>::Zero();
//       work->g = Eigen::Matrix<tinytype, StateDim, NHORIZON>::Zero();
//
//       // 4. 解决MPC问题
//       tiny_solve(solver);
//       auto x_step = work->x.col(0);
//       auto u_step = work->u.col(0);
//
//       // 5. 向前模拟
//
//       x0 = work->Adyn * x0 + work->Bdyn * u_step;
//       //      std::cout << "x sim " << k << ": \n" << x0.transpose() <<
//       //      std::endl;
//       state_.push_back(x_step);
//       control_.push_back(u_step);
//       //      std::cout << "x_opt " << k << ": \n" << work->x << std::endl;
//     }
//   }
//
//   void SetBoundConstraints(Eigen::MatrixXd& x_min, Eigen::MatrixXd& x_max,
//                            Eigen::MatrixXd& u_min, Eigen::MatrixXd& u_max,
//                            int horizon_size) {
//     Eigen::VectorXd state_min(StateDim), state_max(StateDim);
//     Eigen::VectorXd control_min(ControlDim), control_max(ControlDim);
//
//     double min_a = vehicle_.min_a;
//     double max_a = vehicle_.max_a;
//     double min_delta = vehicle_.min_delta;
//     double max_delta = vehicle_.max_delta;
//     double min_v = vehicle_.min_speed;
//     double max_v = vehicle_.max_speed;
//
//     double max_alpha = vehicle_.max_alpha;
//     double min_alpha = vehicle_.min_alpha;
//     double max_jerk = vehicle_.max_jerk;
//     double min_jerk = vehicle_.min_jerk;
//
//     double inf = 1e12;
//
//     // state bounds
//     state_min << -inf, -inf, min_v, -inf, min_delta, -inf, -inf, min_a;
//     state_max << inf, inf, max_v, inf, max_delta, inf, inf, max_a;
//
//     //    state_min << -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf;
//     //    state_max << inf, inf, inf, inf, inf, inf, inf, inf;
//
//     // control bounds
//     control_min << min_jerk, min_alpha;
//     control_max << max_jerk, max_alpha;
//
//     // 将边界扩展到整个预测窗口
//     x_min = state_min.replicate(1, horizon_size);
//     x_max = state_max.replicate(1, horizon_size);
//     u_min = control_min.replicate(1, horizon_size - 1);
//     u_max = control_max.replicate(1, horizon_size - 1);
//
//     //    AINFO << "State bounds (x_min): \n" << x_min;
//     //    AINFO << "State bounds (x_max): \n" << x_max;
//     //    AINFO << "Control bounds (u_min): \n" << u_min;
//     //    AINFO << "Control bounds (u_max): \n" << u_max << '\n';
//   }
//
//   vector<State> getState() { return state_; }
//   std::vector<Control>& getControl() { return control_; }
//
//  public:
//   Eigen::MatrixXd Q_;
//   Eigen::MatrixXd R_;
//
//  private:
//   Vehicle<StateDim, ControlDim>& vehicle_;
//   int horizon_;
//   double dt_;
//   std::vector<State> state_;
//   std::vector<Control> control_;
//
//   Eigen::MatrixXd Q_N_;
// };

// 定义计算误差的函数
template <int StateDim, int ControlDim>
double computeError(
    const std::vector<typename Vehicle<StateDim, ControlDim>::State>&
        trajectory,
    const std::vector<typename Vehicle<StateDim, ControlDim>::State>
        reference) {
  double error = 0.0;
  for (size_t i = 0; i < trajectory.size(); ++i) {
    error += (trajectory[i] - reference[i]).squaredNorm();
  }
  return error / trajectory.size();
}

// 调参函数：用于找到最优的 Q 和 R
template <int StateDim, int ControlDim>
void tuneMPC(MPC<StateDim, ControlDim>& mpc,
             const std::vector<typename Vehicle<StateDim, ControlDim>::State>&
                 reference_trajectory) {
  double best_error = std::numeric_limits<double>::max();
  Eigen::MatrixXd best_Q = mpc.Q_, best_R = mpc.R_;

  // 定义每个状态量和控制输入的权重搜索范围
  std::vector<std::vector<double>> q_values = {
      {0.1, 1, 50, 100},   // X_POS 权重
      {0.1, 1, 50, 100},   // Y_POS 权重
      {0.1, 1, 50},        // SPEED 权重
      {0.1, 1, 20},        // THETA 权重
      {0.1, 1, 50},        // DELTAV 权重
      {0.1, 1, 20},        // OMEGA 权重
      {0.01, 0.1, 1, 20},  // ODOM 权重
      {0.1, 1, 50}         // ACCEL 权重
  };

  // 控制输入的权重设置，增加更多的调节空间
  std::vector<std::vector<double>> r_values = {
      {0.1, 1, 20, 100},  // DELTAV (速度变化) 权重
      {0.1, 1, 20, 100}   // OMEGA (角速度变化) 权重
  };

  int iter = 0;
  // 遍历所有 Q 和 R 的组合
  for (const auto& q_x : q_values[0]) {
    for (const auto& q_y : q_values[1]) {
      for (const auto& q_speed : q_values[2]) {
        for (const auto& q_theta : q_values[3]) {
          for (const auto& q_deltav : q_values[4]) {
            for (const auto& q_omega : q_values[5]) {
              for (const auto& q_odom : q_values[6]) {
                for (const auto& q_accel : q_values[7]) {
                  // 构建 Q 矩阵，直接调整对应的对角线
                  Eigen::MatrixXd Q =
                      Eigen::MatrixXd::Identity(StateDim, StateDim);
                  Q(0, 0) = q_x;       // X_POS 权重
                  Q(1, 1) = q_y;       // Y_POS 权重
                  Q(2, 2) = q_speed;   // SPEED 权重
                  Q(3, 3) = q_theta;   // THETA 权重
                  Q(4, 4) = q_deltav;  // DELTAV 权重
                  Q(5, 5) = q_omega;   // OMEGA 权重
                  Q(6, 6) = q_odom;    // ODOM 权重
                  Q(7, 7) = q_accel;   // ACCEL 权重

                  for (const auto& r_deltav : r_values[0]) {
                    for (const auto& r_omega : r_values[1]) {
                      // 构建 R 矩阵，直接调整对应的对角线
                      Eigen::MatrixXd R =
                          Eigen::MatrixXd::Identity(ControlDim, ControlDim);
                      R(0, 0) = r_deltav;  // DELTAV 控制输入权重
                      R(1, 1) = r_omega;   // OMEGA 控制输入权重

                      // 更新 MPC 的 Q 和 R
                      mpc.Q_ = Q;
                      mpc.R_ = R;

                      // 运行 MPC 并计算误差
                      mpc.solve(reference_trajectory[0], reference_trajectory);
                      double error = computeError<StateDim, ControlDim>(
                          mpc.getState(), reference_trajectory);
                      std::cout << "iter: " << iter << ", error: " << error
                                << std::endl;
                      iter++;

                      // 记录最优结果
                      if (error < best_error) {
                        best_error = error;
                        best_Q = Q;
                        best_R = R;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // 更新为最优参数
  mpc.Q_ = best_Q;
  mpc.R_ = best_R;

  std::cout << "Best Q:\n"
            << best_Q << "\nBest R:\n"
            << best_R << "\nBest Error: " << best_error << std::endl;
}

template <size_t StateDim, size_t ControlDim>
void loadTrajectoryFromCSV(
    std::vector<typename Vehicle<StateDim, ControlDim>::State>& traj,
    const std::string& filename) {
  std::ifstream file(filename);
  std::string line;

  // 跳过表头
  std::getline(file, line);

  size_t i = 0;  // 索引
  while (std::getline(file, line) && i < traj.size()) {
    std::stringstream ss(line);
    std::string item;

    // 读取并解析每一列的数据
    std::getline(ss, item, ',');
    traj[i][Vehicle<StateDim, ControlDim>::X_POS] = std::stod(item);

    std::getline(ss, item, ',');
    traj[i][Vehicle<StateDim, ControlDim>::Y_POS] = std::stod(item);

    std::getline(ss, item, ',');
    traj[i][Vehicle<StateDim, ControlDim>::THETA] = std::stod(item);

    std::getline(ss, item, ',');
    traj[i][Vehicle<StateDim, ControlDim>::SPEED] = std::stod(item);

    std::getline(ss, item, ',');
    traj[i][Vehicle<StateDim, ControlDim>::ACCEL] = std::stod(item);

    std::getline(ss, item, ',');
    traj[i][Vehicle<StateDim, ControlDim>::DELTAV] = std::stod(item);

    std::getline(ss, item, ',');
    traj[i][Vehicle<StateDim, ControlDim>::OMEGA] = std::stod(item);

    std::getline(ss, item, ',');
    traj[i][Vehicle<StateDim, ControlDim>::ODOM] = std::stod(item);

    ++i;  // 更新索引
  }
}
template <int StateDim, int ControlDim>
void normalizeTrajectory(
    std::vector<typename Vehicle<StateDim, ControlDim>::State>& trajectory) {
  // 获取起始点
  double x_start = trajectory.front()[Vehicle<StateDim, ControlDim>::X_POS];
  double y_start = trajectory.front()[Vehicle<StateDim, ControlDim>::Y_POS];
  double theta_start = trajectory.front()[Vehicle<StateDim, ControlDim>::THETA];

  // 预计算sin和cos值以避免重复计算
  double cos_theta_start = cos(theta_start);
  double sin_theta_start = sin(theta_start);

  // 遍历轨迹，进行坐标变换和平移
  for (auto& point : trajectory) {
    // 计算相对于起点的坐标差值
    double dx = point[Vehicle<StateDim, ControlDim>::X_POS] - x_start;
    double dy = point[Vehicle<StateDim, ControlDim>::Y_POS] - y_start;

    // 更新坐标，执行旋转变换
    point[Vehicle<StateDim, ControlDim>::X_POS] =
        dx * cos_theta_start + dy * sin_theta_start;
    point[Vehicle<StateDim, ControlDim>::Y_POS] =
        -dx * sin_theta_start + dy * cos_theta_start;

    // 更新航向角
    point[Vehicle<StateDim, ControlDim>::THETA] -= theta_start;

    //    // 归一化航向角到[-π, π]范围
    //    point[Vehicle<StateDim, ControlDim>::THETA] =
    //        std::fmod(point[Vehicle<StateDim, ControlDim>::THETA] + M_PI, 2 *
    //        M_PI);
    //
    //    // 如果theta大于π，则减去2π，确保在[-π, π]范围内
    //    if (point[Vehicle<StateDim, ControlDim>::THETA] > M_PI) {
    //      point[Vehicle<StateDim, ControlDim>::THETA] -= 2 * M_PI;
    //    }
  }

  //  for (const auto& point : trajectory) {
  //    std::cout << "x: " << point(Vehicle<StateDim, ControlDim>::X_POS)
  //              << " y: " << point(Vehicle<StateDim, ControlDim>::Y_POS)
  //              << ", speed: " << point(Vehicle<StateDim, ControlDim>::SPEED)
  //              << ", theta: " << point(Vehicle<StateDim, ControlDim>::THETA)
  //              << ", deltav: " << point(Vehicle<StateDim,
  //              ControlDim>::DELTAV)
  //              << ", omega: " << point(Vehicle<StateDim, ControlDim>::OMEGA)
  //              << ", odom: " << point(Vehicle<StateDim, ControlDim>::ODOM)
  //              << ", accel: " << point(Vehicle<StateDim, ControlDim>::ACCEL)
  //              << std::endl;
  //  }
}

template <int StateDim, int ControlDim>
void undoRotation(
    std::vector<typename Vehicle<StateDim, ControlDim>::State>& trajectory,
    const typename Vehicle<StateDim, ControlDim>::State& start_point) {
  // 获取传入的起始点
  double x_start = start_point[Vehicle<StateDim, ControlDim>::X_POS];
  double y_start = start_point[Vehicle<StateDim, ControlDim>::Y_POS];
  double theta_start = start_point[Vehicle<StateDim, ControlDim>::THETA];

  // 预计算sin和cos值以避免重复计算
  double cos_theta_start = cos(theta_start);
  double sin_theta_start = sin(theta_start);

  // 遍历轨迹，进行逆旋转
  for (auto& point : trajectory) {
    // 先恢复旋转后的坐标
    double x_prime = point[Vehicle<StateDim, ControlDim>::X_POS];
    double y_prime = point[Vehicle<StateDim, ControlDim>::Y_POS];

    // 逆旋转恢复坐标
    point[Vehicle<StateDim, ControlDim>::X_POS] =
        x_prime * cos_theta_start - y_prime * sin_theta_start + x_start;
    point[Vehicle<StateDim, ControlDim>::Y_POS] =
        x_prime * sin_theta_start + y_prime * cos_theta_start + y_start;

    // 恢复theta
    point[Vehicle<StateDim, ControlDim>::THETA] += theta_start;

    //    // 归一化theta到[-π, π]范围
    //    point[Vehicle<StateDim, ControlDim>::THETA] =
    //        std::fmod(point[Vehicle<StateDim, ControlDim>::THETA] + M_PI, 2 *
    //        M_PI);
    //
    //    if (point[Vehicle<StateDim, ControlDim>::THETA] > M_PI) {
    //      point[Vehicle<StateDim, ControlDim>::THETA] -= 2 * M_PI;
    //    }
  }

  //  for (const auto& point : trajectory) {
  //    std::cout << "x: " << point(Vehicle<StateDim, ControlDim>::X_POS)
  //              << " y: " << point(Vehicle<StateDim, ControlDim>::Y_POS)
  //              << ", speed: " << point(Vehicle<StateDim, ControlDim>::SPEED)
  //              << ", theta: " << point(Vehicle<StateDim, ControlDim>::THETA)
  //              << ", deltav: " << point(Vehicle<StateDim,
  //              ControlDim>::DELTAV)
  //              << ", omega: " << point(Vehicle<StateDim, ControlDim>::OMEGA)
  //              << ", odom: " << point(Vehicle<StateDim, ControlDim>::ODOM)
  //              << ", accel: " << point(Vehicle<StateDim, ControlDim>::ACCEL)
  //              << std::endl;
  //  }
  //  std::cout << std::endl;
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

  std::vector<Vehicle<StateDim, ControlDim>::State> targetTrajectory;
  targetTrajectory.resize(horizon);

  loadTrajectoryFromCSV<StateDim, ControlDim>(targetTrajectory, row_traj_path);

  auto start_point = targetTrajectory.front();
  auto old_traj = targetTrajectory;
  normalizeTrajectory<StateDim, ControlDim>(targetTrajectory);

  auto start = std::chrono::high_resolution_clock::now();
  // 自动调参
  //  tuneMPC(mpc, targetTrajectory);

  // 使用最优参数运行MPC
  mpc.solve(targetTrajectory[0], targetTrajectory);

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << std::fixed << std::setprecision(6)
            << "All MPC solve time: " << duration.count() << " ms" << std::endl;
  // 提取轨迹
  auto opt_trajectory = mpc.getState();
  auto control_inputs = mpc.getControl();

  auto dyn_traj = vehicle.simulate(opt_trajectory.front(), control_inputs, dt);

  undoRotation<StateDim, ControlDim>(opt_trajectory, start_point);
  undoRotation<StateDim, ControlDim>(targetTrajectory, start_point);
  undoRotation<StateDim, ControlDim>(dyn_traj, start_point);

  std::vector<double> x_ref, y_ref, x_coords, y_coords, v_coords, theta_coords,
      acc_coords, steering_coords, omega_coords, odom_coords;
  for (const auto& state : old_traj) {
    x_ref.push_back(state(Vehicle<StateDim, ControlDim>::X_POS) -
                    start_point(Vehicle<StateDim, ControlDim>::X_POS));
    y_ref.push_back(state(Vehicle<StateDim, ControlDim>::Y_POS) -
                    start_point(Vehicle<StateDim, ControlDim>::Y_POS));
  }
  for (const auto& state : opt_trajectory) {
    x_coords.push_back(state(Vehicle<StateDim, ControlDim>::X_POS) -
                       start_point(Vehicle<StateDim, ControlDim>::X_POS));
    y_coords.push_back(state(Vehicle<StateDim, ControlDim>::Y_POS) -
                       start_point(Vehicle<StateDim, ControlDim>::Y_POS));
    v_coords.push_back(state(Vehicle<StateDim, ControlDim>::SPEED));
    theta_coords.push_back(state(Vehicle<StateDim, ControlDim>::THETA));
    steering_coords.push_back(state(Vehicle<StateDim, ControlDim>::DELTAV));
    omega_coords.push_back(state(Vehicle<StateDim, ControlDim>::OMEGA));
    odom_coords.push_back(state(Vehicle<StateDim, ControlDim>::ODOM));
    acc_coords.push_back(state(Vehicle<StateDim, ControlDim>::ACCEL));
  }
  // 提取控制量中的A和DELTA
  std::vector<double> jerk_values, alpha_values;
  for (const auto& control : control_inputs) {
    jerk_values.push_back(control(Vehicle<StateDim, ControlDim>::JERK));
    alpha_values.push_back(control(Vehicle<StateDim, ControlDim>::ALPHAV));
  }

  // 控制量反推轨迹
  std::vector<double> dynamic_traj_x, dynamic_traj_y;

  for (const auto& state : dyn_traj) {
    dynamic_traj_x.push_back(state(Vehicle<StateDim, ControlDim>::X_POS) -
                             start_point(Vehicle<StateDim, ControlDim>::X_POS));
    dynamic_traj_y.push_back(state(Vehicle<StateDim, ControlDim>::Y_POS) -
                             start_point(Vehicle<StateDim, ControlDim>::Y_POS));
  }

  // 将所有数据写入一个CSV文件
  writeDataToCSV(
      root_path + "data.csv",
      {x_ref, y_ref, x_coords, y_coords, v_coords, theta_coords, acc_coords,
       steering_coords, omega_coords, odom_coords, jerk_values, alpha_values,
       dynamic_traj_x, dynamic_traj_y},
      {"x_ref", "y_ref", "x_coords", "y_coords", "v_coords", "theta_coords",
       "acc_coords", "steering_coords", "omega_coords", "odom_coords",
       "jerk_values", "alpha_values", "dynamic_traj_x", "dynamic_traj_y"});

  return 0;
}