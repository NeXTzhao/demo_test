#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#include "common/log.h"
#include "common/matplotlibcpp.h"
#include "nlohmann/json.hpp"
#include "third_party/tinympc/tiny_api.hpp"
#define NHORIZON 30

namespace plt = matplotlibcpp;
using json = nlohmann::json;

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

  std::vector<State> simulate(const State& initial_state,
                              const std::vector<Control>& controls,
                              double increment, int USE_RK = 4) const {
    std::vector<State> states;
    states.reserve(controls.size() + 1);  // 预留空间以优化性能
    states.push_back(initial_state);      // 第一个状态是初始状态

    State current_state = initial_state;

    for (const auto& control : controls) {
      // 计算下一个状态
      State next_state = EvalOneStep(current_state, control, increment, USE_RK);
      states.push_back(next_state);  // 保存计算出的下一个状态
      current_state = next_state;    // 更新当前状态为下一步的初始状态
    }

    return states;
  }

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

  void solve(const State& initial_state,
             const std::vector<State>& reference_trajectory) {
    TinySolver* solver;

    Eigen::MatrixXd A_t(StateDim, StateDim);
    Eigen::MatrixXd B_t(StateDim, ControlDim);
    vehicle_.linearize(initial_state, dt_, A_t, B_t);

    tinyMatrix x_min, x_max, u_min, u_max;
    SetBoundConstraints(x_min, x_max, u_min, u_max, NHORIZON);

    tinytype rho_value = 5.0;
    tinytype verbose = 0;
    int status =
        tiny_setup(&solver, A_t, B_t, Q_, R_, rho_value, StateDim, ControlDim,
                   NHORIZON, x_min, x_max, u_min, u_max, verbose);

    // Update whichever settings we'd like
    solver->settings->max_iter = 50;

    // Alias solver->work for brevity
    TinyWorkspace* work = solver->work;

    // initial guess
    work->Xref = initial_state;

    // reference trajectory
    int point_num = reference_trajectory.size();
    int state_num = reference_trajectory[0].size();
    Eigen::MatrixXd Xref_total(state_num, point_num);  // 使用动态大小矩阵

    for (size_t i = 0; i < point_num; ++i) {
      Xref_total.col(i) = reference_trajectory[i];
    }

    // 将 reference_matrix 中的数据映射到 work->Xref
    work->Xref = Xref_total.block(0, 0, StateDim, NHORIZON);

    State x0;
    x0 = work->Xref.col(0);
    state_.push_back(x0);

    for (int k = 0; k < point_num; ++k) {
      // 1. 更新测量值
      tiny_set_x0(solver, x0);

      // 2. 更新参考轨迹
      if (k + NHORIZON <= point_num) {
        work->Xref = Xref_total.block(0, k, StateDim, NHORIZON);
      } else {
        int remaining = point_num - k;
        work->Xref.leftCols(remaining) =
            Xref_total.block(0, k, StateDim, remaining);
        work->Xref.rightCols(NHORIZON - remaining) =
            Xref_total.col(point_num - 1).replicate(1, NHORIZON - remaining);
      }

      // 3. 重置对偶变量
      work->y = Eigen::Matrix<tinytype, ControlDim, NHORIZON - 1>::Zero();
      work->g = Eigen::Matrix<tinytype, StateDim, NHORIZON>::Zero();

      // 4. 解决MPC问题
      tiny_solve(solver);

      // 5. 向前模拟
      auto u_step = work->u.col(0);
//      std::cout << "controls: \n" << u_step.transpose() << std::endl;
      x0 = work->Adyn * x0 + work->Bdyn * u_step;
      state_.push_back(x0);
      control_.push_back(u_step);
    }

    std::cout << "state size: " << state_.size() << std::endl;
  }

  void SetBoundConstraints(Eigen::MatrixXd& x_min, Eigen::MatrixXd& x_max,
                           Eigen::MatrixXd& u_min, Eigen::MatrixXd& u_max,
                           int horizon_size) {
    Eigen::VectorXd state_min(StateDim), state_max(StateDim);
    Eigen::VectorXd control_min(ControlDim), control_max(ControlDim);

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

    double inf = 1e10;

    // state bounds
    state_min << -inf, -inf, min_v, -inf, min_delta, -inf, -inf, min_a;
    state_max << inf, inf, max_v, inf, max_delta, inf, inf, max_a;
    // control bounds
    control_min << min_jerk, min_alpha;
    control_max << max_jerk, max_alpha;

    // 将边界扩展到整个预测窗口
    x_min = state_min.replicate(1, horizon_size);
    x_max = state_max.replicate(1, horizon_size);
    u_min = control_min.replicate(1, horizon_size - 1);
    u_max = control_max.replicate(1, horizon_size - 1);

    //    AINFO << "State bounds (x_min): \n" << x_min;
    //    AINFO << "State bounds (x_max): \n" << x_max;
    //    AINFO << "Control bounds (u_min): \n" << u_min;
    //    AINFO << "Control bounds (u_max): \n" << u_max;
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
  //  AINFO << "init state: " << initial_state.transpose() << "\n";

  std::vector<Vehicle<StateDim, ControlDim>::State> targetTrajectory(
      horizon, initial_state);

  loadTrajectoryFromCSV<StateDim, ControlDim>(
      targetTrajectory,
      "/home/next/demo_test/Optimal_Control_test/Convex_mpc/trajectory.csv");

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

  //  // 控制量反推轨迹
  std::vector<double> dynamic_traj_x, dynamic_traj_y;
  auto dyn_traj =
      vehicle.simulate(targetTrajectory.front(), control_inputs, dt);
  for (const auto& state : dyn_traj) {
    dynamic_traj_x.push_back(state(Vehicle<StateDim, ControlDim>::X_POS));
    dynamic_traj_y.push_back(state(Vehicle<StateDim, ControlDim>::Y_POS));
  }

  // 将所有数据写入一个CSV文件
  writeDataToCSV(root_path + "data.csv",
                 {x_ref, y_ref, x_coords, y_coords, dynamic_traj_x,
                  dynamic_traj_y, v_coords, acc_coords, theta_coords,
                  steering_coords, jerk_values, alpha_values},
                 {"x_ref", "y_ref", "x_coords", "y_coords", "dynamic_traj_x",
                  "dynamic_traj_y", "velocity", "acc", "theta", "steering",
                  "d(aa)_jerk", "dd(steering)_alpha"});
  //  writeDataToCSV(root_path + "data.csv",
  //                 {x_ref, y_ref, x_coords, y_coords, v_coords, acc_coords,
  //                  theta_coords, steering_coords},
  //                 {"x_ref", "y_ref", "x_coords", "y_coords", "velocity",
  //                 "acc",
  //                  "theta", "steering"});
  return 0;
}

//		[0, 0], [10, 0], [20, 0], [20, 5], [30, 5], [40, 5]
