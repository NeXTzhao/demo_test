#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#include "bp_path_tracking/lateral_model.hpp"
#include "bp_path_tracking/longitudinal_model.hpp"
#include "common/log.h"
#include "nlohmann/json.hpp"
#include "third_party/tinympc/tiny_api.hpp"

#define USE_RK 4
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
class LongitudinalMPC {
 public:
  using State = typename LongitudinalModel<StateDim, ControlDim>::State;
  using Control = typename LongitudinalModel<StateDim, ControlDim>::Control;

  LongitudinalMPC(LongitudinalModel<StateDim, ControlDim>& vehicle, int horizon,
                  double dt, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
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
    auto [A_, B_] = vehicle_.linearize(initial_state, dt_);
    tinytype rho_value = 1e-5;
    tinytype verbose = 1;
    int status =
        tiny_setup(&solver_, A_, B_, Q_, R_, rho_value, StateDim, ControlDim,
                   horizon_, x_min_, x_max_, u_min_, u_max_, verbose);
    if (status != 0 || solver_ == nullptr) {
      std::cout << "TinyMPC setup failed." << std::endl;
    }

    solver_->settings->max_iter = 5;
    solver_->settings->abs_pri_tol = 0.1;
    solver_->settings->abs_dua_tol = 0.1;
    solver_->settings->check_termination = 1;

    const int point_num = reference_trajectory.size();
    Eigen::MatrixXd Xref_total(StateDim, point_num);
    for (int i = 0; i < point_num; ++i) {
      Xref_total.col(i) = reference_trajectory[i];
    }

    State x0 = initial_state;
    state_.push_back(x0);

    TinyWorkspace* work = solver_->work;
    int total_iterations = 0;

    tinytype total_tracking_error = 0;

    for (int k = 0; k < point_num - 1; ++k) {
      tinytype current_error = (x0 - work->Xref.col(1)).norm();
      total_tracking_error += current_error;

      std::cout << "tracking error: " << (x0 - work->Xref.col(1)).norm()
                << std::endl;
      AINFO << "tracking error: \n" << (x0 - work->Xref.col(1)) << std::endl;

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
      auto u_step = work->u.col(0);

      x0 = vehicle_.EvalOneStep(x0, u_step, dt_);

      total_iterations += solver_->solution->iter;
      printf("Iterations for step %2d: %d (cumulative: %d)\n", k,
             solver_->solution->iter, total_iterations);

      state_.push_back(x_step);

      // double alpha_ff = 0.25 * GetFeedforwardSteering(kappas[k]);
      // u_step(1) += alpha_ff;
      control_.push_back(u_step);
    }

    printf("\nTotal iterations across all MPC solves: %d\n", total_iterations);
    printf("Average tracking error: %.4f\n",
           total_tracking_error / solver_->settings->max_iter);
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

    state_min << -inf, 0, -2, 0;
    state_max << inf, 35, 35, inf;

    control_min << -inf;
    control_max << inf;

    x_min = state_min.replicate(1, horizon_size);
    x_max = state_max.replicate(1, horizon_size);
    u_min = control_min.replicate(1, horizon_size - 1);
    u_max = control_max.replicate(1, horizon_size - 1);
  }

 private:
  LongitudinalModel<StateDim, ControlDim>& vehicle_;
  const int horizon_;
  const double dt_;
  Eigen::MatrixXd Q_, R_, Q_N_;

  std::vector<State> state_;
  std::vector<Control> control_;

  TinySolver* solver_;
  // Eigen::MatrixXd A_, B_;
  Eigen::MatrixXd x_min_, x_max_, u_min_, u_max_;
};

template <int StateDim, int ControlDim>
class LateralMPC {
 public:
  using State = typename LateralModel<StateDim, ControlDim>::State;
  using Control = typename LateralModel<StateDim, ControlDim>::Control;

  LateralMPC(LateralModel<StateDim, ControlDim>& vehicle, int horizon,
             double dt, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
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
    // 线性化系统.

    auto ab_mat = vehicle_.linearize(initial_state, dt_);
    auto A_ = ab_mat.first;
    Eigen::Matrix<double, StateDim, ControlDim> B_ = ab_mat.second;

    tinytype rho_value = 1e-5;
    tinytype verbose = 1;
    int status =
        tiny_setup(&solver_, A_, B_, Q_, R_, rho_value, StateDim, ControlDim,
                   horizon_, x_min_, x_max_, u_min_, u_max_, verbose);
    if (status != 0 || solver_ == nullptr) {
      std::cout << "TinyMPC setup failed." << std::endl;
    }

    solver_->settings->max_iter = 5;
    solver_->settings->abs_pri_tol = 0.1;
    solver_->settings->abs_dua_tol = 0.1;
    solver_->settings->check_termination = 1;

    const int point_num = reference_trajectory.size();
    Eigen::MatrixXd Xref_total(StateDim, point_num);
    for (int i = 0; i < point_num; ++i) {
      Xref_total.col(i) = reference_trajectory[i];
    }

    State x0 = initial_state;
    state_.push_back(x0);

    TinyWorkspace* work = solver_->work;
    int total_iterations = 0;

    tinytype total_tracking_error = 0;

    for (int k = 0; k < point_num - 1; ++k) {
      tinytype current_error = (x0 - work->Xref.col(1)).norm();
      total_tracking_error += current_error;

      std::cout << "tracking error: " << (x0 - work->Xref.col(1)).norm()
                << std::endl;
      AINFO << "tracking error: \n" << (x0 - work->Xref.col(1)) << std::endl;

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
      auto u_step = work->u.col(0);

      x0 = vehicle_.EvalOneStep(x0, u_step, dt_);

      total_iterations += solver_->solution->iter;
      printf("Iterations for step %2d: %d (cumulative: %d)\n", k,
             solver_->solution->iter, total_iterations);

      state_.push_back(x_step);
      control_.push_back(u_step);
    }

    printf("\nTotal iterations across all MPC solves: %d\n", total_iterations);
    printf("Average tracking error: %.4f\n",
           total_tracking_error / solver_->settings->max_iter);
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

    state_min << -inf, -inf, -inf, -inf;
    state_max << inf, inf, inf, inf;

    control_min << -inf;
    control_max << inf;

    x_min = state_min.replicate(1, horizon_size);
    x_max = state_max.replicate(1, horizon_size);
    u_min = control_min.replicate(1, horizon_size - 1);
    u_max = control_max.replicate(1, horizon_size - 1);
  }

 private:
  LateralModel<StateDim, ControlDim> vehicle_;
  const int horizon_;
  const double dt_;
  Eigen::MatrixXd Q_, R_, Q_N_;

  std::vector<State> state_;
  std::vector<Control> control_;

  TinySolver* solver_;
  Eigen::MatrixXd x_min_, x_max_, u_min_, u_max_;
};

template <size_t StateDim, size_t ControlDim>
void loadTrajectoryFromCSV(
    std::vector<typename LongitudinalModel<StateDim, ControlDim>::State>&
        lon_traj,
    std::vector<typename LateralModel<StateDim, ControlDim>::State>& lan_traj,
    const std::string& filename) {
  std::ifstream file(filename);
  std::string line;

  // 跳过表头
  std::getline(file, line);

  size_t i = 0;  // 索引
  while (std::getline(file, line) && i < lon_traj.size()) {
    std::stringstream ss(line);
    std::string item;

    // 读取并解析每一列的数据
    std::getline(ss, item, ',');
    lon_traj[i][LongitudinalModel<StateDim, ControlDim>::X_POS] =
        std::stod(item);

    std::getline(ss, item, ',');
    lan_traj[i][LateralModel<StateDim, ControlDim>::Y_POS] = std::stod(item);

    std::getline(ss, item, ',');
    lan_traj[i][LateralModel<StateDim, ControlDim>::THETA] = std::stod(item);

    std::getline(ss, item, ',');
    lon_traj[i][LongitudinalModel<StateDim, ControlDim>::SPEED] =
        std::stod(item);

    std::getline(ss, item, ',');
    lon_traj[i][LongitudinalModel<StateDim, ControlDim>::ACCEL] =
        std::stod(item);

    std::getline(ss, item, ',');
    lan_traj[i][LateralModel<StateDim, ControlDim>::DELTAV] = std::stod(item);

    std::getline(ss, item, ',');
    lan_traj[i][LateralModel<StateDim, ControlDim>::OMEGA] = std::stod(item);

    std::getline(ss, item, ',');
    lon_traj[i][LongitudinalModel<StateDim, ControlDim>::ODOM] =
        std::stod(item);

    ++i;  // 更新索引
  }
}

int main() {
  // 读取JSON文件
  std::ifstream i(root_path + "params.json");
  json j;
  i >> j;

  const int StateDim = 4;
  const int ControlDim = 1;

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
  Q.diagonal() << j["Q"][0], j["Q"][1], j["Q"][2], j["Q"][3];
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(ControlDim, ControlDim);
  R.diagonal() << j["R"][0];
  Eigen::MatrixXd Q_N = Eigen::MatrixXd::Identity(StateDim, StateDim);
  Q_N.diagonal() << j["Q_N"][0], j["Q_N"][1], j["Q_N"][2], j["Q_N"][3];

  LateralModel<StateDim, ControlDim> lat_model;
  LateralMPC<StateDim, ControlDim> lat_mpc(lat_model, horizon, dt, Q, R, Q_N);

  LongitudinalModel<StateDim, ControlDim> lon_model;
  LongitudinalMPC<StateDim, ControlDim> lon_mpc(lon_model, horizon, dt, Q, R,
                                                Q_N);

  std::vector<LateralModel<StateDim, ControlDim>::State> lat_traj;
  lat_traj.resize(horizon);
  std::vector<LongitudinalModel<StateDim, ControlDim>::State> lon_traj;
  lon_traj.resize(horizon);

  loadTrajectoryFromCSV<StateDim, ControlDim>(lat_traj, lon_traj,
                                              row_traj_path);

  auto start_point = lat_traj.front();
  auto old_lon_traj = lat_traj;
  auto old__lat_traj = lat_traj;

  auto start = std::chrono::high_resolution_clock::now();

  // 使用最优参数运行MPC
  lat_mpc.solve(lat_traj[0], lat_traj);
  auto lat_opt_state = lat_mpc.getState();
  auto lat_opt_control = lat_mpc.getControl();

  // lon_mpc.solve(lon_traj[0], lon_traj);
  // auto lon_opt_state = lon_mpc.getState();
  // auto lon_opt_control = lon_mpc.getControl();

  std::vector<Eigen::Matrix<double, 8, 1>> state_seq;
  std::vector<Eigen::Matrix<double, 2, 1>> control_seq;

  state_seq.clear();
  control_seq.clear();

  // for (int i = 0; i < NHORIZON; ++i) {
  //   Eigen::Matrix<double, 8, 1> full_state =
  //       Eigen::Matrix<double, 8, 1>::Zero();
  //   Eigen::Matrix<double, 2, 1> full_control =
  //       Eigen::Matrix<double, 2, 1>::Zero();
  //
  //   // 合并纵向状态
  //   full_state(0) =
  //       lon_opt_state[i](LongitudinalModel<StateDim, ControlDim>::X_POS);
  //   full_state(1) = lat_opt_state[i](LateralModel<StateDim, ControlDim>::Y_POS);
  //   ;  // Y_POS
  //   full_state(2) =
  //       lon_opt_state[i](LongitudinalModel<StateDim, ControlDim>::SPEED);
  //   full_state(3) = lat_opt_state[i](LateralModel<StateDim, ControlDim>::THETA);
  //   ;
  //   full_state(4) =
  //       lat_opt_state[i](LateralModel<StateDim, ControlDim>::DELTAV);
  //   ;
  //   full_state(5) = lat_opt_state[i](LateralModel<StateDim, ControlDim>::OMEGA);
  //   full_state(6) =
  //       lon_opt_state[i](LongitudinalModel<StateDim, ControlDim>::ODOM);
  //   ;
  //   full_state(7) =
  //       lon_opt_state[i](LongitudinalModel<StateDim, ControlDim>::ACCEL);
  //   ;
  //
  //   state_seq.push_back(full_state);
  //
  //   if (i < NHORIZON - 1) {
  //     full_control(0) =
  //         lon_opt_control[i](LongitudinalModel<StateDim, ControlDim>::JERK);
  //     ;
  //     full_control(1) =
  //         lat_opt_control[i](LateralModel<StateDim, ControlDim>::ALPHAV);
  //     control_seq.push_back(full_control);
  //   }
  // }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "solve time: " << duration.count() << " ms" << std::endl;

  // 提取轨迹
  // auto opt_trajectory = mpc.getState();
  // auto control_inputs = mpc.getControl();

  // auto dyn_traj = lat_mpc.simulate(start_point, control_inputs, dt);

  //
  // std::vector<double> x_ref, y_ref, x_coords, y_coords, v_coords,
  // theta_coords,
  //     acc_coords, steering_coords, omega_coords, odom_coords;
  // for (const auto& state : old_traj) {
  //   x_ref.push_back(state(Vehicle<StateDim, ControlDim>::X_POS) -
  //                   start_point(Vehicle<StateDim, ControlDim>::X_POS));
  //   y_ref.push_back(state(Vehicle<StateDim, ControlDim>::Y_POS) -
  //                   start_point(Vehicle<StateDim, ControlDim>::Y_POS));
  // }
  // for (const auto& state : opt_trajectory) {
  //   x_coords.push_back(state(Vehicle<StateDim, ControlDim>::X_POS) -
  //                      start_point(Vehicle<StateDim, ControlDim>::X_POS));
  //   y_coords.push_back(state(Vehicle<StateDim, ControlDim>::Y_POS) -
  //                      start_point(Vehicle<StateDim, ControlDim>::Y_POS));
  //   v_coords.push_back(state(Vehicle<StateDim, ControlDim>::SPEED));
  //   theta_coords.push_back(state(Vehicle<StateDim, ControlDim>::THETA));
  //   steering_coords.push_back(state(Vehicle<StateDim, ControlDim>::DELTAV));
  //   omega_coords.push_back(state(Vehicle<StateDim, ControlDim>::OMEGA));
  //   odom_coords.push_back(state(Vehicle<StateDim, ControlDim>::ODOM));
  //   acc_coords.push_back(state(Vehicle<StateDim, ControlDim>::ACCEL));
  // }
  // // 提取控制量中的A和DELTA
  // std::vector<double> jerk_values, alpha_values;
  // for (const auto& control : control_inputs) {
  //   jerk_values.push_back(control(Vehicle<StateDim, ControlDim>::JERK));
  //   alpha_values.push_back(control(Vehicle<StateDim, ControlDim>::ALPHAV));
  // }
  //
  // // 控制量反推轨迹
  // std::vector<double> dynamic_traj_x, dynamic_traj_y;
  //
  // for (const auto& state : dyn_traj) {
  //   dynamic_traj_x.push_back(state(Vehicle<StateDim, ControlDim>::X_POS) -
  //                            start_point(Vehicle<StateDim,
  //                            ControlDim>::X_POS));
  //   dynamic_traj_y.push_back(state(Vehicle<StateDim, ControlDim>::Y_POS) -
  //                            start_point(Vehicle<StateDim,
  //                            ControlDim>::Y_POS));
  // }
  //
  // // 将所有数据写入一个CSV文件
  // writeDataToCSV(
  //     root_path + "data.csv",
  //     {x_ref, y_ref, x_coords, y_coords, v_coords, theta_coords, acc_coords,
  //      steering_coords, omega_coords, odom_coords, jerk_values, alpha_values,
  //      dynamic_traj_x, dynamic_traj_y},
  //     {"x_ref", "y_ref", "x_coords", "y_coords", "v_coords", "theta_coords",
  //      "acc_coords", "steering_coords", "omega_coords", "odom_coords",
  //      "jerk_values", "alpha_values", "dynamic_traj_x", "dynamic_traj_y"});

  return 0;
}