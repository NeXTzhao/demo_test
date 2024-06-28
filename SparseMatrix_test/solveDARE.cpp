#include "matplotlibcpp.h"
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseQR>
#include <chrono>
#include <iostream>
#include <vector>

namespace plt = matplotlibcpp;

class Vehicle {
public:
  struct State {
    double x;
    double y;
    double theta;
    double v;
  };

  struct Control {
    double a;
    double delta;
  };

  //  enum StateIndex { X = 0, Y = 1, THETA = 2, V = 3 };

  enum ControlIndex { A = 0, DELTA = 1 };

  explicit Vehicle(double wheelbase) : L(wheelbase) {}

  State updateState(const State &state, const Control &control,
                    double dt) const {
    State newState{};
    newState.x = state.x + state.v * cos(state.theta) * dt;
    newState.y = state.y + state.v * sin(state.theta) * dt;
    newState.theta = state.theta + state.v / L * tan(control.delta) * dt;
    newState.v = state.v + control.a * dt;
    return newState;
  }

  void linearize(const State &state, const Control &control, double dt,
                 Eigen::MatrixXd &A, Eigen::MatrixXd &B) const {
    A = Eigen::MatrixXd::Identity(4, 4);
    B = Eigen::MatrixXd::Zero(4, 2);

    double v = state.v;
    double theta = state.theta;
    double delta = control.delta;
    double sin_theta = sin(theta);
    double cos_theta = cos(theta);
    double cos_delta = cos(delta);
    double tan_delta = tan(delta);

    // clang-format off
    A <<  1,  0,   -v * sin_theta * dt,  cos_theta * dt,
          0,  1,   v * cos_theta * dt,   sin_theta * dt,
          0,  0,        1,               tan_delta * dt / L,
          0,  0,        0,                        1;

    B <<  0,  0,
          0,  0,
          0,   v * dt / (L * cos_delta * cos_delta),
          dt, 0;
    // clang-format on
  }

  void linearizeSparse(const State &state, const Control &control, double dt,
                       Eigen::SparseMatrix<double> &A,
                       Eigen::SparseMatrix<double> &B) const {
    Eigen::MatrixXd A_dense(4, 4);
    Eigen::MatrixXd B_dense(4, 2);

    double v = state.v;
    double theta = state.theta;
    double delta = control.delta;
    double sin_theta = sin(theta);
    double cos_theta = cos(theta);
    double cos_delta = cos(delta);
    double tan_delta = tan(delta);

    // clang-format off
    A_dense <<  1,  0,   -v * sin_theta * dt,  cos_theta * dt,
                0,  1,   v * cos_theta * dt,   sin_theta * dt,
                0,  0,        1,               tan_delta * dt / L,
                0,  0,        0,                        1;

    B_dense <<  0,  0,
                0,  0,
                0,   v * dt / (L * cos_delta * cos_delta),
                dt, 0;
    // clang-format on

    A = A_dense.sparseView();
    B = B_dense.sparseView();
  }

private:
  double L; // 车辆轴距
};

Eigen::MatrixXd solveLQR(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
                         const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R) {
  // 使用A和B矩阵求解LQR增益K
  Eigen::MatrixXd P = Q;
  Eigen::MatrixXd K;
  Eigen::MatrixXd P_new;

  for (int i = 0; i < 500; ++i) {
    K = (R + B.transpose() * P * B).inverse() * B.transpose() * P * A;
    P_new = Q + A.transpose() * P * (A - B * K);

    if ((P_new - P).norm() < 1e-6) {
      break;
    }
    P = P_new;
  }

  return K;
}

Eigen::SparseMatrix<double>
solveLQRSparse(const Eigen::SparseMatrix<double> &A,
               const Eigen::SparseMatrix<double> &B,
               const Eigen::SparseMatrix<double> &Q,
               const Eigen::SparseMatrix<double> &R) {
  // 使用A和B矩阵求解LQR增益K
  Eigen::SparseMatrix<double> P = Q;
  Eigen::SparseMatrix<double> K;
  Eigen::SparseMatrix<double> P_new;

  for (int i = 0; i < 500; ++i) {
    //    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(
    //        R + B.transpose() * P * B);
    Eigen::SparseMatrix<double> temp = R + B.transpose() * P * B;
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>
        solver;
    solver.compute(temp);
    K = solver.solve(B.transpose() * P * A);
    P_new = Q + A.transpose() * P * (A - B * K);

    if ((P_new - P).norm() < 1e-6) {
      break;
    }
    P = P_new;
  }

  return P;
}

std::vector<Vehicle::State>
generateTrajectory(Vehicle &vehicle, Vehicle::State initialState,
                   const std::vector<Vehicle::Control> &controls, double dt,
                   const Eigen::MatrixXd &K,
                   const std::vector<Vehicle::State> &referenceTrajectory) {
  std::vector<Vehicle::State> trajectory;
  Vehicle::State state = initialState;

  for (size_t i = 0; i < controls.size(); ++i) {
    Eigen::VectorXd x(4);
    x << state.x, state.y, state.theta, state.v;

    Eigen::VectorXd ref(4);
    ref << referenceTrajectory[i].x, referenceTrajectory[i].y,
        referenceTrajectory[i].theta, referenceTrajectory[i].v;

    Eigen::VectorXd u = -K * (x - ref);

    Vehicle::Control lqrControl{};
    lqrControl.a = u(Vehicle::A);
    lqrControl.delta = u(Vehicle::DELTA);

    state = vehicle.updateState(state, lqrControl, dt);
    trajectory.push_back(state);
  }

  return trajectory;
}

int main() {
  double wheelbase = 2.0; // 车辆轴距
  Vehicle vehicle(wheelbase);

  // 初始状态
  Vehicle::State initialState = {0.0, 0.0, 0.0, 0.0};

  // 控制输入
  std::vector<Vehicle::Control> controls(
      100, {1.0, 0.1}); // 模拟100个时间步长的控制输入

  // 参考轨迹
  std::vector<Vehicle::State> referenceTrajectory(100);
  for (size_t i = 0; i < referenceTrajectory.size(); ++i) {
    referenceTrajectory[i] = {static_cast<double>(i) * 0.1,
                              static_cast<double>(i) * 0.1, 0.0, 1.0};
  }

  // 状态空间矩阵
  Eigen::MatrixXd A(4, 4);
  Eigen::MatrixXd B(4, 2);
  vehicle.linearize(initialState, controls[0], 0.1, A, B);

  // 稀疏状态空间矩阵
  Eigen::SparseMatrix<double> A_sparse(4, 4);
  Eigen::SparseMatrix<double> B_sparse(4, 2);
  vehicle.linearizeSparse(initialState, controls[0], 0.1, A_sparse, B_sparse);

  // LQR权重矩阵
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(4, 4);
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(2, 2);

  Eigen::SparseMatrix<double> Q_sparse = Q.sparseView();
  Eigen::SparseMatrix<double> R_sparse = R.sparseView();

  // 稠密矩阵求解LQR
  auto start_dense = std::chrono::high_resolution_clock::now();
  Eigen::MatrixXd K_dense = solveLQR(A, B, Q, R);
  std::cout << "k d: " << K_dense << std::endl;

  auto end_dense = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_dense = end_dense - start_dense;

  // 稀疏矩阵求解LQR
  auto start_sparse = std::chrono::high_resolution_clock::now();
  Eigen::SparseMatrix<double> K_sparse =
      solveLQRSparse(A_sparse, B_sparse, Q_sparse, R_sparse);
  std::cout << "k s: " << K_sparse << std::endl;
  auto end_sparse = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_sparse = end_sparse - start_sparse;

  // 输出结果
  std::cout << "Dense matrix LQR solution time: " << elapsed_dense.count()
            << " seconds" << std::endl;
  std::cout << "Sparse matrix LQR solution time: " << elapsed_sparse.count()
            << " seconds" << std::endl;

  // 生成轨迹
  auto trajectory_dense = generateTrajectory(vehicle, initialState, controls,
                                             0.1, K_dense, referenceTrajectory);
  auto trajectory_sparse = generateTrajectory(
      vehicle, initialState, controls, 0.1, K_sparse, referenceTrajectory);

  // 可视化轨迹
  std::vector<double> x_dense, y_dense, x_sparse, y_sparse, x_ref, y_ref;
  for (const auto &state : trajectory_dense) {
    x_dense.push_back(state.x);
    y_dense.push_back(state.y);
  }
  for (const auto &state : trajectory_sparse) {
    x_sparse.push_back(state.x);
    y_sparse.push_back(state.y);
  }
  for (const auto &state : referenceTrajectory) {
    x_ref.push_back(state.x);
    y_ref.push_back(state.y);
  }

  //  plt::named_plot("Sparse Trajectory", x_sparse, y_sparse, "g-");
  plt::named_plot("Dense Trajectory", x_dense, y_dense, "r.");
  plt::named_plot("Reference Trajectory", x_ref, y_ref, "b.");
  plt::legend();
  plt::title("Vehicle Trajectory");
  plt::xlabel("X");
  plt::ylabel("Y");
  plt::show();

  return 0;
}
