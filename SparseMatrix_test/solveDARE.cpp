#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseQR>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

class Vehicle {
 public:
  struct State {
    double x;
    double y;
    double theta;
    double v;

    State operator+(const State &other) const {
      return {x + other.x, y + other.y, theta + other.theta, v + other.v};
    }

    State operator*(double scalar) const {
      return {x * scalar, y * scalar, theta * scalar, v * scalar};
    }

    State &operator+=(const State &other) {
      x += other.x;
      y += other.y;
      theta += other.theta;
      v += other.v;
      return *this;
    }
  };

  struct Control {
    double a;
    double delta;
  };

  enum ControlIndex { A = 0, DELTA = 1 };

  explicit Vehicle(double wheelbase) : L(wheelbase) {}

  State dynamics(const State &state, const Control &control) const {
    State dState{};
    dState.x = state.v * cos(state.theta);
    dState.y = state.v * sin(state.theta);
    dState.theta = state.v / L * tan(control.delta);
    dState.v = control.a;
    return dState;
  }

  State updateState(const State &state, const Control &control,
                    double dt) const {
    State k1 = dynamics(state, control);
    State k2 = dynamics(state + k1 * (0.5 * dt), control);
    State k3 = dynamics(state + k2 * (0.5 * dt), control);
    State k4 = dynamics(state + k3 * dt, control);

    State newState = state + (k1 + k2 * 2 + k3 * 2 + k4) * (dt / 6.0);
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
    Eigen::MatrixXd A_dense, B_dense;
    linearize(state, control, dt, A_dense, B_dense);
    A = A_dense.sparseView();
    B = B_dense.sparseView();
  }

 private:
  double L;  // 车辆轴距
};

Eigen::MatrixXd solveLQRDense(const Eigen::MatrixXd &A,
                              const Eigen::MatrixXd &B,
                              const Eigen::MatrixXd &Q,
                              const Eigen::MatrixXd &R) {
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

Eigen::SparseMatrix<double> solveLQRSparse(
    const Eigen::SparseMatrix<double> &A, const Eigen::SparseMatrix<double> &B,
    const Eigen::SparseMatrix<double> &Q,
    const Eigen::SparseMatrix<double> &R) {
  Eigen::SparseMatrix<double> P = Q;
  Eigen::SparseMatrix<double> K;
  Eigen::SparseMatrix<double> P_new;

  //  for (int i = 0; i < 500; ++i) {
  //    Eigen::SparseMatrix<double> temp = R + B.transpose() * P * B;
  //    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>
  //        solver;
  //    solver.compute(temp);
  //    K = solver.solve(B.transpose() * P * A);
  //    P_new = Q + A.transpose() * P * (A - B * K);
  //
  //    if ((P_new - P).norm() < 1e-6) {
  //      break;
  //    }
  //    P = P_new;
  //  }

  for (int i = 0; i < 500; ++i) {
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(
        R + B.transpose() * P * B);
    K = solver.solve(B.transpose() * P * A);
    P_new = Q + A.transpose() * P * (A - B * K);

    if ((P_new - P).norm() < 1e-6) {
      break;
    }
    P = P_new;
  }

  return K;
}

std::vector<Vehicle::State> generateTrajectory(
    Vehicle &vehicle, Vehicle::State initialState,
    const std::vector<Vehicle::Control> &controls, double dt,
    const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R,
    const std::vector<Vehicle::State> &referenceTrajectory, bool useSparse) {
  std::vector<Vehicle::State> trajectory;
  Vehicle::State state = initialState;

  for (size_t i = 0; i < controls.size(); ++i) {
    Eigen::MatrixXd A_dense(4, 4);
    Eigen::MatrixXd B_dense(4, 2);
    Eigen::SparseMatrix<double> A_sparse(4, 4);
    Eigen::SparseMatrix<double> B_sparse(4, 2);
    Eigen::MatrixXd K_dense;
    Eigen::SparseMatrix<double> K_sparse;

    if (useSparse) {
      vehicle.linearizeSparse(state, controls[i], dt, A_sparse, B_sparse);
      K_sparse =
          solveLQRSparse(A_sparse, B_sparse, Q.sparseView(), R.sparseView());
    } else {
      vehicle.linearize(state, controls[i], dt, A_dense, B_dense);
      K_dense = solveLQRDense(A_dense, B_dense, Q, R);
    }

    Eigen::VectorXd x(4);
    x << state.x, state.y, state.theta, state.v;

    Eigen::VectorXd ref(4);
    ref << referenceTrajectory[i].x, referenceTrajectory[i].y,
        referenceTrajectory[i].theta, referenceTrajectory[i].v;
    Eigen::VectorXd error = x - ref;
    Eigen::VectorXd u = useSparse
                            ? Eigen::VectorXd(-K_sparse * error.sparseView())
                            : -K_dense * error;

    Vehicle::Control lqrControl{u(Vehicle::A), u(Vehicle::DELTA)};

    state = vehicle.updateState(state, lqrControl, dt);
    trajectory.push_back(state);
  }

  return trajectory;
}

void analyzeMatrixDensity(const Eigen::MatrixXd &A,
                          const Eigen::SparseMatrix<double> &A_sparse) {
  double dense_elements = A.size();
  double sparse_elements = A_sparse.nonZeros();
  double density = sparse_elements / dense_elements;

  std::cout << "Matrix Density Analysis:\n";
  std::cout << "Dense elements: " << dense_elements << "\n";
  std::cout << "Sparse elements: " << sparse_elements << "\n";
  std::cout << "Density: " << density << "\n";
}

int main() {
  double wheelbase = 2.0;  // 车辆轴距
  Vehicle vehicle(wheelbase);
  double times = 6;
  double dt = 0.2;
  double steps = times / dt + 1.0;

  // 初始状态
  Vehicle::State initialState = {0.0, 0.0, 1.5, 0.0};

  // 控制输入
  std::vector<Vehicle::Control> controls(
      steps, {0.0, 0.0});  // 模拟100个时间步长的控制输入

  // 参考轨迹
  std::vector<Vehicle::State> referenceTrajectory(steps * 2);
  for (size_t i = 0; i < referenceTrajectory.size(); ++i) {
    referenceTrajectory[i] = {static_cast<double>(i) * 0.5,
                              static_cast<double>(i) * 0.5, 0.75, 1.0};
  }

  // LQR权重矩阵
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(4, 4);
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(2, 2);
  R(1, 1) = 50.0;

  auto start_dense = std::chrono::high_resolution_clock::now();
  auto trajectory_dense = generateTrajectory(
      vehicle, initialState, controls, dt, Q, R, referenceTrajectory, false);
  auto end_dense = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_dense = end_dense - start_dense;

  auto start_sparse = std::chrono::high_resolution_clock::now();
  auto trajectory_sparse = generateTrajectory(
      vehicle, initialState, controls, dt, Q, R, referenceTrajectory, true);
  auto end_sparse = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_sparse = end_sparse - start_sparse;

  std::cout << "Dense LQR time: " << elapsed_dense.count() << " seconds\n";
  std::cout << "Sparse LQR time: " << elapsed_sparse.count() << " seconds\n";

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

  plt::named_plot("Sparse Trajectory", x_sparse, y_sparse, "g.");
  plt::named_plot("Dense Trajectory", x_dense, y_dense, "r.");
  plt::named_plot("Reference Trajectory", x_ref, y_ref, "b.");
  plt::legend();
  plt::title("Vehicle Trajectory");
  plt::xlabel("X");
  plt::ylabel("Y");
  plt::show();

  return 0;
}
