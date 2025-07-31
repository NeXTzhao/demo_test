#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
//
// // ---------- 状态定义 ----------
enum StateIndex {
  X_POS = 0,
  Y_POS = 1,
  THETA = 2,
  SPEED = 3,
  ACCEL = 4,
  DELTAV = 5,
  OMEGA = 6,
  ODOM = 7,
  X_DIM = 8
};
using State = Eigen::Matrix<double, X_DIM, 1>;
//
// // ---------- 构造一阶差分矩阵 ----------
// Eigen::SparseMatrix<double> BuildDiffMatrix(int n) {
//   using T = Eigen::Triplet<double>;
//   std::vector<T> triplets;
//   for (int i = 0; i < n - 1; ++i) {
//     triplets.emplace_back(i, i, -1.0);
//     triplets.emplace_back(i, i + 1, 1.0);
//   }
//   Eigen::SparseMatrix<double> D(n - 1, n);
//   D.setFromTriplets(triplets.begin(), triplets.end());
//   return D;
// }
//
// // ---------- 联合平滑系统：一次性优化所有维度 ----------
//
// // 联合平滑，支持每个维度设置不同的 w_obs 和 w_smooth
// void SmoothXseqJointWeighted(std::vector<State>& x_seq,
//                              const std::vector<StateIndex>& dims,
//                              const std::vector<double>& w_obs_list,
//                              const std::vector<double>& w_smooth_list) {
//   using T = Eigen::Triplet<double>;
//
//   const int n = static_cast<int>(x_seq.size());
//   const int d = static_cast<int>(dims.size());
//   const int N = n * d;
//
//   if (n < 3 || d == 0 || w_obs_list.size() != d || w_smooth_list.size() != d)
//     return;
//
//   // 构造原始列优先向量 x_orig ∈ ℝ^{nd}
//   Eigen::VectorXd x_orig(N);
//   for (int j = 0; j < d; ++j)
//     for (int i = 0; i < n; ++i) x_orig(j * n + i) = x_seq[i][dims[j]];
//
//   Eigen::SparseMatrix<double> D = BuildDiffMatrix(n);
//   Eigen::SparseMatrix<double> DTD = D.transpose() * D;
//
//   std::vector<T> H_triplets;
//   Eigen::VectorXd b = Eigen::VectorXd::Zero(N);
//
//   for (int j = 0; j < d; ++j) {
//     double w_obs = w_obs_list[j];
//     double w_smooth = w_smooth_list[j];
//
//     // 构造每个维度的块 A_j = w_obs * I + w_smooth * DᵀD
//     for (int k = 0; k < DTD.outerSize(); ++k) {
//       for (Eigen::SparseMatrix<double>::InnerIterator it(DTD, k); it; ++it) {
//         int row = j * n + it.row();
//         int col = j * n + it.col();
//         double val = w_smooth * it.value();
//         H_triplets.emplace_back(row, col, val);
//       }
//     }
//     for (int i = 0; i < n; ++i) {
//       int idx = j * n + i;
//       H_triplets.emplace_back(idx, idx, w_obs);
//       b(idx) += w_obs * x_orig(idx);
//     }
//
//     // 固定首点：强约束
//     int idx0 = j * n + 0;
//     H_triplets.emplace_back(idx0, idx0, 1e10);
//     b(idx0) += 1e10 * x_orig(idx0);
//   }
//
//   Eigen::SparseMatrix<double> H(N, N);
//   H.setFromTriplets(H_triplets.begin(), H_triplets.end());
//
//   Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
//   solver.compute(H);
//   Eigen::VectorXd x_opt = solver.solve(b);
//
//   // 回写到 x_seq
//   for (int j = 0; j < d; ++j)
//     for (int i = 0; i < n; ++i) x_seq[i][dims[j]] = x_opt(j * n + i);
// }

class BpPathSmooth {
 public:
  explicit BpPathSmooth() {
    // 内部定义要优化的维度及权重
    dims_ = {X_POS, Y_POS, THETA, SPEED, ACCEL, DELTAV, OMEGA, ODOM};
    w_obs_ = {1.0, 1.0, 0.01, 5.0, 5.0, 0.01, 0.01, 1.0};
    // w_smooth_ = {1.0, 1.0, 0.0, 5.0, 5.0, 2.0, 1.0, 1.0};
    w_smooth_ = {4.5, 4.5, 0.01, 5.0, 1.0, 0.01, 0.01, 1.0};

    d_ = static_cast<int>(dims_.size());
  }

  std::vector<State> solve(const std::vector<State>& x_seq) {
    using T = Eigen::Triplet<double>;
    n_ = static_cast<int>(x_seq.size());
    N_ = n_ * d_;

    Eigen::VectorXd x_orig(N_);
    for (int j = 0; j < d_; ++j)
      for (int i = 0; i < n_; ++i) x_orig(j * n_ + i) = x_seq[i][dims_[j]];

    Eigen::SparseMatrix<double> D = BuildDiffMatrix(n_);
    Eigen::SparseMatrix<double> DTD = D.transpose() * D;

    std::vector<T> H_triplets;
    Eigen::VectorXd b = Eigen::VectorXd::Zero(N_);

    for (int j = 0; j < d_; ++j) {
      double w_obs = w_obs_[j];
      double w_smooth = w_smooth_[j];

      for (int k = 0; k < DTD.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(DTD, k); it; ++it) {
          int row = j * n_ + it.row();
          int col = j * n_ + it.col();
          double val = w_smooth * it.value();
          H_triplets.emplace_back(row, col, val);
        }
      }

      for (int i = 0; i < n_; ++i) {
        int idx = j * n_ + i;
        H_triplets.emplace_back(idx, idx, w_obs);
        b(idx) += w_obs * x_orig(idx);
      }

      // 固定首点
      int idx0 = j * n_ + 0;
      H_triplets.emplace_back(idx0, idx0, 1e10);
      b(idx0) += 1e10 * x_orig(idx0);
    }

    Eigen::SparseMatrix<double> H(N_, N_);
    H.setFromTriplets(H_triplets.begin(), H_triplets.end());

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(H);
    auto x_opt = solver.solve(b);

    std::vector<State> solution(n_);
    for (int j = 0; j < d_; ++j) {
      for (int i = 0; i < n_; ++i) solution[i][dims_[j]] = x_opt(j * n_ + i);
    }
    std::cout << "x_opt: \n" << x_opt << std::endl;
    return solution;
  }

 private:
  static Eigen::SparseMatrix<double> BuildDiffMatrix(int n) {
    using T = Eigen::Triplet<double>;
    std::vector<T> triplets;
    for (int i = 0; i < n - 1; ++i) {
      triplets.emplace_back(i, i, -1.0);
      triplets.emplace_back(i, i + 1, 1.0);
    }
    Eigen::SparseMatrix<double> D(n - 1, n);
    D.setFromTriplets(triplets.begin(), triplets.end());
    return D;
  }

 private:
  int d_, n_{}, N_{};
  std::vector<StateIndex> dims_;
  std::vector<double> w_obs_;
  std::vector<double> w_smooth_;
};

// ---------- 加载轨迹 ----------
void loadTrajectoryFromCSV(std::vector<State>& traj,
                           const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "无法打开文件: " << filename << std::endl;
    return;
  }

  std::string line;
  std::getline(file, line);  // 跳过表头

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string item;
    State s = State::Zero();

    std::getline(ss, item, ',');
    s[X_POS] = std::stod(item);
    std::getline(ss, item, ',');
    s[Y_POS] = std::stod(item);
    std::getline(ss, item, ',');
    s[THETA] = std::stod(item);
    std::getline(ss, item, ',');
    s[SPEED] = std::stod(item);
    std::getline(ss, item, ',');
    s[ACCEL] = std::stod(item);
    std::getline(ss, item, ',');
    s[DELTAV] = std::stod(item);
    std::getline(ss, item, ',');
    s[OMEGA] = std::stod(item);
    std::getline(ss, item, ',');
    s[ODOM] = std::stod(item);

    traj.push_back(s);
  }
}

int main() {
  const std::string path =
      "/home/next/要备份的/demo_test/Optimal_Control_test/"
      "Convex_mpc_adaptive_rho/trajectory.csv";

  std::vector<State> x_seq;
  loadTrajectoryFromCSV(x_seq, path);

  if (x_seq.empty()) {
    std::cerr << "轨迹为空或读取失败" << std::endl;
    return -1;
  }

  // 原始轨迹备份
  std::vector<double> x_raw, y_raw;
  for (const auto& s : x_seq) {
    x_raw.push_back(s[X_POS]);
    y_raw.push_back(s[Y_POS]);
  }

  // 联合优化 + 每个状态维度独立设置权重
  std::vector<StateIndex> dims = {X_POS,  Y_POS, SPEED, THETA,
                                  DELTAV, OMEGA, ODOM,  ACCEL};
  std::vector<double> w_obs_list = {1.0, 1.0, 1.0, 5.0, 5.0, 2.0, 1.0};
  std::vector<double> w_smooth_list = {1.0, 1.0, 0.0, 5.0, 5.0, 2.0, 1.0};

  auto start = std::chrono::high_resolution_clock::now();
  // SmoothXseqJointWeighted(x_seq, dims, w_obs_list, w_smooth_list);
  BpPathSmooth smoother{};
  const auto x_opt = smoother.solve(x_seq);

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "solve time: " << duration.count() << " ms" << std::endl;

  // 平滑后轨迹
  std::vector<double> x_smooth, y_smooth;
  for (const auto& s : x_opt) {
    x_smooth.push_back(s[X_POS]);
    y_smooth.push_back(s[Y_POS]);
  }

  // 可视化
  plt::figure_size(800, 600);
  plt::named_plot("raw", x_raw, y_raw, "-o");
  plt::named_plot("smooth", x_smooth, y_smooth, "-x");
  plt::title("Full-State Joint Smoothing");
  plt::xlabel("X");
  plt::ylabel("Y");
  plt::legend();
  plt::grid(true);
  plt::show();

  return 0;
}
