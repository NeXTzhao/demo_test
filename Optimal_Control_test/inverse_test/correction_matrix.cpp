#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <vector>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

// 计算矩阵的条件数
double condition_number(const Eigen::MatrixXd &matrix) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix);
  return svd.singularValues()(0) / svd.singularValues().tail(1)(0);
}

// 计算残差
double compute_residual(const Eigen::MatrixXd &A, const Eigen::MatrixXd &A_inv) {
  return (Eigen::MatrixXd::Identity(A.rows(), A.cols()) - A * A_inv).norm();
}

// 计算向后误差
double compute_backward_error(const Eigen::MatrixXd &A, const Eigen::MatrixXd &A_inv) {
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(A.rows(), A.cols());
  Eigen::MatrixXd E = A_inv * A - I;
  return E.norm() / I.norm();
}

// 计算向前误差
double compute_forward_error(const Eigen::MatrixXd &A, const Eigen::MatrixXd &A_inv) {
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(A.rows(), A.cols());
  Eigen::MatrixXd E = A_inv - A.inverse();
  return E.norm() / A.inverse().norm();
}

// 计算误差放大因子
double compute_error_amplification_factor(double backward_error, double forward_error) {
  return forward_error / backward_error;
}

// 尝试初始Cholesky分解
bool attemptInitialCholesky(const Eigen::MatrixXd &A, Eigen::MatrixXd &A_inv, std::vector<double> &metrics) {
  Eigen::LLT<Eigen::MatrixXd> llt(A);
  if (llt.info() == Eigen::Success) {
    auto start = std::chrono::high_resolution_clock::now();
    A_inv = llt.solve(Eigen::MatrixXd::Identity(A.rows(), A.cols()));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    metrics[0] = duration.count();
    metrics[1] = condition_number(A_inv);
    metrics[2] = compute_residual(A, A_inv);
    metrics[3] = compute_backward_error(A, A_inv);
    metrics[4] = compute_forward_error(A, A_inv);
    metrics[5] = compute_error_amplification_factor(metrics[3], metrics[4]);
    std::cout << "init matrix success" << std::endl;
    std::cout << "Time: " << duration.count() << " s, "
              << "Condition Number: " << condition_number(A_inv) << ", "
              << "Residual: " << compute_residual(A, A_inv) << ", "
              << "Backward Error: " << compute_backward_error(A, A_inv) << ", "
              << "Forward Error: " << compute_forward_error(A, A_inv) << ", "
              << "Error Amplification Factor: " << compute_error_amplification_factor(metrics[3], metrics[4])
              << std::endl;
    return true;
  } else {
    metrics[1] = condition_number(A);
    std::cout << "Init Matrix Condition Number: " << metrics[1] << std::endl;
  }

  return false;
}

// 尝试Cholesky分解，如果失败则进行松弛法修正
bool attemptCholeskyWithCorrection(const Eigen::MatrixXd &A, Eigen::MatrixXd &A_inv, std::vector<double> &metrics, double epsilon = 1e-6, int max_iterations = 10) {
  Eigen::MatrixXd A_chol = A;
  Eigen::LLT<Eigen::MatrixXd> llt;
  for (int i = 0; i < max_iterations; ++i) {
    llt.compute(A_chol);
    if (llt.info() == Eigen::Success) {
      auto start = std::chrono::high_resolution_clock::now();
      A_inv = llt.solve(Eigen::MatrixXd::Identity(A.rows(), A.cols()));
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = end - start;
      metrics[0] = duration.count();
      metrics[1] = condition_number(A_inv);
      metrics[2] = compute_residual(A, A_inv);
      metrics[3] = compute_backward_error(A, A_inv);
      metrics[4] = compute_forward_error(A, A_inv);
      metrics[5] = compute_error_amplification_factor(metrics[3], metrics[4]);
      std::cout << "Step " << i << " success" << std::endl;
      std::cout << "Time: " << duration.count() << " s, "
                << "Condition Number: " << condition_number(A_inv) << ", "
                << "Residual: " << compute_residual(A, A_inv) << ", "
                << "Backward Error: " << compute_backward_error(A, A_inv) << ", "
                << "Forward Error: " << compute_forward_error(A, A_inv) << ", "
                << "Error Amplification Factor: " << compute_error_amplification_factor(metrics[3], metrics[4])
                << std::endl;
      return true;
    } else {
      A_chol += epsilon * Eigen::MatrixXd::Identity(A.rows(), A.cols());
      epsilon *= 10;// 增加修正幅度
    }
  }
  return false;
}

// 可视化结果
void visualize(const std::vector<double> &metrics_no_correction, const std::vector<double> &metrics_with_correction) {
  plt::figure_size(1200, 800);

  std::vector<std::string> titles = {"Time", "Condition Number", "Residual", "Backward Error", "Forward Error", "Error Amplification Factor"};
  for (size_t i = 0; i < titles.size(); ++i) {
    plt::subplot(2, 3, i + 1);
    std::vector<double> x = {1, 2};// 用于标记修正前后的指标
    std::vector<double> y = {metrics_no_correction[i], metrics_with_correction[i]};
    plt::bar(x, y);
    plt::title(titles[i]);
    plt::xticks(x, std::vector<std::string>{"No Correction", "With Correction"});
    plt::grid(true);
  }
  plt::show();
}


int main() {
  int dim = 500;
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(dim, dim);
  A = A * A.transpose();
  //  A.diagonal().array() -= dim * 10;
  A.diagonal()(0) -= 1;

  std::cout << "Original Matrix A :\n"
            << A << std::endl;

  std::vector<double> metrics_no_correction(6, 0);
  std::vector<double> metrics_with_correction(6, 0);

  Eigen::MatrixXd A_inv;

  // 尝试直接进行 Cholesky 分解
  if (!attemptInitialCholesky(A, A_inv, metrics_no_correction)) {
    std::cout << "Initial Cholesky decomposition failed. Proceeding with relaxation..." << std::endl;
    attemptCholeskyWithCorrection(A, A_inv, metrics_with_correction);
  }

  // 可视化结果
  visualize(metrics_no_correction, metrics_with_correction);

  return 0;
}
