#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <limits>

namespace ocp {

/**
 * @brief Derivative of R -> R
 *
 * @param f
 * @param x
 * @param eps
 * @return double
 */
static double
Derivative(std::function<double(double)> f, double x,
           double eps = std::pow(std::numeric_limits<double>::epsilon(), 0.5)) {
  return (f(x + eps) - f(x - eps)) / (2.0 * eps);
}

/**
 * @brief Derivative of R^n ->R
 *
 * @param f
 * @param x
 * @param eps
 * @return Eigen::VectorXd
 */
static Eigen::VectorXd
Derivative(std::function<double(const Eigen::Ref<const Eigen::VectorXd> &)> f,
           const Eigen::Ref<const Eigen::VectorXd> &x,
           double eps = std::pow(std::numeric_limits<double>::epsilon(), 0.5)) {
  Eigen::VectorXd J(x.rows());
  Eigen::VectorXd diff(x.rows());
  for (Eigen::VectorXd::Index i = 0; i < x.rows(); i++) {
    diff.setZero();
    diff(i) = eps;
    J(i) = (f(x + diff) - f(x - diff)) / (2.0 * eps);
  }
  return J;
}

/**
 * @brief Derivative of R^n -> R^m
 *
 * @param f
 * @param x
 * @param eps
 * @return Eigen::MatrixXd
 */
static Eigen::MatrixXd Derivative(
    std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd> &)> f,
    const Eigen::Ref<const Eigen::VectorXd> &x,
    double eps = std::pow(std::numeric_limits<double>::epsilon(), 0.5)) {
  Eigen::MatrixXd J;
  Eigen::VectorXd J_col;
  Eigen::VectorXd diff(x.rows());
  for (Eigen::VectorXd::Index i = 0; i < x.rows(); i++) {
    diff.setZero();
    diff(i) = eps;
    J_col = (f(x + diff) - f(x - diff)) / (2.0 * eps);
    if (i == 0) {
      J = Eigen::MatrixXd::Zero(J_col.rows(), x.rows());
    }
    J.col(i) = J_col;
  }
  return J;
}

/**
 * @brief Second derivative of R -> R
 *
 * @param f
 * @param x
 * @param eps
 * @return double
 */
static double SecondDerivative(
    std::function<double(double)> f, double x,
    double eps = std::pow(std::numeric_limits<double>::epsilon(), 0.5)) {
  return (f(x + eps) - 2 * f(x) + f(x - eps)) / std::pow(eps, 2);
}

/**
 * @brief Second derivative of  R^n -> R
 *
 * @param f
 * @param x
 * @param eps
 * @return Eigen::MatrixXd
 */
static Eigen::MatrixXd SecondDerivative(
    std::function<double(const Eigen::Ref<const Eigen::VectorXd> &)> f,
    const Eigen::Ref<const Eigen::VectorXd> &x,
    double eps = std::pow(std::numeric_limits<double>::epsilon(), 0.5)) {
  int n = x.size();
  Eigen::MatrixXd H(n, n);
  Eigen::VectorXd e_i(n);
  Eigen::VectorXd e_j(n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      e_i.setZero();
      e_j.setZero();
      e_i[i] = eps;
      e_j[j] = eps;

      double f_ij = f(x + e_i + e_j) - f(x + e_i - e_j) - f(x - e_i + e_j) +
                    f(x - e_i - e_j);
      H(i, j) = f_ij / (4 * eps * eps);
    }
  }
  return H;
}

/**
 * @brief Second derivative of  R^(n, m) -> R^(n, m)
 *
 * @param f
 * @param x
 * @param eps
 * @return Eigen::MatrixXd
 */
static Eigen::MatrixXd MixedDerivative(
    std::function<double(const Eigen::Ref<const Eigen::VectorXd> &,
                         const Eigen::Ref<const Eigen::VectorXd> &)>
        f,
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &y,
    double eps = std::pow(std::numeric_limits<double>::epsilon(), 0.5)) {
  int n = x.size();
  int m = y.size();
  Eigen::MatrixXd H(n, m);
  Eigen::VectorXd e_i(n);
  Eigen::VectorXd e_j(m);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; j++) {
      e_i.setZero();
      e_j.setZero();
      e_i[i] = eps;
      e_j[j] = eps;

      double f_ij = f(x + e_i, y + e_j) - f(x + e_i, y - e_j) -
                    f(x - e_i, y + e_j) + f(x - e_i, y - e_j);
      H(i, j) = f_ij / (4 * eps * eps);
    }
  }
  return H;
}

/**
 * @brief
 *
 * @param f
 * @param x
 * @param eps
 * @return Eigen::MatrixXd
 */
static Eigen::MatrixXd ApproxHessian(
    std::function<double(const Eigen::Ref<const Eigen::VectorXd> &)> f,
    const Eigen::Ref<const Eigen::VectorXd> &x,
    double eps = std::pow(std::numeric_limits<double>::epsilon(), 0.5)) {
  auto j = Derivative(f, x, eps);
  return j * j.transpose();
}

} // namespace ocp
