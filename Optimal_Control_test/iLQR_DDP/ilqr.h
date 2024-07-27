//
// Created by next on 24-7-27.
//

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#include "common/log.h"
#include "martix_math.h"
#include "model.h"

template<int StateDim, int ControlDim>
struct CostParams {
  Eigen::Matrix<double, StateDim, StateDim> Q;
  Eigen::Matrix<double, ControlDim, ControlDim> R;
  Eigen::Matrix<double, StateDim, StateDim> Qf;
};

struct LineSearchParams {
  double alpha_min = 1e-4;
  double alpha_max = 1.0;
  double alpha_decay = 0.5;
  double c1 = 1e-4;
};

template<int StateDim, int ControlDim>
class iLQR {
 public:
  using State = Eigen::Matrix<double, StateDim, 1>;
  using Control = Eigen::Matrix<double, ControlDim, 1>;
  using StateSeq = std::vector<State>;
  using ControlSeq = std::vector<Control>;

  iLQR(Vehicle<StateDim, ControlDim>& vehicle, int max_iter = 100)
      : vehicle_(vehicle), max_iter_(max_iter) {}

  void solve(const State& x0, StateSeq& x_seq, ControlSeq& u_seq, double dt,
             const CostParams<StateDim, ControlDim>& cost_params,
             const LineSearchParams& line_search_params) {
    cost_params_ = cost_params;
    line_search_params_ = line_search_params;

    double prev_cost = computeTotalCost(x_seq, u_seq, dt);

    for (int iter = 0; iter < max_iter_; ++iter) {
      forwardPass(x0, x_seq, u_seq, dt);
      backwardPass(x_seq, u_seq, dt);
      double alpha = lineSearch(x0, x_seq, u_seq, dt);
      updateControls(u_seq, alpha);

      double current_cost = computeTotalCost(x_seq, u_seq, dt);

      // Check for convergence
      if (std::abs(prev_cost - current_cost) < 1e-6) {
        std::cout << "Converged after " << iter + 1 << " iterations" << std::endl;
        break;
      }
      prev_cost = current_cost;
    }
  }

 private:
  void forwardPass(const State& x0, StateSeq& x_seq, ControlSeq& u_seq, double dt) {
    int N = u_seq.size();
    x_seq[0] = x0;
    for (int k = 0; k < N; ++k) {
      x_seq[k + 1] = vehicle_.updateState(x_seq[k], u_seq[k], dt);
    }
  }

  void backwardPass(const StateSeq& x_seq, const ControlSeq& u_seq, double dt) {
    int N = u_seq.size();
    std::vector<Eigen::MatrixXd> K(N);
    std::vector<Control> d(N);

    // 终点的代价梯度和Hessian矩阵
    Eigen::VectorXd s = cost_params_.Qf * (x_seq[N] - State::Zero());
    Eigen::MatrixXd S = cost_params_.Qf;

    for (int k = N - 1; k >= 0; --k) {
      Eigen::MatrixXd A(StateDim, StateDim), B(StateDim, ControlDim);
      vehicle_.linearize(x_seq[k], u_seq[k], dt, A, B);

      Eigen::VectorXd lx = cost_params_.Q * x_seq[k];
      Eigen::VectorXd lu = cost_params_.R * u_seq[k];
      Eigen::MatrixXd lxx = cost_params_.Q;
      Eigen::MatrixXd luu = cost_params_.R;
      Eigen::MatrixXd lux = Eigen::MatrixXd::Zero(ControlDim, StateDim);

      // 计算 Qx, Qu, Qxx, Quu, Qux
      Eigen::VectorXd Qx = lx + A.transpose() * s;
      Eigen::VectorXd Qu = lu + B.transpose() * s;
      Eigen::MatrixXd Qxx = lxx + A.transpose() * S * A;
      Eigen::MatrixXd Quu = luu + B.transpose() * S * B;
      Eigen::MatrixXd Qux = lux + B.transpose() * S * A;

      // 计算反馈矩阵 K 和前馈向量 d
      Eigen::MatrixXd Quu_inv = Quu.inverse();
      d[k] = -Quu_inv * Qu;
      K[k] = -Quu_inv * Qux;

      // 更新 s 和 S
      s = Qx + K[k].transpose() * Quu * d[k] + K[k].transpose() * Qu + Qux.transpose() * d[k];
      S = Qxx + K[k].transpose() * Quu * K[k] + K[k].transpose() * Qux + Qux.transpose() * K[k];
    }

    K_ = K;
    d_ = d;
  }


  double lineSearch(const State& x0, StateSeq& x_seq, ControlSeq& u_seq, double dt) {
    double alpha = line_search_params_.alpha_max;
    double initial_cost = computeTotalCost(x_seq, u_seq, dt);

    while (alpha >= line_search_params_.alpha_min) {
      ControlSeq new_u_seq = u_seq;
      updateControls(new_u_seq, alpha);
      StateSeq new_x_seq(x_seq.size());
      forwardPass(x0, new_x_seq, new_u_seq, dt);
      double new_cost = computeTotalCost(new_x_seq, new_u_seq, dt);

      if (new_cost < initial_cost - line_search_params_.c1 * alpha * computeNorm<2, StateDim, ControlDim>(d_)) {
        return alpha;
      }
      alpha *= line_search_params_.alpha_decay;
    }
    return line_search_params_.alpha_min;
  }

  double computeTotalCost(const StateSeq& x_seq, const ControlSeq& u_seq, double dt) {
    double total_cost = 0.0;
    int N = u_seq.size();

    for (int k = 0; k < N; ++k) {
      total_cost += cost(x_seq[k], u_seq[k]) * dt;
    }
    total_cost += terminalCost(x_seq[N]);
    return total_cost;
  }

  double cost(const State& x, const Control& u) const {
    // Example cost function: quadratic cost
    double state_cost = 0.5 * x.transpose() * cost_params_.Q * x;
    double control_cost = 0.5 * u.transpose() * cost_params_.R * u;
    return state_cost + control_cost;
  }

  double terminalCost(const State& x) const {
    // Example terminal cost: quadratic cost
    return 0.5 * x.transpose() * cost_params_.Qf * x;
  }

  void updateControls(ControlSeq& u_seq, double alpha) {
    int N = u_seq.size();
    for (int k = 0; k < N; ++k) {
      u_seq[k] = u_seq[k] - alpha * d_[k];
    }
  }

  Vehicle<StateDim, ControlDim>& vehicle_;
  int max_iter_;
  CostParams<StateDim, ControlDim> cost_params_;
  LineSearchParams line_search_params_;
  std::vector<Eigen::MatrixXd> K_;
  std::vector<Control> d_;
};
