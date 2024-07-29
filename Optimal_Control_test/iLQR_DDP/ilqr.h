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
struct WeightParams {
  Eigen::Matrix<double, StateDim, StateDim> Q;
  Eigen::Matrix<double, ControlDim, ControlDim> R;
  Eigen::Matrix<double, StateDim, StateDim> Qf;
};

struct LineSearchParams {
  double alpha_min;
  double alpha_max;
  double alpha_decay;
  double c1;
};

struct SimulateParams {
  int horizon;
  double dt;
};

struct ILQRParams {
  double max_iter;
  double tolerance;
};

template<int StateDim, int ControlDim>
class iLQR {
 public:
  using State = Eigen::Matrix<double, StateDim, 1>;
  using Control = Eigen::Matrix<double, ControlDim, 1>;
  using StateSeq = std::vector<State>;
  using ControlSeq = std::vector<Control>;

  explicit iLQR(Vehicle<StateDim, ControlDim>& vehicle,
                const WeightParams<StateDim, ControlDim>& weightParams,
                const LineSearchParams& line_search_params, const ILQRParams& ilqrParams, const SimulateParams& simulateParams)
      : vehicle_(vehicle),
        weight_params_(weightParams),
        line_search_params_(line_search_params),
        ilqrParams_(ilqrParams),
        simulateParams_(simulateParams),
        opt_state_seq_(simulateParams.horizon + 1),
        opt_control_seq_(simulateParams.horizon) {
    dt_ = simulateParams_.dt;
  }


  bool solve(const State& x0, const StateSeq& x_ref_seq, const ControlSeq& u_ref_seq) {
    if (x_ref_seq.size() != u_ref_seq.size() + 1) {
      throw std::invalid_argument("The size of x_ref_seq should be one more than the size of u_ref_seq.");
    }
    forwardPass(x0, opt_state_seq_, opt_control_seq_);
    double prev_cost = computeTotalCost(opt_state_seq_, opt_control_seq_, x_ref_seq);

    for (int iter = 0; iter < ilqrParams_.max_iter; ++iter) {
      forwardPass(x0, opt_state_seq_, opt_control_seq_);
      backwardPass(x_ref_seq);
      double alpha = lineSearch(x0, x_ref_seq, u_ref_seq);
      std::cout << "alpha: " << alpha << std::endl;
      updateControls(opt_control_seq_, alpha);
      forwardPass(x0, opt_state_seq_, opt_control_seq_);
      double current_cost = computeTotalCost(opt_state_seq_, opt_control_seq_, x_ref_seq);

      // Check for convergence
      if (std::abs(prev_cost - current_cost) < ilqrParams_.tolerance) {
        std::cout << "Converged after " << iter + 1 << " iterations" << std::endl;
        return true;
      }
      prev_cost = current_cost;
    }
    return false;
  }

  const StateSeq& getOptStateSeq() const {
    return opt_state_seq_;
  }
  const ControlSeq& getOptControlSeq() const {
    return opt_control_seq_;
  }

 private:
  void forwardPass(const State& x0, StateSeq& x_seq, ControlSeq& u_seq) {
    int N = u_seq.size();
    x_seq[0] = x0;
    for (int k = 0; k < N; ++k) {
      x_seq[k + 1] = vehicle_.updateState(x_seq[k], u_seq[k], dt_);
    }
  }

  void backwardPass(const StateSeq& x_ref_seq) {
    int N = opt_control_seq_.size();
    std::vector<Eigen::MatrixXd> K(N);
    std::vector<Control> d(N);

    // 终点的代价梯度和Hessian矩阵
    Eigen::VectorXd s = weight_params_.Qf * (opt_state_seq_.back() - x_ref_seq.back());
    Eigen::MatrixXd S = weight_params_.Qf;

    for (int k = N - 1; k >= 0; --k) {
      Eigen::MatrixXd A(StateDim, StateDim), B(StateDim, ControlDim);
      vehicle_.linearize(opt_state_seq_[k], opt_control_seq_[k], dt_, A, B);

      // 计算 lx, lu, lxx, luu, lux
      Eigen::VectorXd lx = weight_params_.Q * opt_state_seq_[k];
      Eigen::VectorXd lu = weight_params_.R * opt_control_seq_[k];
      Eigen::MatrixXd lxx = weight_params_.Q;
      Eigen::MatrixXd luu = weight_params_.R;
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

  double lineSearch(const State& x0, const StateSeq& x_ref_seq, const ControlSeq& u_ref_seq) {
    double alpha = line_search_params_.alpha_max;
    double initial_cost = computeTotalCost(opt_state_seq_, opt_control_seq_, x_ref_seq);

    while (alpha >= line_search_params_.alpha_min) {
      ControlSeq new_u_seq = opt_control_seq_;
      updateControls(new_u_seq, alpha);
      StateSeq new_x_seq(opt_state_seq_.size());
      forwardPass(x0, new_x_seq, new_u_seq);
      double new_cost = computeTotalCost(new_x_seq, new_u_seq, x_ref_seq);

      if (new_cost < initial_cost - line_search_params_.c1 * alpha * computeNorm<2, StateDim, ControlDim>(d_)) {
        return alpha;
      }
      alpha *= line_search_params_.alpha_decay;
    }
    return line_search_params_.alpha_min;
  }

  double computeTotalCost(const StateSeq& x_seq, const ControlSeq& u_seq,
                          const StateSeq& x_ref_seq) {
    double total_cost = 0.0;
    int N = u_seq.size();

    for (int k = 0; k < N; ++k) {
      total_cost += stage_cost(x_seq[k], u_seq[k], x_ref_seq[k]);
    }
    total_cost += terminalCost(x_seq[N], x_ref_seq[N]);
    return total_cost;
  }

  double stage_cost(const State& x, const Control& u, const State& x_ref) const {
    // Example cost function: quadratic cost relative to reference
    double state_cost = 0.5 * (x - x_ref).transpose() * weight_params_.Q * (x - x_ref);
    double control_cost = 0.5 * u.transpose() * weight_params_.R * u;
    return state_cost + control_cost;
  }

  double terminalCost(const State& x, const State& x_ref) const {
    // Example terminal cost: quadratic cost relative to reference
    return 0.5 * (x - x_ref).transpose() * weight_params_.Qf * (x - x_ref);
  }

  void updateControls(ControlSeq& u_seq, double alpha) {
    int N = u_seq.size();
    for (int k = 0; k < N; ++k) {
      u_seq[k] = u_seq[k] + alpha * d_[k];
    }
  }

 private:
  Vehicle<StateDim, ControlDim>& vehicle_;
  double dt_;
  WeightParams<StateDim, ControlDim> weight_params_;
  LineSearchParams line_search_params_;
  ILQRParams ilqrParams_;
  SimulateParams simulateParams_;
  std::vector<Eigen::MatrixXd> K_;
  std::vector<Control> d_;
  StateSeq opt_state_seq_;
  ControlSeq opt_control_seq_;
};
