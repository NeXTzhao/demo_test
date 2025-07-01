//
// Created by next on 2025/6/25.
//
#pragma once

#include <Eigen/Dense>

template <int StateDim, int ControlDim>
class LateralModel {
public:
  enum LateralStateIndex {
    Y_POS = 0,
    THETA = 1,
    DELTAV = 2,
    OMEGA = 3,
    LATERAL_X_DIM = 4
  };

  enum LateralControlIndex { ALPHAV = 0, U_LATERAL_DIM = 1 };

 public:
  using State = Eigen::Matrix<double, StateDim, 1>;
  using Control = Eigen::Matrix<double, ControlDim, 1>;
  using DifferenceMatrix =
      std::pair<Eigen::Matrix<double, StateDim, StateDim>,
                Eigen::Matrix<double, StateDim, ControlDim>>;

  State Diff(const State& x, const Control& u) const {
    State dx;
    dx << speed * std::sin(x(THETA)), speed * std::tan(x(DELTAV)) / wheel_base_,
        x(OMEGA), u(ALPHAV);
    return dx;
  }

  DifferenceMatrix linearize(const State& x, double h) const {
    const double L = wheel_base_;
    const double delta = x(DELTAV);
    const double theta = x(THETA);
    const double sec_delta_2 = 1.0 / (std::cos(delta) * std::cos(delta));

    Eigen::Matrix<double, StateDim, StateDim> f_x =
        Eigen::Matrix<double, StateDim, StateDim>::Identity();
    Eigen::Matrix<double, StateDim, ControlDim> f_u =
        Eigen::Matrix<double, StateDim, ControlDim>::Zero();

    f_x(Y_POS, THETA) = h * speed * std::cos(theta);
    f_x(THETA, DELTAV) = h * speed * sec_delta_2 / L;
    f_x(DELTAV, OMEGA) = h;

    f_u(OMEGA, ALPHAV) = h;

    return {f_x, f_u};
  }

  State EvalOneStep(const State& x, const Control& u, double increment,
                    int USE_RK = 2) const {
    const State limited_x = x;
    const Control limited_u = u;
    State next_state;

    // RK-2 method
    if (USE_RK == 2) {
      State k1 = Diff(limited_x, limited_u);
      State k2 = Diff(limited_x + 0.5 * increment * k1, limited_u);
      next_state = limited_x + increment * k2;
    }

    // RK-4 method
    if (USE_RK == 4) {
      const State k1 = Diff(limited_x, limited_u);
      const State k2 = Diff(limited_x + 0.5 * increment * k1, limited_u);
      const State k3 = Diff(limited_x + 0.5 * increment * k2, limited_u);
      const State k4 = Diff(limited_x + increment * k3, limited_u);
      next_state = limited_x + (increment / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
    }
    return next_state;
  }

 private:
  double wheel_base_ = 2.5;
  double speed = 3.0;
};