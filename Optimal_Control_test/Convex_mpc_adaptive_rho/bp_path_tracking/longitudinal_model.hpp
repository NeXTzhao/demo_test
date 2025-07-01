//
// Created by next on 2025/6/25.
//
#include <Eigen/Dense>

#ifndef LONGITUDINAL_MODEL_HPP
#define LONGITUDINAL_MODEL_HPP
template <int StateDim, int ControlDim>
class LongitudinalModel {
public:
  enum LongitudinalStateIndex {
    X_POS = 0,
    SPEED = 1,
    ACCEL = 2,
    ODOM = 3,
    LONG_X_DIM = 4
  };
  enum LongitudinalControlIndex {
    JERK = 0,
    U_LONG_DIM = 1
  };

 public:
  using State = Eigen::Matrix<double, StateDim, 1>;
  using Control = Eigen::Matrix<double, ControlDim, 1>;
  using DifferenceMatrix =
      std::pair<Eigen::Matrix<double, StateDim, StateDim>,
                Eigen::Matrix<double, StateDim, ControlDim>>;

  State Diff(const State& x, const Control& u) const {
    State dx;
    dx << x(SPEED), x(ACCEL), u(JERK), x(SPEED);
    return dx;
  }

  DifferenceMatrix linearize(const State& x, double h) const {
    Eigen::Matrix<double, StateDim, StateDim> f_x =
        Eigen::Matrix<double, StateDim, StateDim>::Identity();
    Eigen::Matrix<double, StateDim, ControlDim> f_u =
        Eigen::Matrix<double, StateDim, ControlDim>::Zero();

    f_x(X_POS, SPEED) = h;
    f_x(SPEED, ACCEL) = h;
    f_x(ODOM, SPEED) = h;

    f_u(ACCEL, JERK) = h;

    return {f_x, f_u};
  }

  State EvalOneStep(const State& x, const Control& u, double increment, int USE_RK = 2) const {
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
};

#endif  // LONGITUDINAL_MODEL_HPP
