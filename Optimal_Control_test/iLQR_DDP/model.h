//
// Created by next on 24-7-27.
//

#pragma once

template<int StateDim, int ControlDim>
class Vehicle {
 public:
  using State = Eigen::Matrix<double, StateDim, 1>;
  using Control = Eigen::Matrix<double, ControlDim, 1>;

  enum StateIndex { X = 0,
                    Y = 1,
                    THETA = 2,
                    V = 3 };
  enum ControlIndex { A = 0,
                      DELTA = 1 };

  explicit Vehicle(double wheelbase, double _max_speed, double _min_speed,
                   double _max_acceleration, double _min_acceleration,
                   double _max_steering_angle, double _min_steering_angle)
      : L(wheelbase),
        max_speed(_max_speed),
        min_speed(_min_speed),
        max_a(_max_acceleration),
        min_a(_min_acceleration),
        max_delta(_max_steering_angle),
        min_delta(_min_steering_angle) {}

  State dynamics(const State& state, const Control& control) const {
    State dState;
    const double v = state(V);
    const double theta = state(THETA);
    const double delta = control(DELTA);
    dState(X) = v * cos(theta);        // dx
    dState(Y) = v * sin(theta);        // dy
    dState(THETA) = v / L * tan(delta);// dtheta
    dState(V) = control(A);            // dv
    return dState;
  }

  State updateState(const State& state, Control& control, double dt) const {
    // Apply control limits
    control = applyControlLimits(control);

    State k1 = dynamics(state, control);
    State k2 = dynamics(state + 0.5 * dt * k1, control);
    State k3 = dynamics(state + 0.5 * dt * k2, control);
    State k4 = dynamics(state + dt * k3, control);

    State newState = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
    return newState;
  }

  void linearize(const State& state, const Control& control, double dt,
                 Eigen::MatrixXd& A, Eigen::MatrixXd& B) const {
    A = Eigen::MatrixXd::Identity(StateDim, StateDim);
    B = Eigen::MatrixXd::Zero(StateDim, ControlDim);

    double v = state(V);
    double theta = state(THETA);
    double delta = control(DELTA);
    double sin_theta = sin(theta);
    double cos_theta = cos(theta);
    double cos_delta = cos(delta);
    double tan_delta = tan(delta);

    // clang-format off
    A << 1, 0, -v * sin_theta * dt, cos_theta * dt,
        0, 1, v * cos_theta * dt, sin_theta * dt,
        0, 0, 1, tan_delta * dt / L,
        0, 0, 0, 1;

    B << 0, 0,
        0, 0,
        0, v * dt / (L * cos_delta * cos_delta),
        dt, 0;
    // clang-format on
  }

  Control applyControlLimits(const Control& control) const {
    Control limited_control = control;
    limited_control(A) = std::max(min_a, std::min(max_a, control(A)));
    limited_control(DELTA) =
        std::max(min_delta, std::min(max_delta, control(DELTA)));
    return limited_control;
  }

 private:
  double L;// 车辆轴距

 public:
  double max_speed, min_speed, max_a, min_a, max_delta,
      min_delta;// 控制输入的限制
};
