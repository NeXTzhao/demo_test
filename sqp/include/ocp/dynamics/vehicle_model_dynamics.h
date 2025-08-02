#pragma once

#include "ocp/dynamics/dynamics_base.h"

namespace ocp {
/**
 *
 * A vehicle model system with the following flow map:
 *
 * - \f$ \dot{x} = f(x, u) \f$
 *
 */
class VehicleModelDynamics final : public DynamicsAbstract {
public:
  enum StateIndex {
    X_POS = 0,
    Y_POS = 1,
    SPEED = 2,
    THETA = 3,  /* heading */
    DELTAV = 4, /* steering */
    OMEGA = 5,  /* d(steering) */
    ODOM = 6,
    ACCEL = 7,
    X_DIM = 8
  };

  enum ControlIndex {
    JERK = 0,   /* d(a) */
    ALPHAV = 1, /* dd(steering) */
    U_DIM = 2
  };

  static constexpr int StateDim = static_cast<int>(X_DIM);
  static constexpr int ControlDim = static_cast<int>(U_DIM);

  using State = Eigen::Matrix<double, StateDim, 1>;
  using Control = Eigen::Matrix<double, ControlDim, 1>;

  using StateSequence = std::vector<State>;
  using ControlSequence = std::vector<Control>;

public:
  explicit VehicleModelDynamics(double dt);

  ~VehicleModelDynamics() override = default;

  VehicleModelDynamics *clone() const override;

  static Eigen::VectorXd Dynamics(const Eigen::Ref<const Eigen::VectorXd> &x,
                                  const Eigen::Ref<const Eigen::VectorXd> &u);

  Eigen::VectorXd
  ComputeFlowMap(const Eigen::Ref<const Eigen::VectorXd> &x,
                 const Eigen::Ref<const Eigen::VectorXd> &u) override;

protected:
  VehicleModelDynamics(const VehicleModelDynamics &other) = default;

private:
  double dt_ = 0.0;
};
} // namespace ocp
