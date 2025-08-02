#include "ocp/dynamics/vehicle_model_dynamics.h"
#include "ocp/integrator/dynamics_integrate.h"

namespace ocp {

VehicleModelDynamics::VehicleModelDynamics(const double dt)
    : DynamicsAbstract(StateIndex::X_DIM, ControlIndex::U_DIM), dt_(dt) {}

VehicleModelDynamics *VehicleModelDynamics::clone() const {
  return new VehicleModelDynamics(*this);
}

Eigen::VectorXd
VehicleModelDynamics::Dynamics(const Eigen::Ref<const Eigen::VectorXd> &x,
                               const Eigen::Ref<const Eigen::VectorXd> &u) {
  assert(x.size() == StateDim);
  assert(u.size() == ControlDim);

  // x := [x, y, v, θ, κ, d(κ), odom, acc]^T
  // u := [jerk, dd(κ)]^T
  Eigen::VectorXd x_dot(StateDim);
  x_dot << x(StateIndex::SPEED) * std::cos(x(StateIndex::THETA)), // NOLINT
      x(StateIndex::SPEED) * std::sin(x(StateIndex::THETA)),      // NOLINT
      x(StateIndex::ACCEL),                                       // NOLINT
      x(StateIndex::DELTAV) * x(StateIndex::SPEED),               // NOLINT
      x(StateIndex::OMEGA),                                       // NOLINT
      u(ControlIndex::ALPHAV),                                    // NOLINT
      x(StateIndex::SPEED),                                       // NOLINT
      u(ControlIndex::JERK);                                      // NOLINT

  return x_dot;
}

Eigen::VectorXd VehicleModelDynamics::ComputeFlowMap(
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &u) {
  assert(x.size() == StateDim);
  assert(u.size() == ControlDim);

  std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd> &,
                                const Eigen::Ref<const Eigen::VectorXd> &)>
      dynf = [this](const Eigen::Ref<const Eigen::VectorXd> &x_input,
                    const Eigen::Ref<const Eigen::VectorXd> &u_input) {
        return ocp::VehicleModelDynamics::Dynamics(x_input, u_input);
      };

  Eigen::VectorXd x_next = integrate_rk4(dt_, x, u, dynf);
  return x_next;
}

} // namespace ocp
