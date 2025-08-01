#include "ocp/dynamics/linear_system_dynamics.h"

namespace ocp {

LinearSystemDynamics::LinearSystemDynamics(Eigen::MatrixXd A, Eigen::MatrixXd B)
    : DynamicsAbstract(A.rows(), B.cols()), A_(std::move(A)), B_(std::move(B)) {
}

LinearSystemDynamics *LinearSystemDynamics::clone() const {
  return new LinearSystemDynamics(*this);
}

Eigen::VectorXd LinearSystemDynamics::ComputeFlowMap(
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &u) {
  Eigen::VectorXd f = A_ * x;
  f.noalias() += B_ * u;
  return f;
}

void LinearSystemDynamics::GetLinearApproximation(
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &u, Eigen::Ref<Eigen::MatrixXd> Fx,
    Eigen::Ref<Eigen::MatrixXd> Fu) {
  (void)x;
  (void)u;
  Fx = A_;
  Fu = B_;
}

} // namespace ocp
