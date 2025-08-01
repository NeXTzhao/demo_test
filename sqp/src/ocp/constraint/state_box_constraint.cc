#include "ocp/constraint/state_box_constraint.h"

namespace ocp {

StateBoxConstraint::StateBoxConstraint(Eigen::VectorXd lb, Eigen::VectorXd ub)
    : ConstraintAbstract(std::move(lb), std::move(ub)) {}

StateBoxConstraint *StateBoxConstraint::clone() const {
  return new StateBoxConstraint(*this);
}

size_t StateBoxConstraint::GetNumConstraints() const {
  return lower_bounds_.size();
}

Eigen::VectorXd
StateBoxConstraint::GetValue(const Eigen::Ref<const Eigen::VectorXd> &x,
                             const Eigen::Ref<const Eigen::VectorXd> &u) {
  (void)u;
  CHECK_EQ(x.size(), lower_bounds_.size());
  return x;
}

void StateBoxConstraint::GetLinearApproximation(
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &u, Eigen::Ref<Eigen::MatrixXd> Gx,
    Eigen::Ref<Eigen::MatrixXd> Gu) const {
  (void)u;
  (void)Gu;
  Gx = Eigen::MatrixXd::Identity(lower_bounds_.size(), x.size());
  return;
}

} // namespace ocp
