#include "ocp/constraint/input_box_constraint.h"

namespace ocp {

InputBoxConstraint::InputBoxConstraint(Eigen::VectorXd lb, Eigen::VectorXd ub)
    : ConstraintAbstract(std::move(lb), std::move(ub)) {}

InputBoxConstraint *InputBoxConstraint::clone() const {
  return new InputBoxConstraint(*this);
}

size_t InputBoxConstraint::GetNumConstraints() const {
  return lower_bounds_.size();
}

Eigen::VectorXd
InputBoxConstraint::GetValue(const Eigen::Ref<const Eigen::VectorXd> &x,
                             const Eigen::Ref<const Eigen::VectorXd> &u) {
  (void)x;
  CHECK_EQ(u.size(), lower_bounds_.size());
  return u;
}

void InputBoxConstraint::GetLinearApproximation(
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &u, Eigen::Ref<Eigen::MatrixXd> Gx,
    Eigen::Ref<Eigen::MatrixXd> Gu) const {
  (void)x;
  (void)Gx;

  if (u.size() != 0) {
    Gu = Eigen::MatrixXd::Identity(lower_bounds_.size(), u.size());
  }
  return;
}

} // namespace ocp
