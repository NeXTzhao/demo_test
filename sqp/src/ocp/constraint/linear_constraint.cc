#include "ocp/constraint/linear_constraint.h"

namespace ocp {

LinearConstraint::LinearConstraint(Eigen::VectorXd lb, Eigen::VectorXd ub,
                                   Eigen::MatrixXd C, Eigen::MatrixXd D)
    : ConstraintAbstract(std::move(lb), std::move(ub)), C_(std::move(C)),
      D_(std::move(D)) {
  CHECK_EQ(C_.rows(), D_.rows());
  CHECK_EQ(lower_bounds_.size(), C_.rows());
}

LinearConstraint *LinearConstraint::clone() const {
  return new LinearConstraint(*this);
}

size_t LinearConstraint::GetNumConstraints() const { return C_.rows(); }

Eigen::VectorXd
LinearConstraint::GetValue(const Eigen::Ref<const Eigen::VectorXd> &x,
                           const Eigen::Ref<const Eigen::VectorXd> &u) {
  CHECK_EQ(x.size(), C_.cols());
  CHECK_EQ(u.size(), D_.cols());

  Eigen::VectorXd g = C_ * x;
  if (u.size() != 0) {
    g.noalias() += D_ * u;
  }
  return g;
}

void LinearConstraint::GetLinearApproximation(
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &u, Eigen::Ref<Eigen::MatrixXd> Gx,
    Eigen::Ref<Eigen::MatrixXd> Gu) const {
  Gx = C_;
  if (u.size() != 0) {
    Gu = D_;
  }
  return;
}

} // namespace ocp
