#include "ocp/cost/state_repeller_1d.h"

namespace ocp {

StateRepeller1D::StateRepeller1D(const int index, const double x,
                                 const double sign, const double weight)
    : index_(index), x_(x), weight_(weight) {
  CHECK_GE(index_, 0);
  sign_ = (sign < 0.0) ? -1.0 : 1.0;
}

StateRepeller1D *StateRepeller1D::clone() const {
  return new StateRepeller1D(*this);
}

double StateRepeller1D::GetValue(
    const Eigen::Ref<const Eigen::VectorXd> &state,
    const Eigen::Ref<const Eigen::VectorXd> &input) const {
  (void)input;
  CHECK_LT(index_, state.size());

  const double dx = std::max(sign_ * (state(index_) - x_), 0.0);
  return 0.5 * weight_ * dx * dx;
}

void StateRepeller1D::GetQuadraticApproximation(
    const Eigen::Ref<const Eigen::VectorXd> &state,
    const Eigen::Ref<const Eigen::VectorXd> &input,
    Eigen::Ref<Eigen::VectorXd> Lx, Eigen::Ref<Eigen::VectorXd> Lu,
    Eigen::Ref<Eigen::MatrixXd> Lxx, Eigen::Ref<Eigen::MatrixXd> Luu,
    Eigen::Ref<Eigen::MatrixXd> Lxu) const {
  (void)input;
  (void)Lu;
  (void)Luu;
  (void)Lxu;

  const double dx = state(index_) - x_;
  if (sign_ * dx > 0.0) {
    Lx(index_) = weight_ * dx;
    Lxx(index_, index_) = weight_;
  }

  return;
}

} // namespace ocp
