#include "ocp/cost/state_attractor_1d.h"

namespace ocp {

StateAttractor1D::StateAttractor1D(const int index, const double x,
                                   const double weight)
    : index_(index), x_(x), weight_(weight) {
  CHECK_GE(index_, 0);
}

StateAttractor1D *StateAttractor1D::clone() const {
  return new StateAttractor1D(*this);
}

double StateAttractor1D::GetValue(
    const Eigen::Ref<const Eigen::VectorXd> &state,
    const Eigen::Ref<const Eigen::VectorXd> &input) const {
  (void)input;
  CHECK_LT(index_, state.size());
  const double dx = state(index_) - x_;
  return 0.5 * weight_ * dx * dx;
}

void StateAttractor1D::GetQuadraticApproximation(
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
  Lx(index_) = weight_ * dx;
  Lxx(index_, index_) = weight_;

  return;
}

} // namespace ocp
