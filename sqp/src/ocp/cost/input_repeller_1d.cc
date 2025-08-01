#include "ocp/cost/input_repeller_1d.h"

namespace ocp {

InputRepeller1D::InputRepeller1D(const int index, const double x,
                                 const double sign, const double weight)
    : index_(index), x_(x), weight_(weight) {
  CHECK_GE(index_, 0);
  sign_ = (sign < 0.0) ? -1.0 : 1.0;
}

InputRepeller1D *InputRepeller1D::clone() const {
  return new InputRepeller1D(*this);
}

double InputRepeller1D::GetValue(
    const Eigen::Ref<const Eigen::VectorXd> &state,
    const Eigen::Ref<const Eigen::VectorXd> &input) const {
  (void)state;
  if (input.size() == 0) {
    return 0.0;
  }

  CHECK_LT(index_, input.size());
  const double dx = std::max(sign_ * (input(index_) - x_), 0.0);
  return 0.5 * weight_ * dx * dx;
}

void InputRepeller1D::GetQuadraticApproximation(
    const Eigen::Ref<const Eigen::VectorXd> &state,
    const Eigen::Ref<const Eigen::VectorXd> &input,
    Eigen::Ref<Eigen::VectorXd> Lx, Eigen::Ref<Eigen::VectorXd> Lu,
    Eigen::Ref<Eigen::MatrixXd> Lxx, Eigen::Ref<Eigen::MatrixXd> Luu,
    Eigen::Ref<Eigen::MatrixXd> Lxu) const {
  (void)state;
  (void)Lx;
  (void)Lxx;
  (void)Lxu;

  if (input.size() == 0) {
    return;
  }

  const double dx = input(index_) - x_;
  if (sign_ * dx > 0.0) {
    Lu(index_) = weight_ * dx;
    Luu(index_, index_) = weight_;
  }

  return;
}

} // namespace ocp
