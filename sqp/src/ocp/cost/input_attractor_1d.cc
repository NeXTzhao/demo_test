#include "ocp/cost/input_attractor_1d.h"

namespace ocp {

InputAttractor1D::InputAttractor1D(const int index, const double x,
                                   const double weight)
    : index_(index), x_(x), weight_(weight) {
  CHECK_GE(index_, 0);
}

InputAttractor1D *InputAttractor1D::clone() const {
  return new InputAttractor1D(*this);
}

double InputAttractor1D::GetValue(
    const Eigen::Ref<const Eigen::VectorXd> &state,
    const Eigen::Ref<const Eigen::VectorXd> &input) const {
  (void)state;

  if (input.size() == 0) {
    return 0.0;
  }

  CHECK_LT(index_, input.size());
  const double dx = input(index_) - x_;
  return 0.5 * weight_ * dx * dx;
}

void InputAttractor1D::GetQuadraticApproximation(
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
  Lu(index_) = weight_ * dx;
  Luu(index_, index_) = weight_;

  return;
}

} // namespace ocp
