#include "ocp/cost/quadratic_cost.h"

namespace ocp {

QuadraticCost::QuadraticCost(Eigen::MatrixXd Q, Eigen::MatrixXd R,
                             Eigen::MatrixXd P /* = Eigen::MatrixXd () */)
    : QuadraticCost(Eigen::VectorXd(), Eigen::VectorXd(), std::move(Q),
                    std::move(R), std::move(P)) {}

QuadraticCost::QuadraticCost(Eigen::VectorXd state_ref,
                             Eigen::VectorXd input_ref, Eigen::MatrixXd Q,
                             Eigen::MatrixXd R,
                             Eigen::MatrixXd P /* = Eigen::MatrixXd () */)
    : state_ref_(std::move(state_ref)), input_ref_(std::move(input_ref)),
      Q_(std::move(Q)), R_(std::move(R)), P_(std::move(P)) {
  if (P_.size() > 0) {
    CHECK_EQ(P_.rows(), R_.rows());
    CHECK_EQ(P_.cols(), Q_.rows());
  }

  if (state_ref_.size() == 0) {
    state_ref_ = Eigen::VectorXd::Zero(Q_.rows());
  }
  if (input_ref_.size() == 0) {
    input_ref_ = Eigen::VectorXd::Zero(R_.rows());
  }
  CHECK_EQ(state_ref_.size(), Q_.rows());
  CHECK_EQ(input_ref_.size(), R_.rows());
}

QuadraticCost *QuadraticCost::clone() const { return new QuadraticCost(*this); }

double
QuadraticCost::GetValue(const Eigen::Ref<const Eigen::VectorXd> &state,
                        const Eigen::Ref<const Eigen::VectorXd> &input) const {
  CHECK_EQ(state.size(), state_ref_.size());
  Eigen::VectorXd x_deviation = state - state_ref_;
  if (input.size() == 0) {
    return 0.5 * x_deviation.dot(Q_ * x_deviation);
  }

  CHECK_EQ(input.size(), input_ref_.size());
  Eigen::VectorXd input_deviation = input - input_ref_;
  if (P_.size() == 0) {
    return 0.5 * x_deviation.dot(Q_ * x_deviation) +
           0.5 * input_deviation.dot(R_ * input_deviation);
  }
  return 0.5 * x_deviation.dot(Q_ * x_deviation) +
         0.5 * input_deviation.dot(R_ * input_deviation) +
         input_deviation.dot(P_ * x_deviation);
}

void QuadraticCost::GetQuadraticApproximation(
    const Eigen::Ref<const Eigen::VectorXd> &state,
    const Eigen::Ref<const Eigen::VectorXd> &input,
    Eigen::Ref<Eigen::VectorXd> Lx, Eigen::Ref<Eigen::VectorXd> Lu,
    Eigen::Ref<Eigen::MatrixXd> Lxx, Eigen::Ref<Eigen::MatrixXd> Luu,
    Eigen::Ref<Eigen::MatrixXd> Lxu) const {
  CHECK_EQ(state.size(), state_ref_.size());
  Eigen::VectorXd state_deviation = state - state_ref_;
  Lxx = Q_;
  Lx.noalias() = Q_ * state_deviation;
  if (input.size() == 0) {
    return;
  }

  CHECK_EQ(input.size(), input_ref_.size());
  Eigen::VectorXd input_deviation = input - input_ref_;
  Luu = R_;
  Lu.noalias() = R_ * input_deviation;
  if (P_.size() == 0) {
    Lxu = Eigen::MatrixXd::Zero(state.size(), input.size());
    return;
  }
  Lu.noalias() += P_ * state_deviation;
  Lx.noalias() += P_.transpose() * input_deviation;
  Lxu = P_.transpose();

  return;
}

} // namespace ocp
