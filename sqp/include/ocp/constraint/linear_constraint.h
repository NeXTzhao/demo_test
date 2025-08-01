#pragma once

#include "ocp/constraint/constraint_base.h"

namespace ocp {

/**
 * Linear constraint as the following: \f$ lb <= C * x + D * u <= ub \f$
 */
class LinearConstraint final : public ConstraintAbstract {
public:
  /**
   * Constructor
   *
   * @param [in] lb: lower bound in lb <= C * x + D * u <= ub
   * @param [in] ub: upper bound  lb <= C * x + D * u <= ub
   * @param [in] C: x factor in lb <= C * x + D * u <= ub
   * @param [in] D: u factor in lb <= C * x + D * u <= ub
   */
  LinearConstraint(Eigen::VectorXd lb, Eigen::VectorXd ub, Eigen::MatrixXd C,
                   Eigen::MatrixXd D);

  ~LinearConstraint() override = default;

  LinearConstraint *clone() const override;

  size_t GetNumConstraints() const override;

  Eigen::VectorXd GetValue(const Eigen::Ref<const Eigen::VectorXd> &x,
                           const Eigen::Ref<const Eigen::VectorXd> &u) override;

  void GetLinearApproximation(const Eigen::Ref<const Eigen::VectorXd> &x,
                              const Eigen::Ref<const Eigen::VectorXd> &u,
                              Eigen::Ref<Eigen::MatrixXd> Gx,
                              Eigen::Ref<Eigen::MatrixXd> Gu) const override;

private:
  Eigen::MatrixXd C_; /**< State input constraint derivative wrt. state */
  Eigen::MatrixXd D_; /**< State input constraint derivative wrt. input */
};

} // namespace ocp
