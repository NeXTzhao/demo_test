#pragma once

#include "ocp/constraint/constraint_base.h"

namespace ocp {

/**
 * State box constraint as the following: \f$ lb <= x <= ub \f$
 */
class StateBoxConstraint final : public ConstraintAbstract {
public:
  /**
   * Constructor
   *
   * @param [in] lb: lower bound in lb <= x <= ub
   * @param [in] ub: upper bound in lb <= x <= ub
   */
  StateBoxConstraint(Eigen::VectorXd lb, Eigen::VectorXd ub);

  ~StateBoxConstraint() override = default;

  StateBoxConstraint *clone() const override;

  size_t GetNumConstraints() const override;

  Eigen::VectorXd GetValue(const Eigen::Ref<const Eigen::VectorXd> &x,
                           const Eigen::Ref<const Eigen::VectorXd> &u) override;

  void GetLinearApproximation(const Eigen::Ref<const Eigen::VectorXd> &x,
                              const Eigen::Ref<const Eigen::VectorXd> &u,
                              Eigen::Ref<Eigen::MatrixXd> Gx,
                              Eigen::Ref<Eigen::MatrixXd> Gu) const override;
};

} // namespace ocp
