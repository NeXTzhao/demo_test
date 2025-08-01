#pragma once

#include "ocp/constraint/constraint_base.h"

namespace ocp {

/**
 * Input box constraint as the following: \f$ lb <= u <= ub \f$
 */
class InputBoxConstraint final : public ConstraintAbstract {
public:
  /**
   * Constructor
   *
   * @param [in] lb: lower bound in lb <= u <= ub
   * @param [in] ub: upper bound in lb <= u <= ub
   */
  InputBoxConstraint(Eigen::VectorXd lb, Eigen::VectorXd ub);

  ~InputBoxConstraint() override = default;

  InputBoxConstraint *clone() const override;

  size_t GetNumConstraints() const override;

  Eigen::VectorXd GetValue(const Eigen::Ref<const Eigen::VectorXd> &x,
                           const Eigen::Ref<const Eigen::VectorXd> &u) override;

  void GetLinearApproximation(const Eigen::Ref<const Eigen::VectorXd> &x,
                              const Eigen::Ref<const Eigen::VectorXd> &u,
                              Eigen::Ref<Eigen::MatrixXd> Gx,
                              Eigen::Ref<Eigen::MatrixXd> Gu) const override;
};

} // namespace ocp
