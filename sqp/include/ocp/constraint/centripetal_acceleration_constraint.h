#pragma once

#include "ocp/constraint/constraint_base.h"

namespace ocp {

/**
 * Centripetal acceleration constraint as the following: \f$ min_lat_acc <=
 * v^2*κ <= max_lat_acc \f$
 */
class CentripetalAccelerationConstraint final : public ConstraintAbstract {
public:
  /**
   * Constructor
   *
   * @param [in] min_lat_acc: lower bound in min_lat_acc <= v^2*κ <= max_lat_acc
   * @param [in] max_lat_acc: upper bound in min_lat_acc <= v^2*κ <= max_lat_acc
   */
  CentripetalAccelerationConstraint(const double min_lat_acc,
                                    const double max_lat_acc);

  ~CentripetalAccelerationConstraint() override = default;
  CentripetalAccelerationConstraint *clone() const override;

  size_t GetNumConstraints() const override;

  /** Get cost term value */
  Eigen::VectorXd
  GetValue(const Eigen::Ref<const Eigen::VectorXd> &state,
           const Eigen::Ref<const Eigen::VectorXd> &input) override;

  /** Get cost term quadratic approximation */
  void GetLinearApproximation(const Eigen::Ref<const Eigen::VectorXd> &state,
                              const Eigen::Ref<const Eigen::VectorXd> &input,
                              Eigen::Ref<Eigen::MatrixXd> Gx,
                              Eigen::Ref<Eigen::MatrixXd> Gu) const override;
};

} // namespace ocp
