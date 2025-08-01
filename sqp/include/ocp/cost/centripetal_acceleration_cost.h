#pragma once

#include "ocp/cost/cost_base.h"

namespace ocp {

/**
 * Centripetal acceleration cost function defined as the following: \f$ f =
 * 0.*w*5(v^2*κ) \f$
 */
class CentripetalAccelerationCost : public CostAbstract {
public:
  /**
   * Constructor
   *
   * @param [in] weight: weight
   */
  explicit CentripetalAccelerationCost(double weight);

  ~CentripetalAccelerationCost() override = default;
  CentripetalAccelerationCost *clone() const override;

  /** Get cost term value */
  double GetValue(const Eigen::Ref<const Eigen::VectorXd> &state,
                  const Eigen::Ref<const Eigen::VectorXd> &input) const final;

  /** Get cost term quadratic approximation */
  void GetQuadraticApproximation(const Eigen::Ref<const Eigen::VectorXd> &state,
                                 const Eigen::Ref<const Eigen::VectorXd> &input,
                                 Eigen::Ref<Eigen::VectorXd> Lx,
                                 Eigen::Ref<Eigen::VectorXd> Lu,
                                 Eigen::Ref<Eigen::MatrixXd> Lxx,
                                 Eigen::Ref<Eigen::MatrixXd> Luu,
                                 Eigen::Ref<Eigen::MatrixXd> Lxu) const final;

protected:
  CentripetalAccelerationCost(const CentripetalAccelerationCost &rhs) = default;

private:
  double weight_ = 0.0;
};

} // namespace ocp
