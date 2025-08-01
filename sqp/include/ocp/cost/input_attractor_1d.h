#pragma once

#pragma once

#include "ocp/cost/cost_base.h"

namespace ocp {

/**
 * Input atttractor 1d cost function defined as the the following: \f$ f =
 * 0.5*w*(x - x_)^2 \f$
 */
class InputAttractor1D : public CostAbstract {
public:
  /**
   * Constructor
   *
   * @param [in] index: index of x in state
   * @param [in] x: x_
   * @param [in] weight: weight
   */
  InputAttractor1D(const int index, const double x, const double weight);

  ~InputAttractor1D() override = default;
  InputAttractor1D *clone() const override;

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
  InputAttractor1D(const InputAttractor1D &rhs) = default;

private:
  int index_ = 0;
  double x_ = 0.0;
  double weight_ = 0.0;
};

} // namespace ocp
