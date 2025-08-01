#pragma once

#include "ocp/collection/collection.h"
#include "ocp/cost/cost_base.h"

namespace ocp {

/**
 * State Input Cost function combining a collection of cost terms.
 *
 * This class collects a variable number of cost terms and provides methods to
 * get the summed cost values and quadratic approximations. Each cost term can
 * be accessed through its string name and can be activated or deactivated.
 */
class CostCollection : public Collection<CostAbstract> {
public:
  CostCollection() = default;
  ~CostCollection() override = default;
  CostCollection *clone() const override;

  /** Get state-input cost value */
  virtual double GetValue(const Eigen::Ref<const Eigen::VectorXd> &state,
                          const Eigen::Ref<const Eigen::VectorXd> &input);

  /** Get state-input cost quadratic approximation */
  virtual void GetQuadraticApproximation(
      const Eigen::Ref<const Eigen::VectorXd> &state,
      const Eigen::Ref<const Eigen::VectorXd> &input,
      Eigen::Ref<Eigen::VectorXd> Lx, Eigen::Ref<Eigen::VectorXd> Lu,
      Eigen::Ref<Eigen::MatrixXd> Lxx, Eigen::Ref<Eigen::MatrixXd> Luu,
      Eigen::Ref<Eigen::MatrixXd> Lxu) const;

protected:
  /** Copy constructor */
  CostCollection(const CostCollection &other);
};

} // namespace ocp
