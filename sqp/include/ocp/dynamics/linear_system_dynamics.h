#pragma once

#include "ocp/dynamics/dynamics_base.h"

namespace ocp {

/**
 *
 * A linear time invariant system with the following flow map:
 *
 * - \f$ \dot{x} = A * x + B * u \f$
 *
 */
class LinearSystemDynamics : public DynamicsAbstract {
public:
  LinearSystemDynamics(Eigen::MatrixXd A, Eigen::MatrixXd B);

  ~LinearSystemDynamics() override = default;

  LinearSystemDynamics *clone() const override;

  Eigen::VectorXd
  ComputeFlowMap(const Eigen::Ref<const Eigen::VectorXd> &x,
                 const Eigen::Ref<const Eigen::VectorXd> &u) override;

  void GetLinearApproximation(const Eigen::Ref<const Eigen::VectorXd> &x,
                              const Eigen::Ref<const Eigen::VectorXd> &u,
                              Eigen::Ref<Eigen::MatrixXd> Fx,
                              Eigen::Ref<Eigen::MatrixXd> Fu) override;

protected:
  LinearSystemDynamics(const LinearSystemDynamics &other) = default;

  Eigen::MatrixXd A_;
  Eigen::MatrixXd B_;
};

} // namespace ocp
