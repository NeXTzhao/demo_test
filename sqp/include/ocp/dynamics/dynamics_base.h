#pragma once

#include "ocp/utils/derivative.h"

namespace ocp {

/**
 * The system dynamics and linearization class.
 * The linearized system flow map is defined as: \n
 * \f$ dx/dt = A(t) \delta x + B(t) \delta u \f$ \n
 */
class DynamicsAbstract {
public:
  double eps = 1e-6;
  /**
   * Constructor
   *
   * @param [in] x: Dimension of state.
   * @param [in] u: Dimension of input.
   */
  DynamicsAbstract(const size_t nx, const size_t nu) : nx_(nx), nu_(nu) {}

  /** Default destructor */
  virtual ~DynamicsAbstract() = default;

  /** Clone */
  virtual DynamicsAbstract *clone() const = 0;

  /**
   * Computes the flow map of a system with exogenous input.
   *
   * @param [in] x: The current state.
   * @param [in] u: The current input.
   */
  virtual Eigen::VectorXd
  ComputeFlowMap(const Eigen::Ref<const Eigen::VectorXd> &x,
                 const Eigen::Ref<const Eigen::VectorXd> &u) = 0;

  /**
   * Computes the linear approximation.
   *
   * @param [in] x: The current state.
   * @param [in] u: The current input.
   * @param [in] Fx: Derivative w.r.t state
   * @param [in] Fu: Derivative w.r.t input
   */
  virtual void
  GetLinearApproximation(const Eigen::Ref<const Eigen::VectorXd> &x,
                         const Eigen::Ref<const Eigen::VectorXd> &u,
                         Eigen::Ref<Eigen::MatrixXd> Fx,
                         Eigen::Ref<Eigen::MatrixXd> Fu) {
    Fx = Derivative(std::bind(&DynamicsAbstract::ComputeFlowMap, this,
                              std::placeholders::_1, u),
                    x, eps);
    Fu = Derivative(std::bind(&DynamicsAbstract::ComputeFlowMap, this, x,
                              std::placeholders::_1),
                    u, eps);
  }

  size_t state_size() const { return nx_; }
  size_t input_size() const { return nu_; }

private:
  const size_t nx_;
  const size_t nu_;
};

} // namespace ocp
