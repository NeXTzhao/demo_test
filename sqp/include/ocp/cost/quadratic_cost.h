#pragma once

#include "ocp/cost/cost_base.h"

namespace ocp {

/** Quadratic cost term */
class QuadraticCost : public CostAbstract {
public:
  /**
   * Constructor for the quadratic cost function defined as the following:
   * \f$ L = 0.5(x-x_{n})' Q (x-x_{n}) + 0.5(u-u_{n})' R (u-u_{n}) + (u-u_{n})'
   * P (x-x_{n}) \f$
   * @param [in] Q: \f$ Q \f$
   * @param [in] R: \f$ R \f$
   * @param [in] P: \f$ P \f$
   */
  QuadraticCost(Eigen::MatrixXd Q, Eigen::MatrixXd R,
                Eigen::MatrixXd P = Eigen::MatrixXd());

  QuadraticCost(Eigen::VectorXd state_ref, Eigen::VectorXd input_ref,
                Eigen::MatrixXd Q, Eigen::MatrixXd R,
                Eigen::MatrixXd P = Eigen::MatrixXd());

  ~QuadraticCost() override = default;
  QuadraticCost *clone() const override;

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
  QuadraticCost(const QuadraticCost &rhs) = default;

private:
  Eigen::MatrixXd Q_;
  Eigen::MatrixXd R_;
  Eigen::MatrixXd P_;
  Eigen::VectorXd state_ref_;
  Eigen::VectorXd input_ref_;
};

} // namespace ocp
