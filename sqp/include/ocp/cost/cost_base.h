#pragma once

#include <Eigen/Core>
#include <type_traits>

#include "glog/logging.h"

namespace ocp {

/** Cost term */
class CostAbstract {
public:
  CostAbstract() = default;
  virtual ~CostAbstract() = default;
  virtual CostAbstract *clone() const = 0;

  /** Check if cost term is active */
  virtual bool IsActive() const { return true; }

  /** Get cost term value */
  virtual double
  GetValue(const Eigen::Ref<const Eigen::VectorXd> &state,
           const Eigen::Ref<const Eigen::VectorXd> &input) const = 0;

  /** Get cost term quadratic approximation */
  virtual void GetQuadraticApproximation(
      const Eigen::Ref<const Eigen::VectorXd> &state,
      const Eigen::Ref<const Eigen::VectorXd> &input,
      Eigen::Ref<Eigen::VectorXd> Lx, Eigen::Ref<Eigen::VectorXd> Lu,
      Eigen::Ref<Eigen::MatrixXd> Lxx, Eigen::Ref<Eigen::MatrixXd> Luu,
      Eigen::Ref<Eigen::MatrixXd> Lxu) const = 0;

protected:
  CostAbstract(const CostAbstract &rhs) = default;
};

// Template for conditional compilation using SFINAE
template <typename T>
using EnableIfCostAbstract_t =
    typename std::enable_if<std::is_same<T, CostAbstract>::value, bool>::type;

} // namespace ocp
