#pragma once

#include <Eigen/Core>
#include <type_traits>

#include "glog/logging.h"

namespace ocp {
/** Constraint function base class */
class ConstraintAbstract {
public:
  explicit ConstraintAbstract() = default;

  explicit ConstraintAbstract(Eigen::VectorXd lower_bounds,
                              Eigen::VectorXd upper_bounds)
      : lower_bounds_(std::move(lower_bounds)),
        upper_bounds_(std::move(upper_bounds)) {
    CHECK_EQ(lower_bounds_.size(), upper_bounds_.size());
  }

  virtual ~ConstraintAbstract() = default;

  virtual ConstraintAbstract *clone() const = 0;

  /** Check constraint activity */
  virtual bool IsActive() const { return true; }

  /** Get the size of the constraint vector at given time */
  virtual size_t GetNumConstraints() const = 0;

  /** Get the lower bounds of the constraints */
  virtual const Eigen::VectorXd &GetLowerBounds() const {
    return lower_bounds_;
  }

  /** Get the upper bounds of the constraints */
  virtual const Eigen::VectorXd &GetUpperBounds() const {
    return upper_bounds_;
  }

  /** Get the constraint vector value */
  virtual Eigen::VectorXd
  GetValue(const Eigen::Ref<const Eigen::VectorXd> &state,
           const Eigen::Ref<const Eigen::VectorXd> &input) = 0;

  /** Get the constraint linear approximation */
  virtual void
  GetLinearApproximation(const Eigen::Ref<const Eigen::VectorXd> &state,
                         const Eigen::Ref<const Eigen::VectorXd> &input,
                         Eigen::Ref<Eigen::MatrixXd> Gx,
                         Eigen::Ref<Eigen::MatrixXd> Gu) const = 0;

protected:
  ConstraintAbstract(const ConstraintAbstract &rhs) = default;
  Eigen::VectorXd lower_bounds_;
  Eigen::VectorXd upper_bounds_;
};

// Template for conditional compilation using SFINAE
template <typename T>
using EnableIfConstraintAbstract_t =
    std::enable_if_t<std::is_same_v<T, ConstraintAbstract>, bool>;

} // namespace ocp
