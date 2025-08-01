#pragma once

#include "ocp/collection/collection.h"
#include "ocp/constraint/constraint_base.h"

namespace ocp {

/**
 * Constraint collection class
 *
 * This class collects a variable number of constraint functions and provides
 * methods to get the concatenated constraint vectors and approximations. Each
 * constraint can be accessed through its string name and can be activated or
 * deactivated.
 */
class ConstraintCollection : public Collection<ConstraintAbstract> {
public:
  ConstraintCollection() = default;
  ~ConstraintCollection() override = default;
  ConstraintCollection *clone() const override;

  /** Returns the number of active constraints. */
  size_t GetNumConstraints() const;

  /** Returns the number of active constraintsfor each term. If a term is
   * inactive, it will be ignored. */
  std::vector<size_t> GetTermsSize() const;

  /** Returns the lower bounds of all constraints. If a term is inactive, it
   * will be ignored.*/
  virtual Eigen::VectorXd GetLowerBounds() const;

  /** Returns the upper bounds of all constraints. If a term is inactive, it
   * will be ignored.*/
  virtual Eigen::VectorXd GetUpperBounds() const;

  /** Get an array of all constraints. If a term is inactive, it will be
   * ignored. */
  virtual Eigen::VectorXd
  GetValue(const Eigen::Ref<const Eigen::VectorXd> &state,
           const Eigen::Ref<const Eigen::VectorXd> &input);

  /** Get the constraint linear approximation */
  virtual void
  GetLinearApproximation(const Eigen::Ref<const Eigen::VectorXd> &state,
                         const Eigen::Ref<const Eigen::VectorXd> &input,
                         Eigen::Ref<Eigen::MatrixXd> Gx,
                         Eigen::Ref<Eigen::MatrixXd> Gu) const;

protected:
  /** Copy constructor */
  ConstraintCollection(const ConstraintCollection &other);
};

} // namespace ocp
