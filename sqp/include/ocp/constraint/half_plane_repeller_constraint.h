#pragma once

#include "ocp/constraint/constraint_base.h"

// forward declaration
struct DiskInfo;
class HalfplaneRepeller;

namespace ocp {

/**
 * Half plane repeller constraint as the following: \f$ Ax - b <= offset \f$
 */
class HalfplaneRepellerConstraint final : public ConstraintAbstract {
public:
  /**
   * Constructor
   *
   * @param [in] disks: x in Ax - b <= offset
   * @param [in] half_planes: A,b in Ax - b <= offset
   * @param [in] offset: offset in Ax - b <= offset
   */
  HalfplaneRepellerConstraint(std::vector<DiskInfo> disks,
                              std::vector<HalfplaneRepeller> half_planes,
                              const double offset = 0.0);

  HalfplaneRepellerConstraint(std::vector<DiskInfo> disks,
                              std::vector<HalfplaneRepeller> half_planes,
                              Eigen::VectorXd offsets);

  ~HalfplaneRepellerConstraint() override = default;

  HalfplaneRepellerConstraint *clone() const override;

  size_t GetNumConstraints() const override;

  Eigen::VectorXd GetValue(const Eigen::Ref<const Eigen::VectorXd> &x,
                           const Eigen::Ref<const Eigen::VectorXd> &u) override;

  void GetLinearApproximation(const Eigen::Ref<const Eigen::VectorXd> &x,
                              const Eigen::Ref<const Eigen::VectorXd> &u,
                              Eigen::Ref<Eigen::MatrixXd> Gx,
                              Eigen::Ref<Eigen::MatrixXd> Gu) const override;

private:
  std::vector<DiskInfo> disks_;
  std::vector<HalfplaneRepeller> half_planes_;
};

} // namespace ocp
