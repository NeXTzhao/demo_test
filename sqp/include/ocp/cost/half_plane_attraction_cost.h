#pragma once

#include "ocp/cost/cost_base.h"

// forward declaration
struct DiskInfo;
class HalfplaneRepeller;

namespace ocp {

/**
 * Half plane attraction cost function defined as the the following: \f$ f =
 * 0.5*w*(Ax - b - offset)^2 \f$
 */
class HalfPlaneAttractionCost : public CostAbstract {
public:
  /**
   * Constructor
   *
   * @param [in] weight: weight
   * @param [in] disks: x in Ax - b - offset
   * @param [in] half_planes: A,b in Ax - b - offset
   * @param [in] offset: offset in Ax - b - offset
   */
  HalfPlaneAttractionCost(const double weight, std::vector<DiskInfo> disks,
                          std::vector<HalfplaneRepeller> half_planes,
                          const double offset = 0.0);

  HalfPlaneAttractionCost(const double weight, std::vector<DiskInfo> disks,
                          std::vector<HalfplaneRepeller> half_planes,
                          Eigen::VectorXd offsets);

  ~HalfPlaneAttractionCost() override = default;
  HalfPlaneAttractionCost *clone() const override;

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
  HalfPlaneAttractionCost(const HalfPlaneAttractionCost &rhs) = default;

private:
  double weight_ = 0.0;
  std::vector<DiskInfo> disks_;
  std::vector<HalfplaneRepeller> half_planes_;
  Eigen::VectorXd offsets_;
};

} // namespace ocp
