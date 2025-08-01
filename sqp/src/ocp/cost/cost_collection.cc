#include "ocp/cost/cost_collection.h"

namespace ocp {

CostCollection::CostCollection(const CostCollection &other) = default;

CostCollection *CostCollection::clone() const {
  return new CostCollection(*this);
}

double
CostCollection::GetValue(const Eigen::Ref<const Eigen::VectorXd> &state,
                         const Eigen::Ref<const Eigen::VectorXd> &input) {
  double cost = 0.0;

  // accumulate cost terms
  for (const auto &cost_term : this->terms_) {
    if (cost_term->IsActive()) {
      cost += cost_term->GetValue(state, input);
    }
  }

  return cost;
}

void CostCollection::GetQuadraticApproximation(
    const Eigen::Ref<const Eigen::VectorXd> &state,
    const Eigen::Ref<const Eigen::VectorXd> &input,
    Eigen::Ref<Eigen::VectorXd> Lx, Eigen::Ref<Eigen::VectorXd> Lu,
    Eigen::Ref<Eigen::MatrixXd> Lxx, Eigen::Ref<Eigen::MatrixXd> Luu,
    Eigen::Ref<Eigen::MatrixXd> Lxu) const {
  const auto first_active =
      std::find_if(terms_.begin(), terms_.end(),
                   [](const std::unique_ptr<CostAbstract> &cost_term) {
                     return cost_term->IsActive();
                   });

  Lx = Eigen::VectorXd::Zero(state.size());
  Lxx = Eigen::MatrixXd::Zero(state.size(), state.size());
  if (input.size() != 0) {
    Lu = Eigen::VectorXd::Zero(input.size());
    Luu = Eigen::MatrixXd::Zero(input.size(), input.size());
    Lxu = Eigen::MatrixXd::Zero(state.size(), input.size());
  }

  // No active terms (or terms is empty).
  if (first_active == terms_.end()) {
    return;
  }

  // Initialize with first active term, accumulate potentially other active
  // terms.
  (*first_active)
      ->GetQuadraticApproximation(state, input, Lx, Lu, Lxx, Luu, Lxu);

  Eigen::VectorXd tmp_Lx = Eigen::VectorXd::Zero(state.size());
  Eigen::MatrixXd tmp_Lxx = Eigen::MatrixXd::Zero(state.size(), state.size());
  Eigen::VectorXd tmp_Lu;
  Eigen::MatrixXd tmp_Luu;
  Eigen::MatrixXd tmp_Lxu;
  if (input.size() != 0) {
    tmp_Lu = Eigen::VectorXd::Zero(input.size());
    tmp_Luu = Eigen::MatrixXd::Zero(input.size(), input.size());
    tmp_Lxu = Eigen::MatrixXd::Zero(state.size(), input.size());
  }
  std::for_each(std::next(first_active), terms_.end(),
                [&](const std::unique_ptr<CostAbstract> &cost_term) {
                  if (cost_term->IsActive()) {
                    tmp_Lx.setZero();
                    tmp_Lxx.setZero();
                    if (input.size() != 0) {
                      tmp_Lu.setZero();
                      tmp_Luu.setZero();
                      tmp_Lxu.setZero();
                    }
                    cost_term->GetQuadraticApproximation(state, input, tmp_Lx,
                                                         tmp_Lu, tmp_Lxx,
                                                         tmp_Luu, tmp_Lxu);
                    Lx += tmp_Lx;
                    Lxx += tmp_Lxx;
                    if (input.size() != 0) {
                      Lu += tmp_Lu;
                      Luu += tmp_Luu;
                      Lxu += tmp_Lxu;
                    }
                  }
                });
  return;
}

} // namespace ocp
