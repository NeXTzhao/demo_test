#include "planning_context.h"

PlanningContext::PlanningContext(const StateSequence &state_sequence,
                                 const ControlSequence &control_sequence,
                                 const ModelParameter &mp,
                                 const std::vector<CostingTerm> &costing_terms)
    : xs_init(state_sequence), us_init(control_sequence),
      vehicle_model_parameter_(mp), costing_terms_(costing_terms) {}

// === 统一检查：返回是否通过；若提供 reason，则写明失败原因 ===
bool PlanningContext::CheckPlanningContext(std::string *reason) const {
  // 1) 车辆参数
  if (!&vehicle_model_parameter_) {
    WriteReason(reason, "vehicle_model_parameter is null");
    return false;
  }

  // 2) 轨迹长度
  const std::size_t nx = xs_init.size();
  const std::size_t nu = us_init.size();
  if (nx == 0) {
    WriteReason(reason, "xs_init is empty");
    return false;
  }
  if (nu == 0) {
    WriteReason(reason, "us_init is empty");
    return false;
  }

  // 常见离散 OCP：nu == nx 或 nu + 1 == nx
  if (!(nu == nx || nu + 1 == nx)) {
    WriteReason(reason,
                "size mismatch: xs_init.size()=%zu, us_init.size()=%zu "
                "(expect nu==nx or nu+1==nx)",
                nx, nu);
    return false;
  }

  // 3) 检查 state/control 是否含 NaN/Inf
  for (std::size_t i = 0; i < nx; ++i) {
    if (!xs_init[i].allFinite()) {
      WriteReason(reason, "xs_init[%zu] contains NaN/Inf", i);
      return false;
    }
  }
  for (std::size_t i = 0; i < nu; ++i) {
    if (!us_init[i].allFinite()) {
      WriteReason(reason, "us_init[%zu] contains NaN/Inf", i);
      return false;
    }
  }

  // 4) 代价项检查
  if (costing_terms_.empty()) {
    WriteReason(reason, "costing_terms_ is empty");
    return false;
  }
  for (std::size_t k = 0; k < costing_terms_.size(); ++k) {
    const auto &term = costing_terms_[k];
    const auto &w = term.spec.weights; // 若命名不同，替换这里
    if (w.size() != nx) {
      WriteReason(reason,
                  "cost term[%zu] weights size=%zu != xs_init.size()=%zu", k,
                  w.size(), nx);
      return false;
    }
    for (std::size_t i = 0; i < w.size(); ++i) {
      if (!std::isfinite(w[i])) {
        WriteReason(reason, "cost term[%zu] weights[%zu] is not finite", k, i);
        return false;
      }
    }
  }
  return true;
}
