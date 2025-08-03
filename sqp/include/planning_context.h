#pragma once

#include "common.h"
#include "ocp/csqp/constrained_sqp.h"
#include "ocp/dynamics/vehicle_model_dynamics.h"

#include <Eigen/Core>
#include <cmath>  // std::isfinite
#include <cstdio> // std::snprintf
#include <string>
#include <vector>

#pragma once

// ====== 你的项目头文件（按需调整）======
#include "common.h"
#include "cost_weight_util.h"
#include "ocp/csqp/constrained_sqp.h"
#include "ocp/dynamics/vehicle_model_dynamics.h"

#include <Eigen/Core>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

// ===================================//
//  PlanningContext：实例化数据容器     //
// ===================================//
struct PlanningContext {
  static constexpr int X_DIM = ocp::VehicleModelDynamics::StateIndex::X_DIM;
  static constexpr int U_DIM = ocp::VehicleModelDynamics::ControlIndex::U_DIM;

  using State = Eigen::Matrix<double, X_DIM, 1>;
  using Control = Eigen::Matrix<double, U_DIM, 1>;

  using StateSequence = std::vector<State>;
  using ControlSequence = std::vector<Control>;

  struct StateControlSequence {
    StateSequence state_sequence;
    ControlSequence control_sequence;
  };

  explicit PlanningContext(const StateSequence &state_sequence,
                           const ControlSequence &control_sequence,
                           const ModelParameter &mp,
                           const std::vector<CostingTerm> &costing_terms);
  ~PlanningContext() = default;

  const ModelParameter &get_vehicle_model_parameter() const {
    return vehicle_model_parameter_;
  }
  std::vector<CostingTerm> get_costing_terms() const { return costing_terms_; }

  StateSequence get_xs_init() const { return xs_init; }
  StateSequence get_xs_opt() const { return xs_opt; }
  void set_xs_opt(const StateSequence &xs_opt) { this->xs_opt = xs_opt; }

  ControlSequence get_us_init() const { return us_init; }
  ControlSequence get_us_opt() const { return us_opt; }
  void set_us_opt(const ControlSequence &us_opt) { this->us_opt = us_opt; }

  void set_xs_us_opt(const StateSequence &xs_opt,
                     const ControlSequence &us_opt) {
    set_xs_opt(xs_opt);
    set_us_opt(us_opt);
  }

  // === 统一检查：返回是否通过；若提供 reason，则写明失败原因 ===
  bool CheckPlanningContext(std::string *reason = nullptr) const;

private:
  // 轻量格式化到 reason（避免依赖 <format>）
  static void WriteReason(std::string *reason, const char *msg) {
    if (reason)
      *reason = msg;
  }
  template <typename... Args>
  static void WriteReason(std::string *reason, const char *fmt, Args... args) {
    if (!reason)
      return;
    char buf[256];
    std::snprintf(buf, sizeof(buf), fmt, args...);
    *reason = buf;
  }

private:
  // 注意：const 引用成员必须在构造函数里初始化
  const ModelParameter &vehicle_model_parameter_;
  std::vector<CostingTerm> costing_terms_;
  StateSequence xs_init;
  ControlSequence us_init;
  StateSequence xs_opt;
  ControlSequence us_opt;
};
