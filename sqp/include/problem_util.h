#pragma once

#include <Eigen/Core>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "math/disk.h"
#include "ocp/constraint/constraint_collection.h"
#include "ocp/cost/cost_collection.h"
#include "ocp/csqp/constrained_sqp.h"
#include "ocp/dynamics/vehicle_model_dynamics.h"

struct PlanningContext;

inline std::vector<DiskInfo> ComputeEgoDisks(const ModelParameter &param,
                                             const Eigen::VectorXd &state) {
  using SI = ocp::VehicleModelDynamics::StateIndex;
  return GetCoveringDiskInfo(param.measurement, state(SI::X_POS),
                             state(SI::Y_POS), state(SI::THETA));
}

// =========== 可选：检查工具 ===========
inline bool AllFinite(const Eigen::VectorXd &v) {
  for (int i = 0; i < v.size(); ++i)
    if (!std::isfinite(v[i]))
      return false;
  return true;
}

// ================ 定义问题的抽象基类 ================
class IOcpModels {
public:
  virtual ~IOcpModels() = default;

  // 生命周期
  virtual void Clear() = 0;
  virtual void Reserve(int horizon) = 0;

  // 构建：running_cost + terminal_cost
  // 说明：
  //  - running 模型个数 = N-1，terminal 模型 1 个
  //  - 返回 false 时 why 写明原因
  virtual bool Build(const PlanningContext &ctx,
                     const std::shared_ptr<ocp::DynamicsAbstract> &dynamics,
                     std::string *why = nullptr) = 0;

  // 访问
  virtual int Horizon() const = 0;
  virtual const std::vector<std::shared_ptr<ocp::ActionModel>> &
  Running() const = 0;
  virtual const std::shared_ptr<ocp::ActionModel> &Terminal() const = 0;
};

// ================ 实现：面向 SQP 的 ActionModel ================
class SqpActionModels final : public IOcpModels {
public:
  SqpActionModels() = default;

  void Clear() override {
    running_.clear();
    terminal_.reset();
    horizon_ = 0;
  }

  void Reserve(int horizon) override {
    if (horizon > 1)
      running_.reserve(horizon - 1);
  }

  int Horizon() const override { return horizon_; }
  const std::vector<std::shared_ptr<ocp::ActionModel>> &
  Running() const override {
    return running_;
  }
  const std::shared_ptr<ocp::ActionModel> &Terminal() const override {
    return terminal_;
  }

  bool Build(const PlanningContext &ctx,
             const std::shared_ptr<ocp::DynamicsAbstract> &dynamics,
             std::string *why = nullptr) override {
    Clear();
    horizon_ = static_cast<int>(xs.size());
    running_.resize(horizon_ - 1); // N 状态 → N-1 运行步

    // 第 0 步：无 primitives 的 GenerateFormulaCosts 重载
    running_[0] = BuildRunningStep(
        /*k=*/0, xs[0], *ctx.model_param, ctx.costing_terms, dynamics,
        /*primitives=*/nullptr);

#ifdef CSQP_WITH_MULTITHREADING
#pragma omp parallel for num_threads(kNumOfThreads)
#endif
    for (int k = 1; k < horizon_ - 1; ++k) {
      auto model = BuildRunningStep(k, xs[k], *ctx.model_param,
                                    ctx.costing_terms, dynamics, primitives);
#ifdef CSQP_WITH_MULTITHREADING
#pragma omp critical
#endif
      { running_[k] = std::move(model); }
    }

    // 终端
    terminal_ = BuildTerminalStep(horizon_ - 1, xs.back(), *ctx.model_param,
                                  ctx.costing_terms, dynamics, primitives);

    if (!terminal_)
      return Fail(why, "failed to build terminal model");
    return true;
  }

private:
  static bool Fail(std::string *why, const char *msg) {
    if (why)
      *why = msg;
    return false;
  }
  template <typename... Args>
  static bool FailFmt(std::string *why, const char *fmt, Args... args) {
    if (!why)
      return false;
    char buf[256];
    std::snprintf(buf, sizeof(buf), fmt, args...);
    *why = buf;
    return false;
  }

  std::shared_ptr<ocp::ActionModel> BuildRunningStep(
      int k, const Eigen::VectorXd &state, const ModelParameter &mp,
      const std::vector<CostingTerm> &terms,
      const std::shared_ptr<ocp::DynamicsAbstract> &dynamics) const {

    const auto ego_disks = GetCoveringDiskInfo(mp, state);

    auto costs = std::make_unique<ocp::CostCollection>();
    auto cons = std::make_unique<ocp::ConstraintCollection>();

    if (k == 0 || !primitives) {
      // 第 0 步（或没有 primitives）使用无 primitives 的重载
      GenerateFormulaCosts(k, state, mp, ego_disks, terms, costs.get());
    } else {
      GenerateFormulaCosts(k, state, mp, ego_disks, terms, *primitives,
                           costs.get());
    }

    if (primitives) {
      GenerateFormulaConstraints(k, state, mp, ego_disks, terms, *primitives,
                                 cons.get());
    } else {
      // 没有 primitives 时，按需留空或提供默认约束
    }

    return std::make_shared<ocp::ActionModel>(
        std::move(costs), std::move(cons), dynamics, dynamics->state_size(),
        dynamics->state_size(), dynamics->input_size());
  }

  std::shared_ptr<ocp::ActionModel> BuildTerminalStep(
      int k_terminal, const Eigen::VectorXd &state, const ModelParameter &mp,
      const std::vector<CostingTerm> &terms,
      const std::shared_ptr<ocp::DynamicsAbstract> &dynamics) const {

    const auto ego_disks = ComputeEgoDisks(mp, state);

    auto costs = std::make_unique<ocp::CostCollection>();
    auto cons = std::make_unique<ocp::ConstraintCollection>();

    if (primitives) {
      GenerateFormulaCosts(k_terminal, state, mp, ego_disks, terms, *primitives,
                           costs.get());
      GenerateFormulaConstraints(k_terminal, state, mp, ego_disks, terms,
                                 *primitives, cons.get());
    } else {
      GenerateFormulaCosts(k_terminal, state, mp, ego_disks, terms,
                           costs.get());
      // 终端约束没有 primitives 时可为空
    }

    return std::make_shared<ocp::ActionModel>(
        std::move(costs), std::move(cons), dynamics, dynamics->state_size(),
        dynamics->state_size(), /*input_size=*/0);
  }

private:
  int horizon_{0};
  std::vector<std::shared_ptr<ocp::ActionModel>> running_;
  std::shared_ptr<ocp::ActionModel> terminal_;
};
