#pragma once
#include "planning_context.h"
#include "problem_util.h"

#include <iostream>

// ========== 1) 统一接口 ==========
class AbstractTrajectorySolver {
public:
  virtual ~AbstractTrajectorySolver() = default;
  virtual const char *Name() const = 0;
  virtual bool Solve(PlanningContext &ctx) = 0;
};

// ========== 2) 具体求解器 ==========
class SqpTrajectorySolver final : public AbstractTrajectorySolver {
public:
  const char *Name() const override { return "SQP"; }
  bool Solve(PlanningContext &ctx) override {
    std::string why;
    if (!ctx.CheckPlanningContext(&why)) {
      std::cout << why << std::endl;
    }
    // TODO::
    // SqpSolve(ctx);
    // const std::vector<Eigen::VectorXd> &opt_xs = sqp_solver.get_xs();
    // const std::vector<Eigen::VectorXd> &opt_us = sqp_solver.get_us();
    // ctx.set_xs_us_opt(opt_xs, opt_us);
    return true;
  }
};

class IlqrTrajectorySolver final : public AbstractTrajectorySolver {
public:
  const char *Name() const override { return "iLQR"; }
  bool Solve(PlanningContext &ctx) override {
    std::string why;
    if (!ctx.CheckPlanningContext(&why)) {
      std::cout << why << std::endl;
    }
    // TODO::
    // iLQRSolve(ctx);
    // const std::vector<Eigen::VectorXd> &opt_xs = iLQRSolve.get_xs();
    // const std::vector<Eigen::VectorXd> &opt_us = iLQRSolve.get_us();
    // ctx.set_xs_us_opt(opt_xs, opt_us);
    return true;
  }
};

// ========== 3) 枚举 + switch 选择 ==========
enum class SolverKind {
  SQP,
  ILQR,
};

inline bool SolveWith(SolverKind kind, PlanningContext &ctx) {
  switch (kind) {
  case SolverKind::SQP: {
    SqpTrajectorySolver solver;
    return solver.Solve(ctx);
  }
  case SolverKind::ILQR: {
    IlqrTrajectorySolver solver;
    return solver.Solve(ctx);
  }
  default:
    std::cout << "Unknown Solver Kind " << static_cast<int>(kind) << std::endl;
    return false; // 未支持的类型
  }
}
