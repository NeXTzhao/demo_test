#include "ocp/shooting/action_model.h"
#include "ocp/utils/exception.h"

// cost
#include "ocp/cost/cost_collection.h"
// constraint
#include "ocp/constraint/constraint_collection.h"
// dynamics
#include "ocp/dynamics/dynamics_base.h"

namespace ocp {

ActionModel::ActionModel(std::unique_ptr<CostCollection> costs,
                         std::unique_ptr<ConstraintCollection> constraints,
                         std::shared_ptr<DynamicsAbstract> dynamics,
                         const std::size_t nx, const std::size_t ndx,
                         const std::size_t nu)
    : costs_ptr_(std::move(costs)), constraints_ptr_(std::move(constraints)),
      dynamics_ptr_(dynamics), nx_(nx), ndx_(ndx), nu_(nu),
      unone_(Eigen::VectorXd::Zero(nu)) {
  ng_ = 0;
  if (!constraints_ptr_) {
    return;
  }
  ng_ += constraints_ptr_->GetNumConstraints();
  if (ng_ == 0) {
    return;
  }
  g_lb_ = constraints_ptr_->GetLowerBounds();
  g_ub_ = constraints_ptr_->GetUpperBounds();
}

ActionModel::~ActionModel() {}

void ActionModel::calc(const std::shared_ptr<ActionData> &data,
                       const Eigen::Ref<const Eigen::VectorXd> &x,
                       const Eigen::Ref<const Eigen::VectorXd> &u) {
  // Compute running cost
  data->cost = 0.0;
  if (costs_ptr_) {
    data->cost += costs_ptr_->GetValue(x, u);
  }
  // Compute dynamics
  data->xnext = dynamics_ptr_->ComputeFlowMap(x, u);
  // Compute constraints
  if (constraints_ptr_) {
    data->g = constraints_ptr_->GetValue(x, u);
  }
}

void ActionModel::calc(const std::shared_ptr<ActionData> &data,
                       const Eigen::Ref<const Eigen::VectorXd> &x) {
  // Compute terminal cost
  data->cost = 0.0;
  if (costs_ptr_) {
    data->cost += costs_ptr_->GetValue(x, unone_);
  }
  // Compute dynamics
  data->xnext.setZero();
  // Compute constraints
  if (constraints_ptr_ && ng_ != 0) {
    data->g = constraints_ptr_->GetValue(x, unone_);
  }
}

void ActionModel::calcDiff(const std::shared_ptr<ActionData> &data,
                           const Eigen::Ref<const Eigen::VectorXd> &x,
                           const Eigen::Ref<const Eigen::VectorXd> &u) {
  // Compute running cost derivatives
  if (costs_ptr_) {
    costs_ptr_->GetQuadraticApproximation(x, u, data->Lx, data->Lu, data->Lxx,
                                          data->Luu, data->Lxu);
  }

  // Compute dynamics derivatives
  dynamics_ptr_->GetLinearApproximation(x, u, data->Fx, data->Fu);

  // Compute constraints derivatives
  if (constraints_ptr_ && ng_ != 0) {
    constraints_ptr_->GetLinearApproximation(x, u, data->Gx, data->Gu);
  }
}

void ActionModel::calcDiff(const std::shared_ptr<ActionData> &data,
                           const Eigen::Ref<const Eigen::VectorXd> &x) {
  // Compute terminal cost derivatives
  if (costs_ptr_) {
    costs_ptr_->GetQuadraticApproximation(x, unone_, data->Lx, data->Lu,
                                          data->Lxx, data->Luu, data->Lxu);
  }
  // Compute dynamics derivatives
  data->Fx.setZero();
  // Compute constraints derivatives
  if (constraints_ptr_ && ng_ != 0) {
    constraints_ptr_->GetLinearApproximation(x, unone_, data->Gx, data->Gu);
  }
}

std::shared_ptr<ActionData> ActionModel::createData() {
  return std::allocate_shared<ActionData>(
      Eigen::aligned_allocator<ActionData>(), this);
}

bool ActionModel::checkData(const std::shared_ptr<ActionData> &data) {
  return true;
}

std::size_t ActionModel::get_nx() const { return nx_; }

std::size_t ActionModel::get_ndx() const { return ndx_; }

std::size_t ActionModel::get_nu() const { return nu_; }

std::size_t ActionModel::get_ng() const { return ng_; }

const Eigen::VectorXd &ActionModel::get_g_lb() const { return g_lb_; }

const Eigen::VectorXd &ActionModel::get_g_ub() const { return g_ub_; }

void ActionModel::set_g_lb(const Eigen::VectorXd &g_lb) {
  if (static_cast<std::size_t>(g_lb.size()) != ng_) {
    throw_pretty(
        "Invalid argument: "
        << "inequality lower bound has wrong dimension (it should be " +
               std::to_string(ng_) + ")");
  }
  g_lb_ = g_lb;
}

void ActionModel::set_g_ub(const Eigen::VectorXd &g_ub) {
  if (static_cast<std::size_t>(g_ub.size()) != ng_) {
    throw_pretty(
        "Invalid argument: "
        << "inequality upper bound has wrong dimension (it should be " +
               std::to_string(ng_) + ")");
  }
  g_ub_ = g_ub;
}

} // namespace ocp
