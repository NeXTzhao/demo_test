#ifdef CSQP_WITH_MULTITHREADING
#include <omp.h>
#endif // CSQP_WITH_MULTITHREADING

#include <Eigen/Cholesky>
#include <iomanip>
#include <iostream>

#include "ocp/csqp/constrained_sqp.h"
#include "ocp/dynamics/dynamics_base.h"
#include "ocp/shooting/action_model.h"
#include "ocp/shooting/shooting.h"
#include "ocp/utils/exception.h"
#include "ocp/utils/take_time.h"

namespace ocp {
namespace {
bool raiseIfNaN(const double value) {
  if (std::isnan(value) || std::isinf(value) || value >= 1e30) {
    return true;
  }
  return false;
}
} // namespace

ConstrainedSqp::ConstrainedSqp(SqpSettings settings,
                               std::shared_ptr<ShootingProblem> problem)
    : settings_(std::move(settings)), SolverAbstract(problem), cost_try_(0.) {
  allocateData();

  const std::size_t T = this->problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  constraint_list_.resize(settings_.filter_size);
  gap_list_.resize(settings_.filter_size);
  cost_list_.resize(settings_.filter_size);

  fs_flat_.resize(ndx * (T + 1));
  fs_flat_.setZero();

  xs_try_.resize(T + 1);
  us_try_.resize(T);
  dx_.resize(T + 1);
  du_.resize(T);
  dxtilde_.resize(T + 1);
  dutilde_.resize(T);
  lag_mul_.resize(T + 1);
  fs_try_.resize(T + 1);

  z_.resize(T + 1);
  z_relaxed_.resize(T + 1);
  z_prev_.resize(T + 1);
  y_.resize(T + 1);
  rho_vec_.resize(T + 1);
  inv_rho_vec_.resize(T + 1);
  rho_sparse_ = settings_.rho_sparse_base;

  tmp_Vx_.resize(ndx);
  tmp_Vx_.setZero();
  tmp_vec_x_.resize(ndx);
  tmp_vec_x_.setZero();

  tmp_Cdx_Cdu_.resize(T + 1);
  tmp_dual_cwise_.resize(T + 1);
  tmp_rhoGx_mat_.resize(T + 1);
  tmp_rhoGu_mat_.resize(T);
  tmp_vec_u_.resize(T);
  Vxx_fs_.resize(T);

  const std::vector<std::shared_ptr<ActionModel>> &models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModel> &model = models[t];
    const std::size_t nu = model->get_nu();
    const std::size_t nc = model->get_ng();

    xs_try_[t] = Eigen::VectorXd::Zero(model->get_nx());
    us_try_[t] = Eigen::VectorXd::Zero(nu);
    dx_[t].resize(ndx);
    dx_[t].setZero();
    du_[t].resize(nu);
    du_[t] = Eigen::VectorXd::Zero(nu);
    dxtilde_[t].resize(ndx);
    dxtilde_[t].setZero();
    dutilde_[t].resize(nu);
    dutilde_[t] = Eigen::VectorXd::Zero(nu);
    lag_mul_[t].resize(ndx);
    lag_mul_[t].setZero();
    fs_try_[t].resize(ndx);
    fs_try_[t] = Eigen::VectorXd::Zero(ndx);

    z_[t].resize(nc);
    z_[t].setZero();
    z_relaxed_[t].resize(nc);
    z_relaxed_[t].setZero();
    z_prev_[t].resize(nc);
    z_prev_[t].setZero();
    y_[t].resize(nc);
    y_[t].setZero();

    tmp_Cdx_Cdu_[t].resize(nc);
    tmp_Cdx_Cdu_[t].setZero();
    tmp_dual_cwise_[t].resize(nc);
    tmp_dual_cwise_[t].setZero();
    tmp_rhoGx_mat_[t].resize(nc, ndx);
    tmp_rhoGx_mat_[t].setZero();
    tmp_rhoGu_mat_[t].resize(nc, nu);
    tmp_rhoGu_mat_[t].setZero();
    tmp_vec_u_[t].resize(nu);
    tmp_vec_u_[t].setZero();
    Vxx_fs_[t].resize(ndx);
    Vxx_fs_[t].setZero();

    rho_vec_[t].resize(nc);
    inv_rho_vec_[t].resize(nc);
  }

  xs_try_.back() =
      Eigen::VectorXd::Zero(problem_->get_terminalModel()->get_nx());
  dx_.back().resize(ndx);
  dx_.back().setZero();
  dxtilde_.back().resize(ndx);
  dxtilde_.back().setZero();
  lag_mul_.back().resize(ndx);
  lag_mul_.back().setZero();
  fs_try_.back().resize(ndx);
  fs_try_.back() = Eigen::VectorXd::Zero(ndx);

  const std::size_t nc = problem_->get_terminalModel()->get_ng();
  z_.back().resize(nc);
  z_.back().setZero();
  z_relaxed_.back().resize(nc);
  z_relaxed_.back().setZero();
  z_prev_.back().resize(nc);
  z_prev_.back().setZero();
  y_.back().resize(nc);
  y_.back().setZero();

  tmp_Cdx_Cdu_.back().resize(nc);
  tmp_Cdx_Cdu_.back().setZero();
  tmp_dual_cwise_.back().resize(nc);
  tmp_dual_cwise_.back().setZero();
  tmp_rhoGx_mat_.back().resize(nc, ndx);
  tmp_rhoGx_mat_.back().setZero();

  rho_vec_.back().resize(nc);
  inv_rho_vec_.back().resize(nc);

  const std::size_t n_alphas = 10;
  alphas_.resize(n_alphas);
  for (std::size_t n = 0; n < n_alphas; ++n) {
    alphas_[n] = 1. / pow(2., static_cast<double>(n));
  }
  if (settings_.th_stepinc < alphas_[n_alphas - 1]) {
    settings_.th_stepinc = alphas_[n_alphas - 1];
    std::cerr << "Warning: th_stepinc has higher value than lowest alpha "
                 "value, set to "
              << std::to_string(alphas_[n_alphas - 1]) << std::endl;
  }
}

void ConstrainedSqp::reset_params() {
  if (settings_.reset_rho) {
    reset_rho_vec();
  }

  const std::size_t T = this->problem_->get_T();
  for (std::size_t t = 0; t < T; ++t) {
    z_[t].setZero();
    z_prev_[t].setZero();
    z_relaxed_[t].setZero();
    if (settings_.reset_y) {
      y_[t].setZero();
    }
  }

  z_.back().setZero();
  z_prev_.back().setZero();
  z_relaxed_.back().setZero();
  if (settings_.reset_y) {
    y_.back().setZero();
  }
}

ConstrainedSqp::~ConstrainedSqp() {}

Convergence ConstrainedSqp::solve(const std::vector<Eigen::VectorXd> &init_xs,
                                  const std::vector<Eigen::VectorXd> &init_us,
                                  const double reginit /* = NAN */) {
  start_time_ = take_time(2);

  if (problem_->is_updated()) {
    resizeData();
  }
  setCandidate(init_xs, init_us, false);
  // Otherwise xs[0] is overwritten by init_xs inside setCandidate()
  xs_[0] = problem_->get_x0();
  // it is needed in case that init_xs[0] is infeasible
  xs_try_[0] = problem_->get_x0();

  // Optionally remove Crocoddyl's regularization
  if (settings_.remove_reg) {
    preg_ = 0.;
    dreg_ = 0.;
  } else {
    if (std::isnan(reginit)) {
      preg_ = settings_.reg_min;
      dreg_ = settings_.reg_min;
    } else {
      preg_ = reginit;
      dreg_ = reginit;
    }
  }

  // Otherwise benchmarks blowup
  // TODO: find cleaner way
  if (settings_.max_iteration == 0) {
    calc(true);
    reset_rho_vec();
  }

  std::stringstream ss;
  // Main SQP loop
  max_solve_time_reached_ = false;
  for (iter_ = 0; iter_ < settings_.max_iteration; ++iter_) {
    if (take_time(2) - start_time_ >= settings_.max_solve_time) {
      max_solve_time_reached_ = true;
      return Convergence::SOLVETIME;
    }
    // Compute gradients
    calc(true);

    // reset rho only at the beginning of each solve if reset_rho_ is false
    // (after calc to get correct lb and ub)
    if (iter_ == 0 && !settings_.reset_rho) {
      reset_rho_vec();
    }

    // Solve QP
    if (settings_.remove_reg) {
      computeDirection(true);
    } else {
      while (!max_solve_time_reached_) {
        try {
          computeDirection(true);
        } catch (std::exception &e) {
          increaseRegularization();
          if (preg_ >= settings_.reg_max) {
            return Convergence::FALSE;
          } else {
            continue;
          }
        }
        break;
      }
    }

    if (qp_iters_ == 0) {
      return Convergence::FALSE;
    }

    // Check KKT criteria
    checkKKTConditions();

    if (KKT_ <= settings_.kkt_tolerance) {
      return Convergence::KKT;
    }

    // Line search
    constraint_list_.push_back(constraint_norm_);
    gap_list_.push_back(gap_norm_);
    cost_list_.push_back(cost_);

    // Calculate the coefficient of the merit function.
    if (settings_.mu_dynamic < 0. || settings_.mu_constraint < 0.) {
      lag_mul_inf_norm_ = 0;
      for (const auto &lag_mul : lag_mul_) {
        lag_mul_inf_norm_ =
            std::max(lag_mul_inf_norm_, lag_mul.lpNorm<Eigen::Infinity>());
      }
      for (const auto &y : y_) {
        lag_mul_inf_norm_ =
            std::max(lag_mul_inf_norm_, y.lpNorm<Eigen::Infinity>());
      }
      merit_ = cost_ + settings_.lag_mul_inf_norm_coef * lag_mul_inf_norm_ *
                           (gap_norm_ + constraint_norm_);
    } else {
      merit_ = cost_ + settings_.mu_dynamic * gap_norm_ +
               settings_.mu_constraint * constraint_norm_;
    }

    // We need to recalculate the derivatives when the step length passes
    double step_size = 0.0;
    // less than filter_size_, less or equal iter_
    const std::size_t max_count = std::min(settings_.filter_size, iter_ + 1);
    for (const double steplength_ : alphas_) {
      try {
        merit_try_ = tryStep(steplength_);
      } catch (std::exception &e) {
        continue;
      }

      // Filter line search criteria
      if (settings_.use_filter_line_search) {
        is_worse_than_memory_ = false;
        std::size_t count = 0.;
        while (count < max_count && !is_worse_than_memory_) {
          is_worse_than_memory_ =
              cost_list_[settings_.filter_size - 1 - count] <= cost_try_ &&
              gap_list_[settings_.filter_size - 1 - count] <= gap_norm_try_ &&
              constraint_list_[settings_.filter_size - 1 - count] <=
                  constraint_norm_try_;
          count++;
        }
        if (!is_worse_than_memory_) {
          setCandidate(xs_try_, us_try_, false);
          step_size = steplength_;
          break;
        }
      }
      // Line-search criteria using merit function
      else {
        if (merit_ > merit_try_) {
          setCandidate(xs_try_, us_try_, false);
          step_size = steplength_;
          break;
        }
      }
    }

    merit_diff_ = std::abs(merit_try_ - merit_);

    if (step_size < settings_.alpha_min) {
      constraint_norm_ = constraint_norm_try_;
      // Converged because step size is below the specified minimum
      return Convergence::STEPSIZE;
    }

    if (merit_diff_ < settings_.cost_tolerance &&
        constraint_norm_try_ < settings_.g_min) {
      constraint_norm_ = constraint_norm_try_;
      return Convergence::METRICS;
    }

    if (dx_norm_ < settings_.delta_tolerance &&
        du_norm_ < settings_.delta_tolerance) {
      constraint_norm_ = constraint_norm_try_;
      return Convergence::PRIMAL;
    }

    // update running models and terminal model
    if (devastator_wrapper_info_ptr_ && get_feature_fn_) {
      std::vector<std::shared_ptr<ocp::ActionModel>> running_models;
      std::shared_ptr<ocp::ActionModel> terminal_model;
      get_feature_fn_(*devastator_wrapper_info_ptr_, xs_,
                      problem_->get_terminalModel()->get_dynamics(),
                      &running_models, &terminal_model);
      problem_->set_runningModels(std::move(running_models));
      problem_->set_terminalModel(std::move(terminal_model));
    }

    if (settings_.remove_reg) {
      continue;
    }

    // Regularization logic
    if (steplength_ > settings_.th_stepdec) {
      decreaseRegularization();
    } else {
      increaseRegularization();
      // preg_ equal to reg_max_
      if (preg_ >= settings_.reg_max) {
        return Convergence::FALSE;
      }
    }
  } // end of each iteration

  constraint_norm_ = constraint_norm_try_;
  return Convergence::ITERATIONS;
}

void ConstrainedSqp::calc(const bool recalc) {
  if (recalc) {
    problem_->calc(xs_, us_);
    cost_ = problem_->calcDiff(xs_, us_);
  }

  gap_norm_ = 0.;
  constraint_norm_ = 0.;

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModel>> &models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionData>> &datas =
      problem_->get_runningDatas();

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModel> &m = models[t];
    const std::shared_ptr<ActionData> &d = datas[t];

    // m->get_state()->diff(xs_[t + 1], d->xnext, fs_[t + 1]);
    fs_[t + 1] = d->xnext - xs_[t + 1];

    gap_norm_ += fs_[t + 1].lpNorm<1>();

    const std::size_t nc = m->get_ng();
    constraint_norm_ +=
        (m->get_g_lb() - d->g).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();
    constraint_norm_ +=
        (d->g - m->get_g_ub()).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();
  }

  const std::shared_ptr<ActionData> &d_T = problem_->get_terminalData();
  const std::size_t nc = problem_->get_terminalModel()->get_ng();
  constraint_norm_ += (problem_->get_terminalModel()->get_g_lb() - d_T->g)
                          .cwiseMax(Eigen::VectorXd::Zero(nc))
                          .lpNorm<1>();
  constraint_norm_ += (d_T->g - problem_->get_terminalModel()->get_g_ub())
                          .cwiseMax(Eigen::VectorXd::Zero(nc))
                          .lpNorm<1>();
}

void ConstrainedSqp::computeDirection(const bool /*recalcDiff*/) {
  // CSQP_EIGEN_MALLOC_NOT_ALLOWED();

  reset_params();

  if (settings_.equality_qp_initial_guess) {
    backwardPass_without_constraints();
    forwardPass_without_constraints();
  }

  if (settings_.with_qp_callbacks) {
    printQPCallbacks(0);
  }

  for (qp_iters_ = 1; qp_iters_ < settings_.max_qp_iters + 1; ++qp_iters_) {
    if (take_time(2) - start_time_ >= settings_.max_solve_time) {
      // Reduce number of QP iterations, to match real number of executed loops
      qp_iters_--;
      max_solve_time_reached_ = true;
      break;
    }

    if (qp_iters_ % settings_.rho_update_interval == 1 ||
        settings_.rho_update_interval == 1) {
#ifdef CSQP_WITH_MULTITHREADING
      if (problem_->get_nthreads() > 1)
        backwardPass_mt();
      else
#endif // CSQP_WITH_MULTITHREADING
        backwardPass();
    } else {
#ifdef CSQP_WITH_MULTITHREADING
      if (problem_->get_nthreads() > 1)
        backwardPass_without_rho_update_mt();
      else
#endif // CSQP_WITH_MULTITHREADING
        backwardPass_without_rho_update();
    }
    forwardPass();
    update_lagrangian_parameters(qp_iters_);
    update_rho_vec(qp_iters_);

    // Because (eps_rel=0) x inf = NaN
    if (qp_iters_ % settings_.rho_update_interval == 0) {
      if (settings_.with_qp_callbacks) {
        printQPCallbacks(qp_iters_);
      }
      if (std::fabs(settings_.eps_rel) <=
          std::numeric_limits<double>::epsilon()) {
        norm_primal_tolerance_ = settings_.eps_abs;
        norm_dual_tolerance_ = settings_.eps_abs;
      } else {
        norm_primal_tolerance_ =
            settings_.eps_abs + settings_.eps_rel * norm_primal_rel_;
        norm_dual_tolerance_ =
            settings_.eps_abs + settings_.eps_rel * norm_dual_rel_;
      }
      if (norm_primal_ <= norm_primal_tolerance_ &&
          norm_dual_ <= norm_dual_tolerance_) {
        break;
      }
    }
  }

  // CSQP_EIGEN_MALLOC_ALLOWED();
}

void ConstrainedSqp::update_rho_vec(const int iter) {
  const double scale = std::sqrt((norm_primal_ * norm_dual_rel_) /
                                 (norm_dual_ * norm_primal_rel_));
  rho_estimate_sparse_ = std::min(
      std::max(scale * rho_sparse_, settings_.rho_min), settings_.rho_max);

  if (iter % settings_.rho_update_interval == 0) { // && iter > 1){
    if (rho_estimate_sparse_ > rho_sparse_ * settings_.adaptive_rho_tolerance ||
        rho_estimate_sparse_ < rho_sparse_ / settings_.adaptive_rho_tolerance) {
      rho_sparse_ = rho_estimate_sparse_;
      apply_rho_update(rho_sparse_);
    }
  }
}

void ConstrainedSqp::reset_rho_vec() {
  rho_sparse_ = settings_.rho_sparse_base;
  apply_rho_update(rho_sparse_);
}

void ConstrainedSqp::apply_rho_update(const double rho_sparse_tmp) {
  const std::size_t T = this->problem_->get_T();
  const std::vector<std::shared_ptr<ActionModel>> &models =
      problem_->get_runningModels();
  double infty = std::numeric_limits<double>::infinity();

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModel> &m = models[t];
    const std::size_t nc = m->get_ng();
    for (std::size_t k = 0; k < nc; ++k) {
      if (m->get_g_lb()[k] == -infty && m->get_g_ub()[k] == infty) {
        rho_vec_[t][k] = settings_.rho_min;
        inv_rho_vec_[t][k] = 1. / settings_.rho_min;
      } else if (abs(m->get_g_lb()[k] - m->get_g_ub()[k]) <= 1e-6) {
        rho_vec_[t][k] = 1e3 * rho_sparse_tmp;
        inv_rho_vec_[t][k] = 1. / (1e3 * rho_sparse_tmp);
      } else if (m->get_g_lb()[k] < m->get_g_ub()[k]) {
        rho_vec_[t][k] = rho_sparse_tmp;
        inv_rho_vec_[t][k] = 1. / rho_sparse_tmp;
      }
    }
  }

  const std::size_t nc = problem_->get_terminalModel()->get_ng();
  for (std::size_t k = 0; k < nc; ++k) {
    if (problem_->get_terminalModel()->get_g_lb()[k] == -infty &&
        problem_->get_terminalModel()->get_g_ub()[k] == infty) {
      rho_vec_.back()[k] = settings_.rho_min;
      inv_rho_vec_.back()[k] = 1. / settings_.rho_min;
    } else if (abs(problem_->get_terminalModel()->get_g_lb()[k] -
                   problem_->get_terminalModel()->get_g_ub()[k]) <= 1e-6) {
      rho_vec_.back()[k] = 1e3 * rho_sparse_tmp;
      inv_rho_vec_.back()[k] = 1. / (1e3 * rho_sparse_tmp);
    } else if (problem_->get_terminalModel()->get_g_lb()[k] <
               problem_->get_terminalModel()->get_g_ub()[k]) {
      rho_vec_.back()[k] = rho_sparse_tmp;
      inv_rho_vec_.back()[k] = 1. / rho_sparse_tmp;
    }
  }
}

void ConstrainedSqp::checkKKTConditions() {
  KKT_ = 0.;
  const std::size_t T = problem_->get_T();
  x_grad_norm_ = 0.;
  u_grad_norm_ = 0.;

  for (std::size_t t = 0; t < T + 1; ++t) {
    lag_mul_[t] = Vx_[t];
    lag_mul_[t].noalias() += Vxx_[t] * dxtilde_[t];
  }

  const std::size_t ndx = problem_->get_ndx();
  const std::vector<std::shared_ptr<ActionData>> &datas =
      problem_->get_runningDatas();

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionData> &d = datas[t];
    tmp_vec_x_ = d->Lx;
    tmp_vec_x_.noalias() += d->Fx.transpose() * lag_mul_[t + 1];
    tmp_vec_x_ -= lag_mul_[t];
    if (t > 0) {
      tmp_vec_x_.noalias() += d->Gx.transpose() * y_[t];
    }
    KKT_ = std::max(KKT_, tmp_vec_x_.lpNorm<Eigen::Infinity>());

    tmp_vec_u_[t] = d->Lu;
    tmp_vec_u_[t].noalias() += d->Fu.transpose() * lag_mul_[t + 1];
    tmp_vec_u_[t].noalias() += d->Gu.transpose() * y_[t];
    KKT_ = std::max(KKT_, tmp_vec_u_[t].lpNorm<Eigen::Infinity>());

    fs_flat_.segment(t * ndx, ndx) = fs_[t];
    x_grad_norm_ += dxtilde_[t].lpNorm<1>();
    u_grad_norm_ += dutilde_[t].lpNorm<1>();
  }
  fs_flat_.tail(ndx) = fs_.back();

  const std::shared_ptr<ActionData> &d_ter = problem_->get_terminalData();
  tmp_vec_x_ = d_ter->Lx;
  tmp_vec_x_ -= lag_mul_.back();
  tmp_vec_x_.noalias() += d_ter->Gx.transpose() * y_.back();
  KKT_ = std::max(KKT_, tmp_vec_x_.lpNorm<Eigen::Infinity>());

  KKT_ = std::max(KKT_, fs_flat_.lpNorm<Eigen::Infinity>());
  KKT_ = std::max(KKT_, constraint_norm_);
  x_grad_norm_ += dxtilde_.back().lpNorm<1>();
  x_grad_norm_ = x_grad_norm_ / static_cast<double>(T + 1);
  u_grad_norm_ = u_grad_norm_ / static_cast<double>(T);
}

void ConstrainedSqp::forwardPass(const double /*stepLength*/) {
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionData>> &datas =
      problem_->get_runningDatas();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionData> &d = datas[t];

    dutilde_[t] = -k_[t];
    dutilde_[t].noalias() -= K_[t] * dxtilde_[t];
    dxtilde_[t + 1] = fs_[t + 1];
    dxtilde_[t + 1].noalias() += d->Fx * dxtilde_[t];
    dxtilde_[t + 1].noalias() += d->Fu * dutilde_[t];
  }
}

void ConstrainedSqp::forwardPass_without_constraints() {
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionData>> &datas =
      problem_->get_runningDatas();

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionData> &d = datas[t];

    du_[t] = -k_[t];
    du_[t].noalias() -= K_[t] * dx_[t];
    dx_[t + 1] = fs_[t + 1];
    dx_[t + 1].noalias() += d->Fx * dx_[t];
    dx_[t + 1].noalias() += d->Fu * du_[t];
  }
}

void ConstrainedSqp::backwardPass() {
  const std::shared_ptr<ActionData> &d_T = problem_->get_terminalData();

  Vxx_.back() = d_T->Lxx;
  Vxx_.back().diagonal().array() += settings_.sigma;
  Vx_.back() = d_T->Lx;
  Vx_.back().noalias() -= settings_.sigma * dx_.back();

  if (problem_->get_terminalModel()->get_ng()) {
    tmp_rhoGx_mat_.back().noalias() = rho_vec_.back().asDiagonal() * d_T->Gx;
    Vxx_.back().noalias() += d_T->Gx.transpose() * tmp_rhoGx_mat_.back();
    tmp_dual_cwise_.back() =
        y_.back() - rho_vec_.back().cwiseProduct(z_.back());
    Vx_.back().noalias() += d_T->Gx.transpose() * tmp_dual_cwise_.back();
  }
  if (!std::isnan(preg_)) {
    Vxx_.back().diagonal().array() += preg_;
  }

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModel>> &models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionData>> &datas =
      problem_->get_runningDatas();

  for (int t = static_cast<int>(T - 1); t >= 0; --t) {
    const std::shared_ptr<ActionModel> &m = models[t];
    const std::shared_ptr<ActionData> &d = datas[t];
    const Eigen::MatrixXd &Vxx_p = Vxx_[t + 1];

    Vxx_fs_[t].noalias() = Vxx_[t + 1] * fs_[t + 1];
    tmp_Vx_ = Vxx_fs_[t] + Vx_[t + 1];

    const std::size_t nu = m->get_nu();
    const std::size_t nc = m->get_ng();
    FxTVxx_p_.noalias() = d->Fx.transpose() * Vxx_p;
    Qx_[t] = d->Lx;
    Qx_[t].noalias() -= settings_.sigma * dx_[t];
    if (nc != 0) {
      if (t > 0 || nu != 0) {
        tmp_dual_cwise_[t] = y_[t];
        tmp_dual_cwise_[t].noalias() -= rho_vec_[t].cwiseProduct(z_[t]);
      }
      if (t > 0) {
        Qx_[t].noalias() += d->Gx.transpose() * tmp_dual_cwise_[t];
      }
    }

    Qx_[t].noalias() += d->Fx.transpose() * tmp_Vx_;

    Qxx_[t] = d->Lxx;
    Qxx_[t].diagonal().array() += settings_.sigma;
    if (t > 0 && nc != 0) {
      tmp_rhoGx_mat_[t].noalias() = rho_vec_[t].asDiagonal() * d->Gx;
      Qxx_[t].noalias() += d->Gx.transpose() * tmp_rhoGx_mat_[t];
    }
    Qxx_[t].noalias() += FxTVxx_p_ * d->Fx;

    if (nu != 0) {
      FuTVxx_p_[t].noalias() = d->Fu.transpose() * Vxx_p;
      Qu_[t] = d->Lu - settings_.sigma * du_[t];
      if (nc != 0) {
        Qu_[t].noalias() += d->Gu.transpose() * tmp_dual_cwise_[t];
      }
      Qu_[t].noalias() += d->Fu.transpose() * tmp_Vx_;

      Quu_[t] = d->Luu;
      Quu_[t].diagonal().array() += settings_.sigma;
      Quu_[t].noalias() += FuTVxx_p_[t] * d->Fu;
      if (nc != 0) {
        tmp_rhoGu_mat_[t].noalias() = rho_vec_[t].asDiagonal() * d->Gu;
        Quu_[t].noalias() += d->Gu.transpose() * tmp_rhoGu_mat_[t];
      }
      if (!std::isnan(dreg_)) {
        Quu_[t].diagonal().array() += dreg_;
      }

      Qxu_[t] = d->Lxu;
      if (t > 0 && nc != 0) {
        Qxu_[t].noalias() += d->Gx.transpose() * tmp_rhoGu_mat_[t];
      }
      Qxu_[t].noalias() += FxTVxx_p_ * d->Fu;
    }
    computeGains(t);
    Vx_[t] = Qx_[t];
    Vxx_[t] = Qxx_[t];
    if (nu != 0) {
      // Quuk_[t].noalias() = Quu_[t] * k_[t];
      Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
      Vxx_[t].noalias() -= Qxu_[t] * K_[t];
    }
    Vxx_tmp_ = 0.5 * (Vxx_[t] + Vxx_[t].transpose());
    Vxx_[t] = Vxx_tmp_;
    if (!std::isnan(preg_)) {
      Vxx_[t].diagonal().array() += preg_;
    }
  }
}

void ConstrainedSqp::backwardPass_without_constraints() {
  const std::shared_ptr<ActionData> &d_T = problem_->get_terminalData();

  Vxx_.back() = d_T->Lxx;
  Vx_.back() = d_T->Lx;

  if (!std::isnan(preg_)) {
    Vxx_.back().diagonal().array() += preg_;
  }

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModel>> &models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionData>> &datas =
      problem_->get_runningDatas();

  for (int t = static_cast<int>(T - 1); t >= 0; --t) {
    const std::shared_ptr<ActionModel> &m = models[t];
    const std::shared_ptr<ActionData> &d = datas[t];
    const Eigen::MatrixXd &Vxx_p = Vxx_[t + 1];
    tmp_Vx_.noalias() = Vxx_[t + 1] * fs_[t + 1];
    tmp_Vx_ += Vx_[t + 1];

    const std::size_t nu = m->get_nu();
    FxTVxx_p_.noalias() = d->Fx.transpose() * Vxx_p;
    Qx_[t] = d->Lx;
    Qx_[t].noalias() += d->Fx.transpose() * tmp_Vx_;
    Qxx_[t] = d->Lxx;

    Qxx_[t].noalias() += FxTVxx_p_ * d->Fx;
    if (nu != 0) {
      FuTVxx_p_[t].noalias() = d->Fu.transpose() * Vxx_p;
      Qu_[t] = d->Lu;
      Qu_[t].noalias() += d->Fu.transpose() * tmp_Vx_;

      Quu_[t] = d->Luu;
      Quu_[t].noalias() += FuTVxx_p_[t] * d->Fu;
      Qxu_[t] = d->Lxu;
      Qxu_[t].noalias() += FxTVxx_p_ * d->Fu;

      if (!std::isnan(dreg_)) {
        Quu_[t].diagonal().array() += dreg_;
      }
    }

    computeGains(t);

    Vx_[t] = Qx_[t];
    Vxx_[t] = Qxx_[t];
    if (nu != 0) {
      // Quuk_[t].noalias() = Quu_[t] * k_[t];
      Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
      Vxx_[t].noalias() -= Qxu_[t] * K_[t];
    }
    Vxx_tmp_ = 0.5 * (Vxx_[t] + Vxx_[t].transpose());
    Vxx_[t] = Vxx_tmp_;

    if (!std::isnan(preg_)) {
      Vxx_[t].diagonal().array() += preg_;
    }
  }
}

void ConstrainedSqp::backwardPass_mt() {
  const std::shared_ptr<ActionData> &d_T = problem_->get_terminalData();

  Vxx_.back() = d_T->Lxx;
  Vxx_.back().diagonal().array() += settings_.sigma;
  Vx_.back() = d_T->Lx;
  Vx_.back().noalias() -= settings_.sigma * dx_.back();

  if (problem_->get_terminalModel()->get_ng()) {
    tmp_rhoGx_mat_.back().noalias() = rho_vec_.back().asDiagonal() * d_T->Gx;
    Vxx_.back().noalias() += d_T->Gx.transpose() * tmp_rhoGx_mat_.back();
    tmp_dual_cwise_.back() =
        y_.back() - rho_vec_.back().cwiseProduct(z_.back());
    Vx_.back().noalias() += d_T->Gx.transpose() * tmp_dual_cwise_.back();
  }
  if (!std::isnan(preg_)) {
    Vxx_.back().diagonal().array() += preg_;
  }

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModel>> &models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionData>> &datas =
      problem_->get_runningDatas();

#pragma omp parallel for num_threads(problem_->get_nthreads())
  for (int t = static_cast<int>(T - 1); t >= 0; --t) {
    const std::shared_ptr<ActionModel> &m = models[t];
    const std::shared_ptr<ActionData> &d = datas[t];
    const std::size_t nu = m->get_nu();
    const std::size_t nc = m->get_ng();

    Qx_[t] = d->Lx;
    Qx_[t].noalias() -= settings_.sigma * dx_[t];
    if (nc != 0) {
      if (t > 0 || nu != 0) {
        tmp_dual_cwise_[t] = y_[t];
        tmp_dual_cwise_[t].noalias() -= rho_vec_[t].cwiseProduct(z_[t]);
      }
      if (t > 0) {
        Qx_[t].noalias() += d->Gx.transpose() * tmp_dual_cwise_[t];
      }
    }

    Qxx_[t] = d->Lxx;
    Qxx_[t].diagonal().array() += settings_.sigma;
    if (t > 0 && nc != 0) {
      tmp_rhoGx_mat_[t].noalias() = rho_vec_[t].asDiagonal() * d->Gx;
      Qxx_[t].noalias() += d->Gx.transpose() * tmp_rhoGx_mat_[t];
    }

    if (nu != 0) {
      Qu_[t] = d->Lu - settings_.sigma * du_[t];
      if (nc != 0) {
        Qu_[t] += d->Gu.transpose() * tmp_dual_cwise_[t];
      }

      Quu_[t] = d->Luu;
      Quu_[t].diagonal().array() += settings_.sigma;
      if (nc != 0) {
        tmp_rhoGu_mat_[t].noalias() = rho_vec_[t].asDiagonal() * d->Gu;
        Quu_[t].noalias() += d->Gu.transpose() * tmp_rhoGu_mat_[t];
      }
      if (!std::isnan(dreg_)) {
        Quu_[t].diagonal().array() += dreg_;
      }

      Qxu_[t] = d->Lxu;
      if (t > 0 && nc != 0) {
        Qxu_[t].noalias() += d->Gx.transpose() * tmp_rhoGu_mat_[t];
      }
    }
  }

  for (int t = static_cast<int>(T - 1); t >= 0; --t) {
    const std::shared_ptr<ActionModel> &m = models[t];
    const std::shared_ptr<ActionData> &d = datas[t];
    const std::size_t nu = m->get_nu();

    const Eigen::MatrixXd &Vxx_p = Vxx_[t + 1];
    FxTVxx_p_.noalias() = d->Fx.transpose() * Vxx_p;
    Qxx_[t].noalias() += FxTVxx_p_ * d->Fx;

    Vxx_fs_[t].noalias() = Vxx_[t + 1] * fs_[t + 1];
    tmp_Vx_ = Vxx_fs_[t] + Vx_[t + 1];
    Qx_[t].noalias() += d->Fx.transpose() * tmp_Vx_;
    if (nu != 0) {
      FuTVxx_p_[0].noalias() = d->Fu.transpose() * Vxx_p;
      Quu_[t].noalias() += FuTVxx_p_[0] * d->Fu;
      Qu_[t].noalias() += d->Fu.transpose() * tmp_Vx_;
      Qxu_[t].noalias() += FxTVxx_p_ * d->Fu;
    }

    computeGains(t);
    Vx_[t] = Qx_[t];
    Vxx_[t] = Qxx_[t];
    if (nu != 0) {
      // Quuk_[t].noalias() = Quu_[t] * k_[t];
      Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];

      Vxx_[t].noalias() -= Qxu_[t] * K_[t];
    }
    // The commented version is theoretically slower.
    // Vxx_tmp_ = 0.5 * (Vxx_[t] + Vxx_[t].transpose());
    Vxx_tmp_.triangularView<Eigen::Upper>() =
        (0.5 * (Vxx_[t] + Vxx_[t].transpose())).triangularView<Eigen::Upper>();
    // Vxx_[t] = Vxx_tmp_;
    Vxx_[t] = Vxx_tmp_.selfadjointView<Eigen::Upper>();

    if (!std::isnan(preg_)) {
      Vxx_[t].diagonal().array() += preg_;
    }
  }
}

void ConstrainedSqp::backwardPass_without_rho_update() {
  const std::shared_ptr<ActionModel> &m_T = problem_->get_terminalModel();
  const std::shared_ptr<ActionData> &d_T = problem_->get_terminalData();

  Vx_.back() = d_T->Lx;
  Vx_.back().noalias() -= settings_.sigma * dx_.back();

  if (m_T->get_ng()) { // constraint model
    tmp_dual_cwise_.back() = y_.back();
    tmp_dual_cwise_.back().noalias() -= rho_vec_.back().cwiseProduct(z_.back());
    Vx_.back().noalias() += d_T->Gx.transpose() * tmp_dual_cwise_.back();
  }

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModel>> &models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionData>> &datas =
      problem_->get_runningDatas();

  for (int t = static_cast<int>(T - 1); t >= 0; --t) {
    const std::shared_ptr<ActionModel> &m = models[t];
    const std::shared_ptr<ActionData> &d = datas[t];
    const std::size_t nu = m->get_nu();
    const std::size_t nc = m->get_ng();

    tmp_Vx_ = Vxx_fs_[t] + Vx_[t + 1];
    Qx_[t] = d->Lx;
    Qx_[t].noalias() -= settings_.sigma * dx_[t];
    if (nc != 0) {
      if (t > 0 || nu != 0) {
        tmp_dual_cwise_[t] = y_[t];
        tmp_dual_cwise_[t].noalias() -= rho_vec_[t].cwiseProduct(z_[t]);
      }
      if (t > 0) {
        Qx_[t].noalias() += d->Gx.transpose() * tmp_dual_cwise_[t];
      }
    }
    Qx_[t].noalias() += d->Fx.transpose() * tmp_Vx_;

    if (nu != 0) {
      Qu_[t] = d->Lu;
      Qu_[t].noalias() -= settings_.sigma * du_[t];
      if (nc != 0) {
        Qu_[t].noalias() += d->Gu.transpose() * tmp_dual_cwise_[t];
      }
      Qu_[t].noalias() += d->Fu.transpose() * tmp_Vx_;
    }

    k_[t] = Qu_[t];
    Quu_llt_[t].solveInPlace(k_[t]);

    Vx_[t] = Qx_[t];
    if (nu != 0) {
      Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
    }
  }
}

void ConstrainedSqp::backwardPass_without_rho_update_mt() {
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModel>> &models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionData>> &datas =
      problem_->get_runningDatas();
  const std::shared_ptr<ActionModel> &m_T = problem_->get_terminalModel();
  const std::shared_ptr<ActionData> &d_T = problem_->get_terminalData();

  Vx_.back().noalias() = d_T->Lx - settings_.sigma * dx_.back();

  if (m_T->get_ng()) { // constraint model
    tmp_dual_cwise_.back().noalias() =
        y_.back() - rho_vec_.back().cwiseProduct(z_.back());
    Vx_.back().noalias() += d_T->Gx.transpose() * tmp_dual_cwise_.back();
  }

#ifdef CSQP_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif // CSQP_WITH_MULTITHREADING
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModel> &m = models[t];
    const std::shared_ptr<ActionData> &d = datas[t];
    const std::size_t nu = m->get_nu();
    const std::size_t nc = m->get_ng();

    Qx_[t].noalias() = d->Lx - settings_.sigma * dx_[t];
    if (nc != 0 && (t > 0 || nu != 0)) {
      tmp_dual_cwise_[t].noalias() = y_[t] - rho_vec_[t].cwiseProduct(z_[t]);
    }
    if (nc != 0 && t > 0) {
      Qx_[t].noalias() += d->Gx.transpose() * tmp_dual_cwise_[t];
    }

    if (nu != 0) {
      Qu_[t].noalias() = d->Lu - settings_.sigma * du_[t];
      if (nc != 0) {
        Qu_[t].noalias() += d->Gu.transpose() * tmp_dual_cwise_[t];
      }
    }
  }

  for (int t = static_cast<int>(T - 1); t >= 0; --t) {
    const std::shared_ptr<ActionModel> &m = models[t];
    const std::shared_ptr<ActionData> &d = datas[t];
    const std::size_t nu = m->get_nu();

    tmp_Vx_ = Vxx_fs_[t] + Vx_[t + 1];
    Qx_[t].noalias() += d->Fx.transpose() * tmp_Vx_;
    Vx_[t] = Qx_[t];

    if (nu != 0) {
      Qu_[t].noalias() += d->Fu.transpose() * tmp_Vx_;
      Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
    }
  }

#ifdef CSQP_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif // CSQP_WITH_MULTITHREADING
  for (std::size_t t = 0; t < T; ++t) {
    k_[t] = Qu_[t];
    Quu_llt_[t].solveInPlace(k_[t]);
  }
}

void ConstrainedSqp::update_lagrangian_parameters(const int iter) {
  norm_primal_ = -std::numeric_limits<double>::infinity();
  norm_dual_ = -std::numeric_limits<double>::infinity();
  norm_primal_rel_ = -std::numeric_limits<double>::infinity();
  norm_dual_rel_ = -std::numeric_limits<double>::infinity();

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModel>> &models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionData>> &datas =
      problem_->get_runningDatas();

#ifdef CSQP_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())                 \
    reduction(max                                                              \
              : norm_primal_, norm_dual_, norm_primal_rel_, norm_dual_rel_)
#endif // CSQP_WITH_MULTITHREADING
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModel> &m = models[t];
    const std::shared_ptr<ActionData> &d = datas[t];

    if (m->get_ng() == 0) {
      dx_[t] = dxtilde_[t];
      du_[t] = dutilde_[t];
      continue;
    }

    z_prev_[t] = z_[t];
    tmp_Cdx_Cdu_[t].noalias() = d->Gx * dxtilde_[t];
    tmp_Cdx_Cdu_[t].noalias() += d->Gu * dutilde_[t];
    z_relaxed_[t].noalias() = settings_.alpha * tmp_Cdx_Cdu_[t];
    z_relaxed_[t].noalias() += (1. - settings_.alpha) * z_[t];

    tmp_dual_cwise_[t] = y_[t].cwiseProduct(inv_rho_vec_[t]);

    z_[t] = z_relaxed_[t] + tmp_dual_cwise_[t];
    z_[t] = z_[t].cwiseMax(m->get_g_lb() - d->g).cwiseMin(m->get_g_ub() - d->g);

    y_[t] += rho_vec_[t].cwiseProduct(z_relaxed_[t] - z_[t]);

    dx_[t] = dxtilde_[t];
    du_[t] = dutilde_[t];

    if (iter % settings_.rho_update_interval == 0) {
      if (settings_.update_rho_with_heuristic) {
        tmp_dual_cwise_[t] = rho_vec_[t].cwiseProduct(z_[t] - z_prev_[t]);
        norm_dual_ =
            std::max(norm_dual_, tmp_dual_cwise_[t].lpNorm<Eigen::Infinity>());
        norm_primal_ = std::max(
            norm_primal_, (tmp_Cdx_Cdu_[t] - z_[t]).lpNorm<Eigen::Infinity>());

        norm_primal_rel_ = std::max(norm_primal_rel_,
                                    tmp_Cdx_Cdu_[t].lpNorm<Eigen::Infinity>());
        norm_primal_rel_ =
            std::max(norm_primal_rel_, z_[t].lpNorm<Eigen::Infinity>());
        norm_dual_rel_ =
            std::max(norm_dual_rel_, y_[t].lpNorm<Eigen::Infinity>());
      } else {
        tmp_dual_cwise_[t] = rho_vec_[t].cwiseProduct(z_[t] - z_prev_[t]);
        norm_dual_ = std::max(
            norm_dual_,
            (d->Gx.transpose() * tmp_dual_cwise_[t]).lpNorm<Eigen::Infinity>());
        norm_dual_ = std::max(
            norm_dual_,
            (d->Gu.transpose() * tmp_dual_cwise_[t]).lpNorm<Eigen::Infinity>());
        norm_primal_ = std::max(
            norm_primal_, (tmp_Cdx_Cdu_[t] - z_[t]).lpNorm<Eigen::Infinity>());

        norm_primal_rel_ = std::max(norm_primal_rel_,
                                    tmp_Cdx_Cdu_[t].lpNorm<Eigen::Infinity>());
        norm_primal_rel_ =
            std::max(norm_primal_rel_, z_[t].lpNorm<Eigen::Infinity>());
        norm_dual_rel_ =
            std::max(norm_dual_rel_,
                     (d->Gx.transpose() * y_[t]).lpNorm<Eigen::Infinity>());
        norm_dual_rel_ =
            std::max(norm_dual_rel_,
                     (d->Gu.transpose() * y_[t]).lpNorm<Eigen::Infinity>());
      }
    }
  }

  dx_.back() = dxtilde_.back();
  const std::shared_ptr<ActionModel> &m_T = problem_->get_terminalModel();
  const std::shared_ptr<ActionData> &d_T = problem_->get_terminalData();
  const std::size_t nc = m_T->get_ng();

  if (nc != 0) {
    z_prev_.back() = z_.back();
    tmp_Cdx_Cdu_.back().noalias() = d_T->Gx * dxtilde_.back();
    z_relaxed_.back().noalias() = settings_.alpha * tmp_Cdx_Cdu_.back();
    z_relaxed_.back().noalias() += (1. - settings_.alpha) * z_.back();

    tmp_dual_cwise_.back() = y_.back().cwiseProduct(inv_rho_vec_.back());
    z_.back() = (z_relaxed_.back() + tmp_dual_cwise_.back());
    z_.back() = z_.back()
                    .cwiseMax(m_T->get_g_lb() - d_T->g)
                    .cwiseMin(m_T->get_g_ub() - d_T->g);
    y_.back() += rho_vec_.back().cwiseProduct(z_relaxed_.back() - z_.back());

    if (iter % settings_.rho_update_interval == 0) {
      if (settings_.update_rho_with_heuristic) {
        tmp_dual_cwise_.back() =
            rho_vec_.back().cwiseProduct(z_.back() - z_prev_.back());
        norm_dual_ = std::max(norm_dual_,
                              tmp_dual_cwise_.back().lpNorm<Eigen::Infinity>());
        norm_primal_ = std::max(
            norm_primal_,
            (tmp_Cdx_Cdu_.back() - z_.back()).lpNorm<Eigen::Infinity>());

        norm_primal_rel_ = std::max(
            norm_primal_rel_, tmp_Cdx_Cdu_.back().lpNorm<Eigen::Infinity>());
        norm_primal_rel_ =
            std::max(norm_primal_rel_, z_.back().lpNorm<Eigen::Infinity>());
        norm_dual_rel_ =
            std::max(norm_dual_rel_, y_.back().lpNorm<Eigen::Infinity>());
      } else {
        tmp_dual_cwise_.back() =
            rho_vec_.back().cwiseProduct(z_.back() - z_prev_.back());
        norm_dual_ =
            std::max(norm_dual_, (d_T->Gx.transpose() * tmp_dual_cwise_.back())
                                     .lpNorm<Eigen::Infinity>());
        norm_primal_ = std::max(
            norm_primal_,
            (tmp_Cdx_Cdu_.back() - z_.back()).lpNorm<Eigen::Infinity>());

        norm_primal_rel_ = std::max(
            norm_primal_rel_, tmp_Cdx_Cdu_.back().lpNorm<Eigen::Infinity>());
        norm_primal_rel_ =
            std::max(norm_primal_rel_, z_.back().lpNorm<Eigen::Infinity>());
        norm_dual_rel_ = std::max(
            norm_dual_rel_,
            (d_T->Gx.transpose() * y_.back()).lpNorm<Eigen::Infinity>());
      }
    }
  }
}

double ConstrainedSqp::tryStep(const double steplength) {
  if (steplength > 1. || steplength < 0.) {
    throw_pretty("Invalid argument: "
                 << "invalid step length, value is between 0. to 1.");
  }

  cost_try_ = 0.;
  merit_try_ = 0.;
  gap_norm_try_ = 0.;
  constraint_norm_try_ = 0.;

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModel>> &models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionData>> &datas =
      problem_->get_runningDatas();

  dx_norm_ = 0.0;
  du_norm_ = 0.0;
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModel> &m = models[t];
    // m->get_state()->integrate(xs_[t], steplength * dx_[t], xs_try_[t]);
    xs_try_[t] = xs_[t] + steplength * dx_[t];

    const std::size_t nu = m->get_nu();

    dx_norm_ += dx_[t].squaredNorm();
    if (nu != 0) {
      us_try_[t] = us_[t] + steplength * du_[t];
      du_norm_ += du_[t].squaredNorm();
    }
  }

  const std::shared_ptr<ActionModel> &m_ter = problem_->get_terminalModel();
  const std::shared_ptr<ActionData> &d_ter = problem_->get_terminalData();

  // m_ter->get_state()->integrate(xs_.back(), steplength * dx_.back(),
  //                               xs_try_.back());
  xs_try_.back() = xs_.back() + steplength * dx_.back();

  dx_norm_ += dx_.back().squaredNorm();
  dx_norm_ = std::sqrt(dx_norm_);
  du_norm_ = std::sqrt(du_norm_);
  dx_norm_ *= steplength;
  du_norm_ *= steplength;

#ifdef CSQP_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())                 \
    reduction(max                                                              \
              : norm_primal_, norm_dual_, norm_primal_rel_, norm_dual_rel_)
#endif // CSQP_WITH_MULTITHREADING
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModel> &m = models[t];
    const std::shared_ptr<ActionData> &d = datas[t];

    m->calc(d, xs_try_[t], us_try_[t]);
    cost_try_ += d->cost;
    // m->get_state()->diff(xs_try_[t + 1], d->xnext, fs_try_[t + 1]);
    fs_try_[t + 1] = d->xnext - xs_try_[t + 1];

    gap_norm_try_ += fs_try_[t + 1].lpNorm<1>();

    const std::size_t nc = m->get_ng();
    constraint_norm_try_ +=
        (m->get_g_lb() - d->g).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();
    constraint_norm_try_ +=
        (d->g - m->get_g_ub()).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();

    if (raiseIfNaN(cost_try_)) {
      throw_pretty("step_error");
    }
  }

  // Terminal state update
  m_ter->calc(d_ter, xs_try_.back());
  cost_try_ += d_ter->cost;

  const std::size_t nc = m_ter->get_ng();

  constraint_norm_try_ += (m_ter->get_g_lb() - d_ter->g)
                              .cwiseMax(Eigen::VectorXd::Zero(nc))
                              .lpNorm<1>();
  constraint_norm_try_ += (d_ter->g - m_ter->get_g_ub())
                              .cwiseMax(Eigen::VectorXd::Zero(nc))
                              .lpNorm<1>();

  if (settings_.mu_dynamic < 0. || settings_.mu_constraint < 0.) {
    merit_try_ = cost_try_ + settings_.lag_mul_inf_norm_coef *
                                 lag_mul_inf_norm_ *
                                 (gap_norm_try_ + constraint_norm_try_);

  } else {
    merit_try_ = cost_try_ + settings_.mu_dynamic * gap_norm_try_ +
                 settings_.mu_constraint * constraint_norm_try_;
  }

  if (raiseIfNaN(cost_try_)) {
    throw_pretty("step_error");
  }

  return merit_try_;
}

void ConstrainedSqp::resizeData() {
  SolverAbstract::resizeData();

  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<std::shared_ptr<ActionModel>> &models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModel> &model = models[t];
    const std::size_t nu = model->get_nu();
    Qxu_[t].conservativeResize(ndx, nu);
    Quu_[t].conservativeResize(nu, nu);
    Qu_[t].conservativeResize(nu);
    K_[t].conservativeResize(nu, ndx);
    k_[t].conservativeResize(nu);
    us_try_[t].conservativeResize(nu);
    FuTVxx_p_[t].conservativeResize(nu, ndx);
    Quuk_[t].conservativeResize(nu);
  }
}

void ConstrainedSqp::computeGains(const std::size_t t) {
  const std::size_t nu = problem_->get_runningModels()[t]->get_nu();
  if (nu > 0) {
    Quu_llt_[t].compute(Quu_[t]);
    const Eigen::ComputationInfo &info = Quu_llt_[t].info();
    if (info != Eigen::Success) {
      throw_pretty("backward_error");
    }
    K_[t] = Qxu_[t].transpose();

    Quu_llt_[t].solveInPlace(K_[t]);
    k_[t] = Qu_[t];
    Quu_llt_[t].solveInPlace(k_[t]);
  }
}

void ConstrainedSqp::increaseRegularization() {
  preg_ *= settings_.reg_incfactor;
  if (preg_ > settings_.reg_max) {
    preg_ = settings_.reg_max;
  }
  dreg_ = preg_;
}

void ConstrainedSqp::decreaseRegularization() {
  preg_ /= settings_.reg_decfactor;
  if (preg_ < settings_.reg_min) {
    preg_ = settings_.reg_min;
  }
  dreg_ = preg_;
}

void ConstrainedSqp::allocateData() {
  const std::size_t T = problem_->get_T();
  Vxx_.resize(T + 1);
  Vx_.resize(T + 1);
  Qxx_.resize(T);
  Qxu_.resize(T);
  Quu_.resize(T);
  Qx_.resize(T);
  Qu_.resize(T);
  K_.resize(T);
  k_.resize(T);

  xs_try_.resize(T + 1);
  us_try_.resize(T);
  dx_.resize(T);

  FuTVxx_p_.resize(T);
  Quu_llt_.resize(T);
  Quuk_.resize(T);

  const std::size_t ndx = problem_->get_ndx();
  const std::vector<std::shared_ptr<ActionModel>> &models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModel> &model = models[t];
    const std::size_t nu = model->get_nu();
    Vxx_[t] = Eigen::MatrixXd::Zero(ndx, ndx);
    Vx_[t] = Eigen::VectorXd::Zero(ndx);
    Qxx_[t] = Eigen::MatrixXd::Zero(ndx, ndx);
    Qxu_[t] = Eigen::MatrixXd::Zero(ndx, nu);
    Quu_[t] = Eigen::MatrixXd::Zero(nu, nu);
    Qx_[t] = Eigen::VectorXd::Zero(ndx);
    Qu_[t] = Eigen::VectorXd::Zero(nu);
    K_[t] = MatrixXdRowMajor::Zero(nu, ndx);
    k_[t] = Eigen::VectorXd::Zero(nu);

    if (t == 0) {
      xs_try_[t] = problem_->get_x0();
    } else {
      xs_try_[t] = Eigen::VectorXd::Zero(model->get_nx());
    }

    us_try_[t] = Eigen::VectorXd::Zero(nu);
    dx_[t] = Eigen::VectorXd::Zero(ndx);

    FuTVxx_p_[t] = MatrixXdRowMajor::Zero(nu, ndx);
    Quu_llt_[t] = Eigen::LLT<Eigen::MatrixXd>(nu);
    Quuk_[t] = Eigen::VectorXd(nu);
  }
  Vxx_.back() = Eigen::MatrixXd::Zero(ndx, ndx);
  Vxx_tmp_ = Eigen::MatrixXd::Zero(ndx, ndx);
  Vx_.back() = Eigen::VectorXd::Zero(ndx);
  xs_try_.back() =
      Eigen::VectorXd::Zero(problem_->get_terminalModel()->get_nx());

  FxTVxx_p_ = MatrixXdRowMajor::Zero(ndx, ndx);
  fTVxx_p_ = Eigen::VectorXd::Zero(ndx);
}

void ConstrainedSqp::printQPCallbacks(const int iter) {
  std::cout << "Iters " << iter;
  std::cout << " norm_primal = " << std::scientific << std::setprecision(4)
            << norm_primal_;
  std::cout << " norm_primal_tol = " << std::scientific << std::setprecision(4)
            << norm_primal_tolerance_;
  std::cout << " norm_dual =  " << std::scientific << std::setprecision(4)
            << norm_dual_;
  std::cout << " norm_dual_tol = " << std::scientific << std::setprecision(4)
            << norm_dual_tolerance_;
  std::cout << std::endl;
  std::cout << std::flush;
}

void ConstrainedSqp::set_alphas(const std::vector<double> &alphas) {
  double prev_alpha = alphas[0];
  if (prev_alpha != 1.) {
    std::cerr << "Warning: alpha[0] should be 1" << std::endl;
  }
  for (std::size_t i = 1; i < alphas.size(); ++i) {
    double alpha = alphas[i];
    if (0. >= alpha) {
      throw_pretty("Invalid argument: "
                   << "alpha values has to be positive.");
    }
    if (alpha >= prev_alpha) {
      throw_pretty("Invalid argument: "
                   << "alpha values are monotonously decreasing.");
    }
    prev_alpha = alpha;
  }
  alphas_ = alphas;
}

} // namespace ocp
