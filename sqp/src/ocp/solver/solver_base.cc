#ifdef CROCODDYL_WITH_MULTITHREADING
#include <omp.h>
#endif // CROCODDYL_WITH_MULTITHREADING

#include "ocp/shooting/action_model.h"
#include "ocp/shooting/shooting.h"
#include "ocp/solver/solver_base.h"
#include "ocp/utils/exception.h"

namespace ocp {

SolverAbstract::SolverAbstract(std::shared_ptr<ShootingProblem> problem)
    : problem_(problem), is_feasible_(false), cost_(0.), merit_(0.), preg_(0.),
      dreg_(0.), steplength_(1.), th_acceptstep_(0.1), feasnorm_(LInf),
      iter_(0), tmp_feas_(0.) {
  // Allocate common data
  const std::size_t ndx = problem_->get_ndx();
  const std::size_t T = problem_->get_T();
  xs_.resize(T + 1);
  us_.resize(T);
  fs_.resize(T + 1);
  const std::vector<std::shared_ptr<ActionModel>> &models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModel> &model = models[t];
    const std::size_t nu = model->get_nu();
    xs_[t] = Eigen::VectorXd::Zero(model->get_nx());
    us_[t] = Eigen::VectorXd::Zero(nu);
    fs_[t] = Eigen::VectorXd::Zero(ndx);
  }
  xs_.back() = Eigen::VectorXd::Zero(problem_->get_terminalModel()->get_nx());
  fs_.back() = Eigen::VectorXd::Zero(ndx);
}

SolverAbstract::~SolverAbstract() {}

void SolverAbstract::resizeData() {
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModel>> &models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModel> &model = models[t];
    const std::size_t nu = model->get_nu();
    us_[t].conservativeResize(nu);
  }
}

double SolverAbstract::computeDynamicFeasibility() {
  tmp_feas_ = 0.;
  if (is_feasible_) {
    return tmp_feas_;
  }

  const std::size_t T = problem_->get_T();
  const Eigen::VectorXd &x0 = problem_->get_x0();
  const std::vector<std::shared_ptr<ActionModel>> &models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionData>> &datas =
      problem_->get_runningDatas();

  // models[0]->get_state()->diff(xs_[0], x0, fs_[0]);
  fs_[0] = x0 - xs_[0];

#ifdef CSQP_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModel> &m = models[t];
    const std::shared_ptr<ActionData> &d = datas[t];
    // m->get_state()->diff(xs_[t + 1], d->xnext, fs_[t + 1]);
    fs_[t + 1] = d->xnext - xs_[t + 1];
  }
  switch (feasnorm_) {
  case LInf:
    tmp_feas_ = std::max(tmp_feas_, fs_[0].lpNorm<Eigen::Infinity>());
    for (std::size_t t = 0; t < T; ++t) {
      tmp_feas_ = std::max(tmp_feas_, fs_[t + 1].lpNorm<Eigen::Infinity>());
    }
    break;
  case L1:
    tmp_feas_ = fs_[0].lpNorm<1>();
    for (std::size_t t = 0; t < T; ++t) {
      tmp_feas_ += fs_[t + 1].lpNorm<1>();
    }
    break;
  }

  return tmp_feas_;
}

double SolverAbstract::computeInequalityFeasibility() {
  tmp_feas_ = 0.;
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModel>> &models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionData>> &datas =
      problem_->get_runningDatas();
  switch (feasnorm_) {
  case LInf:
    for (std::size_t t = 0; t < T; ++t) {
      if (models[t]->get_ng() > 0) {
        tmp_feas_ = std::max(tmp_feas_, datas[t]->g.lpNorm<Eigen::Infinity>());
      }
    }
    if (problem_->get_terminalModel()->get_ng() > 0) {
      tmp_feas_ = std::max(
          tmp_feas_, problem_->get_terminalData()->g.lpNorm<Eigen::Infinity>());
    }
    break;
  case L1:
    for (std::size_t t = 0; t < T; ++t) {
      if (models[t]->get_ng() > 0) {
        tmp_feas_ += datas[t]->g.lpNorm<1>();
      }
    }
    if (problem_->get_terminalModel()->get_ng() > 0) {
      tmp_feas_ += problem_->get_terminalData()->g.lpNorm<1>();
    }
    break;
  }
  return tmp_feas_;
}

void SolverAbstract::setCandidate(const std::vector<Eigen::VectorXd> &xs_warm,
                                  const std::vector<Eigen::VectorXd> &us_warm,
                                  bool is_feasible) {
  const std::size_t T = problem_->get_T();

  const std::vector<std::shared_ptr<ActionModel>> &models =
      problem_->get_runningModels();
  if (xs_warm.size() == 0) {
    for (std::size_t t = 0; t < T; ++t) {
      const std::shared_ptr<ActionModel> &model = models[t];
      xs_[t] = Eigen::VectorXd::Zero(model->get_nx());
    }
    xs_.back() = Eigen::VectorXd::Zero(problem_->get_terminalModel()->get_nx());
  } else {
    if (xs_warm.size() != T + 1) {
      throw_pretty("Warm start state vector has wrong dimension, got "
                   << xs_warm.size() << " expecting " << (T + 1));
    }
    for (std::size_t t = 0; t < T; ++t) {
      const std::size_t nx = models[t]->get_nx();
      if (static_cast<std::size_t>(xs_warm[t].size()) != nx) {
        throw_pretty("Invalid argument: "
                     << "xs_init[" + std::to_string(t) +
                            "] has wrong dimension ("
                     << xs_warm[t].size()
                     << " provided - it should be equal to " +
                            std::to_string(nx) + ").");
      }
    }
    const std::size_t nx = problem_->get_terminalModel()->get_nx();
    if (static_cast<std::size_t>(xs_warm[T].size()) != nx) {
      throw_pretty("Invalid argument: "
                   << "xs_init[" + std::to_string(T) +
                          "] (terminal state) has wrong dimension ("
                   << xs_warm[T].size()
                   << " provided - it should be equal to " +
                          std::to_string(nx) + ").");
    }
    std::copy(xs_warm.begin(), xs_warm.end(), xs_.begin());
  }

  if (us_warm.size() == 0) {
    for (std::size_t t = 0; t < T; ++t) {
      const std::shared_ptr<ActionModel> &model = models[t];
      const std::size_t nu = model->get_nu();
      us_[t] = Eigen::VectorXd::Zero(nu);
    }
  } else {
    if (us_warm.size() != T) {
      throw_pretty("Warm start control has wrong dimension, got "
                   << us_warm.size() << " expecting " << T);
    }
    for (std::size_t t = 0; t < T; ++t) {
      const std::shared_ptr<ActionModel> &model = models[t];
      const std::size_t nu = model->get_nu();
      if (static_cast<std::size_t>(us_warm[t].size()) != nu) {
        throw_pretty(
            "Invalid argument: "
            << "us_init[" + std::to_string(t) + "] has wrong dimension ("
            << us_warm[t].size()
            << " provided - it should be equal to " + std::to_string(nu) + ")");
      }
    }
    std::copy(us_warm.begin(), us_warm.end(), us_.begin());
  }
  is_feasible_ = is_feasible;
}

const std::shared_ptr<ShootingProblem> &SolverAbstract::get_problem() const {
  return problem_;
}

const std::vector<Eigen::VectorXd> &SolverAbstract::get_xs() const {
  return xs_;
}

const std::vector<Eigen::VectorXd> &SolverAbstract::get_us() const {
  return us_;
}

const std::vector<Eigen::VectorXd> &SolverAbstract::get_fs() const {
  return fs_;
}

bool SolverAbstract::get_is_feasible() const { return is_feasible_; }

double SolverAbstract::get_cost() const { return cost_; }

double SolverAbstract::get_merit() const { return merit_; }

double SolverAbstract::get_preg() const { return preg_; }

double SolverAbstract::get_dreg() const { return dreg_; }

double SolverAbstract::get_steplength() const { return steplength_; }

double SolverAbstract::get_th_acceptstep() const { return th_acceptstep_; }

SolverAbstract::FeasibilityNorm SolverAbstract::get_feasnorm() const {
  return feasnorm_;
}

std::size_t SolverAbstract::get_iter() const { return iter_; }

void SolverAbstract::set_xs(const std::vector<Eigen::VectorXd> &xs) {
  const std::size_t T = problem_->get_T();
  if (xs.size() != T + 1) {
    throw_pretty("Invalid argument: "
                 << "xs list has to be of length " + std::to_string(T + 1));
  }

  const std::size_t nx = problem_->get_nx();
  for (std::size_t t = 0; t < T; ++t) {
    if (static_cast<std::size_t>(xs[t].size()) != nx) {
      throw_pretty("Invalid argument: "
                   << "xs[" + std::to_string(t) + "] has wrong dimension ("
                   << xs[t].size()
                   << " provided - it should be " + std::to_string(nx) + ")")
    }
  }
  if (static_cast<std::size_t>(xs[T].size()) != nx) {
    throw_pretty("Invalid argument: "
                 << "xs[" + std::to_string(T) +
                        "] (terminal state) has wrong dimension ("
                 << xs[T].size()
                 << " provided - it should be " + std::to_string(nx) + ")")
  }
  xs_ = xs;
}

void SolverAbstract::set_us(const std::vector<Eigen::VectorXd> &us) {
  const std::size_t T = problem_->get_T();
  if (us.size() != T) {
    throw_pretty("Invalid argument: "
                 << "us list has to be of length " + std::to_string(T));
  }

  const std::vector<std::shared_ptr<ActionModel>> &models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModel> &model = models[t];
    const std::size_t nu = model->get_nu();
    if (static_cast<std::size_t>(us[t].size()) != nu) {
      throw_pretty("Invalid argument: "
                   << "us[" + std::to_string(t) + "] has wrong dimension ("
                   << us[t].size()
                   << " provided - it should be " + std::to_string(nu) + ")")
    }
  }
  us_ = us;
}

void SolverAbstract::set_preg(const double preg) {
  if (preg < 0.) {
    throw_pretty("Invalid argument: "
                 << "preg value has to be positive.");
  }
  preg_ = preg;
}

void SolverAbstract::set_dreg(const double dreg) {
  if (dreg < 0.) {
    throw_pretty("Invalid argument: "
                 << "dreg value has to be positive.");
  }
  dreg_ = dreg;
}

void SolverAbstract::set_th_acceptstep(const double th_acceptstep) {
  if (0. >= th_acceptstep || th_acceptstep > 1) {
    throw_pretty("Invalid argument: "
                 << "th_acceptstep value should between 0 and 1.");
  }
  th_acceptstep_ = th_acceptstep;
}

void SolverAbstract::set_feasnorm(const FeasibilityNorm feasnorm) {
  feasnorm_ = feasnorm;
}

} // namespace ocp
