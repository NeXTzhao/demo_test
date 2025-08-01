#ifdef CSQP_WITH_MULTITHREADING
#include <omp.h>
#endif // CSQP_WITH_MULTITHREADING

#include "ocp/shooting/action_model.h"
#include "ocp/shooting/shooting.h"
#include "ocp/utils/exception.h"

namespace ocp {

ShootingProblem::ShootingProblem(
    Eigen::VectorXd x0,
    std::vector<std::shared_ptr<ActionModel>> running_models,
    std::shared_ptr<ActionModel> terminal_model)
    : cost_(0.0), T_(running_models.size()), x0_(std::move(x0)),
      terminal_model_(std::move(terminal_model)),
      running_models_(std::move(running_models)),
      nx_(running_models_[0]->get_nx()), ndx_(running_models_[0]->get_ndx()),
      nu_max_(running_models_[0]->get_nu()), nthreads_(1), is_updated_(false) {
  for (std::size_t i = 1; i < T_; ++i) {
    const std::shared_ptr<ActionModel> &model = running_models_[i];
    const std::size_t nu = model->get_nu();
    if (nu_max_ < nu) {
      nu_max_ = nu;
    }
  }
  if (static_cast<std::size_t>(x0_.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x0 has wrong dimension (it should be " +
                        std::to_string(nx_) + ")");
  }
  for (std::size_t i = 1; i < T_; ++i) {
    const std::shared_ptr<ActionModel> &model = running_models_[i];
    if (model->get_nx() != nx_) {
      throw_pretty("Invalid argument: "
                   << "nx in " << i
                   << " node is not consistent with the other nodes")
    }
    if (model->get_ndx() != ndx_) {
      throw_pretty("Invalid argument: "
                   << "ndx in " << i
                   << " node is not consistent with the other nodes")
    }
  }
  if (terminal_model_->get_nx() != nx_) {
    throw_pretty(
        "Invalid argument: "
        << "nx in terminal node is not consistent with the other nodes")
  }
  if (terminal_model_->get_ndx() != ndx_) {
    throw_pretty(
        "Invalid argument: "
        << "ndx in terminal node is not consistent with the other nodes")
  }
  allocateData();
}

ShootingProblem::ShootingProblem(
    Eigen::VectorXd x0,
    std::vector<std::shared_ptr<ActionModel>> running_models,
    std::shared_ptr<ActionModel> terminal_model,
    std::vector<std::shared_ptr<ActionData>> running_datas,
    std::shared_ptr<ActionData> terminal_data)
    : cost_(0.0), T_(running_models.size()), x0_(std::move(x0)),
      terminal_model_(std::move(terminal_model)),
      terminal_data_(std::move(terminal_data)),
      running_models_(std::move(running_models)),
      running_datas_(std::move(running_datas)),
      nx_(running_models_[0]->get_nx()), ndx_(running_models_[0]->get_ndx()),
      nu_max_(running_models_[0]->get_nu()), nthreads_(1) {
  for (std::size_t i = 1; i < T_; ++i) {
    const std::shared_ptr<ActionModel> &model = running_models_[i];
    const std::size_t nu = model->get_nu();
    if (nu_max_ < nu) {
      nu_max_ = nu;
    }
  }
  if (static_cast<std::size_t>(x0_.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x0 has wrong dimension (it should be " +
                        std::to_string(nx_) + ")");
  }
  const std::size_t Td = running_datas_.size();
  if (Td != T_) {
    throw_pretty(
        "Invalid argument: "
        << "the number of running models and datas are not the same (" +
               std::to_string(T_) + " != " + std::to_string(Td) + ")")
  }
  for (std::size_t i = 0; i < T_; ++i) {
    const std::shared_ptr<ActionModel> &model = running_models_[i];
    const std::shared_ptr<ActionData> &data = running_datas_[i];
    if (model->get_nx() != nx_) {
      throw_pretty("Invalid argument: "
                   << "nx in " << i
                   << " node is not consistent with the other nodes")
    }
    if (model->get_ndx() != ndx_) {
      throw_pretty("Invalid argument: "
                   << "ndx in " << i
                   << " node is not consistent with the other nodes")
    }
    if (!model->checkData(data)) {
      throw_pretty("Invalid argument: "
                   << "action data in " << i
                   << " node is not consistent with the action model")
    }
  }
  if (!terminal_model_->checkData(terminal_data_)) {
    throw_pretty("Invalid argument: "
                 << "terminal action data is not consistent with the terminal "
                    "action model")
  }
}

ShootingProblem::ShootingProblem(const ShootingProblem &problem)
    : cost_(0.0), T_(problem.get_T()), x0_(problem.get_x0()),
      terminal_model_(problem.get_terminalModel()),
      terminal_data_(problem.get_terminalData()),
      running_models_(problem.get_runningModels()),
      running_datas_(problem.get_runningDatas()), nx_(problem.get_nx()),
      ndx_(problem.get_ndx()), nu_max_(problem.get_nu_max()) {}

ShootingProblem::~ShootingProblem() {}

double ShootingProblem::calc(const std::vector<Eigen::VectorXd> &xs,
                             const std::vector<Eigen::VectorXd> &us) {
  if (xs.size() != T_ + 1) {
    throw_pretty("Invalid argument: "
                 << "xs has wrong dimension (it should be " +
                        std::to_string(T_ + 1) + ")");
  }
  if (us.size() != T_) {
    throw_pretty("Invalid argument: "
                 << "us has wrong dimension (it should be " +
                        std::to_string(T_) + ")");
  }

#ifdef CSQP_WITH_MULTITHREADING
#pragma omp parallel for num_threads(nthreads_)
#endif
  for (std::size_t i = 0; i < T_; ++i) {
    running_models_[i]->calc(running_datas_[i], xs[i], us[i]);
  }
  // tbb::parallel_for(0, static_cast<int>(T_), [&](int i) {
  //   running_models_[i]->calc(running_datas_[i], xs[i], us[i]);
  // });
  terminal_model_->calc(terminal_data_, xs.back());

  cost_ = 0.0;
#ifdef CSQP_WITH_MULTITHREADING
#pragma omp simd reduction(+ : cost_)
#endif
  for (std::size_t i = 0; i < T_; ++i) {
    cost_ += running_datas_[i]->cost;
  }
  cost_ += terminal_data_->cost;
  return cost_;
}

double ShootingProblem::calcDiff(const std::vector<Eigen::VectorXd> &xs,
                                 const std::vector<Eigen::VectorXd> &us) {
  if (xs.size() != T_ + 1) {
    throw_pretty("Invalid argument: "
                 << "xs has wrong dimension (it should be " +
                        std::to_string(T_ + 1) + ")");
  }
  if (us.size() != T_) {
    throw_pretty("Invalid argument: "
                 << "us has wrong dimension (it should be " +
                        std::to_string(T_) + ")");
  }

#ifdef CSQP_WITH_MULTITHREADING
#pragma omp parallel for num_threads(nthreads_)
#endif
  for (std::size_t i = 0; i < T_; ++i) {
    running_models_[i]->calcDiff(running_datas_[i], xs[i], us[i]);
  }
  terminal_model_->calcDiff(terminal_data_, xs.back());

  cost_ = 0.0;
#ifdef CSQP_WITH_MULTITHREADING
#pragma omp simd reduction(+ : cost_)
#endif
  for (std::size_t i = 0; i < T_; ++i) {
    cost_ += running_datas_[i]->cost;
  }
  cost_ += terminal_data_->cost;

  return cost_;
}

void ShootingProblem::rollout(const std::vector<Eigen::VectorXd> &us,
                              std::vector<Eigen::VectorXd> &xs) {
  if (xs.size() != T_ + 1) {
    throw_pretty("Invalid argument: "
                 << "xs has wrong dimension (it should be " +
                        std::to_string(T_ + 1) + ")");
  }
  if (us.size() != T_) {
    throw_pretty("Invalid argument: "
                 << "us has wrong dimension (it should be " +
                        std::to_string(T_) + ")");
  }

  xs[0] = x0_;
  for (std::size_t i = 0; i < T_; ++i) {
    const std::shared_ptr<ActionData> &data = running_datas_[i];
    running_models_[i]->calc(data, xs[i], us[i]);
    xs[i + 1] = data->xnext;
  }
  terminal_model_->calc(terminal_data_, xs.back());
}

std::vector<Eigen::VectorXd>
ShootingProblem::rollout_us(const std::vector<Eigen::VectorXd> &us) {
  std::vector<Eigen::VectorXd> xs;
  xs.resize(T_ + 1);
  rollout(us, xs);
  return xs;
}

void ShootingProblem::updateNode(const std::size_t i,
                                 std::shared_ptr<ActionModel> model,
                                 std::shared_ptr<ActionData> data) {
  if (i >= T_ + 1) {
    throw_pretty("Invalid argument: "
                 << "i is bigger than the allocated horizon (it should be less "
                    "than or equal to " +
                        std::to_string(T_ + 1) + ")");
  }
  if (!model->checkData(data)) {
    throw_pretty("Invalid argument: "
                 << "action data is not consistent with the action model")
  }
  if (model->get_nx() != nx_) {
    throw_pretty("Invalid argument: "
                 << "nx is not consistent with the other nodes")
  }
  if (model->get_ndx() != ndx_) {
    throw_pretty("Invalid argument: "
                 << "ndx node is not consistent with the other nodes")
  }
  is_updated_ = true;
  if (i == T_) {
    terminal_model_ = model;
    terminal_data_ = data;
  } else {
    running_models_[i] = model;
    running_datas_[i] = data;
  }
}

void ShootingProblem::updateModel(const std::size_t i,
                                  std::shared_ptr<ActionModel> model) {
  if (i >= T_ + 1) {
    throw_pretty("Invalid argument: "
                 << "i is bigger than the allocated horizon (it should be "
                    "lower than " +
                        std::to_string(T_ + 1) + ")");
  }
  if (model->get_nx() != nx_) {
    throw_pretty("Invalid argument: "
                 << "nx is not consistent with the other nodes")
  }
  if (model->get_ndx() != ndx_) {
    throw_pretty("Invalid argument: "
                 << "ndx is not consistent with the other nodes")
  }
  is_updated_ = true;
  if (i == T_) {
    terminal_model_ = std::move(model);
    terminal_data_ = terminal_model_->createData();
  } else {
    running_models_[i] = std::move(model);
    running_datas_[i] = model->createData();
  }
}

std::size_t ShootingProblem::get_T() const { return T_; }

const Eigen::VectorXd &ShootingProblem::get_x0() const { return x0_; }

const std::vector<std::shared_ptr<ActionModel>> &
ShootingProblem::get_runningModels() const {
  return running_models_;
}

const std::shared_ptr<ActionModel> &ShootingProblem::get_terminalModel() const {
  return terminal_model_;
}

const std::vector<std::shared_ptr<ActionData>> &
ShootingProblem::get_runningDatas() const {
  return running_datas_;
}

const std::shared_ptr<ActionData> &ShootingProblem::get_terminalData() const {
  return terminal_data_;
}

void ShootingProblem::set_x0(Eigen::VectorXd x0_in) {
  if (x0_in.size() != x0_.size()) {
    throw_pretty("Invalid argument: "
                 << "invalid size of x0 provided: Expected " << x0_.size()
                 << ", received " << x0_in.size());
  }
  x0_ = std::move(x0_in);
}

void ShootingProblem::set_runningModels(
    std::vector<std::shared_ptr<ActionModel>> models) {
  if (models.size() != running_models_.size()) {
    throw_pretty(
        "Invalid argument: models size is not consistent with the other models")
  }
  for (std::size_t i = 0; i < T_; ++i) {
    const std::shared_ptr<ActionModel> &model = models[i];
    if (model->get_nx() != nx_) {
      throw_pretty("Invalid argument: "
                   << "nx in " << i
                   << " node is not consistent with the other nodes")
    }
    if (model->get_ndx() != ndx_) {
      throw_pretty("Invalid argument: "
                   << "ndx in " << i
                   << " node is not consistent with the other nodes")
    }
  }
  is_updated_ = true;
  T_ = models.size();
  // running_models_.clear();
  // running_datas_.clear();
  // for (std::size_t i = 0; i < T_; ++i) {
  //   const std::shared_ptr<ActionModel> &model = running_models_[i];
  //   running_models_.push_back(model);
  //   running_datas_.push_back(model->createData());
  // }
  running_models_ = std::move(models);
  for (std::size_t i = 0; i < T_; ++i) {
    const std::shared_ptr<ActionModel> &model = running_models_[i];
    running_datas_[i] = model->createData();
  }
}

void ShootingProblem::set_terminalModel(std::shared_ptr<ActionModel> model) {
  if (model->get_nx() != nx_) {
    throw_pretty("Invalid argument: "
                 << "nx is not consistent with the other nodes")
  }
  if (model->get_ndx() != ndx_) {
    throw_pretty("Invalid argument: "
                 << "ndx is not consistent with the other nodes")
  }
  is_updated_ = true;
  terminal_model_ = std::move(model);
  terminal_data_ = terminal_model_->createData();
}

void ShootingProblem::set_nthreads(const int nthreads) {
#ifndef CSQP_WITH_MULTITHREADING
  (void)nthreads;
  std::cerr << "Warning: the number of threads won't affect the computational "
               "performance as multithreading "
               "support is not enabled."
            << std::endl;
#else
  nthreads_ = std::max(1, nthreads);
#endif
}

std::size_t ShootingProblem::get_nx() const { return nx_; }

std::size_t ShootingProblem::get_ndx() const { return ndx_; }

std::size_t ShootingProblem::get_nu_max() const { return nu_max_; }

std::size_t ShootingProblem::get_nthreads() const {
#ifndef CSQP_WITH_MULTITHREADING
  std::cerr << "Warning: the number of threads won't affect the computational "
               "performance as multithreading "
               "support is not enabled."
            << std::endl;
#endif
  return nthreads_;
}

bool ShootingProblem::is_updated() {
  const bool status = is_updated_;
  is_updated_ = false;
  return status;
}

void ShootingProblem::allocateData() {
  running_datas_.resize(T_);
  for (std::size_t i = 0; i < T_; ++i) {
    const std::shared_ptr<ActionModel> &model = running_models_[i];
    running_datas_[i] = model->createData();
  }
  terminal_data_ = terminal_model_->createData();
}

} // namespace ocp
