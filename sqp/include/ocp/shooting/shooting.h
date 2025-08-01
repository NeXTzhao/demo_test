#pragma once

#define CSQP_WITH_MULTITHREADING 1

#include <Eigen/Core>
#include <memory>

namespace ocp {

// forward declaration
class ActionData;
class ActionModel;

/**
 * @brief This class encapsulates a shooting problem
 *
 * A shooting problem encapsulates the initial state
 * $\mathbf{x}_{0}\in\mathcal{M}$, a set of running action models and a
 * terminal action model for a discretized trajectory into $T$ nodes. It has
 * three main methods - `calc`, `calcDiff` and `rollout`. The first computes the
 * set of next states and cost values per each node $k$. Instead, `calcDiff`
 * updates the derivatives of all action models. Finally, `rollout` integrates
 * the system dynamics. This class is used to decouple problem formulation and
 * resolution.
 */
class ShootingProblem {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the shooting problem and allocate its data
   *
   * @param[in] x0              Initial state
   * @param[in] running_models  Running action models (size $T$)
   * @param[in] terminal_model  Terminal action model
   */
  ShootingProblem(Eigen::VectorXd x0,
                  std::vector<std::shared_ptr<ActionModel>> running_models,
                  std::shared_ptr<ActionModel> terminal_model);

  /**
   * @brief Initialize the shooting problem (models and datas)
   *
   * @param[in] x0              Initial state
   * @param[in] running_models  Running action models (size $T$)
   * @param[in] terminal_model  Terminal action model
   * @param[in] running_datas   Running action datas (size $T$)
   * @param[in] terminal_data   Terminal action data
   */
  ShootingProblem(Eigen::VectorXd x0,
                  std::vector<std::shared_ptr<ActionModel>> running_models,
                  std::shared_ptr<ActionModel> terminal_model,
                  std::vector<std::shared_ptr<ActionData>> running_datas,
                  std::shared_ptr<ActionData> terminal_data);

  /**
   * @brief Initialize the shooting problem
   */
  ShootingProblem(const ShootingProblem &problem);

  ~ShootingProblem();

  /**
   * @brief Compute the cost and the next states
   *
   * For each node $k$, and along the state $\mathbf{x_{s}}$ and control
   * $\mathbf{u_{s}}$ trajectory, it computes the next state
   * $\mathbf{x}_{k+1}$ and cost $l_{k}$.
   *
   * @param[in] xs  time-discrete state trajectory $\mathbf{x_{s}}$ (size
   * $T+1$)
   * @param[in] us  time-discrete control sequence $\mathbf{u_{s}}$ (size
   * $T$)
   * @return The total cost value $l_{k}$
   */
  double calc(const std::vector<Eigen::VectorXd> &xs,
              const std::vector<Eigen::VectorXd> &us);

  /**
   * @brief Compute the derivatives of the cost and dynamics
   *
   * For each node $k$, and along the state $\mathbf{x_{s}}$ and control
   * $\mathbf{u_{s}}$ trajectory, it computes the derivatives of the cost
   * $(\mathbf{l}_{\mathbf{x}}, \mathbf{l}_{\mathbf{u}},
   * \mathbf{l}_{\mathbf{xx}}, \mathbf{l}_{\mathbf{xu}},
   * \mathbf{l}_{\mathbf{uu}})$ and dynamics $(\mathbf{f}_{\mathbf{x}},
   * \mathbf{f}_{\mathbf{u}})$.
   *
   * @param[in] xs  time-discrete state trajectory $\mathbf{x_{s}}$ (size
   * $T+1$)
   * @param[in] us  time-discrete control sequence $\mathbf{u_{s}}$ (size
   * $T$)
   * @return The total cost value $l_{k}$
   */
  double calcDiff(const std::vector<Eigen::VectorXd> &xs,
                  const std::vector<Eigen::VectorXd> &us);

  /**
   * @brief Integrate the dynamics given a control sequence
   *
   * @param[in] xs  time-discrete state trajectory $\mathbf{x_{s}}$ (size
   * $T+1$)
   * @param[in] us  time-discrete control sequence $\mathbf{u_{s}}$ (size
   * $T$)
   */
  void rollout(const std::vector<Eigen::VectorXd> &us,
               std::vector<Eigen::VectorXd> &xs);

  /**
   * @copybrief rollout
   *
   * @param[in] us  time-discrete control sequence $\mathbf{u_{s}}$ (size
   * $T$)
   * @return the time-discrete state trajectory $\mathbf{x_{s}}$ (size
   * $T+1$)
   */
  std::vector<Eigen::VectorXd>
  rollout_us(const std::vector<Eigen::VectorXd> &us);

  /**
   * @brief Update the model and data for a specific node
   *
   * @param[in] i      node index $(0\leq i \lt T+1)$
   * @param[in] model  action model
   * @param[in] data   action data
   */
  void updateNode(const std::size_t i, std::shared_ptr<ActionModel> model,
                  std::shared_ptr<ActionData> data);

  /**
   * @brief Update a model and allocated new data for a specific node
   *
   * @param[in] i      node index $(0\leq i \lt T+1)$
   * @param[in] model  action model
   */
  void updateModel(const std::size_t i, std::shared_ptr<ActionModel> model);

  /**
   * @brief Return the number of running nodes
   */
  std::size_t get_T() const;

  /**
   * @brief Return the initial state
   */
  const Eigen::VectorXd &get_x0() const;

  /**
   * @brief Return the running models
   */
  const std::vector<std::shared_ptr<ActionModel>> &get_runningModels() const;

  /**
   * @brief Return the terminal model
   */
  const std::shared_ptr<ActionModel> &get_terminalModel() const;

  /**
   * @brief Return the running datas
   */
  const std::vector<std::shared_ptr<ActionData>> &get_runningDatas() const;

  /**
   * @brief Return the terminal data
   */
  const std::shared_ptr<ActionData> &get_terminalData() const;

  /**
   * @brief Modify the initial state
   */
  void set_x0(Eigen::VectorXd x0_in);

  /**
   * @brief Modify the running models and allocate new data
   */
  void set_runningModels(std::vector<std::shared_ptr<ActionModel>> models);

  /**
   * @brief Modify the terminal model and allocate new data
   */
  void set_terminalModel(std::shared_ptr<ActionModel> model);

  /**
   * @brief Modify the number of threads using with multithreading support
   *
   * For values lower than 1, the number of threads is chosen by
   * CROCODDYL_WITH_NTHREADS macro
   */
  void set_nthreads(const int nthreads);

  /**
   * @brief Return the dimension of the state tuple
   */
  std::size_t get_nx() const;

  /**
   * @brief Return the dimension of the tangent space of the state manifold
   */
  std::size_t get_ndx() const;

  /**
   * @brief Return the maximum dimension of the control vector
   */
  std::size_t get_nu_max() const;

  /**
   * @brief Return the number of threads
   */
  std::size_t get_nthreads() const;
  /**
   * @brief Return only once true is the shooting problem has been changed,
   * otherwise false
   */
  bool is_updated();

  // clang-format off
 protected:
  double cost_;         //!< Total cost
  std::size_t T_;       //!< number of running nodes
  Eigen::VectorXd x0_;  //!< Initial state
  std::shared_ptr<ActionModel> terminal_model_;  //!< Terminal action model
  std::shared_ptr<ActionData> terminal_data_;    //!< Terminal action data
  std::vector<std::shared_ptr<ActionModel>> running_models_;  //!< Running action model
  std::vector<std::shared_ptr<ActionData>> running_datas_;    //!< Running action data
  std::size_t nx_;        //!< State dimension
  std::size_t ndx_;       //!< State rate dimension
  std::size_t nu_max_;    //!< Maximum control dimension
  std::size_t nthreads_;  //!< Number of threads launch by the multi-threading
                          //!< application
  bool is_updated_;
  // clang-format on

private:
  void allocateData();
};

} // namespace ocp
