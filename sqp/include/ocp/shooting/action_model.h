#pragma once

#include <Eigen/Core>
#include <memory>

namespace ocp {

// forward declaration
class CostCollection;
class ConstraintCollection;
class DynamicsAbstract;

struct ActionData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <class Model>
  explicit ActionData(const Model *const model)
      : cost(0.0), xnext(model->get_nx()),
        Fx(model->get_ndx(), model->get_ndx()),
        Fu(model->get_ndx(), model->get_nu()), Lx(model->get_ndx()),
        Lu(model->get_nu()), Lxx(model->get_ndx(), model->get_ndx()),
        Lxu(model->get_ndx(), model->get_nu()),
        Luu(model->get_nu(), model->get_nu()), g(model->get_ng()),
        Gx(model->get_ng(), model->get_ndx()),
        Gu(model->get_ng(), model->get_nu()) {
    xnext.setZero();
    Fx.setZero();
    Fu.setZero();
    Lx.setZero();
    Lu.setZero();
    Lxx.setZero();
    Lxu.setZero();
    Luu.setZero();
    g.setZero();
    Gx.setZero();
    Gu.setZero();
  }

  virtual ~ActionData() {}

  // clang-format off
  double cost;           //!< cost value
  Eigen::VectorXd xnext; //!< evolution state
  Eigen::MatrixXd Fx;    //!< Jacobian of the dynamics w.r.t. the state $\mathbf{x}$
  Eigen::MatrixXd Fu;    //!< Jacobian of the dynamics w.r.t. the control $\mathbf{u}$
  Eigen::VectorXd Lx;    //!< Jacobian of the cost w.r.t. the state $\mathbf{x}$
  Eigen::VectorXd Lu;    //!< Jacobian of the cost w.r.t. the control $\mathbf{u}$
  Eigen::MatrixXd Lxx;   //!< Hessian of the cost w.r.t. the state $\mathbf{x}$
  Eigen::MatrixXd Lxu;   //!< Hessian of the cost w.r.t. the state $\mathbf{x}$ and control $\mathbf{u}$
  Eigen::MatrixXd Luu;   //!< Hessian of the cost w.r.t. the control $\mathbf{u}$
  Eigen::VectorXd g;     //!< Inequality constraint values
  Eigen::MatrixXd Gx;    //!< Jacobian of the inequality constraint w.r.t. the state $\mathbf{x}$
  Eigen::MatrixXd Gu;    //!< Jacobian of the inequality constraint w.r.t. the control $\mathbf{u}$
  // clang-format on
};

class ActionModel {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the action model
   *
   * @param[in] nx     Dimension of state vector
   * @param[in] ndx     Dimension of state rate vector
   * @param[in] nu     Dimension of control vector
   */
  ActionModel(std::unique_ptr<CostCollection> costs,
              std::unique_ptr<ConstraintCollection> constraints,
              std::shared_ptr<DynamicsAbstract> dynamics, const std::size_t nx,
              const std::size_t ndx, const std::size_t nu);

  virtual ~ActionModel();

  /**
   * @brief Compute the next state and cost value
   *
   * @param[in] data  Action data
   * @param[in] x     State point $\mathbf{x}\in\mathbb{R}^{ndx}$
   * @param[in] u     Control input $\mathbf{u}\in\mathbb{R}^{nu}$
   */
  virtual void calc(const std::shared_ptr<ActionData> &data,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u);

  /**
   * @brief Compute the total cost value for nodes that depends only on the
   * state
   *
   * It updates the total cost and the next state is not computed as it is not
   * expected to change. This function is used in the terminal nodes of an
   * optimal control problem.
   *
   * @param[in] data  Action data
   * @param[in] x     State point $\mathbf{x}\in\mathbb{R}^{ndx}$
   */
  virtual void calc(const std::shared_ptr<ActionData> &data,
                    const Eigen::Ref<const Eigen::VectorXd> &x);
  /**
   * @brief Compute the derivatives of the dynamics and cost functions
   *
   * It computes the partial derivatives of the dynamical system and the cost
   * function. It assumes that `calc()` has been run first. This function
   * builds a linear-quadratic approximation of the action model (i.e.
   * dynamical system and cost function).
   *
   * @param[in] data  Action data
   * @param[in] x     State point $\mathbf{x}\in\mathbb{R}^{ndx}$
   * @param[in] u     Control input $\mathbf{u}\in\mathbb{R}^{nu}$
   */
  virtual void calcDiff(const std::shared_ptr<ActionData> &data,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u);
  /**
   * @brief Compute the derivatives of the cost functions with respect to the
   * state only
   *
   * It updates the derivatives of the cost function with respect to the state
   * only. This function is used in the terminal nodes of an optimal control
   * problem.
   *
   * @param[in] data  Action data
   * @param[in] x     State point $\mathbf{x}\in\mathbb{R}^{ndx}$
   */
  virtual void calcDiff(const std::shared_ptr<ActionData> &data,
                        const Eigen::Ref<const Eigen::VectorXd> &x);

  /**
   * @brief Create the action data
   *
   * @return the action data
   */
  virtual std::shared_ptr<ActionData> createData();

  /**
   * @brief Checks that a specific data belongs to this model
   */
  virtual bool checkData(const std::shared_ptr<ActionData> &data);

  /**
   * @brief Return the dimension of the state
   */
  std::size_t get_nx() const;

  /**
   * @brief Return the dimension of the state rate
   */
  std::size_t get_ndx() const;

  /**
   * @brief Return the dimension of the control input
   */
  std::size_t get_nu() const;

  /**
   * @brief Return the number of inequality constraints
   */
  virtual std::size_t get_ng() const;

  /**
   * @brief Return the lower bound of the inequality constraints
   */
  virtual const Eigen::VectorXd &get_g_lb() const;

  /**
   * @brief Return the upper bound of the inequality constraints
   */
  virtual const Eigen::VectorXd &get_g_ub() const;

  /**
   * @brief Modify the lower bound of the inequality constraints
   */
  void set_g_lb(const Eigen::VectorXd &g_lb);

  /**
   * @brief Modify the upper bound of the inequality constraints
   */
  void set_g_ub(const Eigen::VectorXd &g_ub);

  const std::shared_ptr<DynamicsAbstract> &get_dynamics() const {
    return dynamics_ptr_;
  }

  const std::unique_ptr<CostCollection> &get_costs() const {
    return costs_ptr_;
  }

  const std::unique_ptr<ConstraintCollection> &get_constraints() const {
    return constraints_ptr_;
  }

protected:
  // clang-format off
  std::unique_ptr<CostCollection> costs_ptr_;             //!< Costs pointer
  std::unique_ptr<ConstraintCollection> constraints_ptr_; //!< Constraints pointer
  std::shared_ptr<DynamicsAbstract> dynamics_ptr_;        //!< System dynamics pointer

  std::size_t nx_;         //!< State dimension
  std::size_t ndx_;        //!< State rate dimension
  std::size_t nu_;         //!< Control dimension
  std::size_t ng_;         //!< Number of inequality constraints
  Eigen::VectorXd unone_;  //!< Neutral state
  Eigen::VectorXd g_lb_;   //!< Lower bound of the inequality constraints
  Eigen::VectorXd g_ub_;   //!< Lower bound of the inequality constraints
  // clang-format on
};

} // namespace ocp
