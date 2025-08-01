#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

namespace ocp {
// forward declaration
class ShootingProblem;

/** Different types of convergence */
enum class Convergence {
  FALSE = 0,
  KKT,
  ITERATIONS,
  SOLVETIME,
  STEPSIZE,
  METRICS,
  PRIMAL
};

/** Transforms sqp::Convergence to string */
inline std::string ConvergenceToString(const Convergence &convergence) {
  switch (convergence) {
  case Convergence::KKT:
    return "KKT beblow tolerance";
  case Convergence::ITERATIONS:
    return "Maximum number of iterations reached";
  case Convergence::SOLVETIME:
    return "Maximum solve time reached";
  case Convergence::STEPSIZE:
    return "Step size below minimum";
  case Convergence::METRICS:
    return "Cost decrease and constraint satisfaction below tolerance";
  case Convergence::PRIMAL:
    return "Primal update below tolerance";
  case Convergence::FALSE:
  default:
    return "Not Converged";
  }
  return "Not Converged";
}

/**
 * @brief Abstract class for optimal control solvers
 *
 * A solver resolves an optimal control solver of the form
 * {eqnarray*}{
 * \begin{Bmatrix}
 * 	\mathbf{x}^*_0,\cdots,\mathbf{x}^*_{T} \\
 * 	\mathbf{u}^*_0,\cdots,\mathbf{u}^*_{T-1}
 * \end{Bmatrix} =
 * \arg\min_{\mathbf{x}_s,\mathbf{u}_s} && l_T (\mathbf{x}_T) + \sum_{k=0}^{T-1}
 * l_k(\mathbf{x}_t,\mathbf{u}_t) \\
 * \operatorname{subject}\,\operatorname{to} && \mathbf{x}_0 =
 * \mathbf{\tilde{x}}_0\\
 * &&  \mathbf{x}_{k+1} = \mathbf{f}_k(\mathbf{x}_k,\mathbf{u}_k)\\
 * &&  \mathbf{x}_k\in\mathcal{X}, \mathbf{u}_k\in\mathcal{U}
 * }
 * where $l_T(\mathbf{x}_T)$, $l_k(\mathbf{x}_t,\mathbf{u}_t)$ are the
 * terminal and running cost functions, respectively,
 * $\mathbf{f}_k(\mathbf{x}_k,\mathbf{u}_k)$ describes evolution of the
 * system, and state and control admissible sets are defined by
 * $\mathbf{x}_k\in\mathcal{X}$, $\mathbf{u}_k\in\mathcal{U}$. An action
 * model, defined in the shooting problem, describes each node $k$. Inside
 * the action model, we specialize the cost functions, the system evolution and
 * the admissible sets.
 *
 * The main routines are `computeDirection()` and `tryStep()`. The former finds
 * a search direction and typically computes the derivatives of each action
 * model. The latter rollout the dynamics and cost (i.e., the action) to try the
 * search direction found by `computeDirection`. Both functions used the current
 * guess defined by `setCandidate()`. Finally, `solve()` function is used to
 * define when the search direction and length are computed in each iterate. It
 * also describes the globalization strategy (i.e., regularization) of the
 * numerical optimization.
 *
 * \sa `solve()`, `computeDirection()`, `tryStep()`, `stoppingCriteria()`
 */
class SolverAbstract {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum FeasibilityNorm { LInf = 0, L1 };

  /**
   * @brief Initialize the solver
   *
   * @param[in] problem  shooting problem
   */
  explicit SolverAbstract(std::shared_ptr<ShootingProblem> problem);
  virtual ~SolverAbstract();

  /**
   * @brief Compute the optimal trajectory $\mathbf{x}^*_s,\mathbf{u}^*_s$
   * as lists of $T+1$ and $T$ terms
   *
   * From an initial guess \p init_xs, \p init_us (feasible or not), iterate
   * over `computeDirection()` and `tryStep()` until `stoppingCriteria()` is
   * below threshold. It also describes the globalization strategy used during
   * the numerical optimization.
   *
   * @param[in] init_xs      initial guess for state trajectory with $T+1$
   * elements (default [])
   * @param[in] init_us      initial guess for control trajectory with $T$
   * elements (default [])
   * @param[in] init_reg     initial guess for the regularization value. Very
   * low values are typical used with very good guess points (default 1e-9).
   * @return A type that describes if convergence was reached.
   */
  virtual Convergence solve(const std::vector<Eigen::VectorXd> &init_xs =
                                std::vector<Eigen::VectorXd>(),
                            const std::vector<Eigen::VectorXd> &init_us =
                                std::vector<Eigen::VectorXd>(),
                            const double reg_init = NAN) = 0;

  /**
   * @brief Compute the search direction
   * $(\delta\mathbf{x}^k,\delta\mathbf{u}^k)$ for the current guess
   * $(\mathbf{x}^k_s,\mathbf{u}^k_s)$.
   *
   * You must call `setCandidate()` first in order to define the current guess.
   * A current guess defines a state and control trajectory
   * $(\mathbf{x}^k_s,\mathbf{u}^k_s)$ of $T+1$ and $T$ elements,
   * respectively.
   *
   * @param[in] recalc  true for recalculating the derivatives at current state
   * and control
   * @return  The search direction $(\delta\mathbf{x},\delta\mathbf{u})$ and
   * the dual lambdas as lists of $T+1$, $T$ and $T+1$ lengths,
   * respectively
   */
  virtual void computeDirection(const bool recalc) = 0;

  /**
   * @brief Try a predefined step length $\alpha$ and compute its cost
   * improvement $dV$.
   *
   * It uses the search direction found by `computeDirection()` to try a
   * determined step length $\alpha$. Therefore, it assumes that we have run
   * `computeDirection()` first. Additionally, it returns the cost improvement
   * $dV$ along the predefined step length $\alpha$.
   *
   * @param[in] steplength  applied step length ($0\leq\alpha\leq1$)
   * @return  the cost improvement
   */
  virtual double tryStep(const double steplength = 1) = 0;

  /**
   * @brief Resizing the solver data
   *
   * If the shooting problem has changed after construction, then this function
   * resizes all the data before starting resolve the problem.
   */
  virtual void resizeData();

  /**
   * @brief Compute the dynamic feasibility
   * $\|\mathbf{f}_{\mathbf{s}}\|_{\infty,1}$ for the current guess
   * $(\mathbf{x}^k,\mathbf{u}^k)$
   *
   * The feasibility can be computed using different norms (e.g,
   * $\ell_\infty$ or $\ell_1$ norms). By default we use the
   * $\ell_\infty$ norm, however, we can change the type of norm using
   * `set_feasnorm`. Note that $\mathbf{f}_{\mathbf{s}}$ are the gaps on the
   * dynamics, which are computed at each node as
   * $\mathbf{x}^{'}-\mathbf{f}(\mathbf{x},\mathbf{u})$.
   */
  double computeDynamicFeasibility();

  /**
   * @brief Compute the feasibility of the inequality constraints for the
   * current guess
   *
   * The feasibility can be computed using different norms (e.g,
   * $\ell_\infty$ or $\ell_1$ norms). By default we use the
   * $\ell_\infty$ norm, however, we can change the type of norm using
   * `set_feasnorm`.
   */
  double computeInequalityFeasibility();

  /**
   * @brief Set the solver candidate trajectories
   * $(\mathbf{x}_s,\mathbf{u}_s)$
   *
   * The solver candidates are defined as a state and control trajectories
   * $(\mathbf{x}_s,\mathbf{u}_s)$ of $T+1$ and $T$ elements,
   * respectively. Additionally, we need to define the dynamic feasibility of
   * the $(\mathbf{x}_s,\mathbf{u}_s)$ pair. Note that the trajectories are
   * feasible if $\mathbf{x}_s$ is the resulting trajectory from the system
   * rollout with $\mathbf{u}_s$ inputs.
   *
   * @param[in] xs          state trajectory of $T+1$ elements (default [])
   * @param[in] us          control trajectory of $T$ elements (default [])
   * @param[in] isFeasible  true if the \p xs are obtained from integrating the
   * \p us (rollout)
   */
  void setCandidate(const std::vector<Eigen::VectorXd> &xs_warm =
                        std::vector<Eigen::VectorXd>(),
                    const std::vector<Eigen::VectorXd> &us_warm =
                        std::vector<Eigen::VectorXd>(),
                    const bool is_feasible = false);

  /**
   * @brief Return the shooting problem
   */
  const std::shared_ptr<ShootingProblem> &get_problem() const;

  /**
   * @brief Return the state trajectory $\mathbf{x}_s$
   */
  const std::vector<Eigen::VectorXd> &get_xs() const;

  /**
   * @brief Return the control trajectory $\mathbf{u}_s$
   */
  const std::vector<Eigen::VectorXd> &get_us() const;

  /**
   * @brief Return the dynamic infeasibility $\mathbf{f}_{s}$
   */
  const std::vector<Eigen::VectorXd> &get_fs() const;

  /**
   * @brief Return the feasibility status of the
   * $(\mathbf{x}_s,\mathbf{u}_s)$ trajectory
   */
  bool get_is_feasible() const;

  /**
   * @brief Return the cost for the current guess
   */
  double get_cost() const;

  /**
   * @brief Return the merit for the current guess
   */
  double get_merit() const;

  /**
   * @brief Return the primal-variable regularization
   */
  double get_preg() const;

  /**
   * @brief Return the dual-variable regularization
   */
  double get_dreg() const;

  /**
   * @brief Return the step length $\alpha$
   */
  double get_steplength() const;

  /**
   * @brief Return the threshold used for accepting a step
   */
  double get_th_acceptstep() const;

  /**
   * @brief Return the type of norm used to evaluate the dynamic and constraints
   * feasibility
   */
  FeasibilityNorm get_feasnorm() const;

  /**
   * @brief Return the number of iterations performed by the solver
   */
  std::size_t get_iter() const;

  /**
   * @brief Modify the state trajectory $\mathbf{x}_s$
   */
  void set_xs(const std::vector<Eigen::VectorXd> &xs);

  /**
   * @brief Modify the control trajectory $\mathbf{u}_s$
   */
  void set_us(const std::vector<Eigen::VectorXd> &us);

  /**
   * @brief Modify the primal-variable regularization value
   */
  void set_preg(const double preg);

  /**
   * @brief Modify the dual-variable regularization value
   */
  void set_dreg(const double dreg);

  /**
   * @brief Modify the threshold used for accepting step
   */
  void set_th_acceptstep(const double th_acceptstep);

  /**
   * @brief Modify the current norm used for computed the dynamic and constraint
   * feasibility
   */
  void set_feasnorm(const FeasibilityNorm feas_norm);

protected:
  std::shared_ptr<ShootingProblem> problem_; //!< optimal control problem
  std::vector<Eigen::VectorXd> xs_;          //!< State trajectory
  std::vector<Eigen::VectorXd> us_;          //!< Control trajectory
  std::vector<Eigen::VectorXd> fs_; //!< Gaps/defects between shooting nodes

  bool is_feasible_; //!< Label that indicates is the iteration is feasible

  double cost_;  //!< Cost for the current guess
  double merit_; //!< Merit for the current guess

  double preg_;          //!< Current primal-variable regularization value
  double dreg_;          //!< Current dual-variable regularization value
  double steplength_;    //!< Current applied step length
  double th_acceptstep_; //!< Threshold used for accepting step
  enum FeasibilityNorm feasnorm_; //!< Type of norm used to evaluate the
                                  //!< dynamics and constraints feasibility
  std::size_t iter_; //!< Number of iteration performed by the solver
  double tmp_feas_;  //!< Temporal variables used for computed the feasibility
};

} // namespace ocp
