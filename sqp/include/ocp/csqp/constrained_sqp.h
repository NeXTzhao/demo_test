#pragma once

#include <boost/circular_buffer.hpp>
#include <limits>
#include <vector>

#include "ocp/csqp/csqp_settings.h"
#include "ocp/solver/solver_base.h"

// forward declaration
class DevastatorWrapperInfo;

namespace ocp {

// forward declaration
class ActionModel;
class DynamicsAbstract;

// clang-format off
typedef typename Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXdRowMajor;
// clang-format on

// clang-format off
/**
 * @brief Constrained Sequential Quadratic Programming (CSQP) solver
 *
 * The CSQP solver computes an optimal trajectory and control commands by
 * iterates running `backwardPass()` and `forwardPass()`. The backward pass
 * accepts infeasible guess as described in the `SolverDDP::backwardPass()`.
 * Additionally, the forward pass handles infeasibility simulations that
 * resembles the numerical behaviour of a multiple-shooting formulation, i.e.:
 * {eqnarray}
 *   \mathbf{\hat{x}}_0 &=& \mathbf{\tilde{x}}_0 - (1 -
 * \alpha)\mathbf{\bar{f}}_0,\\
 *   \mathbf{\hat{u}}_k &=& \mathbf{u}_k + \alpha\mathbf{k}_k +
 * \mathbf{K}_k(\mathbf{\hat{x}}_k-\mathbf{x}_k),\\
 *   \mathbf{\hat{x}}_{k+1} &=&
 * \mathbf{f}_k(\mathbf{\hat{x}}_k,\mathbf{\hat{u}}_k) - (1 -
 * \alpha)\mathbf{\bar{f}}_{k+1}.
 * }
 * Note that the forward pass keeps the gaps $\mathbf{\bar{f}}_s$ open
 * according to the step length $\alpha$ that has been accepted. This solver
 * has shown empirically greater globalization strategy. Additionally, the
 * expected improvement computation considers the gaps in the dynamics:
 * {equation}
 *   \Delta J(\alpha) = \Delta_1\alpha + \frac{1}{2}\Delta_2\alpha^2,
 * }
 * with
 * {eqnarray}
 *   \Delta_1 = \sum_{k=0}^{N-1} \mathbf{k}_k^\top\mathbf{Q}_{\mathbf{u}_k}
 * +\mathbf{\bar{f}}_k^\top(V_{\mathbf{x}_k} -
 *   V_{\mathbf{xx}_k}\mathbf{x}_k),\nonumber\\ \Delta_2 = \sum_{k=0}^{N-1}
 *   \mathbf{k}_k^\top\mathbf{Q}_{\mathbf{uu}_k}\mathbf{k}_k +
 * \mathbf{\bar{f}}_k^\top(2 V_{\mathbf{xx}_k}\mathbf{x}_k
 * - V_{\mathbf{xx}_k}\mathbf{\bar{f}}_k). }
 *
 */
// clang-format on
class ConstrainedSqp : public SolverAbstract {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the fadmm solver
   *
   * @param[in] problem  shooting problem
   */
  explicit ConstrainedSqp(SqpSettings settings,
                          std::shared_ptr<ShootingProblem> problem);
  virtual ~ConstrainedSqp();

  virtual Convergence solve(const std::vector<Eigen::VectorXd> &init_xs =
                                std::vector<Eigen::VectorXd>(),
                            const std::vector<Eigen::VectorXd> &init_us =
                                std::vector<Eigen::VectorXd>(),
                            const double regInit = NAN);

  void set_devastator_wrapper_function(
      DevastatorWrapperInfo *devastator_wrapper_info_ptr,
      std::function<void(
          DevastatorWrapperInfo & /* devastator_wrapper_info */,
          const std::vector<Eigen::VectorXd> & /* xs */,
          const std::shared_ptr<DynamicsAbstract> & /* vehicle_dynamics */,
          std::vector<std::shared_ptr<ActionModel>> *const /* running_models */,
          std::shared_ptr<ActionModel> *const /* terminal_model*/)>
          get_feature_fn) {
    devastator_wrapper_info_ptr_ = devastator_wrapper_info_ptr;
    get_feature_fn_ = get_feature_fn;
  }

  /**
   * @brief Update internal values for computing the expected improvement
   */
  void updateExpectedImprovement();

  virtual void forwardPass(const double stepLength = 0.);
  virtual void forwardPass_without_constraints();
  virtual void backwardPass();
  virtual void backwardPass_without_rho_update();
  virtual void backwardPass_without_constraints();
  virtual void backwardPass_mt();
  virtual void backwardPass_without_rho_update_mt();

  /**
   * @brief Computes the merit function, gaps at the given xs, us along with
   * delta x and delta u
   */
  virtual void computeDirection(const bool recalcDiff);

  virtual double tryStep(const double stepLength);

  virtual void calc(const bool recalc = true);

  virtual void resizeData();

  /**
   * @brief Compute the feedforward and feedback terms using a Cholesky
   * decomposition
   *
   * To compute the feedforward $\mathbf{k}_k$ and feedback
   * $\mathbf{K}_k$ terms, we use a Cholesky decomposition to solve
   * $\mathbf{Q}_{\mathbf{uu}_k}^{-1}$ term:
   * {eqnarray}
   * \mathbf{k}_k &=& \mathbf{Q}_{\mathbf{uu}_k}^{-1}\mathbf{Q}_{\mathbf{u}},\\
   * \mathbf{K}_k &=& \mathbf{Q}_{\mathbf{uu}_k}^{-1}\mathbf{Q}_{\mathbf{ux}}.
   * }
   *
   * Note that if the Cholesky decomposition fails, then we re-start the
   * backward pass and increase the state and control regularization values.
   */
  virtual void computeGains(const std::size_t t);

  /**
   * @brief Increase the state and control regularization values by a
   * `regfactor_` factor
   */
  void increaseRegularization();

  /**
   * @brief Decrease the state and control regularization values by a
   * `regfactor_` factor
   */
  void decreaseRegularization();

  /**
   * @brief Allocate all the internal data needed for the solver
   */
  virtual void allocateData();

  virtual void reset_params();

  virtual void reset_rho_vec();

  /**
   * @brief Compute the KKT conditions residual
   */
  virtual void checkKKTConditions();

  const std::vector<Eigen::VectorXd> &get_xs_try() const { return xs_try_; };
  const std::vector<Eigen::VectorXd> &get_us_try() const { return us_try_; };

  const std::vector<Eigen::VectorXd> &get_xs() const { return xs_; };
  const std::vector<Eigen::VectorXd> &get_us() const { return us_; };

  const std::vector<Eigen::VectorXd> &get_dx_tilde() const { return dxtilde_; };
  const std::vector<Eigen::VectorXd> &get_du_tilde() const { return dutilde_; };

  const std::vector<Eigen::VectorXd> &get_dx() const { return dx_; };
  const std::vector<Eigen::VectorXd> &get_du() const { return du_; };

  const std::vector<Eigen::VectorXd> &get_y() const { return y_; };
  const std::vector<Eigen::VectorXd> &get_z() const { return z_; };

  const std::vector<Eigen::VectorXd> &get_rho_vec() const { return rho_vec_; };

  double get_KKT() const { return KKT_; };
  double get_gap_norm() const { return gap_norm_; };
  double get_constraint_norm() const { return constraint_norm_; };
  double get_qp_iters() const { return qp_iters_; };
  double get_xgrad_norm() const { return x_grad_norm_; };
  double get_ugrad_norm() const { return u_grad_norm_; };
  double get_merit() const { return merit_; };

  double get_dx_norm() const { return dx_norm_; }
  double get_du_norm() const { return du_norm_; }
  double get_merit_diff() const { return merit_diff_; }

  double get_rho_sparse() const { return rho_sparse_; };

  double get_norm_primal() const { return norm_primal_; };
  double get_norm_primal_tolerance() const { return norm_primal_tolerance_; };
  double get_norm_dual() const { return norm_dual_; };
  double get_norm_dual_tolerance() const { return norm_dual_tolerance_; };

  bool get_max_solve_time_reached() const { return max_solve_time_reached_; };

  const std::vector<double> &get_alphas() const { return alphas_; }
  const std::vector<Eigen::MatrixXd> &get_Vxx() const { return Vxx_; }
  const std::vector<Eigen::VectorXd> &get_Vx() const { return Vx_; }
  const std::vector<Eigen::MatrixXd> &get_Qxx() const { return Qxx_; }
  const std::vector<Eigen::MatrixXd> &get_Qxu() const { return Qxu_; }
  const std::vector<Eigen::MatrixXd> &get_Quu() const { return Quu_; }
  const std::vector<Eigen::VectorXd> &get_Qx() const { return Qx_; }
  const std::vector<Eigen::VectorXd> &get_Qu() const { return Qu_; }
  const std::vector<MatrixXdRowMajor> &get_K() const { return K_; }
  const std::vector<Eigen::VectorXd> &get_k() const { return k_; }

  void printQPCallbacks(const int iter);

  void update_lagrangian_parameters(const int iter);
  void set_rho_sparse(const double rho_sparse) { rho_sparse_ = rho_sparse; };
  void update_rho_vec(const int iter);
  void apply_rho_update(const double rho_sparse);

  void set_alphas(const std::vector<double> &alphas);
  void set_th_grad(const double th_grad);

  // clang-format off
  public:
    SqpSettings settings_;
 
   // allocate data
   Eigen::MatrixXd Vxx_tmp_;           //!< Temporary variable for ensuring symmetry of Vxx
   std::vector<Eigen::MatrixXd> Vxx_;  //!< Hessian of the Value function $\mathbf{V_{xx}}$
   std::vector<Eigen::VectorXd> Vx_;   //!< Gradient of the Value function $\mathbf{V_x}$
   std::vector<Eigen::MatrixXd> Qxx_;  //!< Hessian of the Hamiltonian $\mathbf{Q_{xx}}$
   std::vector<Eigen::MatrixXd> Qxu_;  //!< Hessian of the Hamiltonian $\mathbf{Q_{xu}}$
   std::vector<Eigen::MatrixXd> Quu_;  //!< Hessian of the Hamiltonian $\mathbf{Q_{uu}}$
   std::vector<Eigen::VectorXd> Qx_;   //!< Gradient of the Hamiltonian $\mathbf{Q_x}$
   std::vector<Eigen::VectorXd> Qu_;   //!< Gradient of the Hamiltonian $\mathbf{Q_u}$
   std::vector<MatrixXdRowMajor> K_;   //!< Feedback gains $\mathbf{K}$
   std::vector<Eigen::VectorXd> k_;    //!< Feed-forward terms $\mathbf{l}$
 
   Eigen::VectorXd xnext_;      //!< Next state $\mathbf{x}^{'}$
   MatrixXdRowMajor FxTVxx_p_;  //!< Store the value of
                                //!< $\mathbf{f_x}^T\mathbf{V_{xx}}^{'}$
   std::vector<MatrixXdRowMajor>
       FuTVxx_p_;             //!< Store the values of
                              //!< $\mathbf{f_u}^T\mathbf{V_{xx}}^{'}$
                              //!< per each running node
   Eigen::VectorXd fTVxx_p_;  //!< Store the value of
                              //!< $\mathbf{\bar{f}}^T\mathbf{V_{xx}}^{'}$
   std::vector<Eigen::LLT<Eigen::MatrixXd> > Quu_llt_;  //!< Cholesky LLT solver
   std::vector<Eigen::VectorXd> Quuk_;  //!< Store the values of $\mathbf{Q_{uu}\mathbf{k}} per each running node
 
   std::vector<double> alphas_;                      //!< Set of step lengths using by the line-search procedure
   boost::circular_buffer<double> constraint_list_;  //!< memory buffer of constraint norms (used in filter line-search)
   boost::circular_buffer<double> gap_list_;         //!< memory buffer of gap norms (used in filter line-search)
   boost::circular_buffer<double> cost_list_;        //!< memory buffer of gap norms (used in filter line-search)
 
   double cost_try_;  //!< Total cost computed by line-search procedure
   std::vector<Eigen::VectorXd> xs_try_;  //!< State trajectory computed by line-search procedure
   std::vector<Eigen::VectorXd> us_try_;  //!< Control trajectory computed by line-search procedure
   std::vector<Eigen::VectorXd> fs_try_;   //!< Gaps/defects between shooting nodes
   std::vector<Eigen::VectorXd> dx_;       //!< the descent direction for x
   std::vector<Eigen::VectorXd> du_;       //!< the descent direction for u
   std::vector<Eigen::VectorXd> lag_mul_;  //!< the Lagrange multiplier of the dynamics constraint
 
   double lag_mul_inf_norm_;               //!< the infinite norm of Lagrange multiplier
 
   Eigen::VectorXd fs_flat_;               //!< Gaps/defects between shooting nodes (1D array)
 
   std::vector<Eigen::VectorXd> dxtilde_;  //!< the descent direction for x
   std::vector<Eigen::VectorXd> dutilde_;  //!< the descent direction for u
 
   // ADMM parameters
   std::vector<Eigen::VectorXd> y_;            //!< lagrangian dual variable
   std::vector<Eigen::VectorXd> z_;            //!< second admm variable
   std::vector<Eigen::VectorXd> z_prev_;       //!< second admm variable previous
   std::vector<Eigen::VectorXd> z_relaxed_;    //!< relaxed step of z
   std::vector<Eigen::VectorXd> rho_vec_;      //!< rho vector
   std::vector<Eigen::VectorXd> inv_rho_vec_;  //!< rho vector
 
   double norm_primal_ = 0.0;      //!< norm primal residual
   double norm_dual_ = 0.0;        //!< norm dual residual
   double norm_primal_rel_ = 0.0;  //!< norm primal relative residual
   double norm_dual_rel_ = 0.0;    //!< norm dual relative residual
   double norm_primal_tolerance_ = 0.0;  //!< tolerance of the primal residual norm
   double norm_dual_tolerance_ = 0.0;    //!< tolerance of the primal residual norm
 
  protected:
   double merit_ = 0;                //!< merit function at nominal traj
   double merit_try_ = 0;            //!< merit function for the step length tried
   double x_grad_norm_ = 0;          //!< 1 norm of the delta x
   double u_grad_norm_ = 0;          //!< 1 norm of the delta u
   double gap_norm_ = 0;             //!< 1 norm of the gaps
   double constraint_norm_ = 0;      //!< 1 norm of constraint violation
   double constraint_norm_try_ = 0;  //!< 1 norm of constraint violation try
   double gap_norm_try_ = 0;         //!< 1 norm of the gaps
 
   std::size_t qp_iters_ = 0;
 
   double rho_estimate_sparse_ = 0.0;  //!< rho estimate
   double rho_sparse_;                 //!< rho
 
   double KKT_ =
       std::numeric_limits<double>::infinity();  //!< KKT conditions residual
 
   double merit_before_step_ = 0.0;
   double merit_after_step_ = 0.0;
 
   double dx_norm_ = 0.0;
   double du_norm_ = 0.0;
   double merit_diff_ = 0.0;

  // clang-format on
private:
  bool is_worse_than_memory_ =
      false; //!< Boolean for filter line-search criteria

  Eigen::VectorXd tmp_vec_x_;                   //!< Temporary variable
  std::vector<Eigen::VectorXd> tmp_vec_u_;      //!< Temporary variable
  std::vector<Eigen::VectorXd> tmp_dual_cwise_; //!< Temporary variable
  Eigen::VectorXd tmp_Vx_;                      //!< Temporary variable
  std::vector<Eigen::VectorXd> tmp_Cdx_Cdu_;    //!< Temporary variable
  std::vector<Eigen::MatrixXd> tmp_rhoGx_mat_;  //!< Temporary variable
  std::vector<Eigen::MatrixXd> tmp_rhoGu_mat_;  //!< Temporary variable
  std::vector<Eigen::VectorXd> Vxx_fs_;         //!< Temporary variable

  double start_time_ = 0.0; // Time when the solve function was called
  bool max_solve_time_reached_ = false; // Flag indicating solver timedout

  DevastatorWrapperInfo *devastator_wrapper_info_ptr_;
  std::function<void(
      DevastatorWrapperInfo & /* devastator_wrapper_info */,
      const std::vector<Eigen::VectorXd> & /* xs */,
      const std::shared_ptr<DynamicsAbstract> & /* vehicle_dynamics */,
      std::vector<std::shared_ptr<ActionModel>> *const /* running_models */,
      std::shared_ptr<ActionModel> *const /* terminal_model*/)>
      get_feature_fn_;
};

// To-do: move definitions to a dedicated file

// Same logic as in Proxsuite and Pinocchio to check eigen malloc
#ifdef CSQP_EIGEN_CHECK_MALLOC
#ifndef EIGEN_RUNTIME_NO_MALLOC
#define EIGEN_RUNTIME_NO_MALLOC_WAS_NOT_DEFINED
#define EIGEN_RUNTIME_NO_MALLOC
#endif
#endif

// #include <Eigen/Core>
// #include <cassert>

#ifdef CSQP_EIGEN_CHECK_MALLOC
#ifdef EIGEN_RUNTIME_NO_MALLOC_WAS_NOT_DEFINED
#undef EIGEN_RUNTIME_NO_MALLOC
#undef EIGEN_RUNTIME_NO_MALLOC_WAS_NOT_DEFINED
#endif
#endif

// Check memory allocation for Eigen
#ifdef CSQP_EIGEN_CHECK_MALLOC
#define CSQP_EIGEN_MALLOC(allowed)                                             \
  ::Eigen::internal::set_is_malloc_allowed(allowed)
#define CSQP_EIGEN_MALLOC_ALLOWED() CSQP_EIGEN_MALLOC(true)
#define CSQP_EIGEN_MALLOC_NOT_ALLOWED() CSQP_EIGEN_MALLOC(false)
#else
#define CSQP_EIGEN_MALLOC(allowed)
#define CSQP_EIGEN_MALLOC_ALLOWED()
#define CSQP_EIGEN_MALLOC_NOT_ALLOWED()
#endif

} // namespace ocp
