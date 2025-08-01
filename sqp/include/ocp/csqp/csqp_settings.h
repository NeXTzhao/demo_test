#pragma once

#include <limits>

namespace ocp {

// Constrained Sqp settings
struct SqpSettings {
  // Maximum number of SQP iterations
  std::size_t max_iteration = 100;
  // Termination condition : RMS update of x(t) and u(t) are both below this
  // value
  double delta_tolerance = 1e-6;
  // Termination condition : (cost{i+1} - (cost{i}) < costTol AND
  // constraints{i+1} < g_min
  double cost_tolerance = 1e-4;
  double g_min = 1e-6;
  // Termination tolerance
  double kkt_tolerance = 1e-6;

  // terminate linesearch if the attempted step size is below this threshold
  double alpha_min = 1e-4;

  // Maximum time in seconds used to stop execution of the solver
  double max_solve_time = std::numeric_limits<double>::infinity();

  // Regularization factor used to increase the damping value
  double reg_incfactor = 10.0;
  // Regularization factor used to decrease the damping value
  double reg_decfactor = 10.0;
  // Minimum allowed regularization value
  double reg_min = 1e-9;
  // Maximum allowed regularization value
  double reg_max = 1e9;

  // Step-length threshold used to decrease regularization \in (0, 1]
  double th_stepdec = 0.5;
  // Step-length threshold used to increase regularization \in (0, 1]
  double th_stepinc = 0.01;

  // merit function coefficient scaling the infinite norm of Lagrange
  // multiplier
  double lag_mul_inf_norm_coef = 10.;

  // Use filter line search
  bool use_filter_line_search = true;
  // Filter size for line-search (do not change the default value !)
  std::size_t filter_size = 1;

  bool reset_y = false;
  bool reset_rho = false;
  bool update_rho_with_heuristic = false;
  bool remove_reg = false; //!< Removes regularization (preg,dreg)

  // penalty weight for dymanic violation in the merit function
  double mu_dynamic = 1e1;
  // penalty weight for constraint violation in the merit function
  double mu_constraint = 1e1;

  // With QP callbacks
  bool with_qp_callbacks = false;

  // proximal term
  double sigma = 1e-6;
  // relaxed step size
  double alpha = 1.6;
  // max qp iters
  std::size_t max_qp_iters = 1000;

  double rho_sparse_base = 1e-1;
  double rho_min = 1e-6;                //!< rho min
  double rho_max = 1e3;                 //!< rho max
  std::size_t rho_update_interval = 25; //!< frequency of update of rho
  double adaptive_rho_tolerance = 5;

  // absolute termination criteria
  double eps_abs = 1e-4;
  // relative termination criteria
  double eps_rel = 1e-4;
  // warm-start the QP with unconstrained solution
  double equality_qp_initial_guess = true;
};

} // namespace ocp
