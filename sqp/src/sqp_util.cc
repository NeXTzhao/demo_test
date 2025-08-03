// #ifdef CSQP_WITH_MULTITHREADING
// #include <omp.h>
// #endif // CSQP_WITH_MULTITHREADING

// #include <algorithm>
// #include <cmath>
// #include <iomanip>
// #include <iostream>
// #include <vector>

// // cost
// #include "ocp/cost/centripetal_acceleration_cost.h"
// #include "ocp/cost/cost_collection.h"
// #include "ocp/cost/half_plane_attraction_cost.h"
// #include "ocp/cost/half_plane_repeller_cost.h"
// #include "ocp/cost/input_attractor_1d.h"
// #include "ocp/cost/quadratic_cost.h"
// #include "ocp/cost/state_attractor_1d.h"
// #include "ocp/cost/state_repeller_1d.h"
// // constraint
// #include "ocp/constraint/centripetal_acceleration_constraint.h"
// #include "ocp/constraint/constraint_collection.h"
// #include "ocp/constraint/half_plane_repeller_constraint.h"
// #include "ocp/constraint/input_box_constraint.h"
// #include "ocp/constraint/state_box_constraint.h"
// // dynamics
// #include "ocp/dynamics/vehicle_model_dynamics.h"
// // shooting
// #include "ocp/shooting/action_model.h"
// #include "ocp/shooting/shooting.h"
// // csqp
// #include "ocp/csqp/constrained_sqp.h"

// #include "common.h"
// #include "cost_weight_util.h"
// #include "math/disk.h"

// static constexpr int X_DIM = ocp::VehicleModelDynamics::StateIndex::X_DIM;
// static constexpr int U_DIM = ocp::VehicleModelDynamics::ControlIndex::U_DIM;
// using StateIndex = ocp::VehicleModelDynamics::StateIndex;
// using ControlIndex = ocp::VehicleModelDynamics::ControlIndex;

// static constexpr int StateDim = static_cast<int>(X_DIM);
// static constexpr int ControlDim = static_cast<int>(U_DIM);

// using State = Eigen::Matrix<double, StateDim, 1>;
// using Control = Eigen::Matrix<double, ControlDim, 1>;

// using StateSequence = std::vector<State>;
// using ControlSequence = std::vector<Control>;

// void GenerateFormulaCosts(const int step, const Eigen::VectorXd &state,
//                           const ModelParameter &param,
//                           const std::vector<DiskInfo> &ego_disks,
//                           const std::vector<TrajectoryCostingTerm>
//                           &cost_terms, ocp::CostCollection *const
//                           cost_collection_ptr) {
//   // Sanity check
//   if (cost_features.empty() || cost_terms.empty()) {
//     return;
//   }
//   if (!cost_collection_ptr) {
//     return;
//   }
//   if (state.size() != StateIndex::X_DIM) {
//     return;
//   }

//   const bool is_termal_node = (step + 1 == kTrajectoryStateNum);

//   cost_collection_ptr->clear();
//   cost_collection_ptr->reserve(cost_features.size());

//   for (const auto &term : cost_terms) {
//     if (!cost_features.count(term.feature_type)) {
//       continue;
//     }
//     // const double weight = term.costing_spec.weight_vec[step];
//     const double weight = 10;
//     // process other cost
//     switch (term.feature_type) {
//     case FeatureType::CONSTRAINT_ACCELERATION: {
//       cost_collection_ptr->add(FeatureTypeToString(term.feature_type),
//                                std::make_unique<ocp::StateAttractor1D>(
//                                    StateIndex::ACCEL, 0.0, weight));
//       break;
//     }
//     case FeatureType::CONSTRAINT_DELTA: {
//       cost_collection_ptr->add(FeatureTypeToString(term.feature_type),
//                                std::make_unique<ocp::StateAttractor1D>(
//                                    StateIndex::DELTAV, 0.0, weight));
//       break;
//     }
//     case FeatureType::CONSTRAINT_OMEGA: {
//       cost_collection_ptr->add(FeatureTypeToString(term.feature_type),
//                                std::make_unique<ocp::StateAttractor1D>(
//                                    StateIndex::OMEGA, 0.0, weight));
//       break;
//     }
//     case FeatureType::CONSTRAINT_JERK: {
//       if (!is_termal_node) {
//         cost_collection_ptr->add(FeatureTypeToString(term.feature_type),
//                                  std::make_unique<ocp::InputAttractor1D>(
//                                      ControlIndex::JERK, 0.0, weight));
//       }
//       break;
//     }
//     case FeatureType::CONSTRAINT_ALPHA: {
//       if (!is_termal_node) {
//         cost_collection_ptr->add(FeatureTypeToString(term.feature_type),
//                                  std::make_unique<ocp::InputAttractor1D>(
//                                      ControlIndex::ALPHAV, 0.0, weight));
//       }
//       break;
//     }
//     case FeatureType::LATERAL_ACCEL: {
//       cost_collection_ptr->add(
//           FeatureTypeToString(term.feature_type),
//           std::make_unique<ocp::CentripetalAccelerationCost>(weight));
//       break;
//     }
//     default:
//       std::cout << "Do not have this cost feature, please check" <<
//       std::endl; break;
//     }
//   }

//   return;
// }

// void GenerateFormulaConstraints(
//     const int step, const Eigen::VectorXd &state, const ModelParameter
//     &param, const std::vector<DiskInfo> &ego_disks, const
//     std::vector<TrajectoryCostingTerm> &cost_terms, ocp::ConstraintCollection
//     *const constraint_collection_ptr) {
//   // Sanity check
//   if (constraint_features.empty() || cost_terms.empty()) {
//     return;
//   }
//   if (!constraint_collection_ptr) {
//     return;
//   }
//   if (state.size() != StateIndex::X_DIM) {
//     return;
//   }

//   const bool is_terminal_node = (step + 1 == kTrajectoryStateNum);

//   constraint_collection_ptr->reserve(constraint_features.size());

//   // namely: x_min <= x <= x_max
//   Eigen::VectorXd x_min = Eigen::VectorXd::Constant(StateIndex::X_DIM,
//   -1.0e20); Eigen::VectorXd x_max =
//   Eigen::VectorXd::Constant(StateIndex::X_DIM, 1.0e20);
//   // theta bounds
//   x_min(StateIndex::THETA) = -M_PI;
//   x_max(StateIndex::THETA) = M_PI;

//   // speed bounds
//   {
//     const double speed_min = std::max(0.0, param.limit.speed_limit.min_val);
//     x_min(StateIndex::SPEED) = speed_min;
//     x_max(StateIndex::SPEED) =
//         std::max(param.limit.speed_limit.max_val, speed_min + 1e-2);
//   }
//   // acc bounds
//   x_min(StateIndex::ACCEL) = param.limit.accel_limit.min_val;
//   x_max(StateIndex::ACCEL) = param.limit.accel_limit.max_val;
//   // In sqp, delta means kappa, kappa := tan(delta) / L
//   const double min_tan_delta = std::tan(param.limit.steering_limit.min_val);
//   const double max_tan_delta = std::tan(param.limit.steering_limit.max_val);
//   // kappa bounds
//   x_min(StateIndex::DELTAV) = min_tan_delta / param.measurement.wheel_base;
//   x_max(StateIndex::DELTAV) = max_tan_delta / param.measurement.wheel_base;
//   // dkappa := omega / (L * cos^2(delta));
//   const double min_cos_delta = std::cos(param.limit.steering_limit.min_val);
//   const double max_cos_delta = std::cos(param.limit.steering_limit.max_val);
//   const double min_L_sec_square_inv =
//       1.0 / (min_cos_delta * min_cos_delta * param.measurement.wheel_base);
//   const double max_L_sec_square_inv =
//       1.0 / (max_cos_delta * max_cos_delta * param.measurement.wheel_base);
//   // dkappa bounds
//   x_min(StateIndex::OMEGA) =
//       param.limit.steering_speed_limit.min_val * min_L_sec_square_inv;
//   x_max(StateIndex::OMEGA) =
//       param.limit.steering_speed_limit.max_val * max_L_sec_square_inv;
//   // odom bounds
//   {
//     const double min_odom = 0.0;
//     const double max_odom = 100.0;

//     x_min(StateIndex::ODOM) = min_odom;
//     x_max(StateIndex::ODOM) = std::max(max_odom, min_odom + 1e-2);
//   }
//   constraint_collection_ptr->add("state_box_constraint",
//                                  std::make_unique<ocp::StateBoxConstraint>(
//                                      std::move(x_min), std::move(x_max)));

//   // CONSTRAINT_JERK,CONSTRAINT_ALPHA,
//   // namely: u_min <= u <= u_max
//   if (!is_terminal_node) {
//     Eigen::VectorXd u_min =
//         Eigen::VectorXd::Constant(ControlIndex::U_DIM, -1.0e20);
//     Eigen::VectorXd u_max =
//         Eigen::VectorXd::Constant(ControlIndex::U_DIM, 1.0e20);
//     // jerk bounds
//     u_min(ControlIndex::JERK) = param.limit.jerk_limit.min_val;
//     u_max(ControlIndex::JERK) = param.limit.jerk_limit.max_val;
//     // ddkappa bounds
//     // ddkappa := 1.0 / (L * cos^2(delta)) * (alpha + 2 * tan(delta) *
//     omega^2) u_min(ControlIndex::ALPHAV) =
//         min_L_sec_square_inv *
//         (param.limit.steering_accel_limit.min_val +
//          2.0 * min_tan_delta * param.limit.steering_speed_limit.min_val *
//              param.limit.steering_speed_limit.min_val);
//     u_max(ControlIndex::ALPHAV) =
//         max_L_sec_square_inv *
//         (param.limit.steering_accel_limit.max_val +
//          2.0 * max_tan_delta * param.limit.steering_speed_limit.max_val *
//              param.limit.steering_speed_limit.max_val);
//     constraint_collection_ptr->add("control_box_constraint",
//                                    std::make_unique<ocp::InputBoxConstraint>(
//                                        std::move(u_min), std::move(u_max)));
//   }

//   for (const auto &term : cost_terms) {
//     if (!constraint_features.count(term.feature_type)) {
//       continue;
//     }

//     // process other cost
//     switch (term.feature_type) {
//     default:
//       std::cout << "Do not have this constraint feature, please check"
//                 << std::endl;
//       break;
//     }
//   }
// }

// void GenerateSqpModels(
//     Solve &solve, const std::vector<Eigen::VectorXd> &xs,
//     const std::shared_ptr<ocp::DynamicsAbstract> &dynamics,
//     const ModelParameter &vehicle_model_parameter,
//     std::vector<std::shared_ptr<ocp::ActionModel>> *const running_models,
//     std::shared_ptr<ocp::ActionModel> *const terminal_model) {
//   if (!running_models || !terminal_model) {
//     return;
//   }

//   clock_t start = clock();

//   StateSequence state_seq;
//   state_seq.reserve(xs.size());
//   for (const auto &x : xs) {
//     state_seq.push_back(x);
//   }

//   // const auto &vehicle_model_parameter =
//   //     solve.vehicle_model_parameter();
//   // const auto &formula_ptr =
//   //     solve.formula_ptr();

//   // GenerateContraints(
//   //     devastator_wrapper_info.config(),
//   //     devastator_wrapper_info.reference_line_info(),
//   //     devastator_wrapper_info.convex_space(), vehicle_model_parameter,
//   //     state_seq, devastator_wrapper_info.sequence_specification(),
//   //     *formula_ptr, primitives,
//   //     devastator_wrapper_info.last_obs(),
//   //     devastator_wrapper_info.frame_ptr());

//   {
//     clock_t end = clock();
//     double cost_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
//     std::cout << "GenerateSqpModels GenerateContraints cost time: "
//               << cost_time;
//   }

//   start = clock();
//   running_models->clear();
//   running_models->resize(kTrajectoryControlNum);
//   // 0th step to update init_acc_rewards and init_speed_rewards
//   {
//     const Eigen::VectorXd &state = xs[0];
//     const std::vector<DiskInfo> ego_disks = GetCoveringDiskInfo(
//         vehicle_model_parameter.measurement, state(StateIndex::X_POS),
//         state(StateIndex::Y_POS), state(StateIndex::THETA));
//     std::unique_ptr<ocp::CostCollection> running_costs(new
//     ocp::CostCollection); GenerateFormulaCosts(0, state,
//     vehicle_model_parameter, ego_disks,
//                          formula_ptr->costing_terms(), running_costs.get());
//     std::unique_ptr<ocp::ConstraintCollection> running_constraints(
//         new ocp::ConstraintCollection);
//     GenerateFormulaConstraints(0, state, vehicle_model_parameter, ego_disks,
//                                formula_ptr->costing_terms(), primitives,
//                                running_constraints.get());
//     (*running_models)[0] = std::make_shared<ocp::ActionModel>(
//         std::move(running_costs), std::move(running_constraints), dynamics,
//         dynamics->state_size(), dynamics->state_size(),
//         dynamics->input_size());
//   }

// #ifdef CSQP_WITH_MULTITHREADING
// #pragma omp parallel for num_threads(kNumOfThreads)
// #endif
//   for (int i = 1; i < kTrajectoryControlNum; ++i) {
//     const Eigen::VectorXd &state = xs[i];
//     const std::vector<DiskInfo> ego_disks = GetCoveringDiskInfo(
//         vehicle_model_parameter.measurement, state(StateIndex::X_POS),
//         state(StateIndex::Y_POS), state(StateIndex::THETA));
//     std::unique_ptr<ocp::CostCollection> running_costs(new
//     ocp::CostCollection); GenerateFormulaCosts(i, state,
//     vehicle_model_parameter, ego_disks,
//                          formula_ptr->costing_terms(), primitives,
//                          running_costs.get());
//     std::unique_ptr<ocp::ConstraintCollection> running_constraints(
//         new ocp::ConstraintCollection);
//     GenerateFormulaConstraints(i, state, vehicle_model_parameter, ego_disks,
//                                formula_ptr->costing_terms(), primitives,
//                                running_constraints.get());
//     (*running_models)[i] = std::make_shared<ocp::ActionModel>(
//         std::move(running_costs), std::move(running_constraints), dynamics,
//         dynamics->state_size(), dynamics->state_size(),
//         dynamics->input_size());
//   }

//   {
//     const Eigen::VectorXd &state = xs[kTrajectoryStateNum - 1];
//     const std::vector<DiskInfo> ego_disks = GetCoveringDiskInfo(
//         vehicle_model_parameter.measurement, state(StateIndex::X_POS),
//         state(StateIndex::Y_POS), state(StateIndex::THETA));
//     std::unique_ptr<ocp::CostCollection> terminal_costs(
//         new ocp::CostCollection);
//     GenerateFormulaCosts(
//         kTrajectoryStateNum - 1, state, vehicle_model_parameter, ego_disks,
//         formula_ptr->costing_terms(), primitives, terminal_costs.get());
//     std::unique_ptr<ocp::ConstraintCollection> terminal_constraints(
//         new ocp::ConstraintCollection);
//     GenerateFormulaConstraints(
//         kTrajectoryStateNum - 1, state, vehicle_model_parameter, ego_disks,
//         formula_ptr->costing_terms(), primitives,
//         terminal_constraints.get());
//     *terminal_model = std::make_shared<ocp::ActionModel>(
//         std::move(terminal_costs), std::move(terminal_constraints), dynamics,
//         dynamics->state_size(), dynamics->state_size(), 0);
//   }

//   {
//     clock_t end = clock();
//     double cost_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
//     std::cout << "GenerateSqpModels GenerateFormulaCosts cost time: "
//               << cost_time;
//   }

//   return;
// }

// bool SqpSolve(const StateSequence &xs_init, const ControlSequence &us_init) {
//   std::vector<Eigen::VectorXd> init_x;
//   init_x.reserve(xs_init.size());
//   std::vector<Eigen::VectorXd> init_u;
//   init_u.reserve(us_init.size());

//   for (size_t i = 0; i < xs_init.size(); ++i) {
//     Eigen::VectorXd x_tmp = xs_init[i];
//     const double cos_delta = std::cos(xs_init[i](StateIndex::DELTAV));
//     const double tan_delta = std::tan(xs_init[i](StateIndex::DELTAV));
//     // 1.0 / (L * cos^2(delta))
//     const double L_sec_square_inv =
//         1.0 / (cos_delta * cos_delta *
//                vehicle_model_parameter.measurement.wheel_base);
//     // kappa := tan(delta) / L
//     x_tmp(StateIndex::DELTAV) =
//         tan_delta / vehicle_model_parameter.measurement.wheel_base;
//     // dkappa := omega / (L * cos^2(delta))
//     x_tmp(StateIndex::OMEGA) = xs_init[i](StateIndex::OMEGA) *
//     L_sec_square_inv;

//     if (i + 1 != xs_init.size()) {
//       Eigen::VectorXd u_tmp = us_init[i];
//       // ddkappa := 1.0 / (L * cos^2(delta)) * (alpha + 2 * tan(delta) *
//       // omega^2)
//       u_tmp(ControlIndex::ALPHAV) =
//           L_sec_square_inv * (us_init[i](ControlIndex::ALPHAV) +
//                               2.0 * tan_delta * xs_init[i](StateIndex::OMEGA)
//                               *
//                                   xs_init[i](StateIndex::OMEGA));
//       init_u.push_back(std::move(u_tmp));
//     }
//     init_x.push_back(std::move(x_tmp));
//   }

//   WrapperInfo wrapper_info;

//   ocp::SqpSettings settings;
//   settings.eps_abs = 1e-2;
//   settings.eps_rel = 1e-2;
//   settings.kkt_tolerance = 1e-2;
//   settings.delta_tolerance = 1e-2;
//   settings.cost_tolerance = 1e-3;
//   settings.g_min = 1e-2;
//   settings.max_qp_iters = 200;
//   settings.max_iteration = 20;
//   settings.filter_size = 3;
//   settings.max_solve_time = 0.03; // 30ms

//   clock_t start = clock();
//   std::shared_ptr<ocp::ShootingProblem> shooting_problem;
//   const auto vehicle_model =
//       std::make_shared<ocp::VehicleModelDynamics>(kTrajectoryIntervalInSec);
//   {
//     std::vector<std::shared_ptr<ocp::ActionModel>> running_models;
//     std::shared_ptr<ocp::ActionModel> terminal_model;
//     GenerateSqpModels(devastator_wrapper_info, init_x, vehicle_model,
//                       &running_models, &terminal_model);
//     shooting_problem = std::make_shared<ocp::ShootingProblem>(
//         init_x[0], std::move(running_models), std::move(terminal_model));
//   }

//   shooting_problem->set_nthreads(kNumOfThreads);
//   ocp::ConstrainedSqp sqp_solver(settings, shooting_problem);
//   sqp_solver.set_devastator_wrapper_function(&devastator_wrapper_info,
//                                              GenerateSqpModels);
//   ocp::Convergence ret = sqp_solver.solve(init_x, init_u);

//   clock_t end = clock();
//   double cost_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
//   std::cout << "Sqp cost time: " << cost_time;

//   const std::vector<Eigen::VectorXd> &opt_xs = sqp_solver.get_xs();
//   const std::vector<Eigen::VectorXd> &opt_us = sqp_solver.get_us();

//   if (ret == ocp::Convergence::FALSE) {
//     return false;
//   }
//   return true;
// }