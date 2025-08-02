// #ifdef CSQP_WITH_MULTITHREADING
// #include <omp.h>
// #endif // CSQP_WITH_MULTITHREADING
//
// #include <algorithm>
// #include <cmath>
// #include <iomanip>
// #include <iostream>
// #include <vector>
//
// #include "sqp_util.h"
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
//
// namespace {
//
// // The planning node frequency 0.1s.
// constexpr double kPlanningCycleTime = 0.1;
//
// // Trajectory horizon, in seconds.
// constexpr double kTrajectoryHorizonInSec = 6.8;
//
// // Trajectory states interval, in seconds.
// constexpr double kTrajectoryIntervalInSec = 0.2;
//
// // Trajectory state/control #.
// constexpr int kTrajectoryStateNum =
//     static_cast<int>(kTrajectoryHorizonInSec / kTrajectoryIntervalInSec) + 1;
//
// constexpr int kTrajectoryControlNum = kTrajectoryStateNum - 1;
//
// constexpr double kEpsilon = 1e-3;
//
// // Enable debug flag.
// constexpr bool kEnableDebug = false;
//
// // The maximum replan interval from last trajectory.
// constexpr double kMaxReplanIntervalSec = 0.5;
// constexpr double kStopLineViolationMaxToleratnInMeters = 1.0;
// constexpr double kMaxLateralAcceleration = 4.0;
// constexpr int kNumOfThreads = 4;
//
// void GenerateSqpModels(
//     DevastatorWrapperInfo &devastator_wrapper_info,
//     const std::vector<Eigen::VectorXd> &xs,
//     const std::shared_ptr<ocp::DynamicsAbstract> &dynamics,
//     std::vector<std::shared_ptr<ocp::ActionModel>> *const running_models,
//     std::shared_ptr<ocp::ActionModel> *const terminal_model) {
//   if (!running_models || !terminal_model) {
//     return;
//   }
//
//   clock_t start = clock();
//
//   StateSequence state_seq;
//   state_seq.reserve(xs.size());
//   for (const auto &x : xs) {
//     state_seq.push_back(x);
//   }
//
//   const auto &vehicle_model_parameter =
//       devastator_wrapper_info.vehicle_model_parameter();
//   const auto &trajectory_formula_ptr =
//       devastator_wrapper_info.trajectory_formula_ptr();
//   auto &primitives = devastator_wrapper_info.primitives();
//
//   // GenerateContraints(
//   //     devastator_wrapper_info.trajectory_config(),
//   //     devastator_wrapper_info.reference_line_info(),
//   //     devastator_wrapper_info.convex_space(), vehicle_model_parameter,
//   //     state_seq, devastator_wrapper_info.sequence_specification(),
//   //     *trajectory_formula_ptr, primitives,
//   devastator_wrapper_info.last_obs(),
//   //     devastator_wrapper_info.frame_ptr());
//
//   {
//     clock_t end = clock();
//     double cost_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
//     std::cout << "GenerateSqpModels GenerateContraints cost time: " <<
//     cost_time;
//   }
//
//   start = clock();
//   running_models->clear();
//   running_models->resize(kTrajectoryControlNum);
//   // 0th step to update init_acc_rewards and init_speed_rewards
//   {
//     const Eigen::VectorXd &state = xs[0];
//     // const std::vector<DiskInfo> ego_disks = GetCoveringDiskInfo(
//     //     vehicle_model_parameter.measurement, state(StateIndex::X_POS),
//     //     state(StateIndex::Y_POS), state(StateIndex::THETA));
//     std::unique_ptr<ocp::CostCollection> running_costs(new
//     ocp::CostCollection);
//     // GenerateFormulaCosts(0, state, vehicle_model_parameter, ego_disks,
//     //                      trajectory_formula_ptr->costing_terms(),
//     primitives,
//     //                      running_costs.get());
//     std::unique_ptr<ocp::ConstraintCollection> running_constraints(
//     //     new ocp::ConstraintCollection);
//     // GenerateFormulaConstraints(0, state, vehicle_model_parameter,
//     ego_disks,
//     //                            trajectory_formula_ptr->costing_terms(),
//     //                            primitives, running_constraints.get());
//     (*running_models)[0] = std::make_shared<ocp::ActionModel>(
//         std::move(running_costs), std::move(running_constraints), dynamics,
//         dynamics->state_size(), dynamics->state_size(),
//         dynamics->input_size());
//   }
//
// #ifdef CSQP_WITH_MULTITHREADING
// #pragma omp parallel for num_threads(kNumOfThreads)
// #endif
//   for (int i = 1; i < kTrajectoryControlNum; ++i) {
//     const Eigen::VectorXd &state = xs[i];
//     // const std::vector<DiskInfo> ego_disks = GetCoveringDiskInfo(
//     //     vehicle_model_parameter.measurement, state(StateIndex::X_POS),
//     //     state(StateIndex::Y_POS), state(StateIndex::THETA));
//     std::unique_ptr<ocp::CostCollection> running_costs(new
//     ocp::CostCollection);
//     // GenerateFormulaCosts(i, state, vehicle_model_parameter, ego_disks,
//     //                      trajectory_formula_ptr->costing_terms(),
//     primitives,
//     //                      running_costs.get());
//     std::unique_ptr<ocp::ConstraintCollection> running_constraints(
//         new ocp::ConstraintCollection);
//     // GenerateFormulaConstraints(i, state, vehicle_model_parameter,
//     ego_disks,
//     //                            trajectory_formula_ptr->costing_terms(),
//     //                            primitives, running_constraints.get());
//     (*running_models)[i] = std::make_shared<ocp::ActionModel>(
//         std::move(running_costs), std::move(running_constraints), dynamics,
//         dynamics->state_size(), dynamics->state_size(),
//         dynamics->input_size());
//   }
//
//   {
//     const Eigen::VectorXd &state = xs[kTrajectoryStateNum - 1];
//     // const std::vector<DiskInfo> ego_disks = GetCoveringDiskInfo(
//     //     vehicle_model_parameter.measurement, state(StateIndex::X_POS),
//     //     state(StateIndex::Y_POS), state(StateIndex::THETA));
//     std::unique_ptr<ocp::CostCollection> terminal_costs(
//         new ocp::CostCollection);
//     // GenerateFormulaCosts(kTrajectoryStateNum - 1, state,
//     //                      vehicle_model_parameter, ego_disks,
//     //                      trajectory_formula_ptr->costing_terms(),
//     primitives,
//     //                      terminal_costs.get());
//     std::unique_ptr<ocp::ConstraintCollection> terminal_constraints(
//         new ocp::ConstraintCollection);
//     // GenerateFormulaConstraints(kTrajectoryStateNum - 1, state,
//     //                            vehicle_model_parameter, ego_disks,
//     //                            trajectory_formula_ptr->costing_terms(),
//     //                            primitives, terminal_constraints.get());
//     *terminal_model = std::make_shared<ocp::ActionModel>(
//         std::move(terminal_costs), std::move(terminal_constraints), dynamics,
//         dynamics->state_size(), dynamics->state_size(), 0);
//   }
//
//   {
//     clock_t end = clock();
//     double cost_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
//     std::cout << "GenerateSqpModels GenerateFormulaCosts cost time: "
//               << cost_time;
//   }
//
//   return;
// }
//
// bool SqpSolve(
//     const apollo::planning::ContingencyTrajectoryPlanningCfg
//     &trajectory_config, const
//     std::shared_ptr<apollo::planning::ReferenceLineInfo>
//         &reference_line_info,
//     const apollo::planning::convex_space::ConvexSpace &convex_space,
//     const std::shared_ptr<Formula<FormulaType::kTrajectory>>
//         &trajectory_formula_ptr,
//     const Model<FormulaType::kTrajectory>::SequenceSpecification
//         &sequence_specification,
//     const Formula<FormulaType::kTrajectory>::ControlSequence &us_init,
//     const ModelParameter &vehicle_model_parameter,
//     TrajectoryPrimitives &primitives,
//     std::unordered_map<std::string, std::vector<double>> &last_obs,
//     apollo::planning::MotionFrame *frame_ptr,
//     apollo::planning::SqpDebugInfo *const sqp_debug_info_ptr) {
//   StateSequence xs_init =
//       trajectory_formula_ptr->model().EvaluateControlSequence(
//           sequence_specification, us_init);
//
//   std::vector<Eigen::VectorXd> init_x;
//   init_x.reserve(xs_init.size());
//   std::vector<Eigen::VectorXd> init_u;
//   init_u.reserve(us_init.size());
//
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
//
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
//
//   DevastatorWrapperInfo devastator_wrapper_info(
//       trajectory_config, reference_line_info, convex_space,
//       vehicle_model_parameter, sequence_specification,
//       trajectory_formula_ptr, primitives, last_obs, frame_ptr);
//
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
//
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
//
//   shooting_problem->set_nthreads(kNumOfThreads);
//   ocp::ConstrainedSqp sqp_solver(settings, shooting_problem);
//   sqp_solver.set_devastator_wrapper_function(&devastator_wrapper_info,
//                                              GenerateSqpModels);
//   ocp::Convergence ret = sqp_solver.solve(init_x, init_u);
//
//   clock_t end = clock();
//   double cost_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
//   std::cout << "Sqp cost time: "<<  cost_time;
//
//   const std::vector<Eigen::VectorXd> &opt_xs = sqp_solver.get_xs();
//   const std::vector<Eigen::VectorXd> &opt_us = sqp_solver.get_us();
//
//
//   std::stringstream ss;
//   ss << ocp::ConvergenceToString(ret);
//   ss << "\ndx_norm:" << std::setprecision(2) << sqp_solver.get_dx_norm();
//   ss << ", du_norm:" << std::setprecision(2) << sqp_solver.get_du_norm();
//   ss << "\nconstraint_norm:" << std::setprecision(2)
//      << sqp_solver.get_constraint_norm();
//   ss << "\nmerit_diff:" << std::setprecision(2) <<
//   sqp_solver.get_merit_diff(); ss << "\ngap_norm:" << std::setprecision(2) <<
//   sqp_solver.get_gap_norm(); ss << "\nKKT:" << sqp_solver.get_KKT(); ss <<
//   "\ncost_time:" << std::setprecision(2) << cost_time; ss << "ms\niter:" <<
//   sqp_solver.get_iter(); ss << "\ncost:" << std::setprecision(5) <<
//   sqp_solver.get_cost(); ss << "\n";
//
//   *sqp_debug_info_ptr->mutable_debug_str() = ss.str();
//
//   if (ret == ocp::Convergence::FALSE) {
//     return false;
//   }
//   return true;
// }