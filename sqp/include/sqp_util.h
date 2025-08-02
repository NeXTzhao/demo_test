#pragma once

#include "ocp/csqp/constrained_sqp.h"
#include "ocp/model/vehicle_model_dynamics.h"

class WrapperInfo {
  static constexpr int X_DIM = VehicleModelDynamics::StateIndex::X_DIM;
  static constexpr int U_DIM = VehicleModelDynamics::ControlIndex::U_DIM;
  using StateIndex = VehicleModelDynamics::StateIndex;
  using ControlIndex = VehicleModelDynamics::ControlIndex;

  static constexpr int StateDim = static_cast<int>(X_DIM);
  static constexpr int ControlDim = static_cast<int>(U_DIM);

  using State = Eigen::Matrix<double, StateDim, 1>;
  using Control = Eigen::Matrix<double, ControlDim, 1>;

  using StateSequence = std::vector<State>;
  using ControlSequence = std::vector<Control>;

public:
  // explicit WrapperInfo(
  //     const apollo::planning::ContingencyTrajectoryPlanningCfg
  //         &trajectory_config,
  //     const std::shared_ptr<apollo::planning::ReferenceLineInfo>
  //         &reference_line_info,
  //     const apollo::planning::convex_space::ConvexSpace &convex_space,
  //     const ModelParameter &vehicle_model_parameter,
  //     const Model<FormulaType::kTrajectory>::SequenceSpecification
  //         &sequence_specification,
  //     const std::shared_ptr<Formula<FormulaType::kTrajectory>>
  //         &trajectory_formula_ptr,
  //     TrajectoryPrimitives &primitives,
  //     std::unordered_map<std::string, std::vector<double>> &last_obs,
  //     apollo::planning::MotionFrame *frame_ptr)
  //     : trajectory_config_(trajectory_config),
  //       reference_line_info_(reference_line_info),
  //       convex_space_(convex_space),
  //       vehicle_model_parameter_(vehicle_model_parameter),
  //       sequence_specification_(sequence_specification),
  //       trajectory_formula_ptr_(trajectory_formula_ptr),
  //       primitives_(primitives), last_obs_(last_obs), frame_ptr_(frame_ptr)
  //       {}
  WrapperInfo() = delete;
  ~WrapperInfo() = default;
  //
  // inline const apollo::planning::ContingencyTrajectoryPlanningCfg &
  // trajectory_config() const {
  //   return trajectory_config_;
  // }
  // inline const std::shared_ptr<apollo::planning::ReferenceLineInfo> &
  // reference_line_info() const {
  //   return reference_line_info_;
  // }
  // inline const apollo::planning::convex_space::ConvexSpace &
  // convex_space() const {
  //   return convex_space_;
  // }
  // inline const ModelParameter &vehicle_model_parameter() const {
  //   return vehicle_model_parameter_;
  // }
  // inline const Model<FormulaType::kTrajectory>::SequenceSpecification &
  // sequence_specification() const {
  //   return sequence_specification_;
  // }
  // inline const std::shared_ptr<Formula<FormulaType::kTrajectory>> &
  // trajectory_formula_ptr() const {
  //   return trajectory_formula_ptr_;
  // }
  // inline TrajectoryPrimitives &primitives() { return primitives_; }
  // inline std::unordered_map<std::string, std::vector<double>> &last_obs() {
  //   return last_obs_;
  // }
  // inline apollo::planning::MotionFrame *frame_ptr() { return frame_ptr_; }

private:
  // const apollo::planning::ContingencyTrajectoryPlanningCfg
  // &trajectory_config_; const
  // std::shared_ptr<apollo::planning::ReferenceLineInfo>
  //     &reference_line_info_;
  // const apollo::planning::convex_space::ConvexSpace &convex_space_;
  // const ModelParameter &vehicle_model_parameter_;
  // const Model<FormulaType::kTrajectory>::SequenceSpecification
  //     &sequence_specification_;
  // const std::shared_ptr<Formula<FormulaType::kTrajectory>>
  //     &trajectory_formula_ptr_;
  // TrajectoryPrimitives &primitives_;
  // std::unordered_map<std::string, std::vector<double>> &last_obs_;
  // apollo::planning::MotionFrame *frame_ptr_;
};

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
//     apollo::planning::SqpDebugInfo *const sqp_debug_info_ptr);
