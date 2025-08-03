#pragma once

#include "common.h"
#include <unordered_set>
#include <vector>

enum FeatureType {
  EFFORT_ACCELERATION = 0,
  EFFORT_JERK = 1,
  EFFORT_DELTA = 2,
  EFFORT_OMEGA = 3,
  EFFORT_ALPHA = 4,
  EFFORT_SPEED = 5,

  CONSTRAINT_ACCELERATION = 6,
  CONSTRAINT_JERK = 7,
  CONSTRAINT_DELTA = 8,
  CONSTRAINT_OMEGA = 9,
  CONSTRAINT_ALPHA = 10,
  CONSTRAINT_SPEED = 11,

  LATERAL_ACCEL = 12,
  LATERAL_JERK = 13,

  VIOLATION_STOP_LINE = 14,
  VIOLATION_SPEED_ZONE = 15,

  SOFT_S_REWARD_PROGRESS = 16,
  SOFT_ACC_S_COMFORT = 17,

  VIOLATION_LEFT_ROAD_BOUNDARY = 18,
  VIOLATION_RIGHT_ROAD_BOUNDARY = 19,
  VIOLATION_LEFT_LANE_BOUNDARY = 20,
  VIOLATION_RIGHT_LANE_BOUNDARY = 21,

  VIOLATION_STATIC_OBSTACLE = 22,
  VIOLATION_DYNAMIC_OBSTACLE = 23,

  SOFT_NOMINAL_HALF_ATTRACTION = 24,

  LAST_CYCLE_ATTRACTION = 25,

  FEATURE_DIM = 26
};

const std::unordered_set<FeatureType> cost_features = {
    FeatureType::EFFORT_ACCELERATION,
    FeatureType::EFFORT_DELTA,
    FeatureType::EFFORT_OMEGA,
    FeatureType::EFFORT_JERK,
    FeatureType::EFFORT_ALPHA,
    FeatureType::EFFORT_SPEED,

    FeatureType::SOFT_S_REWARD_PROGRESS,
    FeatureType::SOFT_ACC_S_COMFORT,
    FeatureType::SOFT_NOMINAL_HALF_ATTRACTION,

    FeatureType::VIOLATION_LEFT_LANE_BOUNDARY,
    FeatureType::VIOLATION_RIGHT_LANE_BOUNDARY,

    FeatureType::VIOLATION_SPEED_ZONE,

    FeatureType::VIOLATION_LEFT_ROAD_BOUNDARY,
    FeatureType::VIOLATION_RIGHT_ROAD_BOUNDARY,
    FeatureType::VIOLATION_DYNAMIC_OBSTACLE,
};

const std::unordered_set<FeatureType> constraint_features = {
    FeatureType::VIOLATION_LEFT_ROAD_BOUNDARY,
    FeatureType::VIOLATION_RIGHT_ROAD_BOUNDARY,
    FeatureType::VIOLATION_STATIC_OBSTACLE,
    FeatureType::VIOLATION_DYNAMIC_OBSTACLE,
};

inline std::string FeatureTypeToString(const FeatureType &feature_type) {
  switch (feature_type) {
  case FeatureType::EFFORT_ACCELERATION:
    return "acceleration_effort_cost";
  case FeatureType::EFFORT_JERK:
    return "jerk_effort_cost";
  case FeatureType::EFFORT_DELTA:
    return "delta_effort_cost";
  case FeatureType::EFFORT_OMEGA:
    return "omega_effort_cost";
  case FeatureType::EFFORT_ALPHA:
    return "alpha_effort_cost";
  case FeatureType::EFFORT_SPEED:
    return "speed_effort_cost";

  case FeatureType::CONSTRAINT_ACCELERATION:
    return "constraint_acceleration";
  case FeatureType::CONSTRAINT_JERK:
    return "constraint_jerk";
  case FeatureType::CONSTRAINT_DELTA:
    return "constraint_delta";
  case FeatureType::CONSTRAINT_OMEGA:
    return "constraint_omega";
  case FeatureType::CONSTRAINT_ALPHA:
    return "constraint_alpha";
  case FeatureType::CONSTRAINT_SPEED:
    return "constraint_speed";

  case FeatureType::LATERAL_ACCEL:
    return "lateral_acceleration";
  case FeatureType::LATERAL_JERK:
    return "lateral_jerk";

  case FeatureType::VIOLATION_STOP_LINE:
    return "violation_stop_line";
  case FeatureType::VIOLATION_SPEED_ZONE:
    return "violation_speed_zone";

  case FeatureType::SOFT_S_REWARD_PROGRESS:
    return "s_reward_progress_cost";
  case FeatureType::SOFT_ACC_S_COMFORT:
    return "acc_comfort_cost";

  case FeatureType::VIOLATION_LEFT_ROAD_BOUNDARY:
    return "left_road_boundary_cost";
  case FeatureType::VIOLATION_RIGHT_ROAD_BOUNDARY:
    return "right_road_boundary_cost";
  case FeatureType::VIOLATION_LEFT_LANE_BOUNDARY:
    return "left_lane_boundary_cost";
  case FeatureType::VIOLATION_RIGHT_LANE_BOUNDARY:
    return "right_lane_boundary_cost";
  case FeatureType::VIOLATION_STATIC_OBSTACLE:
    return "static_obstacle_cost";
  case FeatureType::VIOLATION_DYNAMIC_OBSTACLE:
    return "dynamic_obstacle_cost";

  case FeatureType::SOFT_NOMINAL_HALF_ATTRACTION:
    return "nominal_half_attraction_cost";
  case FeatureType::LAST_CYCLE_ATTRACTION:
    return "last_cycle_attraction_cost";
  }

  throw std::invalid_argument(
      "Unknown FeatureType enum value in FeatureTypeToString");
}

enum CostingType { QUADRATIC = 0, LINEAR = 1 };

struct CostingSpefic {
  CostingType type = CostingType::QUADRATIC;
  std::vector<double> weight_vec;
  double aditional_param = 0.0;
};
struct CostingTerm {
  FeatureType feature_type;
  CostingSpefic costing_spec;
};

inline std::vector<CostingTerm> GetCostingTerm() {
  const int N = kTrajectoryStateNum;
  // clang-format off

  // 1. 为 cost_features 准备容器
  std::vector<double> w_effort_acceleration;          w_effort_acceleration.reserve(N);
  std::vector<double> w_effort_delta;                 w_effort_delta.reserve(N);
  std::vector<double> w_effort_omega;                 w_effort_omega.reserve(N);
  std::vector<double> w_effort_jerk;                  w_effort_jerk.reserve(N);
  std::vector<double> w_effort_alpha;                 w_effort_alpha.reserve(N);
  std::vector<double> w_effort_speed;                 w_effort_speed.reserve(N);

  std::vector<double> w_soft_s_reward_progress;       w_soft_s_reward_progress.reserve(N);
  std::vector<double> w_soft_acc_s_comfort;           w_soft_acc_s_comfort.reserve(N);
  std::vector<double> w_soft_nominal_half_attraction; w_soft_nominal_half_attraction.reserve(N);

  std::vector<double> w_violation_left_lane_boundary;  w_violation_left_lane_boundary.reserve(N);
  std::vector<double> w_violation_right_lane_boundary; w_violation_right_lane_boundary.reserve(N);
  std::vector<double> w_violation_speed_zone;          w_violation_speed_zone.reserve(N);
  std::vector<double> w_violation_left_road_boundary;  w_violation_left_road_boundary.reserve(N);
  std::vector<double> w_violation_right_road_boundary; w_violation_right_road_boundary.reserve(N);
  std::vector<double> w_violation_dynamic_obstacle;    w_violation_dynamic_obstacle.reserve(N);

  // 2. 一个大的 for：所有权重暂定设为 1.0
  for (int i = 0; i < N; ++i) {
    w_effort_acceleration.push_back(1.0);
    w_effort_delta.push_back(1.0);
    w_effort_omega.push_back(1.0);
    w_effort_jerk.push_back(1.0);
    w_effort_alpha.push_back(1.0);
    w_effort_speed.push_back(1.0);

    w_soft_s_reward_progress.push_back(1.0);
    w_soft_acc_s_comfort.push_back(1.0);
    w_soft_nominal_half_attraction.push_back(1.0);

    w_violation_left_lane_boundary.push_back(1.0);
    w_violation_right_lane_boundary.push_back(1.0);
    w_violation_speed_zone.push_back(1.0);
    w_violation_left_road_boundary.push_back(1.0);
    w_violation_right_road_boundary.push_back(1.0);
    w_violation_dynamic_obstacle.push_back(1.0);
  }

  // 3. 组装 terms
  std::vector<CostingTerm> terms;
  terms.reserve(static_cast<int>(FeatureType::FEATURE_DIM));

  terms.push_back({FeatureType::EFFORT_ACCELERATION,{CostingType::QUADRATIC, std::move(w_effort_acceleration), 0.0}});
  terms.push_back({FeatureType::EFFORT_DELTA, {CostingType::QUADRATIC, std::move(w_effort_delta), 0.0}});
  terms.push_back({FeatureType::EFFORT_OMEGA, {CostingType::QUADRATIC, std::move(w_effort_omega), 0.0}});
  terms.push_back({FeatureType::EFFORT_JERK, {CostingType::QUADRATIC, std::move(w_effort_jerk), 0.0}});
  terms.push_back({FeatureType::EFFORT_ALPHA, {CostingType::QUADRATIC, std::move(w_effort_alpha), 0.0}});
  terms.push_back({FeatureType::EFFORT_SPEED, {CostingType::QUADRATIC, std::move(w_effort_speed), 0.0}});

  terms.push_back({FeatureType::SOFT_S_REWARD_PROGRESS, {CostingType::QUADRATIC, std::move(w_soft_s_reward_progress), 0.0}});
  terms.push_back({FeatureType::SOFT_ACC_S_COMFORT, {CostingType::QUADRATIC, std::move(w_soft_acc_s_comfort), 0.0}});
  terms.push_back({FeatureType::SOFT_NOMINAL_HALF_ATTRACTION, {CostingType::QUADRATIC, std::move(w_soft_nominal_half_attraction), 0.0}});

  terms.push_back({FeatureType::VIOLATION_LEFT_LANE_BOUNDARY, {CostingType::QUADRATIC, std::move(w_violation_left_lane_boundary), 0.0}});
  terms.push_back({FeatureType::VIOLATION_RIGHT_LANE_BOUNDARY, {CostingType::QUADRATIC, std::move(w_violation_right_lane_boundary), 0.0}});
  terms.push_back({FeatureType::VIOLATION_SPEED_ZONE, {CostingType::QUADRATIC, std::move(w_violation_speed_zone), 0.0}});
  terms.push_back({FeatureType::VIOLATION_LEFT_ROAD_BOUNDARY, {CostingType::QUADRATIC, std::move(w_violation_left_road_boundary), 0.0}});
  terms.push_back({FeatureType::VIOLATION_RIGHT_ROAD_BOUNDARY, {CostingType::QUADRATIC, std::move(w_violation_right_road_boundary), 0.0}});
  terms.push_back({FeatureType::VIOLATION_DYNAMIC_OBSTACLE, {CostingType::QUADRATIC, std::move(w_violation_dynamic_obstacle), 0.0}});

//clang-format on
  return terms;
}