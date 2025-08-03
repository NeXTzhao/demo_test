#pragma once

#include <string>
#include <vector>
// The planning node frequency 0.1s.
constexpr double kPlanningCycleTime = 0.1;

// Trajectory horizon, in seconds.
constexpr double kTrajectoryHorizonInSec = 6.8;

// Trajectory states interval, in seconds.
constexpr double kTrajectoryIntervalInSec = 0.2;

// Trajectory state/control #.
constexpr int kTrajectoryStateNum =
    static_cast<int>(kTrajectoryHorizonInSec / kTrajectoryIntervalInSec) + 1;

constexpr int kTrajectoryControlNum = kTrajectoryStateNum - 1;

constexpr double kEpsilon = 1e-3;

constexpr double kEgoDiskSize = 3.0;

// Car measurement.
struct Measurement {
  double wheel_base = 2.85;
  double front_bumper_to_front_axle = 1.0;
  double rear_bumper_to_rear_axle = 1.0;
  double width = 1.740;
  double length = 4.7405;
  double steering_gear_ratio = 18.8;
};

// Model Parameter.
struct MinMaxLimit {
  double min_val = 0.0;
  double max_val = 0.0;
};

struct Limit {
  MinMaxLimit steering_limit{-0.5, 0.5};
  MinMaxLimit steering_speed_limit{-0.5, 0.5};
  MinMaxLimit steering_accel_limit{-2.0, 2.0};
  MinMaxLimit soft_steering_limit{-0.5, 0.5};
  MinMaxLimit soft_steering_speed_limit{-0.5, 0.5};
  MinMaxLimit soft_steering_accel_limit{-2.0, 2.0};

  MinMaxLimit speed_limit{0.0, 35};
  MinMaxLimit accel_limit{-5.0, 2.0};
  MinMaxLimit jerk_limit{-5.0, 2.0};
  MinMaxLimit accel_soft_limit{-2.0, 2.0};
  MinMaxLimit jerk_soft_limit{-2.0, 2.0};
  double extreme_lateral_acceleration = 3.0;
};
struct ModelParameter {
  std::string vechile_type;
  Measurement measurement;
  Limit limit;
  Limit modified_limit;
};
