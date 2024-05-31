#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

struct TrajectoryPoint {
  double x, y;
  double theta;
  double s;
  double steer;
};

double calculateDistance(const TrajectoryPoint& p1, const TrajectoryPoint& p2) {
  double posDistance = std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
  double thetaDiff = std::fmod(std::fabs(p1.theta - p2.theta), 360.0);
  if (thetaDiff > 180.0) thetaDiff = 360.0 - thetaDiff;

  double steerDiff = std::fmod(std::fabs(p1.steer - p2.steer), 360.0);
  if (steerDiff > 180.0) steerDiff = 360.0 - steerDiff;

  // 考虑弧长s的差异
  double sDiff = std::fabs(p1.s - p2.s);

  // 组合差异
  return posDistance + thetaDiff + steerDiff + sDiff;// 可根据需要调整权重
}

double frechetDistance(const std::vector<TrajectoryPoint>& traj1, const std::vector<TrajectoryPoint>& traj2) {
  size_t n = traj1.size(), m = traj2.size();
  std::vector<std::vector<double>> dp(n + 1, std::vector<double>(m + 1, std::numeric_limits<double>::infinity()));

  dp[0][0] = 0;// Starting point

  for (size_t i = 1; i <= n; ++i) {
    for (size_t j = 1; j <= m; ++j) {
      double cost = calculateDistance(traj1[i - 1], traj2[j - 1]);
      dp[i][j] = std::min({dp[i - 1][j] + cost, dp[i][j - 1] + cost, dp[i - 1][j - 1] + cost});
    }
  }

  return dp[n][m];
}

std::vector<TrajectoryPoint> generateSpiralArc(double startRadius,
                                               double endRadius,
                                               double startAngle,
                                               double endAngle,
                                               int numPoints) {
  std::vector<TrajectoryPoint> trajectory;
  double deltaAngle = (endAngle - startAngle) / (numPoints - 1);
  double radiusIncrement = (endRadius - startRadius) / (numPoints - 1);
  TrajectoryPoint previousPoint{};

  for (int i = 0; i < numPoints; ++i) {
    double angle = startAngle + i * deltaAngle;// Angle in degrees
    double radius = startRadius + i * radiusIncrement;
    double radian = angle * M_PI / 180;// Convert angle to radians for trigonometric functions

    TrajectoryPoint point{};
    point.x = radius * cos(radian);
    point.y = radius * sin(radian);

    if (i == 0) {         // Initial orientation and steering angle
      point.theta = angle;// Initial direction aligned with the first segment
      point.steer = 0;    // Initial steering angle
    } else {
      point.theta = atan2(point.y - previousPoint.y, point.x - previousPoint.x) * 180 / M_PI;
      point.steer = (angle - previousPoint.theta);// Simplified steering calculation
      point.steer = std::fmin(std::fmax(point.steer, -180.0), 180.0);
    }

    // Calculating path length s (approximation)
    if (i == 0) {
      point.s = 0;// Starting point
    } else {
      double dx = point.x - previousPoint.x;
      double dy = point.y - previousPoint.y;
      point.s = previousPoint.s + sqrt(dx * dx + dy * dy);// Incremental path length
    }

    trajectory.push_back(point);
    previousPoint = point;
  }

  return trajectory;
}


struct Thresholds {
  double posThreshold;  // 位置阈值
  double thetaThreshold;// 朝向阈值
  double sThreshold;    // 弧长阈值
};

bool compareTrajectoriesWithAveragePoints(const std::vector<TrajectoryPoint>& traj1, const std::vector<TrajectoryPoint>& traj2, const Thresholds& thresholds) {
  if (traj1.size() != traj2.size()) {
    std::cout << "Trajectories do not have the same number of points.\n";
    return false;
  }

  for (size_t i = 0; i < traj1.size(); ++i) {
    // Initialize average points with current point
    TrajectoryPoint avg1 = traj1[i], avg2 = traj2[i];
    int count = 1;

    // Add previous point if exists
    if (i > 0) {
      avg1.x += traj1[i - 1].x;
      avg1.y += traj1[i - 1].y;
      avg1.theta += traj1[i - 1].theta;
      avg1.s += traj1[i - 1].s;
      avg2.x += traj2[i - 1].x;
      avg2.y += traj2[i - 1].y;
      avg2.theta += traj2[i - 1].theta;
      avg2.s += traj2[i - 1].s;
      count++;
    }
    // Add next point if exists
    if (i < traj1.size() - 1) {
      avg1.x += traj1[i + 1].x;
      avg1.y += traj1[i + 1].y;
      avg1.theta += traj1[i + 1].theta;
      avg1.s += traj1[i + 1].s;
      avg2.x += traj2[i + 1].x;
      avg2.y += traj2[i + 1].y;
      avg2.theta += traj2[i + 1].theta;
      avg2.s += traj2[i + 1].s;
      count++;
    }
    // Calculate the average
    avg1.x /= count;
    avg1.y /= count;
    avg1.theta /= count;
    avg1.s /= count;
    avg2.x /= count;
    avg2.y /= count;
    avg2.theta /= count;
    avg2.s /= count;
    
    // Calculate differences
    double dist = std::sqrt(std::pow(avg1.x - avg2.x, 2) + std::pow(avg1.y - avg2.y, 2));
    double thetaDiff = std::fabs(avg1.theta - avg2.theta);
    double sDiff = std::fabs(avg1.s - avg2.s);

    // Check against thresholds
    if (dist > thresholds.posThreshold || thetaDiff > thresholds.thetaThreshold || sDiff > thresholds.sThreshold) {
      std::cout << "Difference exceeded at point " << i << ":\n";
      if (dist > thresholds.posThreshold) {
        std::cout << "  Position difference: " << dist << " (Threshold: " << thresholds.posThreshold << ")\n";
      }
      if (thetaDiff > thresholds.thetaThreshold) {
        std::cout << "  Orientation difference: " << thetaDiff << " (Threshold: " << thresholds.thetaThreshold << ")\n";
      }
      if (sDiff > thresholds.sThreshold) {
        std::cout << "  Path length difference: " << sDiff << " (Threshold: " << thresholds.sThreshold << ")\n";
      }
      return false;
    }
  }

  return true;
}

int main() {

  double startRadius = 0;
  double endRadius = 23;
  double startAngle = 0;
  double endAngle = 32.5;

  double startRadius1 = 0;
  double endRadius1 = 23;
  double startAngle1 = -20;
  double endAngle1 = 60;

  int numPoints = 50;
  auto old_traj = generateSpiralArc(startRadius, endRadius, startAngle, endAngle, numPoints);
  auto new_traj = generateSpiralArc(startRadius1, endRadius1, startAngle1, endAngle1, numPoints);

  std::cout << "Frechet distance: " << frechetDistance(old_traj, new_traj) << std::endl;

  Thresholds thresholds = {3.75, 30, 5.0};// Example thresholds for position, orientation, and path length

  if (compareTrajectoriesWithAveragePoints(old_traj, new_traj, thresholds)) {
    std::cout << "Trajectories are similar.\n";
  } else {
    std::cout << "Trajectories are not similar.\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  //  for (const auto &point : old_spiralArc) {
  //    std::cout << "x: " << point.x << ", y: " << point.y << ", theta: " << point.theta
  //              << ", s: " << point.s << ", steer: " << point.steer << std::endl;
  //  }
  std::cout << "////////////////////////////////////" << std::endl;

  //  for (const auto &point : new_spiralArc) {
  //    std::cout << "x: " << point.x << ", y: " << point.y << ", theta: " << point.theta
  //              << ", s: " << point.s << ", steer: " << point.steer << std::endl;
  //  }

  std::vector<double> old_x, old_y, new_x, new_y;
  for (int i = 0; i < numPoints; ++i) {
    old_x.push_back(old_traj.at(i).x);
    old_y.push_back(old_traj.at(i).y);
    new_x.push_back(new_traj.at(i).x);
    new_y.push_back(new_traj.at(i).y);
  }

  // Plotting
  plt::named_plot("old", old_x, old_y, "r.");
  plt::named_plot("new", new_x, new_y, "b.");
  plt::xlabel("X");
  plt::ylabel("Y");
  plt::legend();
  plt::axis("equal");// Ensure equal aspect ratio
  plt::show();
  return 0;
}
