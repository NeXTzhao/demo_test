//
// Created by next on 24-7-27.
//

#pragma once

// 通用范数计算函数模板
template<int Order, int StateDim, int ControlDim>
typename std::enable_if<Order != 0, double>::type computeNorm(
    const std::vector<Eigen::Matrix<double, ControlDim, 1>>& delta_u) {
  double norm = 0.0;
  for (const auto& control : delta_u) {
    norm += std::pow(control.template lpNorm<Order>(), Order);
  }
  return std::pow(norm, 1.0 / Order);
}

// 特化无穷范数
template<int Order, int StateDim, int ControlDim>
typename std::enable_if<Order == 0, double>::type computeNorm(
    const std::vector<Eigen::Matrix<double, ControlDim, 1>>& delta_u) {
  double max_norm = 0.0;
  for (const auto& control : delta_u) {
    max_norm = std::max(max_norm, control.template lpNorm<Eigen::Infinity>());
  }
  return max_norm;
}