#pragma once

#include <Eigen/Core>
#include <functional>

namespace ocp {

template <class T>
static T integrate_forward_euler(
    const double dt, const Eigen::Ref<const T> &x, const Eigen::Ref<const T> &u,
    std::function<T(const Eigen::Ref<const T> &, const Eigen::Ref<const T> &)>
        dynamics) {
  return x + dt * dynamics(x, u);
}

template <class T>
static T integrate_modified_euler(
    const double dt, const Eigen::Ref<const T> &x, const Eigen::Ref<const T> &u,
    std::function<T(const Eigen::Ref<const T> &, const Eigen::Ref<const T> &)>
        dynamics) {
  T k1 = dynamics(x, u);
  T k2 = dynamics(x + dt * k1, u);
  return x + dt * (k1 + k2) / 2;
}

template <class T>
static T integrate_rk2(
    const double dt, const Eigen::Ref<const T> &x, const Eigen::Ref<const T> &u,
    std::function<T(const Eigen::Ref<const T> &, const Eigen::Ref<const T> &)>
        dynamics) {
  T k1 = dynamics(x, u);
  T k2 = dynamics(x + 0.5 * dt * k1, u);
  return x + dt * k2;
}

template <class T>
static T integrate_rk4(
    const double dt, const Eigen::Ref<const T> &x, const Eigen::Ref<const T> &u,
    std::function<T(const Eigen::Ref<const T> &, const Eigen::Ref<const T> &)>
        dynamics) {
  T k1 = dt * dynamics(x, u);
  T k2 = dt * dynamics(x + k1 / 2, u);
  T k3 = dt * dynamics(x + k2 / 2, u);
  T k4 = dt * dynamics(x + k3, u);
  return x + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
}

} // namespace ocp
