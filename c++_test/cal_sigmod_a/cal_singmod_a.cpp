#include <cmath>
#include <iostream>

// 计算 a，使得 sigmoid(x + bias, a) ≈ epsilon
double SolveAForSigmoidEpsilon(double x, double epsilon, double bias) {
  const double default_a = 1.0;  // 你可以根据需要更改默认值
  const double min_epsilon = 1e-8;
  const double max_epsilon = 1.0 - 1e-8;

  // 手动 clamp epsilon 范围，兼容 C++11/C++14
  if (epsilon < min_epsilon) {
    std::cerr << "Warning: epsilon too small, clamped to " << min_epsilon << std::endl;
    epsilon = min_epsilon;
  } else if (epsilon > max_epsilon) {
    std::cerr << "Warning: epsilon too large, clamped to " << max_epsilon << std::endl;
    epsilon = max_epsilon;
  }

  double offset = x + bias;
  if (offset == 0.0) {
    std::cerr << "Warning: x + bias == 0. Returning default a = " << default_a << std::endl;
    return default_a;
  }

  double log_term = std::log((1.0 - epsilon) / epsilon);
  double a = -log_term / offset;

  return a;
}

// 示例
int main() {
  double x = 0.0;
  double bias = 0.5;
  double epsilon = 0.01;

  double a = SolveAForSigmoidEpsilon(x, epsilon, bias);
  std::cout << "Computed a: " << a << std::endl;

  return 0;
}
