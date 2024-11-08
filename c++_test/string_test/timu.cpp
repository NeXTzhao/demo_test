////
//// Created by next on 2024/10/30.
////
//
//#include <iostream>
//#include <cmath>
//#include <stdexcept>
//
//double f(double x) {
//  return std::pow(x, 5) + x - 1;
//}
//
//double bisection_method(double a, double b, double tolerance = 1e-6, int max_iterations = 100) {
//  if (f(a) * f(b) >= 0) {
//    throw std::invalid_argument("初始区间[a, b]未能包含根：f(a) 和 f(b) 的符号应相反。");
//  }
//
//  double midpoint;
//  for (int iteration = 0; iteration < max_iterations; ++iteration) {
//    midpoint = (a + b) / 2.0;
//    double f_mid = f(midpoint);
//
//    // 检查是否达到精度要求
//    if (std::fabs(f_mid) < tolerance || (b - a) / 2.0 < tolerance) {
//      return midpoint;
//    }
//
//    // 根据符号缩小区间
//    if (f(a) * f_mid < 0) {
//      b = midpoint;
//    } else {
//      a = midpoint;
//    }
//  }
//  throw std::runtime_error("未能在最大迭代次数内收敛");
//}
//
//int main() {
//  double a = 0.0;   // 初始左端点
//  double b = 1.0;   // 初始右端点
//  try {
//    double solution = bisection_method(a, b);
//    std::cout << "解为 x = " << solution << std::endl;
//  } catch (const std::exception& e) {
//    std::cerr << e.what() << std::endl;
//  }
//  return 0;
//}

#include <iostream>
#include <cmath>
#include <stdexcept>

double f(double x) {
  return std::pow(x, 5) + x - 1;
}

double f_prime(double x) {
  return 5 * std::pow(x, 4) + 1;
}

double newton_method(double initial_guess, double tolerance = 1e-6, int max_iterations = 100) {
  double x = initial_guess;
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    double fx = f(x);
    double fpx = f_prime(x);

    // 更新 x 的值
    double x_new = x - fx / fpx;

    // 检查收敛条件
    if (std::fabs(x_new - x) < tolerance) {
      return x_new;  // 返回解
    }

    x = x_new;
  }
  throw std::runtime_error("未能在最大迭代次数内收敛");
}

int main() {
  double initial_guess = 0.5;  // 初始猜测值
  try {
    double solution = newton_method(initial_guess);
    std::cout << "解为 x = " << solution << std::endl;
  } catch (const std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
  }
  return 0;
}
