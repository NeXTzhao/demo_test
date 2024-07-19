#include <iostream>
#include <cmath>
#include "linesearch.hpp"

using namespace linesearch;

// 示例目标函数：f(x) = (x - 2)^2
void MeritFunction(double alpha, double* phi, double* dphi) {
  double x = alpha; // 假设我们从 x = 2.0 开始，沿着方向 1 搜索
  *phi = (x - 2.0) * (x - 2.0); // f(x) = (x - 2)^2
  if (dphi) {
    *dphi = 2.0 * (x - 2.0); // f'(x) = 2(x - 2)
  }
//  std::cout << "MeritFunction: alpha = " << alpha << ", phi = " << *phi << ", dphi = " << *dphi << std::endl;
}

int main() {
  // 创建一个 CubicLineSearch 对象
  CubicLineSearch line_search;

  // 设置 Wolfe 条件参数
  line_search.SetOptimalityTolerances(1e-4, 0.9);
  line_search.SetVerbose(true);

  // 初始步长
  double alpha0 = 1.0;
  // 初始目标函数值和导数值
  double phi0, dphi0;
  MeritFunction(0, &phi0, &dphi0); // 在 alpha = 0 处计算目标函数值和导数值
//  std::cout << "Initial phi0: " << phi0 << ", Initial dphi0: " << dphi0 << std::endl;

  // 执行线搜索
  double alpha = line_search.Run(MeritFunction, alpha0, phi0, dphi0);

  // 输出结果
  double final_phi, final_dphi;
  line_search.GetFinalMeritValues(&final_phi, &final_dphi);
  int iterations = line_search.Iterations();
  std::cout << "Optimal step size: " << alpha << std::endl;
  std::cout << "Function value at optimal step: " << final_phi << std::endl;
  std::cout << "Derivative value at optimal step: " << final_dphi << std::endl;
  std::cout << "Number of iterations: " << iterations << std::endl;

  return 0;
}
