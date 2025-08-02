#include <iostream>

#include "Jx.h"
#include "casadi_interface.h"
#include "f.h"  // 由 CasADi 导出的函数

int main() {
  // 初始化
  CasadiFunction<double> func(f, f_sparsity_out, f_work);

  std::vector<std::vector<double>> inputs = {
      {1.0, 2.0, 3.0},  // 输入1
      {4.0, 5.0}        // 输入2
  };

  // 执行计算
  auto outputs = func.evaluate(inputs);

  // 访问结果
  for (const auto& output : outputs) {
    for (double val : output) {
      std::cout << val << " ";
    }
    std::cout << "\n";
  }

  return 0;
}
