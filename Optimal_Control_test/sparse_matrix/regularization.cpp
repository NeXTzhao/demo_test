#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

// 创建 Hilbert 矩阵
MatrixXd createHilbertMatrix(int n) {
  MatrixXd H(n, n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      H(i, j) = 1.0 / (i + j + 1);
    }
  }
  return H;
}

// 计算矩阵的条件数
double computeConditionNumber(const MatrixXd& matrix) {
  JacobiSVD<MatrixXd> svd(matrix);
  double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
  return cond;
}

int main() {
  // 创建 10x10 的 Hilbert 矩阵
  int n = 10;
  MatrixXd H = createHilbertMatrix(n);
  std::cout << "Original Hilbert matrix H:\n" << H << "\n\n";

  // 计算特征值
  SelfAdjointEigenSolver<MatrixXd> eigensolver(H);
  if (eigensolver.info() != Success) {
    std::cerr << "Eigenvalue decomposition failed!\n";
    return -1;
  }
  std::cout << "Eigenvalues of H:\n" << eigensolver.eigenvalues() << "\n\n";

  // 计算条件数
  double condH = computeConditionNumber(H);
  std::cout << "Condition number of H: " << condH << "\n\n";

  // 在矩阵中引入一个微小的扰动
  double epsilon = 1e-10;
  MatrixXd H_perturbed = H;
  H_perturbed(4, 4) -= epsilon; // 修改 H 的 (5, 5) 元素
  std::cout << "Perturbed Hilbert matrix H:\n" << H_perturbed << "\n\n";

  // 计算扰动后矩阵的特征值
  SelfAdjointEigenSolver<MatrixXd> perturbed_eigensolver(H_perturbed);
  if (perturbed_eigensolver.info() != Success) {
    std::cerr << "Eigenvalue decomposition failed for perturbed matrix!\n";
    return -1;
  }
  std::cout << "Eigenvalues of perturbed H:\n" << perturbed_eigensolver.eigenvalues() << "\n\n";

  // 计算扰动后矩阵的条件数
  double condH_perturbed = computeConditionNumber(H_perturbed);
  std::cout << "Condition number of perturbed H: " << condH_perturbed << "\n\n";

  return 0;
}
