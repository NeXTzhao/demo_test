#include <Eigen/Dense>
#include <Eigen/SparseLU>
#include <iostream>
#include <chrono>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

// 生成稠密矩阵
MatrixXd generateDenseMatrix(int size) {
  return MatrixXd::Random(size, size);
}

// 生成稀疏矩阵
SparseMatrix<double> generateSparseMatrix(int size, double density) {
  SparseMatrix<double> mat(size, size);
  for (int k = 0; k < mat.outerSize(); ++k) {
    for (SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
      if ((double) rand() / RAND_MAX < density) {
        it.valueRef() = (double) rand() / RAND_MAX;
      }
    }
  }
  return mat;
}

// 稠密矩阵直接求解
double solveDenseDirect(const MatrixXd& matrix, const VectorXd& b) {
  auto start = high_resolution_clock::now();
  VectorXd x = matrix.colPivHouseholderQr().solve(b);
  auto end = high_resolution_clock::now();
  return duration_cast<duration<double>>(end - start).count();
}

// 稠密矩阵转稀疏再求解
double solveDenseToSparse(const MatrixXd& matrix, const VectorXd& b) {
  SparseMatrix<double> sparse_matrix = matrix.sparseView();
  auto start = high_resolution_clock::now();
  SparseLU<SparseMatrix<double>> solver;
  solver.compute(sparse_matrix);
  VectorXd x = solver.solve(b);
  auto end = high_resolution_clock::now();
  return duration_cast<duration<double>>(end - start).count();
}

// 稀疏矩阵直接求解
double solveSparseDirect(const SparseMatrix<double>& matrix, const VectorXd& b) {
  auto start = high_resolution_clock::now();
  SparseLU<SparseMatrix<double>> solver;
  solver.compute(matrix);
  VectorXd x = solver.solve(b);
  auto end = high_resolution_clock::now();
  return duration_cast<duration<double>>(end - start).count();
}

int main() {
  int size = 1000;
  double density = 0.01;

  // 生成矩阵和向量
  MatrixXd dense_matrix = generateDenseMatrix(size);
  SparseMatrix<double> sparse_matrix = generateSparseMatrix(size, density);
  VectorXd b = VectorXd::Random(size);

  // 计算并输出耗时
  double time_dense_direct = solveDenseDirect(dense_matrix, b);
  double time_dense_to_sparse = solveDenseToSparse(dense_matrix, b);
  double time_sparse_direct = solveSparseDirect(sparse_matrix, b);

  cout << "Dense Direct Solve Time: " << time_dense_direct << " seconds" << endl;
  cout << "Dense to Sparse Solve Time: " << time_dense_to_sparse << " seconds" << endl;
  cout << "Sparse Direct Solve Time: " << time_sparse_direct << " seconds" << endl;

  return 0;
}
