#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <chrono>

using namespace std;
using namespace Eigen;

int main() {
  // 问题规模
  int n = 2000;

  // 生成一个随机的稠密矩阵Q和向量b
  MatrixXd Q = MatrixXd::Random(n, n);
  Q = Q.transpose() * Q;  // 使Q对称正定
  VectorXd b = VectorXd::Random(n);

  // 稠密矩阵求解
  auto start_dense = chrono::high_resolution_clock::now();
  VectorXd x_dense = Q.ldlt().solve(-b);
  auto end_dense = chrono::high_resolution_clock::now();
  chrono::duration<double> elapsed_dense = end_dense - start_dense;

  // 创建一个稀疏矩阵
//  SparseMatrix<double> Q_sparse(n, n);
//  vector<Triplet<double>> tripletList;
//  for (int i = 0; i < n; ++i) {
//    for (int j = 0; j < n; ++j) {
//      if (i == j || abs(i - j) <= 10) { // 保持对角线及其附近的稀疏性
//        tripletList.emplace_back(i, j, Q(i, j));
//      }
//    }
//  }
//  Q_sparse.setFromTriplets(tripletList.begin(), tripletList.end());

  // 稀疏矩阵求解
  // 将稠密矩阵Q转换为稀疏矩阵
  SparseMatrix<double> Q_sparse = Q.sparseView();
  SimplicialLDLT<SparseMatrix<double>> solver;
  auto start_sparse = chrono::high_resolution_clock::now();
  solver.compute(Q_sparse);  // 进行分解
  if (solver.info() != Success) {
    cerr << "Decomposition failed" << endl;
    return -1;
  }
  VectorXd x_sparse = solver.solve(-b);  // 使用分解结果求解
  if (solver.info() != Success) {
    cerr << "Solving failed" << endl;
    return -1;
  }
  auto end_sparse = chrono::high_resolution_clock::now();
  chrono::duration<double> elapsed_sparse = end_sparse - start_sparse;

  // 输出结果
  cout << "Dense solution time: " << elapsed_dense.count() << " seconds" << endl;
  cout << "Sparse solution time: " << elapsed_sparse.count() << " seconds" << endl;

  return 0;
}
