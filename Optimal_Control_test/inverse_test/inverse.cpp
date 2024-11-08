//#include <Eigen/Dense>
//#include <chrono>
//#include <iostream>
//#include <vector>
//#include <map>
//#include "matplotlibcpp.h"
//
//namespace plt = matplotlibcpp;
//
//// 计算矩阵的条件数
//double condition_number(const Eigen::MatrixXd &A) {
//  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);
//  double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
//  return cond;
//}
//
//// 计算残差
//double compute_residual(const Eigen::MatrixXd &A, const Eigen::MatrixXd &A_inv) {
//  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(A.rows(), A.cols());
//  Eigen::MatrixXd residual = I - A * A_inv;
//  return residual.norm();
//}
//
//void compare_inversion_methods(const std::vector<double> &dimensions) {
//  std::vector<double> direct_times, lu_times, cholesky_times, qr_times, svd_times;
//  std::vector<double> direct_conds, lu_conds, cholesky_conds, qr_conds, svd_conds;
//  std::vector<double> direct_residuals, lu_residuals, cholesky_residuals, qr_residuals, svd_residuals;
//  const double initial_epsilon = 1e-6;  // 初始修正参数
//
//  for (const auto dim : dimensions) {
//    // 生成随机矩阵
//    Eigen::MatrixXd A = Eigen::MatrixXd::Random(dim, dim);
//    Eigen::MatrixXd A_inv;
//
//    // 直接求逆
//    auto start = std::chrono::high_resolution_clock::now();
//    A_inv = A.inverse();
//    auto end = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> duration = end - start;
//    direct_times.push_back(duration.count());
//    direct_conds.push_back(condition_number(A_inv));
//    direct_residuals.push_back(compute_residual(A, A_inv));
//    std::cout << "Dimension: " << dim << " Direct Inversion Time: " << duration.count() << " s, Condition Number: "
//              << direct_conds.back() << ", Residual: " << direct_residuals.back() << std::endl;
//
//    // LU 分解求逆
//    start = std::chrono::high_resolution_clock::now();
//    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
//    //    A_inv = lu.inverse();
//    A_inv = lu.solve(Eigen::MatrixXd::Identity(dim, dim));
//    end = std::chrono::high_resolution_clock::now();
//    duration = end - start;
//    lu_times.push_back(duration.count());
//    lu_conds.push_back(condition_number(A_inv));
//    lu_residuals.push_back(compute_residual(A, A_inv));
//    std::cout << "Dimension: " << dim << " LU Inversion Time: " << duration.count() << " s, Condition Number: "
//              << lu_conds.back() << ", Residual: " << lu_residuals.back() << std::endl;
//
//    // Cholesky 分解求逆
//    Eigen::MatrixXd A_chol = A;
//    bool success = false;
//    double epsilon = initial_epsilon;
//    for (int i = 0; i < 10; ++i) {  // 尝试多次修正矩阵
//      Eigen::LLT<Eigen::MatrixXd> llt(A_chol);
//      if (llt.info() == Eigen::Success) {
//        start = std::chrono::high_resolution_clock::now();
//        A_inv = llt.solve(Eigen::MatrixXd::Identity(dim, dim));
//        end = std::chrono::high_resolution_clock::now();
//        duration = end - start;
//        cholesky_times.push_back(duration.count());
//        cholesky_conds.push_back(condition_number(A_inv));
//        cholesky_residuals.push_back(compute_residual(A, A_inv));
//        success = true;
//        std::cout << "Dimension: " << dim << " Cholesky Inversion Time: " << duration.count()
//                  << " s, Condition Number: " << cholesky_conds.back() << ", Residual: " << cholesky_residuals.back()
//                  << std::endl;
//        break;
//      } else {
//        A_chol += epsilon * Eigen::MatrixXd::Identity(dim, dim);
//        epsilon *= 10;  // 增加修正幅度
//      }
//    }
//    if (!success) {
//      cholesky_times.push_back(0);  // 如果修正多次仍不成功，记录0作为占位符
//      cholesky_conds.push_back(0);
//      cholesky_residuals.push_back(0);
//      std::cout << "Dimension: " << dim << " Cholesky Inversion Failed after " << 10 << " attempts." << std::endl;
//    }
//
//    // QR 分解求逆
//    start = std::chrono::high_resolution_clock::now();
//    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
//    A_inv = qr.solve(Eigen::MatrixXd::Identity(dim, dim));
//    end = std::chrono::high_resolution_clock::now();
//    duration = end - start;
//    qr_times.push_back(duration.count());
//    qr_conds.push_back(condition_number(A_inv));
//    qr_residuals.push_back(compute_residual(A, A_inv));
//    std::cout << "Dimension: " << dim << " QR Inversion Time: " << duration.count() << " s, Condition Number: "
//              << qr_conds.back() << ", Residual: " << qr_residuals.back() << std::endl;
//
//    // SVD 分解求逆
//    start = std::chrono::high_resolution_clock::now();
//    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
//    A_inv = svd.solve(Eigen::MatrixXd::Identity(dim, dim));
//    end = std::chrono::high_resolution_clock::now();
//    duration = end - start;
//    svd_times.push_back(duration.count());
//    svd_conds.push_back(condition_number(A_inv));
//    svd_residuals.push_back(compute_residual(A, A_inv));
//    std::cout << "Dimension: " << dim << " SVD Inversion Time: " << duration.count() << " s, Condition Number: "
//              << svd_conds.back() << ", Residual: " << svd_residuals.back() << std::endl;
//  }
//
//  // 可视化结果
//  plt::figure();
//
//  // 直接求逆时间
//  std::map<std::string, std::string> direct_keywords;
//  direct_keywords["color"] = "r";
//  direct_keywords["label"] = "Direct Inversion";
//  plt::plot(dimensions, direct_times, direct_keywords);
//
//  // LU 分解时间
//  std::map<std::string, std::string> lu_keywords;
//  lu_keywords["color"] = "g";
//  lu_keywords["label"] = "LU Inversion";
//  plt::plot(dimensions, lu_times, lu_keywords);
//
//  // Cholesky 分解时间
//  std::map<std::string, std::string> cholesky_keywords;
//  cholesky_keywords["color"] = "b";
//  cholesky_keywords["label"] = "Cholesky Inversion";
//  plt::plot(dimensions, cholesky_times, cholesky_keywords);
//
//  // QR 分解时间
//  std::map<std::string, std::string> qr_keywords;
//  qr_keywords["color"] = "c";
//  qr_keywords["label"] = "QR Inversion";
//  plt::plot(dimensions, qr_times, qr_keywords);
//
//  // SVD 分解时间
//  std::map<std::string, std::string> svd_keywords;
//  svd_keywords["color"] = "m";
//  svd_keywords["label"] = "SVD Inversion";
//  plt::plot(dimensions, svd_times, svd_keywords);
//
//  plt::xlabel("Matrix Dimension");
//  plt::ylabel("Time (seconds)");
//  plt::legend();
//  plt::title("Comparison of Matrix Inversion Methods");
//  plt::show();
//
//  // 可视化条件数
//  plt::figure();
//  plt::plot(dimensions, direct_conds, {{"color", "r"}, {"label", "Direct Inversion"}});
//  plt::plot(dimensions, lu_conds, {{"color", "g"}, {"label", "LU Inversion"}});
//  plt::plot(dimensions, cholesky_conds, {{"color", "b"}, {"label", "Cholesky Inversion"}});
//  plt::plot(dimensions, qr_conds, {{"color", "c"}, {"label", "QR Inversion"}});
//  plt::plot(dimensions, svd_conds, {{"color", "m"}, {"label", "SVD Inversion"}});
//  plt::xlabel("Matrix Dimension");
//  plt::ylabel("Condition Number");
//  plt::legend();
//  plt::title("Condition Numbers of Inversion Methods");
//  plt::show();
//
//  // 可视化残差
//  plt::figure();
//  plt::plot(dimensions, direct_residuals, {{"color", "r"}, {"label", "Direct Inversion"}});
//  plt::plot(dimensions, lu_residuals, {{"color", "g"}, {"label", "LU Inversion"}});
//  plt::plot(dimensions, cholesky_residuals, {{"color", "b"}, {"label", "Cholesky Inversion"}});
//  plt::plot(dimensions, qr_residuals, {{"color", "c"}, {"label", "QR Inversion"}});
//  plt::plot(dimensions, svd_residuals, {{"color", "m"}, {"label", "SVD Inversion"}});
//  plt::xlabel("Matrix Dimension");
//  plt::ylabel("Residual");
//  plt::legend();
//  plt::title("Residuals of Inversion Methods");
//  plt::show();
//}
//
//int main() {
//  // 定义矩阵维度范围
//  std::vector<double> dimensions = {10, 50, 100, 200, 500};
//
//  // 比较不同求逆方法的耗时
//  compare_inversion_methods(dimensions);
//
//  return 0;
//}

#include <Eigen/Dense>
#include <iostream>

int main() {
  // 假设 hamiltonian 是一个 5x5 的矩阵
  Eigen::MatrixXd hamiltonian(5, 5);

  // 初始化 hamiltonian 矩阵
  hamiltonian << 1, 2, 3, 4, 5,
                 6, 7, 8, 9, 10,
                 11, 12, 13, 14, 15,
                 16, 17, 18, 19, 20,
                 21, 22, 23, 24, 25;

  // 假设 X_DIM = 3, U_DIM = 2
  int X_DIM = 3;
  int U_DIM = 2;

  // 提取右上角的子矩阵
  Eigen::MatrixXd Q_xu = hamiltonian.topRightCorner(1 + X_DIM, U_DIM);

  // 输出 Q_xu
  std::cout << "Q_xu:" << std::endl << Q_xu << std::endl;

  return 0;
}
