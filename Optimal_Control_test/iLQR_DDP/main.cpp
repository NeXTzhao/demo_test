//
// Created by next on 24-7-27.
//

#include "ilqr.h"
#include "iostream"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

// 写入数据到CSV文件的函数
void writeDataToCSV(const std::string &filename,
                    const std::vector<std::vector<double>> &data,
                    const std::vector<std::string> &headers) {
  std::ofstream file(filename);
  for (const auto &header : headers) {
    file << header << ",";
  }
  file << "\n";

  for (size_t i = 0; i < data[0].size(); ++i) {
    for (const auto &column : data) {
      file << column[i] << ",";
    }
    file << "\n";
  }

  file.close();
}

// 写入Eigen矩阵到CSV文件的函数
void writeEigenToCSV(const std::string &filename,
                     const Eigen::MatrixXd &matrix) {
  std::ofstream file(filename);
  if (file.is_open()) {
    for (int i = 0; i < matrix.rows(); ++i) {
      for (int j = 0; j < matrix.cols(); ++j) {
        file << matrix(i, j);
        if (j < matrix.cols() - 1) {
          file << ",";
        }
      }
      file << "\n";
    }
    file.close();
  } else {
    std::cerr << "Unable to open file " << filename << std::endl;
  }
}

// 读取参数文件
json readParams(const std::string &filename) {
  std::ifstream file(filename);
  json params;
  file >> params;
  return params;
}

int main() {
  const std::string root_path = "/home/next/Documents/demo_test/Optimal_Control_test/iLQR_DDP/data/";
  const std::string file_path = root_path + "params.json";

  // 读取JSON文件
  std::ifstream i(file_path);
  if (!i.is_open()) {
    std::cerr << "Failed to open file: " << file_path << std::endl;
    return 1;
  }

  json j;
  i >> j;

  const int StateDim = 4;
  const int ControlDim = 2;

  // 提取车辆参数
  const auto &vehicle_params = j["vehicle_params"];
  double wheelbase = vehicle_params["wheelbase"];
  double max_speed = vehicle_params["max_speed"];
  double min_speed = vehicle_params["min_speed"];
  double max_acceleration = vehicle_params["max_acceleration"];
  double min_acceleration = vehicle_params["min_acceleration"];
  double max_steering_angle = vehicle_params["max_steering_angle"];
  double min_steering_angle = vehicle_params["min_steering_angle"];

  // 提取仿真参数
  int horizon = j["simulate"]["horizon"];
  double dt = j["simulate"]["dt"];

  Vehicle<StateDim, ControlDim> vehicle(wheelbase, max_speed, min_speed,
                                        max_acceleration, min_acceleration,
                                        max_steering_angle, min_steering_angle);

  int N = 50;// Number of time steps
  Eigen::Matrix<double, StateDim, 1> x0;
  x0 << 0, 0, 0, 0;

  // 提取成本参数
  auto extractDiagonal = [](const json &j_array, int dim) {
    Eigen::MatrixXd mat = Eigen::MatrixXd::Identity(dim, dim);
    for (int i = 0; i < dim; ++i) {
      mat(i, i) = j_array[i];
    }
    return mat;
  };

  Eigen::MatrixXd Q = extractDiagonal(j["Q"], StateDim);
  Eigen::MatrixXd R = extractDiagonal(j["R"], ControlDim);
  Eigen::MatrixXd Q_N = extractDiagonal(j["Q_N"], StateDim);

  CostParams<StateDim, ControlDim> cost_params;
  cost_params.Q = Q;
  cost_params.R = R;
  cost_params.Qf = Q_N;

  // 提取线搜索参数
  const auto &line_search = j["line_search_param"];
  LineSearchParams line_search_params;
  line_search_params.alpha_min = line_search["alpha_min"];
  line_search_params.alpha_max = line_search["alpha_max"];
  line_search_params.alpha_decay = line_search["alpha_decay"];
  line_search_params.c1 = line_search["c1"];

  // 初始化状态和控制序列
  std::vector<Eigen::Matrix<double, StateDim, 1>> x_seq(N + 1, Eigen::Matrix<double, StateDim, 1>::Zero());
  std::vector<Eigen::Matrix<double, ControlDim, 1>> u_seq(N, Eigen::Matrix<double, ControlDim, 1>::Zero());

  iLQR<StateDim, ControlDim> ilqr(vehicle);
  ilqr.solve(x0, x_seq, u_seq, dt, cost_params, line_search_params);

  // Print final state sequence
  for (int i = 0; i < N + 1; ++i) {
    std::cout << "x[" << i << "] = " << x_seq[i].transpose() << std::endl;
  }

  return 0;
}