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

  // 提取ilqr求解参数
  ILQRParams ilqrParams{};
  ilqrParams.tolerance = j["ilqr_params"]["tolerance"];
  ilqrParams.max_iter = j["ilqr_params"]["max_iter"];

  Vehicle<StateDim, ControlDim> vehicle(wheelbase, max_speed, min_speed,
                                        max_acceleration, min_acceleration,
                                        max_steering_angle, min_steering_angle);

  Eigen::Matrix<double, StateDim, 1> x0;
  x0 << 0, 0, 1, 1;

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

  WeightParams<StateDim, ControlDim> cost_params;
  cost_params.Q = Q;
  cost_params.R = R;
  cost_params.Qf = Q_N;

  // 提取线搜索参数
  const auto &line_search = j["line_search_param"];
  LineSearchParams line_search_params{};
  line_search_params.alpha_min = line_search["alpha_min"];
  line_search_params.alpha_max = line_search["alpha_max"];
  line_search_params.alpha_decay = line_search["alpha_decay"];
  line_search_params.c1 = line_search["c1"];

  // 提取仿真参数
  SimulateParams simulateParams{};
  simulateParams.horizon = j["simulate_params"]["horizon"];
  simulateParams.dt = j["simulate_params"]["dt"];

  // 初始化状态和控制序列
  std::vector<Eigen::Matrix<double, StateDim, 1>> xRefSeq(simulateParams.horizon + 1, Eigen::Matrix<double, StateDim, 1>::Zero());
  std::vector<Eigen::Matrix<double, ControlDim, 1>> uRefSeq(simulateParams.horizon, Eigen::Matrix<double, ControlDim, 1>::Zero());

  // 设置初始状态
  xRefSeq[0] = x0;

  // 设置参考轨迹为简单的直线
  for (int k = 0; k < simulateParams.horizon; ++k) {
    double t = k * simulateParams.dt;
    xRefSeq[k + 1][0] = t;                // 更新 x 位置
    xRefSeq[k + 1][1] = t * 0.25;         // 更新 y 位置
    xRefSeq[k + 1][2] = j["ref"]["theta"];// 更新方向角
    xRefSeq[k + 1][3] = j["ref"]["v"];    // 速度保持不变
  }

  for (const auto &s : xRefSeq) {
    std::cout << "xref " << s.transpose() << std::endl;
  }

  iLQR<StateDim, ControlDim> iLqr(vehicle, cost_params, line_search_params, ilqrParams, simulateParams);
  bool solver_flag = iLqr.solve(x0, xRefSeq, uRefSeq);

  if (solver_flag) {
    std::cout << "solver success" << std::endl;
  }
  // 假设优化后的轨迹为 x_opt_seq 和 u_opt_seq
  std::vector<Eigen::Matrix<double, StateDim, 1>> x_opt_seq = iLqr.getOptStateSeq();
//  for (const auto &s : x_opt_seq) {
//    std::cout << "x " << s.transpose() << std::endl;
//  }
  std::vector<Eigen::Matrix<double, ControlDim, 1>> u_opt_seq = iLqr.getOptControlSeq();

  // 准备写入CSV文件的数据
  std::vector<std::vector<double>> data(8);// 8列数据
  for (int k = 0; k < x_opt_seq.size(); ++k) {
    data[0].push_back(xRefSeq[k][0]);// x_ref
    data[1].push_back(xRefSeq[k][1]);// y_ref

    data[2].push_back(x_opt_seq[k][0]);// x_coords
    data[3].push_back(x_opt_seq[k][1]);// y_coords
    data[4].push_back(x_opt_seq[k][3]);// velocity
    data[5].push_back(x_opt_seq[k][2]);// theta
    data[6].push_back(u_opt_seq[k][0]);// acceleration
    data[7].push_back(u_opt_seq[k][1]);// steering_angle
  }
  // 写入CSV文件
  std::vector<std::string> headers = {"x_ref", "y_ref", "x_coords", "y_coords", "velocity", "theta", "acceleration", "steering_angle"};
  writeDataToCSV(root_path + "data.csv", data, headers);

  return 0;
}