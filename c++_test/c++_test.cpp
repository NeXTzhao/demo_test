#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// Vec2d 类，用于表示二维点
class Vec2d {
 public:
  Vec2d(double x, double y) : x_(x), y_(y) {}
  double x() const { return x_; }
  double y() const { return y_; }

 private:
  double x_;
  double y_;
};

// 风险方向枚举
enum class RiskSide { LEFT, RIGHT };

// 风险场导数枚举
enum class RiskFieldDerivative {
  // gradient
  df_dx = 0,
  df_dy = 1,
  df_dv = 2,
  df_dq = 3,

  // hessian
  ddf_dxdx = 4,
  ddf_dxdy = 5,
  ddf_dxdv = 6,
  ddf_dxdq = 7,

  ddf_dydx = 8,
  ddf_dydy = 9,
  ddf_dydv = 10,
  ddf_dydq = 11,

  ddf_dvdx = 12,
  ddf_dvdy = 13,
  ddf_dvdv = 14,
  ddf_dvdq = 15,

  ddf_dqdx = 16,
  ddf_dqdy = 17,
  ddf_dqdv = 18,
  ddf_dqdq = 19,
 
  derivative_dim = 20
};
// 风险场类
class RiskField {
 public:
  RiskField(const Vec2d& ego_point, const Vec2d& ref_point, double v,
            double theta, const RiskSide& side)
      : ego_point_(ego_point),
        ref_point_(ref_point),
        v_(v),
        theta_(theta),
        side_(side) {
    derivatives_.resize(static_cast<int>(RiskFieldDerivative::derivative_dim),
                        0.0);
  }

void CalculateDerivatives() {
    // 计算位置差异
    double delta_x = ego_point_.x() - ref_point_.x();
    double delta_y = ego_point_.y() - ref_point_.y();
    double distance_squared = delta_x * delta_x + delta_y * delta_y;
    double distance = std::sqrt(distance_squared);

    // 风险方向调整因子
    double a = 0.01;
    double b = 0.01;

    // 预计算三角函数值
    double sin_theta = std::sin(theta_);
    double cos_theta = std::cos(theta_);

    // 速度和方向调整公式
    double velocity_component = b * v_ * cos_theta + a * v_ * sin_theta;

    // 指数项计算
    double exp_term = std::exp(distance * velocity_component);

    // 按照枚举顺序依次赋值

    // Gradient
    derivatives_.at(static_cast<int>(RiskFieldDerivative::df_dx)) =
        (exp_term * delta_x * velocity_component) / distance;
    derivatives_.at(static_cast<int>(RiskFieldDerivative::df_dy)) =
        (exp_term * delta_y * velocity_component) / distance;
    derivatives_.at(static_cast<int>(RiskFieldDerivative::df_dv)) =
        exp_term * distance * (b * cos_theta + a * sin_theta);
    derivatives_.at(static_cast<int>(RiskFieldDerivative::df_dq)) =
        exp_term * distance * (-b * v_ * sin_theta + a * v_ * cos_theta);

    // Hessian
    derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dxdx)) =
        -((exp_term * std::pow(delta_x, 2)* velocity_component) / std::pow(distance_squared, 1.5)) +
        (exp_term * velocity_component) / distance +
        (exp_term * std::pow(delta_x, 2) * std::pow(velocity_component, 2)) / distance_squared;

    derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dxdy)) =
        -((exp_term * delta_x * delta_y * velocity_component) / std::pow(distance_squared, 1.5)) +
        (exp_term * delta_x * delta_y * std::pow(velocity_component, 2)) / distance_squared;

    derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dxdv)) =
        (exp_term * delta_x * (b * cos_theta + a * sin_theta)) / distance +
        exp_term * delta_x * velocity_component * (b * cos_theta + a * sin_theta);

    derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dxdq)) =
        (exp_term * delta_x * (a * v_ * cos_theta - b * v_ * sin_theta)) / distance +
        exp_term * delta_x * velocity_component * (a * v_ * cos_theta - b * v_ * sin_theta);

    // y hession
    derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dydx)) =
        derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dxdy));

    derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dydy)) =
          (exp_term * velocity_component) / std::pow(distance_squared, 1.5) -
          (exp_term * std::pow(delta_y, 2) * velocity_component) / std::pow(distance_squared, 1.5) +
          (exp_term * std::pow(delta_y, 2) * std::pow(velocity_component, 2)) / distance_squared;

    derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dydv)) =
        (exp_term * delta_y * (b * cos_theta + a * sin_theta)) / distance +
        exp_term * delta_y * velocity_component * (b * cos_theta + a * sin_theta);

    derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dydq)) =
        (exp_term * delta_y * (a * v_ * cos_theta - b * v_ * sin_theta)) / distance +
        exp_term * delta_y * (b * v_ * cos_theta + a * v_ * sin_theta) * (a * v_ * cos_theta - b * v_ * sin_theta);

    // v hessian
    derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dvdx)) =
        derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dxdv));

    derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dvdy)) =
        derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dydv));

    derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dvdv)) =
        exp_term * distance_squared * std::pow(b * cos_theta + a * sin_theta, 2);
    
    derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dvdq)) =
        exp_term * distance * (a * cos_theta - b * sin_theta) +
        exp_term * distance_squared * (b * cos_theta + a * sin_theta) * (a * v_  * cos_theta - b * v_ *sin_theta);

    // theta hessian
    derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dqdx)) =
        derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dxdq));

    derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dqdy)) =
        derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dydq));

    derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dqdv)) =
        derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dvdq));

    derivatives_.at(static_cast<int>(RiskFieldDerivative::ddf_dqdq)) =
        exp_term * distance * (-b * v_ * cos_theta - a * v_ * sin_theta) +
        exp_term * distance_squared * std::pow(a * cos_theta - b * sin_theta, 2);
}

  double GetDerivative(int index) const {
    assert(index >= 0 &&
           index < static_cast<int>(RiskFieldDerivative::derivative_dim));
    return derivatives_.at(index);
  }

 private:
  Vec2d ego_point_;
  Vec2d ref_point_;
  double v_;
  double theta_;
  RiskSide side_;
  std::vector<double> derivatives_;
};

int main() {
  // 定义参考点
  Vec2d ref_point(0.0, 0.0);

  // 定义速度和角度
  double v = 5.0;           // 固定速度
  double theta = M_PI / 4;  // 固定角度

  // 定义风险侧
  RiskSide side = RiskSide::LEFT;

  // 生成35个轨迹点
  std::vector<Vec2d> trajectory;
  for (int i = 0; i < 35; ++i) {
    double x = i * 0.5;      // 轨迹点x坐标间隔为0.5
    double y = std::sin(x);  // y坐标为sin(x)
    trajectory.emplace_back(x, y);
  }

  // 计算每个轨迹点的风险
  auto start_time0 = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < trajectory.size(); ++i) {
    RiskField risk_field(trajectory[i], ref_point, v, theta, side);

    // 计时计算
    auto start_time = std::chrono::high_resolution_clock::now();
    risk_field.CalculateDerivatives();
    auto end_time = std::chrono::high_resolution_clock::now();

    // 计算耗时（微秒）
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_time - start_time)
                        .count();

    // std::cout << std::fixed << std::setprecision(6) << "Risk calculation took "
    //           << duration << " μs." << std::endl;  // 修改为微秒输出
  }
  auto end_time0 = std::chrono::high_resolution_clock::now();
  // 计算总耗时（微秒）
  auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(
                       end_time0 - start_time0)
                       .count();
  std::cout << std::fixed << std::setprecision(6)
            << "Total risk calculation took " << duration0 << " μs."
            << std::endl;  // 修改为微秒输出

  return 0;
}
