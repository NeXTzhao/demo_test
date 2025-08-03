#pragma once
#include "math/vec2d.h"
#include <limits>
#include <string>
#include <vector>

class GeometryShape {
public:
  virtual std::shared_ptr<GeometryShape> ShiftXY(double, double) const = 0;
  virtual std::shared_ptr<GeometryShape> ShiftDT(double, double) const = 0;
  virtual std::shared_ptr<GeometryShape> Rotate(Vec2d, double) const = 0;
  virtual std::shared_ptr<GeometryShape> Transform(Vec2d, double, double,
                                                   double) const = 0;

  virtual ~GeometryShape(){};
};

class Disk2d : public GeometryShape {
public:
  Disk2d() : center_(0.0, 0.0), radius_(0.0) {}

  Disk2d(const Vec2d center, double radius) {
    center_ = center;
    radius_ = radius;
  }

  Disk2d(double x, double y, double r) {
    center_.set_x(x);
    center_.set_y(y);
    radius_ = r;
  }

  // accessors.
  const Vec2d &center() const { return center_; }
  double radius() const { return radius_; }

  std::shared_ptr<GeometryShape> ShiftXY(double dx, double dy) const {
    Vec2d shifted_center;
    shifted_center.set_x(center_.x() + dx);
    shifted_center.set_y(center_.y() + dy);
    return std::make_shared<Disk2d>(shifted_center, radius_);
  }

  std::shared_ptr<GeometryShape> ShiftDT(double dist, double theta) const {
    Vec2d shifted_center;
    shifted_center.set_x(center_.x() + dist * cos(theta));
    shifted_center.set_y(center_.y() + dist * sin(theta));
    return std::make_shared<Disk2d>(shifted_center, radius_);
  }

  std::shared_ptr<GeometryShape> Rotate(Vec2d Rcenter, double angle) const {
    Vec2d rotated_center;
    double relativeX = center_.x() - Rcenter.x();
    double relativeY = center_.y() - Rcenter.y();
    rotated_center.set_x(Rcenter.x() + relativeX * cos(angle) -
                         relativeY * sin(angle));
    rotated_center.set_y(Rcenter.y() + relativeX * sin(angle) +
                         relativeY * cos(angle));
    return std::make_shared<Disk2d>(rotated_center, radius_);
    ;
  }

  std::shared_ptr<GeometryShape> Transform(Vec2d Rcenter, double angle,
                                           double dist, double theta) const {
    std::shared_ptr<GeometryShape> rotated_disk = Rotate(Rcenter, angle);
    std::shared_ptr<GeometryShape> transformed_disk =
        rotated_disk->ShiftDT(dist, theta);
    return transformed_disk;
  }

  // Setters
  void SetCenter(double x_init, double y_init) {
    center_.set_x(x_init);
    center_.set_y(y_init);
  }

private:
  Vec2d center_;
  double radius_;
};

class Capsule2d : public GeometryShape {
public:
  Capsule2d() {}

  Capsule2d(const std::vector<Disk2d> disks) { disks_ = disks; }
  const std::vector<Disk2d> &Disks() const { return disks_; }
  void AppendDisk(Disk2d disk) { disks_.push_back(disk); }
  void SetDisks(const std::vector<Disk2d> disks) { disks_ = disks; }
  int DisksNum() const { return disks_.size(); }

  const Vec2d &center_i(int index) const {
    CHECK_GE(index, 0);
    CHECK_LT(index, disks_.size());
    return disks_.at(index).center();
  }

  double radius_i(int index) const {
    CHECK_GE(index, 0);
    CHECK_LT(index, disks_.size());
    return disks_.at(index).radius();
  }

  std::shared_ptr<GeometryShape> ShiftXY(double dx, double dy) const {
    std::vector<Disk2d> shifted_disks;
    for (const auto &disk : disks_) {
      Disk2d shifted_disk =
          *std::dynamic_pointer_cast<Disk2d>(disk.ShiftXY(dx, dy));
      shifted_disks.push_back(shifted_disk);
    }
    return std::make_shared<Capsule2d>(shifted_disks);
  }

  std::shared_ptr<GeometryShape> ShiftDT(double dist, double theta) const {
    std::vector<Disk2d> shifted_disks;
    for (const auto &disk : disks_) {
      Disk2d shifted_disk =
          *std::dynamic_pointer_cast<Disk2d>(disk.ShiftDT(dist, theta));
      shifted_disks.push_back(shifted_disk);
    }
    return std::make_shared<Capsule2d>(shifted_disks);
  }

  std::shared_ptr<GeometryShape> Rotate(Vec2d Rcenter, double angle) const {
    std::vector<Disk2d> shifted_disks;
    for (const auto &disk : disks_) {
      Disk2d shifted_disk =
          *std::dynamic_pointer_cast<Disk2d>(disk.Rotate(Rcenter, angle));
      shifted_disks.push_back(shifted_disk);
    }
    return std::make_shared<Capsule2d>(shifted_disks);
  }

  std::shared_ptr<GeometryShape> Transform(Vec2d Rcenter, double angle,
                                           double dist, double theta) const {
    std::vector<Disk2d> shifted_disks;
    for (const auto &disk : disks_) {
      Disk2d shifted_disk = *std::dynamic_pointer_cast<Disk2d>(
          disk.Transform(Rcenter, angle, dist, theta));
      shifted_disks.push_back(shifted_disk);
    }
    return std::make_shared<Capsule2d>(shifted_disks);
  }

private:
  std::vector<Disk2d> disks_;
};

// DiskInfo struct contains the information using circular disks to cover
// the actual shape of a vehicle.
struct DiskInfo {
  DiskInfo(Disk2d disk_in, double lon_offset_in, double lat_offset_in)
      : disk(std::move(disk_in)), lon_offset(lon_offset_in),
        lat_offset(lat_offset_in) {}

  Disk2d disk;

  // The longitudinal offset from rear differential.
  double lon_offset = 0.0;

  // The lateral offset from rear differential.
  double lat_offset = 0.0;
};

struct CapsuleInfo {
  CapsuleInfo(Capsule2d capsule_in, double lon_offset_in, double lat_offset_in)
      : capsule(std::move(capsule_in)), lon_offset(lon_offset_in),
        lat_offset(lat_offset_in) {}

  Capsule2d capsule;

  // The longitudinal offset from rear differential.
  double lon_offset = 0.0;

  // The lateral offset from rear differential.
  double lat_offset = 0.0;
};

inline std::vector<DiskInfo> GetCoveringDiskInfo(const Measurement &measurement,
                                                 double x, double y,
                                                 double heading) {
  // Compute the parameters needed to represent vehicle with disks.
  const double L = measurement.length; // Total length (m).
  const double W = measurement.width;  // Total width (m).
  const double disk_num =
      kEgoDiskSize; // Total number of longitudinal covering disks.

  const double dl =
      L / (2 * disk_num);  // The length used to compute disk radius (m).
  const double dw = W / 2; // The width used to compute disk radius (m).

  const double radius = std::hypot(dl, dw);

  // Get pose of rear bumper.
  const double sin_theta = sin(heading);
  const double cos_theta = cos(heading);
  const double xr = x - measurement.rear_bumper_to_rear_axle * cos_theta;
  const double yr = y - measurement.rear_bumper_to_rear_axle * sin_theta;

  // Construct longitudinal covering disks.
  std::vector<DiskInfo> disk_info;
  for (int i = 0; i < disk_num; ++i) {
    const double l = (2 * i + 1) * dl;
    disk_info.emplace_back(
        Disk2d({xr + l * cos_theta, yr + l * sin_theta}, radius),
        /*lon_offset=*/l - measurement.rear_bumper_to_rear_axle,
        /*lat_offset=*/0.0);
  }

  return disk_info;
}
