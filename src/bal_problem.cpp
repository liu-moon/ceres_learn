// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2023 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "bal_problem.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "ceres/rotation.h"
#include "glog/logging.h"

namespace ceres::examples {
namespace {
using VectorRef = Eigen::Map<Eigen::VectorXd>;
using ConstVectorRef = Eigen::Map<const Eigen::VectorXd>;

template <typename T>
void FscanfOrDie(FILE* fptr, const char* format, T* value) {
  int num_scanned = fscanf(fptr, format, value);
  if (num_scanned != 1) {
    LOG(FATAL) << "Invalid UW data file.";
  }
}

void PerturbPoint3(std::function<double()> dist, double* point) {
  for (int i = 0; i < 3; ++i) {
    point[i] += dist();
  }
}

double Median(std::vector<double>* data) { // 返回数据中中位数
  auto mid_point = data->begin() + data->size() / 2;
  std::nth_element(data->begin(), mid_point, data->end());
  return *mid_point;
}

}  // namespace

BALProblem::BALProblem(const std::string& filename, bool use_quaternions) {
  FILE* fptr = fopen(filename.c_str(), "r");

  if (fptr == nullptr) {
    LOG(FATAL) << "Error: unable to open file " << filename;
    return;
  };

  // This will die horribly on invalid files. Them's the breaks.
  FscanfOrDie(fptr, "%d", &num_cameras_); // 相机数量
  FscanfOrDie(fptr, "%d", &num_points_); // 点数量
  FscanfOrDie(fptr, "%d", &num_observations_); // 观测数量

  VLOG(1) << "Header: " << num_cameras_ << " " << num_points_ << " "
          << num_observations_;

  point_index_ = new int[num_observations_]; // 堆区创建
  camera_index_ = new int[num_observations_];
  observations_ = new double[2 * num_observations_];

  num_parameters_ = 9 * num_cameras_ + 3 * num_points_; // 参数数量
  parameters_ = new double[num_parameters_];

  for (int i = 0; i < num_observations_; ++i) {
    FscanfOrDie(fptr, "%d", camera_index_ + i);
    FscanfOrDie(fptr, "%d", point_index_ + i);
    for (int j = 0; j < 2; ++j) {
      FscanfOrDie(fptr, "%lf", observations_ + 2 * i + j);
    }
  }

  for (int i = 0; i < num_parameters_; ++i) {
    FscanfOrDie(fptr, "%lf", parameters_ + i);
  }

  fclose(fptr);

  use_quaternions_ = use_quaternions; // 默认不启用
  if (use_quaternions) {
    // Switch the angle-axis rotations to quaternions.
    num_parameters_ = 10 * num_cameras_ + 3 * num_points_;
    auto* quaternion_parameters = new double[num_parameters_];
    double* original_cursor = parameters_;
    double* quaternion_cursor = quaternion_parameters;
    for (int i = 0; i < num_cameras_; ++i) {
      AngleAxisToQuaternion(original_cursor, quaternion_cursor);
      quaternion_cursor += 4;
      original_cursor += 3;
      for (int j = 4; j < 10; ++j) {
        *quaternion_cursor++ = *original_cursor++;
      }
    }
    // Copy the rest of the points.
    for (int i = 0; i < 3 * num_points_; ++i) {
      *quaternion_cursor++ = *original_cursor++;
    }
    // Swap in the quaternion parameters.
    delete[] parameters_;
    parameters_ = quaternion_parameters;
  }
}

// This function writes the problem to a file in the same format that
// is read by the constructor.
void BALProblem::WriteToFile(const std::string& filename) const {
  FILE* fptr = fopen(filename.c_str(), "w");

  if (fptr == nullptr) {
    LOG(FATAL) << "Error: unable to open file " << filename;
    return;
  };

  fprintf(fptr, "%d %d %d\n", num_cameras_, num_points_, num_observations_);

  for (int i = 0; i < num_observations_; ++i) {
    fprintf(fptr, "%d %d", camera_index_[i], point_index_[i]);
    for (int j = 0; j < 2; ++j) {
      fprintf(fptr, " %g", observations_[2 * i + j]);
    }
    fprintf(fptr, "\n");
  }

  for (int i = 0; i < num_cameras(); ++i) {
    double angleaxis[9];
    if (use_quaternions_) {
      // Output in angle-axis format.
      QuaternionToAngleAxis(parameters_ + 10 * i, angleaxis);
      memcpy(angleaxis + 3, parameters_ + 10 * i + 4, 6 * sizeof(double));
    } else {
      memcpy(angleaxis, parameters_ + 9 * i, 9 * sizeof(double));
    }
    for (double coeff : angleaxis) {
      fprintf(fptr, "%.16g\n", coeff);
    }
  }

  const double* points = parameters_ + camera_block_size() * num_cameras_;
  for (int i = 0; i < num_points(); ++i) {
    const double* point = points + i * point_block_size();
    for (int j = 0; j < point_block_size(); ++j) {
      fprintf(fptr, "%.16g\n", point[j]);
    }
  }

  fclose(fptr);
}

// Write the problem to a PLY file for inspection in Meshlab or CloudCompare.
void BALProblem::WriteToPLYFile(const std::string& filename) const {
  std::ofstream of(filename.c_str());

  of << "ply" << '\n'
     << "format ascii 1.0" << '\n'
     << "element vertex " << num_cameras_ + num_points_ << '\n'
     << "property float x" << '\n'
     << "property float y" << '\n'
     << "property float z" << '\n'
     << "property uchar red" << '\n'
     << "property uchar green" << '\n'
     << "property uchar blue" << '\n'
     << "end_header" << std::endl;

  // Export extrinsic data (i.e. camera centers) as green points.
  double angle_axis[3];
  double center[3];
  for (int i = 0; i < num_cameras(); ++i) {
    const double* camera = cameras() + camera_block_size() * i;
    CameraToAngleAxisAndCenter(camera, angle_axis, center);
    of << center[0] << ' ' << center[1] << ' ' << center[2] << " 0 255 0"
       << '\n';
  }

  // Export the structure (i.e. 3D Points) as white points.
  const double* points = parameters_ + camera_block_size() * num_cameras_;
  for (int i = 0; i < num_points(); ++i) {
    const double* point = points + i * point_block_size();
    for (int j = 0; j < point_block_size(); ++j) {
      of << point[j] << ' ';
    }
    of << "255 255 255\n";
  }
  of.close();
}

void BALProblem::CameraToAngleAxisAndCenter(const double* camera,
                                            double* angle_axis,
                                            double* center) const {
  VectorRef angle_axis_ref(angle_axis, 3);
  if (use_quaternions_) {
    QuaternionToAngleAxis(camera, angle_axis);
  } else {
    angle_axis_ref = ConstVectorRef(camera, 3); // 取前三维旋转R
  }

  // c = -R't
  Eigen::VectorXd inverse_rotation = -angle_axis_ref; // 构建R'
  AngleAxisRotatePoint(
      inverse_rotation.data(), camera + camera_block_size() - 6, center);  // camera + camera_block_size() - 6 构建t
  VectorRef(center, 3) *= -1.0;
}

void BALProblem::AngleAxisAndCenterToCamera(const double* angle_axis,
                                            const double* center,
                                            double* camera) const {
  ConstVectorRef angle_axis_ref(angle_axis, 3); // 
  if (use_quaternions_) {
    AngleAxisToQuaternion(angle_axis, camera);
  } else {
    VectorRef(camera, 3) = angle_axis_ref; // 直接将角轴向量 angle_axis_ref 存入 camera R 更新 R？ 应该是R不变
  }

  // t = -R * c
  AngleAxisRotatePoint(angle_axis, center, camera + camera_block_size() - 6); // 旋转 输入点 输出点
  VectorRef(camera + camera_block_size() - 6, 3) *= -1.0;
}

void BALProblem::Normalize() {
  // Compute the marginal median of the geometry.
  std::vector<double> tmp(num_points_);
  Eigen::Vector3d median;
  double* points = mutable_points();
  for (int i = 0; i < 3; ++i) { // 分别遍历三维点的 X Y Z坐标
    for (int j = 0; j < num_points_; ++j) { // 遍历三维点数量次 7776
      tmp[j] = points[3 * j + i];
    }
    median(i) = Median(&tmp); // mediam存储了所有三维点的 X Y Z坐标 的中位数
  }

  for (int i = 0; i < num_points_; ++i) { // 遍历所有三维点
    VectorRef point(points + 3 * i, 3); // 将 points映射为Eigen向量 point
    tmp[i] = (point - median).lpNorm<1>(); //计算每个点到 median 的曼哈顿距离 衡量离散程度
  }

  const double median_absolute_deviation = Median(&tmp); // 返回离散程度的中位数

  // Scale so that the median absolute deviation of the resulting
  // reconstruction is 100. // 缩放以使生成的重建的中位绝对偏差为 100。
  const double scale = 100.0 / median_absolute_deviation;

  VLOG(2) << "median: " << median.transpose();
  VLOG(2) << "median absolute deviation: " << median_absolute_deviation;
  VLOG(2) << "scale: " << scale;

  // X = scale * (X - median) // 归一化：将每个三维点平移到几何中心，并按照比例缩放
  for (int i = 0; i < num_points_; ++i) {
    VectorRef point(points + 3 * i, 3);
    point = scale * (point - median);
  }
  // 归一化相机位置
  double* cameras = mutable_cameras();
  double angle_axis[3];
  double center[3];
  for (int i = 0; i < num_cameras_; ++i) { // 遍历所有相机
    double* camera = cameras + camera_block_size() * i; // 遍历每个相机数据
    CameraToAngleAxisAndCenter(camera, angle_axis, center); // angle_axis 是 R center是相机中心
    // center = scale * (center - median)
    VectorRef(center, 3) = scale * (VectorRef(center, 3) - median); // 将相机中心 center 以 median 为几何中心，按比例 scale 进行平移和缩放
    AngleAxisAndCenterToCamera(angle_axis, center, camera); // 将角轴旋转和相机中心位置转换为相机的参数表示
  }
}

void BALProblem::Perturb(const double rotation_sigma,
                         const double translation_sigma,
                         const double point_sigma) {
  CHECK_GE(point_sigma, 0.0);
  CHECK_GE(rotation_sigma, 0.0);
  CHECK_GE(translation_sigma, 0.0);
  std::mt19937 prng;
  std::normal_distribution<double> point_noise_distribution(0.0, point_sigma);
  double* points = mutable_points();
  if (point_sigma > 0) {
    for (int i = 0; i < num_points_; ++i) {
      PerturbPoint3(std::bind(point_noise_distribution, std::ref(prng)),
                    points + 3 * i);
    }
  }

  std::normal_distribution<double> rotation_noise_distribution(0.0,
                                                               point_sigma);
  std::normal_distribution<double> translation_noise_distribution(
      0.0, translation_sigma);
  for (int i = 0; i < num_cameras_; ++i) {
    double* camera = mutable_cameras() + camera_block_size() * i;

    double angle_axis[3];
    double center[3];
    // Perturb in the rotation of the camera in the angle-axis
    // representation.
    CameraToAngleAxisAndCenter(camera, angle_axis, center);
    if (rotation_sigma > 0.0) {
      PerturbPoint3(std::bind(rotation_noise_distribution, std::ref(prng)),
                    angle_axis);
    }
    AngleAxisAndCenterToCamera(angle_axis, center, camera);

    if (translation_sigma > 0.0) {
      PerturbPoint3(std::bind(translation_noise_distribution, std::ref(prng)),
                    camera + camera_block_size() - 6);
    }
  }
}

BALProblem::~BALProblem() {
  delete[] point_index_;
  delete[] camera_index_;
  delete[] observations_;
  delete[] parameters_;
}

}  // namespace ceres::examples
