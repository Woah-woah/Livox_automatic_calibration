/*
The mapping algorithm is an advanced implementation of the following open source project:
  [blam](https://github.com/erik-nelson/blam). 
Modifier: livox               dev@livoxtech.com


Copyright (c) 2015, The Regents of the University of California (Regents).
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

   3. Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <chrono>
#include <sstream>
#include <fstream>
#include <string>
#include <iostream>
#include <limits>
#include <vector>
#include <array>
#include <map>
#include <algorithm>
#include <random>
#include <cstdint>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <small_gicp/pcl/pcl_registration.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <geometry_utils/Transform3.h>

using namespace std;
namespace gu = geometry_utils;

#define PI (3.1415926535897932346f)
namespace {

constexpr double kDegToRad = PI / 180.0;
constexpr double kMaxFitnessScore = 0.05;
constexpr double kMaxCorrectionTranslation = 0.25;
constexpr double kMaxCorrectionRotation = 6.0 * kDegToRad;
constexpr float kRegistrationMapVoxel = 0.05f;
constexpr float kFineTargetVoxel = 0.06f;
constexpr float kFineSourceVoxel = 0.04f;
constexpr double kLocalMapRadius = 6.0;
constexpr std::size_t kMinLocalMapPoints = 500;
constexpr std::size_t kLocalMapDownsampleThreshold = 10000;
constexpr std::size_t kFineMinCorrespondences = 80;
constexpr std::size_t kFineMaxCorrespondences = 900;
constexpr double kFineMaxDistance = 0.12;
constexpr double kFineHuber = 0.03;
constexpr double kFineNormalRadius = 0.20;
constexpr double kFineNormalMaxCurvature = 0.03;
constexpr double kFineNormalMinSpreadRatio = 0.08;
constexpr double kFineRejectTranslation = 0.03;
constexpr double kFineRejectRotation = 1.0 * kDegToRad;
constexpr int kFineOuterIterations = 2;
constexpr int kFinePhaseARotationIterations = 10;
constexpr int kFinePhaseBTranslationIterations = 5;
constexpr int kFinePhaseCJointIterations = 5;

bool ReadMatrix4f(std::istream& input, Eigen::Matrix4f* matrix)
{
    for (int row = 0; row != 4; ++row)
    {
        for (int col = 0; col != 4; ++col)
        {
            if (!(input >> (*matrix)(row, col)))
            {
                return false;
            }
        }
    }
    return true;
}

Eigen::Matrix4f InverseRigidTransform(const Eigen::Matrix4f& transform)
{
    Eigen::Matrix4f inverse = Eigen::Matrix4f::Identity();
    const Eigen::Matrix3f rotation = transform.block<3, 3>(0, 0);
    const Eigen::Vector3f translation = transform.block<3, 1>(0, 3);
    inverse.block<3, 3>(0, 0) = rotation.transpose();
    inverse.block<3, 1>(0, 3) = -rotation.transpose() * translation;
    return inverse;
}

double RotationAngle(const Eigen::Matrix3f& rotation)
{
    const double trace = static_cast<double>(rotation.trace());
    double cos_angle = 0.5 * (trace - 1.0);
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
    return std::acos(cos_angle);
}

Eigen::Matrix3d AngleAxisToMatrix(const double angle_axis[3])
{
    const Eigen::Vector3d v(angle_axis[0], angle_axis[1], angle_axis[2]);
    const double angle = v.norm();
    if (angle < 1e-12)
    {
        return Eigen::Matrix3d::Identity();
    }
    return Eigen::AngleAxisd(angle, v / angle).toRotationMatrix();
}

Eigen::Vector3f RpyDegrees(const Eigen::Matrix4f& transform)
{
    const Eigen::Matrix3f r = transform.block<3, 3>(0, 0);
    const double pitch = std::asin(std::max(-1.0, std::min(1.0, -static_cast<double>(r(2, 0)))));
    double roll = 0.0;
    double yaw = std::atan2(-static_cast<double>(r(0, 1)), static_cast<double>(r(1, 1)));
    if (std::abs(std::cos(pitch)) > 1e-12)
    {
        roll = std::atan2(static_cast<double>(r(2, 1)), static_cast<double>(r(2, 2)));
        yaw = std::atan2(static_cast<double>(r(1, 0)), static_cast<double>(r(0, 0)));
    }
    return Eigen::Vector3f(
        static_cast<float>(roll / kDegToRad),
        static_cast<float>(pitch / kDegToRad),
        static_cast<float>(yaw / kDegToRad));
}

pcl::PointCloud<pcl::PointXYZ>::Ptr VoxelizeMap(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input,
    const float leaf)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setLeafSize(leaf, leaf, leaf);
    voxel.setInputCloud(input);
    pcl::PointXYZ map_min, map_max;
    pcl::getMinMax3D(*input, map_min, map_max);
    const double inverse_leaf = 1.0 / leaf;
    const double grid_cells =
        (std::floor((map_max.x - map_min.x) * inverse_leaf) + 1.0) *
        (std::floor((map_max.y - map_min.y) * inverse_leaf) + 1.0) *
        (std::floor((map_max.z - map_min.z) * inverse_leaf) + 1.0);
    if (grid_cells <= std::numeric_limits<std::int32_t>::max())
    {
        voxel.filter(*output);
        return output;
    }

    std::map<std::array<double, 3>, pcl::IndicesPtr> tiles;
    for (std::size_t i = 0; i < input->size(); ++i)
    {
        const auto& point = (*input)[i];
        if (!pcl::isFinite(point))
        {
            continue;
        }
        const std::array<double, 3> key = {
            std::floor(std::floor(point.x * static_cast<float>(inverse_leaf)) / 512.0),
            std::floor(std::floor(point.y * static_cast<float>(inverse_leaf)) / 512.0),
            std::floor(std::floor(point.z * static_cast<float>(inverse_leaf)) / 512.0)};
        auto& indices = tiles[key];
        if (!indices)
        {
            indices.reset(new std::vector<int>);
        }
        indices->push_back(static_cast<int>(i));
    }
    for (const auto& tile : tiles)
    {
        pcl::PointCloud<pcl::PointXYZ> filtered_tile;
        voxel.setIndices(tile.second);
        voxel.filter(filtered_tile);
        *output += filtered_tile;
    }
    return output;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr DownsampleCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input,
    const float leaf)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>);
    if (input->empty())
    {
        return output;
    }
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setLeafSize(leaf, leaf, leaf);
    voxel.setInputCloud(input);
    voxel.filter(*output);
    return output;
}

struct FineNormal
{
    Eigen::Vector3f n = Eigen::Vector3f::Zero();
    bool valid = false;
};

std::vector<FineNormal> EstimateFineNormals(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const pcl::KdTreeFLANN<pcl::PointXYZ>& tree)
{
    std::vector<FineNormal> normals(cloud->size());
    std::vector<int> ids;
    std::vector<float> distances;
    for (std::size_t i = 0; i < cloud->size(); ++i)
    {
        if (tree.radiusSearch((*cloud)[i], kFineNormalRadius, ids, distances, 80) < 12)
        {
            continue;
        }
        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        for (int id : ids)
        {
            mean += (*cloud)[id].getVector3fMap().cast<double>();
        }
        mean /= static_cast<double>(ids.size());
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (int id : ids)
        {
            const Eigen::Vector3d d =
                (*cloud)[id].getVector3fMap().cast<double>() - mean;
            covariance.noalias() += d * d.transpose();
        }
        covariance /= static_cast<double>(ids.size());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        if (solver.info() != Eigen::Success)
        {
            continue;
        }
        const auto values = solver.eigenvalues();
        if (!values.allFinite() ||
            values[1] < 1e-8 ||
            values[2] <= 0.0 ||
            values[0] / values.sum() > kFineNormalMaxCurvature ||
            values[1] / values[2] < kFineNormalMinSpreadRatio)
        {
            continue;
        }
        normals[i].n = solver.eigenvectors().col(0).cast<float>().normalized();
        normals[i].valid = true;
    }
    return normals;
}

int NormalDirection(const Eigen::Vector3f& n)
{
    Eigen::Index axis = 0;
    n.cwiseAbs().maxCoeff(&axis);
    return static_cast<int>(axis);
}

double FineRangeWeight(const Eigen::Vector3f& source_point)
{
    const double range = source_point.norm();
    if (range < 3.0)
    {
        return 1.0;
    }
    if (range < 6.0)
    {
        return 1.2;
    }
    return 1.5;
}

struct FineCorrespondence
{
    Eigen::Vector3d p;
    Eigen::Vector3d q;
    Eigen::Vector3d n;
    double weight = 1.0;
};

std::vector<FineCorrespondence> BuildFineCorrespondences(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    const pcl::KdTreeFLANN<pcl::PointXYZ>& target_tree,
    const std::vector<FineNormal>& normals,
    const Eigen::Matrix4f& transform,
    const unsigned seed)
{
    std::array<std::vector<FineCorrespondence>, 3> by_direction;
    std::vector<int> ids(1);
    std::vector<float> distances(1);
    for (const auto& point : *source)
    {
        if (!pcl::isFinite(point))
        {
            continue;
        }
        const Eigen::Vector3f p = point.getVector3fMap();
        const Eigen::Vector3f transformed =
            (transform * p.homogeneous()).head<3>();
        if (target_tree.nearestKSearch(
                pcl::PointXYZ(transformed.x(), transformed.y(), transformed.z()),
                1, ids, distances) != 1 ||
            distances[0] > kFineMaxDistance * kFineMaxDistance ||
            ids[0] < 0 ||
            static_cast<std::size_t>(ids[0]) >= normals.size() ||
            !normals[ids[0]].valid)
        {
            continue;
        }
        FineCorrespondence correspondence;
        correspondence.p = p.cast<double>();
        correspondence.q = (*target)[ids[0]].getVector3fMap().cast<double>();
        correspondence.n = normals[ids[0]].n.cast<double>();
        correspondence.weight = FineRangeWeight(p);
        by_direction[NormalDirection(normals[ids[0]].n)].push_back(correspondence);
    }

    std::mt19937 rng(seed);
    for (auto& bucket : by_direction)
    {
        std::shuffle(bucket.begin(), bucket.end(), rng);
    }

    std::array<std::size_t, 3> available = {
        by_direction[0].size(), by_direction[1].size(), by_direction[2].size()};
    std::size_t total_available = available[0] + available[1] + available[2];
    const std::size_t target_count = std::min(kFineMaxCorrespondences, total_available);
    std::array<std::size_t, 3> quota = {0, 0, 0};
    std::size_t remaining = target_count;
    int active = 0;
    for (auto n : available)
    {
        if (n)
        {
            ++active;
        }
    }
    while (remaining && active)
    {
        const std::size_t base = std::max<std::size_t>(1, remaining / active);
        bool progress = false;
        for (int d = 0; d < 3 && remaining; ++d)
        {
            if (quota[d] >= available[d])
            {
                continue;
            }
            const std::size_t take = std::min(base, available[d] - quota[d]);
            quota[d] += take;
            remaining -= take;
            progress = progress || take > 0;
        }
        active = 0;
        for (int d = 0; d < 3; ++d)
        {
            if (quota[d] < available[d])
            {
                ++active;
            }
        }
        if (!progress)
        {
            break;
        }
    }

    std::vector<FineCorrespondence> result;
    result.reserve(target_count);
    for (int d = 0; d < 3; ++d)
    {
        for (std::size_t i = 0; i < quota[d]; ++i)
        {
            result.push_back(by_direction[d][i]);
        }
    }
    std::shuffle(result.begin(), result.end(), rng);
    return result;
}

double PointToPlaneRmse(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    const pcl::KdTreeFLANN<pcl::PointXYZ>& target_tree,
    const std::vector<FineNormal>& normals,
    const Eigen::Matrix4f& transform)
{
    const auto correspondences = BuildFineCorrespondences(
        source, target, target_tree, normals, transform, 20260915u);
    if (correspondences.empty())
    {
        return std::numeric_limits<double>::infinity();
    }
    double squared_sum = 0.0;
    for (const auto& c : correspondences)
    {
        const Eigen::Vector3d p =
            (transform.cast<double>() * c.p.homogeneous()).head<3>();
        const double r = c.n.dot(p - c.q);
        squared_sum += r * r;
    }
    return std::sqrt(squared_sum / static_cast<double>(correspondences.size()));
}

struct RotationOnlyResidual
{
    Eigen::Vector3d rp;
    Eigen::Vector3d t;
    Eigen::Vector3d q;
    Eigen::Vector3d n;
    template <typename T>
    bool operator()(const T* const angle_axis, T* residual) const
    {
        const T p0[3] = {T(rp.x()), T(rp.y()), T(rp.z())};
        T rotated[3];
        ceres::AngleAxisRotatePoint(angle_axis, p0, rotated);
        residual[0] =
            T(n.x()) * (rotated[0] + T(t.x()) - T(q.x())) +
            T(n.y()) * (rotated[1] + T(t.y()) - T(q.y())) +
            T(n.z()) * (rotated[2] + T(t.z()) - T(q.z()));
        return true;
    }
};

struct TranslationOnlyResidual
{
    Eigen::Vector3d rp;
    Eigen::Vector3d t;
    Eigen::Vector3d q;
    Eigen::Vector3d n;
    template <typename T>
    bool operator()(const T* const delta_t, T* residual) const
    {
        residual[0] =
            T(n.x()) * (T(rp.x()) + T(t.x()) + delta_t[0] - T(q.x())) +
            T(n.y()) * (T(rp.y()) + T(t.y()) + delta_t[1] - T(q.y())) +
            T(n.z()) * (T(rp.z()) + T(t.z()) + delta_t[2] - T(q.z()));
        return true;
    }
};

struct JointResidual
{
    Eigen::Vector3d current_p;
    Eigen::Vector3d q;
    Eigen::Vector3d n;
    template <typename T>
    bool operator()(const T* const xi, T* residual) const
    {
        const T p0[3] = {T(current_p.x()), T(current_p.y()), T(current_p.z())};
        T rotated[3];
        ceres::AngleAxisRotatePoint(xi + 3, p0, rotated);
        residual[0] =
            T(n.x()) * (rotated[0] + xi[0] - T(q.x())) +
            T(n.y()) * (rotated[1] + xi[1] - T(q.y())) +
            T(n.z()) * (rotated[2] + xi[2] - T(q.z()));
        return true;
    }
};

ceres::Solver::Options FineSolverOptions(const int max_iterations)
{
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = max_iterations;
    options.function_tolerance = 1e-9;
    options.parameter_tolerance = 1e-10;
    options.gradient_tolerance = 1e-12;
    options.num_threads = 1;
    options.minimizer_progress_to_stdout = false;
    return options;
}

void AddPointToPlaneLoss(ceres::Problem* problem,
                         ceres::CostFunction* cost,
                         double weight,
                         double* parameter)
{
    problem->AddResidualBlock(
        cost,
        new ceres::ScaledLoss(
            new ceres::HuberLoss(kFineHuber),
            weight,
            ceres::TAKE_OWNERSHIP),
        parameter);
}

struct FineResult
{
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    double rmse_before = std::numeric_limits<double>::infinity();
    double rmse_after = std::numeric_limits<double>::infinity();
    double delta_translation = 0.0;
    double delta_rotation = 0.0;
    bool used = false;
};

FineResult RefineFinePointToPlane(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    const Eigen::Matrix4f& gicp_transform,
    const unsigned seed)
{
    FineResult result;
    result.transform = gicp_transform;
    if (source->empty() || target->size() < kMinLocalMapPoints)
    {
        return result;
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> target_tree(false);
    target_tree.setInputCloud(target);
    const auto normals = EstimateFineNormals(target, target_tree);
    result.rmse_before = PointToPlaneRmse(
        source, target, target_tree, normals, gicp_transform);
    if (!std::isfinite(result.rmse_before))
    {
        return result;
    }

    Eigen::Matrix4f current = gicp_transform;

    for (int outer = 0; outer < kFineOuterIterations; ++outer)
    {
        {
            const auto correspondences = BuildFineCorrespondences(
                source, target, target_tree, normals, current, seed + 31u * outer + 1u);
            if (correspondences.size() < kFineMinCorrespondences)
            {
                break;
            }
            double angle_axis[3] = {0.0, 0.0, 0.0};
            ceres::Problem problem;
            const Eigen::Matrix3d r = current.block<3, 3>(0, 0).cast<double>();
            const Eigen::Vector3d t = current.block<3, 1>(0, 3).cast<double>();
            for (const auto& c : correspondences)
            {
                const RotationOnlyResidual residual{
                    r * c.p, t, c.q, c.n};
                AddPointToPlaneLoss(
                    &problem,
                    new ceres::AutoDiffCostFunction<RotationOnlyResidual, 1, 3>(
                        new RotationOnlyResidual(residual)),
                    c.weight,
                    angle_axis);
            }
            ceres::Solver::Summary summary;
            ceres::Solve(FineSolverOptions(kFinePhaseARotationIterations), &problem, &summary);
            if (!summary.IsSolutionUsable())
            {
                break;
            }
            const Eigen::Matrix3d delta_r = AngleAxisToMatrix(angle_axis);
            current.block<3, 3>(0, 0) =
                (delta_r * current.block<3, 3>(0, 0).cast<double>()).cast<float>();
        }

        {
            const auto correspondences = BuildFineCorrespondences(
                source, target, target_tree, normals, current, seed + 31u * outer + 11u);
            if (correspondences.size() < kFineMinCorrespondences)
            {
                break;
            }
            double delta_t[3] = {0.0, 0.0, 0.0};
            ceres::Problem problem;
            const Eigen::Matrix3d r = current.block<3, 3>(0, 0).cast<double>();
            const Eigen::Vector3d t = current.block<3, 1>(0, 3).cast<double>();
            for (const auto& c : correspondences)
            {
                const TranslationOnlyResidual residual{
                    r * c.p, t, c.q, c.n};
                AddPointToPlaneLoss(
                    &problem,
                    new ceres::AutoDiffCostFunction<TranslationOnlyResidual, 1, 3>(
                        new TranslationOnlyResidual(residual)),
                    c.weight,
                    delta_t);
            }
            ceres::Solver::Summary summary;
            ceres::Solve(FineSolverOptions(kFinePhaseBTranslationIterations), &problem, &summary);
            if (!summary.IsSolutionUsable())
            {
                break;
            }
            current.block<3, 1>(0, 3) += Eigen::Vector3f(
                static_cast<float>(delta_t[0]),
                static_cast<float>(delta_t[1]),
                static_cast<float>(delta_t[2]));
        }

        {
            const auto correspondences = BuildFineCorrespondences(
                source, target, target_tree, normals, current, seed + 31u * outer + 21u);
            if (correspondences.size() < kFineMinCorrespondences)
            {
                break;
            }
            double xi[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            ceres::Problem problem;
            const Eigen::Matrix4d current_d = current.cast<double>();
            for (const auto& c : correspondences)
            {
                const JointResidual residual{
                    (current_d * c.p.homogeneous()).head<3>(), c.q, c.n};
                AddPointToPlaneLoss(
                    &problem,
                    new ceres::AutoDiffCostFunction<JointResidual, 1, 6>(
                        new JointResidual(residual)),
                    c.weight,
                    xi);
            }
            for (int i = 0; i < 3; ++i)
            {
                problem.SetParameterLowerBound(xi, i, -kFineRejectTranslation);
                problem.SetParameterUpperBound(xi, i, kFineRejectTranslation);
                problem.SetParameterLowerBound(xi, i + 3, -kFineRejectRotation);
                problem.SetParameterUpperBound(xi, i + 3, kFineRejectRotation);
            }
            ceres::Solver::Summary summary;
            ceres::Solve(FineSolverOptions(kFinePhaseCJointIterations), &problem, &summary);
            if (!summary.IsSolutionUsable())
            {
                break;
            }
            const Eigen::Matrix3d delta_r = AngleAxisToMatrix(xi + 3);
            current.block<3, 3>(0, 0) =
                (delta_r * current.block<3, 3>(0, 0).cast<double>()).cast<float>();
            current.block<3, 1>(0, 3) += Eigen::Vector3f(
                static_cast<float>(xi[0]),
                static_cast<float>(xi[1]),
                static_cast<float>(xi[2]));
        }
    }

    result.rmse_after = PointToPlaneRmse(
        source, target, target_tree, normals, current);
    const Eigen::Matrix4f delta = current * InverseRigidTransform(gicp_transform);
    result.delta_translation = delta.block<3, 1>(0, 3).norm();
    result.delta_rotation = RotationAngle(delta.block<3, 3>(0, 0));
    if (std::isfinite(result.rmse_after) &&
        result.rmse_after < result.rmse_before &&
        result.delta_translation <= kFineRejectTranslation &&
        result.delta_rotation <= kFineRejectRotation)
    {
        result.transform = current;
        result.used = true;
    }
    return result;
}

}  // namespace

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

void printProgress(double percentage)
{
    int val = (int)(percentage * 100);
    int lpad = (int)(percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush(stdout);
}

char framesDir[100] = "../data/Target_LiDAR_Frames";

std::string itos(int i)
{
    std::stringstream s;
    s << i;
    return s.str();
}

int main()
{
    std::cout << "Start calibration..." << std::endl;

    //================== Step.1 Reading H-LiDAR map data =====================//

    std::cout << "Reading H-LiDAR map data..." << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr H_LiDAR_Map(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("../data/H-LiDAR-Map-data/H_LiDAR_Map.pcd", *H_LiDAR_Map) == -1)
    {
        PCL_ERROR("Couldn't read H_LiDAR_Map \n");
        return (-1);
    }
    // Keep the original map intact; registration uses separate voxelized maps.
    pcl::PointCloud<pcl::PointXYZ>::Ptr H_LiDAR_Map_reg(
        new pcl::PointCloud<pcl::PointXYZ>);
    H_LiDAR_Map_reg = VoxelizeMap(H_LiDAR_Map, kRegistrationMapVoxel);
    std::cout << "Registration map: original_points=" << H_LiDAR_Map->size()
              << ", registration_points=" << H_LiDAR_Map_reg->size()
              << ", voxel_m=" << kRegistrationMapVoxel
              << ", fine_target_voxel_m=" << kFineTargetVoxel << std::endl;
    if (H_LiDAR_Map_reg->empty())
    {
        std::cerr << "ERROR: registration map is empty" << std::endl;
        return -1;
    }
    // Build once. Radius results need not be sorted by distance.
    pcl::KdTreeFLANN<pcl::PointXYZ> registration_tree(false);
    registration_tree.setInputCloud(H_LiDAR_Map_reg);
    std::vector<int> local_indices;
    std::vector<float> local_squared_distances;

    //================== Step.2 Reading H-LiDAR's Trajectory and init guess=====================//

    ifstream T_Mat_File("../data/T_Matrix.txt");
    Eigen::Matrix4f T_Matrix = Eigen::Matrix4f::Identity();

    ifstream initFile("../data/Init_Matrix.txt");

    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f init_guess_0 = Eigen::Matrix4f::Identity();

    if (!T_Mat_File.is_open())
    {
        std::cerr << "ERROR: cannot open ../data/T_Matrix.txt" << std::endl;
        return -1;
    }

    if (!initFile.is_open())
    {
        std::cerr << "ERROR: cannot open ../data/Init_Matrix.txt" << std::endl;
        return -1;
    }

    if (!ReadMatrix4f(initFile, &init_guess))
    {
        std::cerr << "ERROR: cannot read a 4x4 matrix from ../data/Init_Matrix.txt" << std::endl;
        return -1;
    }

    init_guess_0 = init_guess;
    //================== Step.3 Reading L-LiDAR frames =====================//

    struct dirent **namelist;
    int framenumbers = scandir(framesDir, &namelist, 0, alphasort) - 2;
    int frame_count = 100000;
    int cframe_count = 0;
    cout << "Loaded " << framenumbers << " frames from Target-LiDAR" << endl;

    //================== small_gicp ==================//
    pcl::PointCloud<pcl::PointXYZ>::Ptr ICP_output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    small_gicp::RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ> icp; // small_gicp PCL-compatible interface
    icp.setNumThreads(8);    // CPU 8 线程
    icp.setRegistrationType("GICP"); // 和原程序一样，先用 GICP，不先引入 VGICP 这个变量
    icp.setCorrespondenceRandomness(20); // GICP 每个点估计协方差时使用的邻居数
    icp.setMaxCorrespondenceDistance(0.5);  // Stage 1 only needs a stable local basin.
    icp.setMaximumIterations(50); // 最大优化迭代次数
    icp.setTransformationEpsilon(1e-6);    // 平移收敛阈值
    icp.setRotationEpsilon(1e-6); // 旋转收敛阈值
    icp.setVerbosity(false); // 不刷 small_gicp 内部日志
    //================================================//

    // //=================================
    // //prepare display
    // boost::shared_ptr<pcl::visualization::PCLVisualizer>
    //     viewer_final(new pcl::visualization::PCLVisualizer("3D Viewer"));
    // viewer_final->setBackgroundColor(0, 0, 0);
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> map_color(H_LiDAR_Map, 255, 0, 0);
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> match_color(H_LiDAR_Map, 0, 255, 0);

    // viewer_final->addPointCloud<pcl::PointXYZ>(H_LiDAR_Map, map_color, "target cloud");
    // viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");
    // viewer_final->addPointCloud<pcl::PointXYZ>(H_LiDAR_Map, match_color, "match cloud"); //display the match cloud

    //=================================
    // prepare save matrix

    char filename[] = "../data/calib_data.txt";
    ofstream fout(filename);
    fout.setf(ios::fixed, ios::floatfield);
    fout.precision(7);
    int accepted_count = 0;
    int rejected_count = 0;

    //=================================
    //              START
    //=================================
    while (true)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr frames(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(string(framesDir) + "/" + itos(frame_count) + ".pcd", *frames) == -1)
        {
            PCL_ERROR("Couldn't read Target_LiDAR frame \n");
            return (-1);
        }
        //std::cout << "Loaded " << frames->size() << " data points from frames" << std::endl;

        //Load H-LiDAR's Trajectory
        if (!ReadMatrix4f(T_Mat_File, &T_Matrix))
        {
            std::cerr << "\nERROR: T_Matrix.txt ended before frame "
                      << frame_count << std::endl;
            return -1;
        }

        //================== Step.4 Start calibration =====================//

        const auto calibration_start = std::chrono::steady_clock::now();
        std::size_t local_map_points = 0;
        std::size_t queried_map_points = 0;
        std::size_t fine_local_map_points = 0;
        bool fine_used = false;
        FineResult fine_result;
        const auto log_calibration_time = [&]() {
            if ((cframe_count + 1) % 100 == 0)
            {
                const double elapsed_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - calibration_start).count();
                std::cout << "\nFrame " << frame_count
                          << ": local_map_points=" << local_map_points
                          << ", fine_local_map_points=" << fine_local_map_points
                          << ", radius_query_points=" << queried_map_points
                          << ", fine_used=" << (fine_used ? 1 : 0)
                          << ", fine_rmse_before=" << fine_result.rmse_before
                          << ", fine_rmse_after=" << fine_result.rmse_after
                          << ", fine_delta_translation_m=" << fine_result.delta_translation
                          << ", fine_delta_rotation_deg=" << fine_result.delta_rotation / kDegToRad
                          << ", calibration_ms=" << elapsed_ms << std::endl;
            }
        };

        pcl::PointCloud<pcl::PointXYZ>::Ptr trans_output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*frames, *trans_output_cloud, init_guess_0);

        if (trans_output_cloud->empty())
        {
            ++rejected_count;
            log_calibration_time();
            frame_count++;
            cframe_count++;
            printProgress((double)cframe_count / (double)framenumbers);

            if (cframe_count == framenumbers)
            {
                std::cout << "\n Matching complete." << std::endl;
                std::cout << "Accepted frames: " << accepted_count
                        << ", rejected frames: " << rejected_count << std::endl;
                break;
            }
            continue;
        }

        // Query real map geometry around the Base LiDAR origin in map frame.
        const pcl::PointXYZ base_position(
            T_Matrix(0, 3), T_Matrix(1, 3), T_Matrix(2, 3));
        registration_tree.radiusSearch(
            base_position, kLocalMapRadius, local_indices, local_squared_distances);
        pcl::PointCloud<pcl::PointXYZ>::Ptr local_map_world(
            new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud(*H_LiDAR_Map_reg, local_indices, *local_map_world);
        queried_map_points = local_map_world->size();
        if (local_map_world->size() > kLocalMapDownsampleThreshold)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_local_map(
                new pcl::PointCloud<pcl::PointXYZ>);
            pcl::VoxelGrid<pcl::PointXYZ> local_voxel;
            local_voxel.setLeafSize(
                kRegistrationMapVoxel, kRegistrationMapVoxel, kRegistrationMapVoxel);
            local_voxel.setInputCloud(local_map_world);
            local_voxel.filter(*filtered_local_map);
            local_map_world.swap(filtered_local_map);
        }
        local_map_points = local_map_world->size();

        // Reject a frame if there is not enough real map geometry.
        if (local_map_world->size() < kMinLocalMapPoints)
        {
            ++rejected_count;
            log_calibration_time();

            frame_count++;
            cframe_count++;
            printProgress((double)cframe_count / (double)framenumbers);

            if (cframe_count == framenumbers)
            {
                std::cout << "\n Matching complete." << std::endl;
                std::cout << "Accepted frames: " << accepted_count
                        << ", rejected frames: " << rejected_count << std::endl;
                break;
            }
            continue;
        }

        // Bring the real local map from map frame back into the Base-LiDAR
        // local frame of this timestamp.
        //
        // p_map  = T_Matrix * p_base
        // p_base = inverse(T_Matrix) * p_map
        const Eigen::Matrix4f T_Matrix_Inverse =
            InverseRigidTransform(T_Matrix);

        pcl::PointCloud<pcl::PointXYZ>::Ptr local_map_base(
            new pcl::PointCloud<pcl::PointXYZ>);

        pcl::transformPointCloud(
            *local_map_world,
            *local_map_base,
            T_Matrix_Inverse);

        // GICP now aligns:
        //   source: Target frame after coarse extrinsic, in Base-LiDAR frame
        //   target: REAL local Base map, also in Base-LiDAR frame
        //
        // Tiny_T therefore remains a residual Target->Base correction,
        // so the existing:
        //   Final_Calib_T = Tiny_T * init_guess_0
        // remains valid.
        icp.setInputSource(trans_output_cloud);
        icp.setInputTarget(local_map_base);
        icp.align(*ICP_output_cloud);
        const Eigen::Matrix4f Tiny_T = icp.getFinalTransformation();
        const double fitness = icp.getFitnessScore();
        const double correction_translation = Tiny_T.block<3, 1>(0, 3).norm();
        const double correction_rotation = RotationAngle(Tiny_T.block<3, 3>(0, 0));

        if (!icp.hasConverged() ||
            !std::isfinite(fitness) ||
            fitness > kMaxFitnessScore ||
            correction_translation > kMaxCorrectionTranslation ||
            correction_rotation > kMaxCorrectionRotation)
        {
            ++rejected_count;
        }
        else
        {
            Eigen::Matrix4f Final_Calib_T = Eigen::Matrix4f::Identity();

            Eigen::Matrix4f T_frame = Tiny_T;
            pcl::PointCloud<pcl::PointXYZ>::Ptr fine_local_map_base =
                DownsampleCloud(local_map_base, kFineTargetVoxel);
            fine_local_map_points = fine_local_map_base->size();
            pcl::PointCloud<pcl::PointXYZ>::Ptr fine_source_cloud =
                DownsampleCloud(trans_output_cloud, kFineSourceVoxel);
            if (fine_local_map_base->size() >= kMinLocalMapPoints &&
                fine_source_cloud->size() >= kFineMinCorrespondences)
            {
                fine_result = RefineFinePointToPlane(
                    fine_source_cloud,
                    fine_local_map_base,
                    Tiny_T,
                    20260915u + static_cast<unsigned>(frame_count));
                fine_used = fine_result.used;
                if (fine_used)
                {
                    T_frame = fine_result.transform;
                }
            }

            Final_Calib_T = T_frame * init_guess_0;

            //===== Out put Euler angle =====//
            gu::Vector3 EulerAngle;
            gu::Rot3 rot_mat(Final_Calib_T(0, 0), Final_Calib_T(0, 1), Final_Calib_T(0, 2),
                             Final_Calib_T(1, 0), Final_Calib_T(1, 1), Final_Calib_T(1, 2),
                             Final_Calib_T(2, 0), Final_Calib_T(2, 1), Final_Calib_T(2, 2));
            EulerAngle = rot_mat.GetEulerZYX();
            const Eigen::Matrix<double, 3, 1> EulerAngle_T = EulerAngle.Eigen();
            //std::cout<<"EulerAngle:  "<<EulerAngle_T(0,0)<<"  "<<EulerAngle_T(1,0)<<"  "<<EulerAngle_T(2,0)<<"  "<<std::endl;

            fout << frame_count - 100000 << " "
                << fitness << " "
                << Final_Calib_T(0, 3) << " "
                << Final_Calib_T(1, 3) << " "
                << Final_Calib_T(2, 3) << " "
                << EulerAngle_T(0, 0) << " "
                << EulerAngle_T(1, 0) << " "
                << EulerAngle_T(2, 0)
                << endl;
            ++accepted_count;

            if ((cframe_count + 1) % 100 == 0)
            {
                const Eigen::Vector3f gicp_rpy = RpyDegrees(Tiny_T * init_guess_0);
                const Eigen::Vector3f fine_rpy = RpyDegrees(Final_Calib_T);
                std::cout << "Frame " << frame_count
                          << ": gicp_fitness=" << fitness
                          << ", gicp_xyz=[" << (Tiny_T * init_guess_0)(0, 3)
                          << "," << (Tiny_T * init_guess_0)(1, 3)
                          << "," << (Tiny_T * init_guess_0)(2, 3) << "]"
                          << ", gicp_rpy_deg=[" << gicp_rpy.x()
                          << "," << gicp_rpy.y()
                          << "," << gicp_rpy.z() << "]"
                          << ", fine_xyz=[" << Final_Calib_T(0, 3)
                          << "," << Final_Calib_T(1, 3)
                          << "," << Final_Calib_T(2, 3) << "]"
                          << ", fine_rpy_deg=[" << fine_rpy.x()
                          << "," << fine_rpy.y()
                          << "," << fine_rpy.z() << "]" << std::endl;
            }
        }

        log_calibration_time();
        frame_count++;
        cframe_count++;

        printProgress((double)cframe_count / (double)framenumbers);
        // viewer_final->updatePointCloud<pcl::PointXYZ>(final_output_cloud, match_color, "match cloud");
        // viewer_final->spinOnce(10);

        if (cframe_count == framenumbers)
        {
            std::cout << "\n Matching complete." << std::endl;
            std::cout << "Accepted frames: " << accepted_count
                      << ", rejected frames: " << rejected_count << std::endl;
            break;
        }
    }

    return 0;
}
