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
#include <sstream>
#include <fstream>
#include <string>
#include <iostream>
#include <limits>
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
#include <small_gicp/pcl/pcl_registration.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <geometry_utils/Transform3.h>
#include <point_cloud_mapper/PointCloudMapper.h>

using namespace std;
namespace gu = geometry_utils;

#define PI (3.1415926535897932346f)
namespace {

constexpr double kDegToRad = PI / 180.0;
constexpr double kMaxFitnessScore = 0.05;
constexpr double kMaxCorrectionTranslation = 0.25;
constexpr double kMaxCorrectionRotation = 6.0 * kDegToRad;

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
    std::cout << "Loaded " << H_LiDAR_Map->size() << " data points from H_LiDAR_Map.pcd" << std::endl;

    //put it into map
    PointCloudMapper maps;
    maps.Initialize();
    pcl::PointCloud<pcl::PointXYZ>::Ptr unused(new pcl::PointCloud<pcl::PointXYZ>);
    maps.InsertPoints(H_LiDAR_Map, unused.get());

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
    icp.setMaxCorrespondenceDistance(1.0);  // 官方地图近邻流程下限制错误结构吸附
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

        pcl::PointCloud<pcl::PointXYZ>::Ptr trans_output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr final_output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*frames, *trans_output_cloud, init_guess_0);

        pcl::transformPointCloud(*trans_output_cloud, *final_output_cloud, T_Matrix);

        pcl::PointCloud<pcl::PointXYZ>::Ptr neighbors_L(new pcl::PointCloud<pcl::PointXYZ>); //201 neighbors points from nap202
        maps.ApproxNearestNeighbors(*final_output_cloud, neighbors_L.get());

        // Bring the local map neighbors back to the base LiDAR frame.
        const Eigen::Matrix4f T_Matrix_Inverse = InverseRigidTransform(T_Matrix);
        pcl::PointCloud<pcl::PointXYZ>::Ptr neighbors_trans(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*neighbors_L, *neighbors_trans, T_Matrix_Inverse);

        //Do ICP and get the tiny trans T
        icp.setInputSource(trans_output_cloud); //201
        icp.setInputTarget(neighbors_trans);    //202 (201's neighbor's point cloud)
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

            Final_Calib_T = Tiny_T * init_guess_0;

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
        }

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
