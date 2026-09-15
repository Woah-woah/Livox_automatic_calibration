CMAKE_MINIMUM_REQUIRED(VERSION 3.10 FATAL_ERROR)

PROJECT(livox_calibration)

set(CMAKE_BUILD_TYPE Release)

find_package(PCL 1.7 REQUIRED)
find_package(small_gicp REQUIRED)
find_package(Ceres REQUIRED)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(
    ${PCL_INCLUDE_DIRS}
    include
)

# 老的 mapping / fitline 继续保持 C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

FILE(GLOB SRCS0 src/mapping/*.cpp)
FILE(GLOB SRCS1 src/calibration/*.cpp)
FILE(GLOB SRCS2 src/ransac/*.cpp)

add_executable(mapping ${SRCS0})
target_link_libraries(
    mapping
    ${PCL_LIBRARIES}
    ${LIBS}
)

add_executable(calibration ${SRCS1})

# small_gicp 需要 C++17，只给 calibration 开
set_property(TARGET calibration PROPERTY CXX_STANDARD 17)
set_property(TARGET calibration PROPERTY CXX_STANDARD_REQUIRED ON)

target_link_libraries(
    calibration
    ${PCL_LIBRARIES}
    ${LIBS}
    Ceres::ceres
    small_gicp::small_gicp
)

add_executable(fitline ${SRCS2})
target_link_libraries(
    fitline
    ${PCL_LIBRARIES}
    ${LIBS}
)