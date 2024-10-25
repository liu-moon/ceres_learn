# 引入 FetchContent 模块
include(FetchContent)

# 使用 FetchContent_Declare 来获取 Ceres 的源码
FetchContent_Declare(
  Ceres
  GIT_REPOSITORY https://github.com/ceres-solver/ceres-solver.git
  GIT_TAG        2.2.0
)
FetchContent_MakeAvailable(Ceres)


find_package(Ceres REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main Ceres::ceres)

add_executable(numeric_diff helloworld_numeric_diff.cpp)
target_link_libraries(numeric_diff Ceres::ceres)

add_executable(analytic_diff helloworld_analytic_diff.cpp)
target_link_libraries(analytic_diff Ceres::ceres)
