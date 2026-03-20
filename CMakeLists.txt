cmake_minimum_required(VERSION 3.8)
project(planner_payload_cpp)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(quadrotor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_srvs REQUIRED)

# ARM NEON flags
if ("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "armv7l")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv7-a -mfpu=neon -mfloat-abi=hard -funsafe-math-optimizations")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mfpu=neon -mfloat-abi=hard -funsafe-math-optimizations")
  message("enabling ARM neon optimizations")
endif()

# flags for speed (should already be enabled by default)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp -O3")

set(acados_path "$ENV{ACADOS_SOURCE_DIR}")
set(acados_lib "${acados_path}/lib")
set(acados_generated_code "${PROJECT_SOURCE_DIR}/c_generated_code")

if("${acados_path}" STREQUAL "")
  message(FATAL_ERROR "ACADOS_SOURCE_DIR is not set. Please export ACADOS_SOURCE_DIR to your acados root path.")
endif()

set(dependencies
  rclcpp
  geometry_msgs
  nav_msgs
  quadrotor_msgs
  std_msgs
  sensor_msgs
  tf2
  tf2_ros
  rclcpp_components
  std_srvs
)

include_directories(include
  ${EIGEN3_INCLUDE_DIR}
  ${acados_generated_code}
  ${acados_path}/interfaces
  ${acados_path}/acados
  ${acados_path}/acados/utils
  ${acados_path}
  ${acados_path}/external
  ${acados_path}/external/blasfeo/include
  ${acados_path}/external/hpipm/include
)

link_directories(${acados_generated_code})

add_library(planner_payload_component SHARED
  src/nodelet.cpp
  src/wrapper.cpp
  src/nmpc_planner.cpp
)

set_target_properties(planner_payload_component PROPERTIES COMPILE_DEFINITIONS "COMPOSITION_BUILDING_DLL")

target_include_directories(planner_payload_component
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(planner_payload_component ${dependencies})
target_link_libraries(planner_payload_component
  Eigen3::Eigen
  yaml-cpp
  libacados_ocp_solver_planner_payload.so
  ${acados_lib}/libacados.so
)

rclcpp_components_register_node(planner_payload_component
  PLUGIN "planner_payload_nodelet::NMPCControlNodelet"
  EXECUTABLE planner_payload_nodelet_exec)
ament_export_libraries(planner_payload_component)



install(TARGETS
  planner_payload_component
  planner_payload_nodelet_exec
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(PROGRAMS
  scripts/build_payload_planner.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(${dependencies})
ament_package()
