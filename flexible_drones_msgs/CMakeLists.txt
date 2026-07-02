# Copyright 2026 Christopher Newport University
# Capable Humanitarian Robotics and Intelligent Systems Lab (CHRISLAB)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)
project(flexible_drones_msgs)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrajectoryPolynomialPiece.msg"
  "msg/TrajectoryPoint.msg"
  "msg/Trajectory.msg"
  "msg/SampledTrajectory.msg"
  "msg/DroneStatus.msg"
  "msg/OnboardStatus.msg"
  "msg/FullState.msg"
  "msg/TrajectoryObstacle.msg"
  "msg/TrajectoryBounds.msg"
  "msg/TrajectoryBoundary.msg"
  "srv/Arm.srv"
  "srv/GetStatus.srv"
  "srv/EmergencyStop.srv"
  "srv/SetLEDColor.srv"
  "srv/SpawnDrone.srv"
  "srv/SwarmArm.srv"
  "srv/Reboot.srv"
  "srv/SetTrajectoryObstacles.srv"
  "srv/SetTrajectoryBounds.srv"
  "action/Takeoff.action"
  "action/Land.action"
  "action/GoTo.action"
  "action/EnableController.action"
  "action/SwarmTakeoff.action"
  "action/SwarmLand.action"
  "action/SwarmGoTo.action"
  "action/UploadTrajectory.action"
  "action/ExecuteTrajectory.action"
  "action/GetTrajectory.action"
  "action/GetSampledTrajectory.action"
  "srv/RealignLocalPosition.srv"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs action_msgs
  ADD_LINTER_TESTS
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_cmake_copyright REQUIRED)
  ament_copyright()
endif()

ament_package()
