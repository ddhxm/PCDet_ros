cmake_minimum_required(VERSION 3.0.2)
project(lidar_objects_msgs)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  sensor_msgs
  std_msgs
  message_generation
)


add_message_files(
  FILES
    Bbox3d.msg
    Other_info.msg
    Object.msg	
    ObjectArray.msg
)

generate_messages(
  DEPENDENCIES
    sensor_msgs
    std_msgs
    nav_msgs
)

catkin_package(
  CATKIN_DEPENDS
    nav_msgs
    sensor_msgs
    std_msgs
    message_runtime
)
