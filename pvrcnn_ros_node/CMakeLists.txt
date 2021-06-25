cmake_minimum_required(VERSION 3.0.2)
project(pvrcnn_ros_node)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  rospy
  roscpp
  sensor_msgs
  std_msgs
  geometry_msgs
  lidar_objects_msgs
  dynamic_reconfigure
)

generate_dynamic_reconfigure_options(
  reconfigure/Thresholds.cfg
)

#add_message_files(
#  FILES
#    Object.msg
#    ObjectArray.msg
#)

#generate_messages(
#  DEPENDENCIES
#  geometry_msgs
#    sensor_msgs
#    std_msgs
#)

catkin_package(
  CATKIN_DEPENDS
    nav_msgs
    sensor_msgs
    std_msgs
    lidar_objects_msgs
)
