cmake_minimum_required(VERSION 3.0.2)
project(person_manager)

find_package(catkin REQUIRED COMPONENTS
    rospy
    std_msgs
    message_generation
)

add_message_files(
  FILES
  Landmark.msg
  Pose.msg
  PoseArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

# add_service_files(
#   FILES
# )

# add_action_files(
#   FILES
# )



catkin_package(
    CATKIN_DEPENDS message_runtime
)

include_directories(

)