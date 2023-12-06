#!/usr/bin/env python3
from typing import Any
import numpy as np
import math
import sys
import rospy

from robot.task.pick_and_place import object_manipulator_with_ur
from robot.mir import mir
from robot.ur import ur
from robot.utils.robot_vision import object_pose_estimation

if __name__ == "__main__":
    try:

        print("Starting new node: robot_vision_test")
        rospy.init_node('robot_vision_test')

        robot_vision = object_pose_estimation()

        rospy.spin()

        if True:
            rospy.loginfo("----------------------- Code finished -----------------------")
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")