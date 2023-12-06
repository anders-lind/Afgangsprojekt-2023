#!/usr/bin/env python3
from typing import Any
import numpy as np
import math
import sys
import rospy

from robot.task.pick_and_place import object_manipulator_with_ur
from robot.mir import mir
from robot.ur import ur


if __name__ == "__main__":
    try:

        print("Starting new node: ur")
        rospy.init_node('ur')

        task_obj = object_manipulator_with_ur()

        # Full pick and place robot task sequence.
        # task_obj.move_to_home()
        object_pose_and_id = task_obj.search_for_object()   

        if not(object_pose_and_id == None):
            task_obj.move_to_object_and_pick(object_pose_and_id)
            task_obj.move_to_start_pose()
            task_obj.move_to_bin_pose()
            task_obj.put_down_object()
            task_obj.move_to_home()

        if True:
            rospy.loginfo("----------------------- Code finished -----------------------")
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")