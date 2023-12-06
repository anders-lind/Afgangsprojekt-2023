#!/usr/bin/env python3
import numpy as np
import math
import sys
import rospy

from robot.mir import mir
from robot.ur import ur


if __name__ == "__main__":
    try:
        print("Starting new node: mir")
        rospy.init_node('mir')


        mir_obj = mir()
        ur_obj = ur("192.168.11.40")

        rospy.spin()
        


        if True:
            rospy.loginfo("Goal execution done!")
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")