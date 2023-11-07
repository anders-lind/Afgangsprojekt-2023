#!/usr/bin/env python3

import rospy

from geometry_msgs.msg import PoseArray, Pose, Point
from nav_msgs.msg import GridCells

import numpy as np


def publish_circle(topic_name: str, cell_size: float, radius: float, center_x: float, center_y: float, height: float):
    # Create GridCells
    grid_cells = GridCells()
    grid_cells.header.stamp = rospy.Time.now()
    grid_cells.header.frame_id = "map"
    grid_cells.cell_height = cell_size
    grid_cells.cell_width = cell_size

    # Algorith to get all pixels in circle
    #
    # (x-h)^2 + (y-k)^2 == r^2 
    # h,k er centrum
    #
    r = radius
    h = center_x
    k = center_y
    for cx in np.arange(h-r, h+r, cell_size):
        for cy in np.arange(k-r, k+r, cell_size):
            if pow(cx-h,2) + pow(cy-k,2) < pow(r,2):
                point = Point(x=cx, y=cy, z=height)
                grid_cells.cells.append(point)
        

    # Publish GridCells
    pub = rospy.Publisher(topic_name, GridCells, queue_size=10)
    pub.publish(grid_cells)




def publish_spaces(pose_array: PoseArray):
    for i in range(0, len(pose_array.poses)):
        pose = pose_array.poses[i] 
        pose:Pose

        publish_circle("/persons/intimate_space", 0.02  , 0.36  , pose.position.x, pose.position.y, 0.3)
        publish_circle("/persons/personal_space", 0.1   , 1.2   , pose.position.x, pose.position.y, 0.2)
        publish_circle("/persons/social_space"  , 0.1   , 3.0   , pose.position.x, pose.position.y, 0.1)



if __name__ == "__main__":
    rospy.init_node("circle_drawer")
    rospy.loginfo("Started circle_drawer")

    sub = rospy.Subscriber (
        "/persons/poses",
        PoseArray,
        publish_spaces
    )

    rospy.spin()