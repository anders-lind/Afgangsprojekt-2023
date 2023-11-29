#!/usr/bin/env python3

from mir import mir

import rospy
from sensor_msgs.msg import LaserScan, PointCloud
from geometry_msgs.msg import Point32
from std_msgs.msg import Header

import math


class Robot_control:
    def __init__(self):
        # Subscribers
        self.hum_sub = rospy.Subscriber(
            name="obstacles_only_people", 
            callback=self.filter_humans, 
            data_class=PointCloud,
            queue_size=10
        )
        self.obs_sub = rospy.Subscriber(
            name="obstacles_without_people", 
            callback=self.filter_obstacles, 
            data_class=PointCloud,
            queue_size=1
        )
        # Publishers
        self.obs_pup = rospy.Publisher(
            name="obstacles_without_people_filtered",
            data_class=PointCloud,
            queue_size=1
        )


    def filter_humans(self, data : PointCloud):
        pass
        # print("update_humans:", data.points.__len__())

    def filter_obstacles(self, data : PointCloud):
        print("Doing stuff:")
        
        print("len:", data.points.__len__())

        # Threshold to how close points can be
        point_th = 1.0

        # Init new data
        new_data = PointCloud()
        h = Header()
        h.stamp = rospy.Time(0)
        h.frame_id = "map"
        new_data.header = h

        # Only add points far away from other points
        i = 0
        while i < data.points.__len__():
            point : Point32 = data.points[i]
            
            # Check if point is close to other points
            j = i+1 # Check only points after current point
            while j < data.points.__len__():
                if math.sqrt((point.x - data.points[j].x)**2 + (point.y - data.points[j].y)**2 + (point.z - data.points[j].z)**2 ) < point_th:
                    data.points.pop(j)
                    print("pop")

                j += 1
            
            new_data.points.append(point)
            i += 1
        


        # Publish new filtered data
        self.obs_pup.publish(new_data)

        print("new len:",new_data.points.__len__())




if __name__ == "__main__":
    # For every timestep
    #   Find obstacles and humans
    #       Filter points which are too close 
    #   Use path planning to calculate next pose
    #   Send next pose to MiR

    rc = Robot_control()
    rospy.init_node("Object_listener")
    rospy.spin()





