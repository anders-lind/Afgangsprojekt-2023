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
            callback=self.filter_human_points, 
            data_class=PointCloud,
            queue_size=1
        )
        self.obs_sub = rospy.Subscriber(
            name="obstacles_without_people", 
            callback=self.filter_obstacle_points, 
            data_class=PointCloud,
            queue_size=1
        )
        # Publishers
        self.obs_pup = rospy.Publisher(
            name="obstacles_without_people_filtered",
            data_class=PointCloud,
            queue_size=1
        )
        self.hum_pup = rospy.Publisher(
            name="obstacles_only_people_filtered",
            data_class=PointCloud,
            queue_size=1
        )



    def filter_points(self, data : PointCloud):        
        # Threshold to how close points can be
        point_th = 0.1

        # Init new data
        new_data = PointCloud()
        h = Header()
        h.stamp = rospy.Time(0)
        h.frame_id = "map"
        new_data.header = h

        # Only add points far away from other points
        i = 0
        while i < data.points.__len__() - 1:
            point : Point32 = data.points[i]
            
            # Check if point is close to other points
            j = i+1 # Check only points after current point
            while j < data.points.__len__():
                if math.sqrt((point.x - data.points[j].x)**2 + (point.y - data.points[j].y)**2) < point_th:
                    data.points.pop(j)
                else:
                    j += 1
            
            new_data.points.append(point)
            i += 1
        
        return new_data
    


    def filter_human_points(self, data : PointCloud):
        print("filter_human_points")
        filtered_data = self.filter_points(data)
        self.hum_pup.publish(filtered_data)




    def filter_obstacle_points(self, data : PointCloud):
        print("filter_obstacle_points")
        filtered_data = self.filter_points(data)
        self.obs_pup.publish(filtered_data)




if __name__ == "__main__":
    # For every timestep
    #   Find obstacles and humans
    #       Filter points which are too close 
    #   Use path planning to calculate next pose
    #   Send next pose to MiR

    print("Starting robot control script")
    rc = Robot_control()
    rospy.init_node("Object_listener")
    rospy.spin()





