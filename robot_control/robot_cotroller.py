#!/usr/bin/env python3
import rospy 
from geometry_msgs.msg import PoseArray, Pose, TwistStamped
from sensor_msgs.msg import PointCloud

from math import cos, sin, atan2

from local_planners.dynamic_window_approach import DWA
from local_planners.safe_artificial_potential_fields import SAPF


class Local_Planner:
    def __init__(self, rate = 10):
        rospy.init_node('obstacle_detector', anonymous=True)
        
        self.obstacles = []
        self.humans = []
         
        self.obstacle_subscriber = rospy.Subscriber('/obstacles_without_people',
                                                PointCloud, self.update_obstacles)
        
        self.human_subscriber = rospy.Subscriber('/people',PoseArray, self.update_humans)
        
        
        self.x = 0
        self.y = 0
        self.theta = 0
        
        self.v = 0
        self.w = 0
        
        self.velocity_subscriber = rospy.Subscriber('/cmd_vel',TwistStamped, self.update_velocities)
        self.pose_subscriber = rospy.Subscriber('/robot_pose',Pose, self.update_pose)
        
        self.rate = rospy.Rate(rate)
        
        self.dwa = DWA()
        self.sapf = SAPF()
                  
        
    
    def update_obstacles(self, data:PointCloud):
        self.obstacles.clear()
        for i in range(len(data.points)):
            point = [data.points[i].x, data.points[i].y]
            self.obstacles.append(point)
            
            
    def update_humans(self, data:PoseArray):
        self.humans.clear()
        for i in range(len(data.poses)):
            orientation = data.poses[i].orientation
            position = data.poses[i].position
            
            point = [position.x, position.y]
            
            t3 = +2.0 * (orientation.w * orientation.z)
            t4 = +1.0 - 2.0 * (orientation.z * orientation.z)
            angle = atan2(t3, t4)
            
            orient = [cos(angle), sin(angle)]
            person = [point, orient] 
            
            self.humans.append(person)
            
            
    def update_velocities(self, data:TwistStamped):
        self.v = data.twist.linear.x
        self.w = data.twist.angular.z
    
    
    
    def update_pose(self, data:Pose):
        self.x = data.position.x
        self.y = data.position.y
        
        t3 = +2.0 * (data.orientation.w * data.orientation.z)
        t4 = +1.0 - 2.0 * (data.orientation.z * data.orientation.z)
        self.theta = atan2(t3, t4)
            
            
    def move_robot(self, v, w, x, y):
        pass



    def planner(self, use_sapf = True):
        
        if use_sapf == True:
            pass

    
    def main(self):
        while not rospy.is_shutdown():
            # DO GOOD WORK HERE
            
            self.rate.sleep()




if __name__ == "__main__":
    try:   
        LOC = Local_Planner()
        LOC.main()
    except rospy.ROSInterruptException:
        print("Error")
        