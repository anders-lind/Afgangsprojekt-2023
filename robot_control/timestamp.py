#!/usr/bin/env python3
import rospy 
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud



class Local_Planner:
    def __init__(self):
        rospy.init_node('obstacle_detector', anonymous=True)
        
        self.obstacles = None
        self.humans = None
         
        self.obstacle_subscriber = rospy.Subscriber('/obstacles_without_people',
                                                PointCloud, self.update_obstacles)
        self.human_subscriber = rospy.Subscriber('/obstacles_only_people',
                                                PointCloud, self.update_humans)
        
        




if __name__ == "__main__":
    publish = rospy.Publisher("pose_test", Pose, queue_size=10)
    
    msg = Pose()
    msg.orientation.x = 1
    msg.orientation.y = 2
    msg.orientation.z = 3
    msg.orientation.w = 4
    
    msg.position.x = 1
    msg.position.y = 3
    msg.position.z = 4
    
    
    rate = rospy.Rate(10)
    
    while not rospy.is_shutdown():
        publish.publish(msg)
        rate.sleep()