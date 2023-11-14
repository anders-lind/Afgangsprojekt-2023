#!/usr/bin/env python3
import rospy
from turtlesim.msg import Pose
from geometry_msgs.msg import Pose, PoseArray, Point32
from math import sqrt
from sensor_msgs.msg import LaserScan, PointCloud
from std_msgs.msg import Header
import tf2_ros
import tf2_geometry_msgs
import math
from fiducial_msgs.msg import FiducialTransformArray
from scipy.spatial.transform import Rotation as R

#http://wiki.ros.org/aruco_detect


class Get_Obstacle_Coordinates:
    def __init__(self, people_distance):
        rospy.init_node('obstacle_detector', anonymous=True)
        
        self.obstacle_publisher = rospy.Publisher('obstacles',
                                                  PointCloud, queue_size=10)
        
        self.obstacle_only_people_publisher = rospy.Publisher('obstacles_only_people',
                                                  PointCloud, queue_size=10)
        
        self.obstacle_without_people_publisher = rospy.Publisher('obstacles_without_people',
                                                  PointCloud, queue_size=10)

        self.people_publisher = rospy.Publisher('people',PoseArray, queue_size=10)

        self.laser_scan_subscriber = rospy.Subscriber('scan',
                                                LaserScan, self.update_obstacles)
        
        self.aruco_subscriber = rospy.Subscriber('fiducial_transforms', FiducialTransformArray, self.update_people_locations)
               
        
        self.people_distance = people_distance
        
        self.rate = rospy.Rate(10)
        
        self.people = {}
                
        self.tf_buffer = tf2_ros.Buffer()
        self.listener =tf2_ros.TransformListener(self.tf_buffer)
        

    def update_people_locations(self, data : FiducialTransformArray):
        #https://docs.ros.org/en/melodic/api/geometry_msgs/html/msg/PoseArray.html
        #https://docs.ros.org/en/kinetic/api/fiducial_msgs/html/msg/FiducialTransformArray.html
        #https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/PointCloud.html
        #https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/LaserScan.html

        for i in range(len(data.transforms)):
            
            fidcuial_id = data.transforms[i].fiducial_id
                
            pose =tf2_geometry_msgs.PoseStamped()
            
            pose.header = data.header
            pose.pose.position = data.transforms[i].transform.translation
            pose.pose.orientation = data.transforms[i].transform.rotation
            
            other_pose = self.tf_buffer.transform(pose, 'map', rospy.Duration(1))

            rot = other_pose.pose.orientation
            trans = other_pose.pose.position
            
            r = R.from_quat([rot.x, rot.y, rot.z, rot.w])
            rot_90 = R.from_euler('y', -90, degrees=True)
            quat = (r*rot_90).as_quat()

            new_pose = Pose()
            new_pose.position = trans
            
            new_pose.orientation.x = quat[0]
            new_pose.orientation.x = quat[0]
            new_pose.orientation.y = quat[1]
            new_pose.orientation.z = quat[2]
            new_pose.orientation.w = quat[3]

            self.people[fidcuial_id] = new_pose
        
        
        h = Header()
        h.stamp = rospy.Time(0)
        h.frame_id = "map"
        
        pose_array = PoseArray()
        poses = []
        pose_array.header = h
        
        for key, val in self.people.items():
            poses.append(val)
        pose_array.poses = poses
        self.people_publisher.publish(pose_array)
    
    
    def update_obstacles(self, data : LaserScan):
        
        obstacles = PointCloud()
        obstacles_without_people = PointCloud()
        people = PointCloud()
        
        for i in range(len(data.ranges)):
            point = tf2_geometry_msgs.PointStamped()
            point.header = data.header
            point.header.stamp =rospy.Time(0)
            
            angle = data.angle_min + i*data.angle_increment
            
            #wrap ange between 0 and 2 pi
            angle = angle % (2*math.pi)
            
            #polar to cartesian coordinates
            point.point.x = math.cos(angle)*data.ranges[i]
            point.point.y = math.sin(angle)*data.ranges[i]
            point.point.z = 0
            
            try:
                point = self.tf_buffer.transform(point, 'map', rospy.Duration(1))
                
                new_point = Point32()
                new_point.x = point.point.x
                new_point.y = point.point.y
                new_point.z = point.point.z
                
                h = Header()
                h.stamp = rospy.Time(0)
                h.frame_id = "map"
                
                obstacles.header = h
                obstacles_without_people.header = h
                people.header = h
                               
                obstacles.points.append(new_point)
                
                people_dict = self.people.copy()
                
                add = True
                for key, val in people_dict.items():
                    dist = sqrt((val.position.x - new_point.x)**2 + (val.position.y - new_point.y)**2 + (val.position.z - new_point.z)**2)
                    if dist < self.people_distance:
                        add = False
                
                if add == True:
                    obstacles_without_people.points.append(new_point)
                else:
                    people.points.append(new_point)
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.Rate(10).sleep()
    
        self.obstacle_publisher.publish(obstacles)
        self.obstacle_without_people_publisher.publish(obstacles_without_people)
        self.obstacle_only_people_publisher.publish(people)
        

        
    def main(self):            
        while not rospy.is_shutdown():
            rospy.Rate(10).sleep()
            


if __name__=='__main__':
    try:
        x = Get_Obstacle_Coordinates(people_distance=0.3)
        x.main()
    except rospy.ROSInterruptException:
        print("Error")
    