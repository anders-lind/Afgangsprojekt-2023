#!/usr/bin/env python3
import numpy as np
import math
import sys

# ros
import rospy

# ros messages
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Pose
import actionlib

# Mathematical transformations
from scipy.spatial.transform import Rotation
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler

# utils
from robot.utils.method_logger import log_method

@log_method()
class mir:
    """[summary]
        This class is for making calls to MiR 
    """
    
    @log_method()
    def __init__(self):
        """[summary]
        Overwrites current goal, and sends new navigation goal to MiR
       
        start values:
        self.client: [description] communicator with move base goal of ros bridge
        self.pose.sub: [description] subsbriber to robot_pose, to get the robot pose
        self.current_pose_recieved: [description] boolean that makes sure, that robot pose has been written at least once before being read.
        self.client.wait_for_server(): [description] Waits for the server before starting mir control
        self.cancel_goal(): [description] Observed that the mir sometimes erroneously have goals, immediately cancels all to make sure it does not.
        """

        self.client = actionlib.SimpleActionClient('move_base',MoveBaseAction)

        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.__joint_cb)
        self.pose_sub = rospy.Subscriber('/robot_pose', Pose, self.__pose_cb)

        self.joint_pub = rospy.Publisher('/mir_jointstate', JointState, queue_size=1)
        self.pose_pub = rospy.Publisher('/mir_pose', Pose2D, queue_size=1)
        print("mir publisher initiated")

        self.current_pose_received = False

        self.client.wait_for_server()
        self.cancel_goal()

    @log_method()
    def send_goal(self, x, y, theta, frame=None):
        """[summary]
        Overwrites current goal, and sends new navigation goal to MiR
       
        Args:
        x (float32): [description] x coordinate in m
        y (float32): [description] y coordinate in m
        theta (float32): [description] theta angle in degrees
        frame (string): [description] goal frame, if not specified use map frame

        Returns:
            string: [description] status


            STATUS DESCRIPTIONS
            SENDING = 0
            ACTIVE = 1
            PREEMPTED = 2
            SUCCEEDED = 3
            ABORTED = 4
            REJECTED = 5
            PREEMPTING = 6
            RECALLING = 7
            RECALLED = 8
            LOST = 9
        """
 
        self.client.wait_for_server()

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        # Expected theta to be in degrees
        theta_radians = theta * (2.0 * math.pi) / 360.0

        yaw = theta_radians
        q = quaternion_from_euler(0, 0, yaw, 'ryxz')

        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]

        self.client.send_goal(goal)

        return self.client.get_state()

    @log_method()
    def get_goal_status(self):
        """[summary]
        Get status of sent goal
        Args:
        Returns:
            int8: [description] integer codes for success/ failure/ in progress
        """
        
        return self.client.get_state()

    @log_method()
    def cancel_goal(self):
        """[summary]
        Cancel sent navigation goal. Sleeps for 0.1 second to make sure that the robot has time to start the given goal before cancellation
        Args:
        Returns:
            int8: [description] integer codes for success/ failure/ in progress
        """

        rospy.sleep(0.1)

        self.client.cancel_goal()

        return self.client.get_state()

    @log_method()
    def move_relative(self, dist, angle): # angle is given en degrees
        """[summary]
        overwrites old goal and moves mir distance "dist" in x direction after turning by angle "angle"
        Args:
            dist (float32): [description] distance in m
            angle (float32): [description] turn angle in radians  
        Returns:
            bool: [description] status
        """

        dist = float(dist)
        angle = float(angle)

        self.client.wait_for_server()

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        # x, y, and theta are set as the robot pose some way, dont know the data format yet.
        if self.current_pose_received:
            local_data = self.current_robot_pose
        else:
            print("ERROR: no pose information received yet, cannot move relative")
            return self.client.get_state

        x = local_data[0][0] # x value
        y = local_data[0][1] # y value
        theta = euler_from_quaternion(local_data[1])[2] # get yaw angle

        # Converting theta into degrees (input angle is seen as degrees) 
        theta = theta * 360.0 / (2.0 * math.pi)
        theta = theta + angle

        theta_radians = theta * (2.0 * math.pi) / 360.0
        x_shift = float(dist)*math.cos(float(theta_radians))
        y_shift = float(dist)*math.sin(float(theta_radians))

        x = x+x_shift
        y = y+y_shift
        
        yaw = theta_radians
        q = quaternion_from_euler(0, 0, yaw, 'ryxz')

        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]

        self.client.send_goal(goal)

        return self.client.get_state()

    def __pose_cb(self, msg):
        """[summary]
        This method is activated everytime communication the rostopic /robot_pose gets new information. It then updates the robot pose variable.
        Args:
            msg (geometry_msgs.msg.Pose): [description] msg gotten from the topic robot_pose
        """        

        if self.current_pose_received == False:
            self.current_pose_received = True
        
        self.current_robot_pose = self.__geometryPose_to_dict(msg)

        x = msg.position.x
        y = msg.position.y

        rx = msg.orientation.x
        ry = msg.orientation.y
        rz = msg.orientation.z
        rw = msg.orientation.w

        euler = Rotation.from_quat([rx, ry, rz, rw]).as_euler("xyz")

        pose_for_publishing = Pose2D()
        pose_for_publishing.x = x
        pose_for_publishing.y = y
        pose_for_publishing.theta = euler[2]

        self.pose_pub.publish(pose_for_publishing)

    def __joint_cb(self, msg):
        """[summary]
        This method is used to listen to the ros topic /joint_states
        Args:
            msg (geometry_msgs.msg.Pose): [description] msg gotten from the topic robot_pose
        """        

        self.joint_pub.publish(msg)

    def __geometryPose_to_dict(self, pose):
        """[summary]
        This method is used to convert the input from the robot pose in a list of float values.
        Args:
            msg (geometry_msgs.msg.Pose): [description] msg gotten from the topic robot_pose
        Returns:
            list =  [[position] [pose]]: [description] The output is given as a list of two values the position and the pose.
                                          Both these values are lists of floats. 
        """     

        pose = str(pose)

        pose = pose.split()

        res1 = []
        res1.append(float(pose[2]))
        res1.append(float(pose[4]))

        res2 = []
        res2.append(float(pose[9]))
        res2.append(float(pose[11]))
        res2.append(float(pose[13]))
        res2.append(float(pose[15]))

        res = [res1, res2]

        return res