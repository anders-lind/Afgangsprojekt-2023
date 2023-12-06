#!/usr/bin/env python3
import os
import time
import sys

# ros
import rospy

# rtde for ur control
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEInterface
from rtde_io import RTDEIOInterface as RTDEIO

# robot gripper control
from robot.gripper.robotiq_gripper_control import RobotiqGripper

# ros messages
from sensor_msgs.msg import JointState

# Multi threading for ur joint messages
import threading 

# for math calculations
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation

# utils
from robot.utils.method_logger import log_method

@log_method()
class ur(RTDEIO, RTDEInterface, RTDEControl):
    """[summary]
    This class is used to control the UR robot through rtde
    """

    @log_method()
    def __init__(self, ip):
        """[summary]
        Initalizer for the ur class
       
        args:
            self.ip [description] is the ip of the Ur robot given as a string
        """

        rtde_frequency = 70.0

        RTDEIO.__init__(self, ip,rtde_frequency)
        RTDEInterface.__init__(self, ip,rtde_frequency)
        RTDEControl.__init__(self, ip, rtde_frequency)
    
        print("Activating gripper...")
        self.gripper = RobotiqGripper(self)

        self.gripper.activate()
        self.gripper.set_force(20)  # from 0 to 100 %
        self.gripper.set_speed(50)  # from 0 to 100 %
        
        self.gripper.open()

        self.joint_pub = rospy.Publisher('/ur_jointstate', JointState, queue_size=1)

        t1 = threading.Thread(target=self.publish_joint_positions)
        t1.daemon = True # Exits the tread when the main program exits
        t1.start()

    def publish_joint_positions(self):
        """[summary]
        Publishes the joint positions of the UR robot
        """

        joints = JointState()

        while(True):
            joints.position = self.getActualQ()
            self.joint_pub.publish(joints)
            rospy.sleep(0.5)

    @log_method()
    def stop_joint_motion(self, val=0.5):
        """[summary]
        Stops the UR robot
        """

        self.stopJ()

    @log_method()
    def move_in_joint_space(self, joints_positions, velocity=0.2, acceleration=0.10, blend=0.01, sync=False):
        """[summary]
        Moves UR robot linearly in joint space
       
        args:
            joint_positions [description] is the joint positions of the robot given as a list
            velocity [description] is the velocity of the robot given as a float
            acceleration [description] is the acceleration of the robot given as a float
            blend [description] is the blend of the robot given as a float
            sync [description] is a boolean that determines if the robot should wait for the motion (true)
                                or continue with the code (false)            
        """

        vab = np.array([velocity, acceleration, blend])
        vab_array = np.tile(vab, (len(joints_positions), 1))

        desired_trajectory = np.hstack((joints_positions, vab_array))

        self.moveJ(desired_trajectory, sync)

    @log_method()
    def get_joint_positions(self):
        """[summary]
        Returns the joint positions of the robot

        return:
            [list]: [description] returns the joint positions of the robot
        """

        joint_positions = self.getActualQ()
        return joint_positions

    @log_method()
    def set_joint_positions(self, j1, j2, j3, j4, j5, j6, sync=False):
        """[summary]
        Moves UR robot linearly in joint space (same as self.move_in_joint_space() given what i understand)
       
        args:
            self.j1-j6 [description] is the joint positions of the robot given as float values
        """

        velocity = 0.1 # 0.20    
        acceleration = 0.10 
        blend = 0.01

        joints_positions = np.array([j1, j2, j3, j4, j5, j6])
        vab = np.array([velocity, acceleration, blend])

        desired_trajectory = np.hstack((joints_positions, vab))

        self.moveJ([desired_trajectory], sync)

    @log_method()
    def move_in_cartesian_space(self, pose, velocity=0.03, acceleration=0.01, blend=0.1):
        """[summary]
        Moves UR robot linearly in joint space (same as self.move_in_joint_space() given what i understand)
       
        args:
            pose [description]: is the desired end pose for the arm given as (x,y,z, rx,ry,rz,rw)
            velocity [description]: is the velocity in which the robot moves
            acceleration [description]: is the acceleration in which the robot moves
            blend [description]: Describes the blend rafius of the robot movement given in meters, (currently not used.)
        """

        (x,y,z,qx,qy,qz,qw) = pose

        velocity = 0.03    
        acceleration = 0.01 
        blend = 0.1

        [rx, ry, rz] = Rotation.from_quat([qx,qy,qz,qw]).as_rotvec()

        trajectory_pose = [x,y,z,rx,ry,rz]
        vab = np.array([velocity, acceleration, blend])

        desired_trajectory = np.hstack((trajectory_pose, vab))

        self.moveL(trajectory_pose, 0.05, 0.05)

    @log_method()
    def get_tcp_pose(self):
        """[summary]
        Returns the pose of the robot in the base frame

        Returns:
            TCP pose: [description] returns the pose of the robot in the base frame given as (x,y,z,qx,qy,qz,qw)
        """

        TCP_pos = self.getActualTCPPose()

        rot = TCP_pos[3:6]
        pos = TCP_pos[0:3]
        
        unit_q = Rotation.from_rotvec(rot).as_quat()

        return [pos[0], pos[1], pos[2], unit_q[0], unit_q[1], unit_q[2], unit_q[3]]

    @log_method()
    def __get_matrix_from_pose(self, pose):
        """[summary]
        Transforms a pose to a transformation matrix
        args:
            pose: [description] pose of the object which myst be transformed to a transformation matrix

        Returns:
            tranformation matrix: [description] returns a matrix presented as nested lists.
        """

        T = np.identity(4)
        T[:3,:3] = Rotation.from_quat(pose[3:7]).as_matrix()
        T[0:3,3] = pose[0:3]
        return T

    @log_method()
    def get_transformation_robot_to_target(self, target_pose):
        """[summary]
        gets the transformation from the robot to the target

        args:
            target_pose: [description] pose of the target in the camera frame

        Returns:
            tranformation matrix: [description] matrix representing transformation from robot to target

        """

        # get the robot pose
        robot_tcp_pose = self.get_tcp_pose()
        self.robot_tcp_pose = robot_tcp_pose

        target_pose[0] = target_pose[0] # /1000.0
        target_pose[1] = target_pose[1] # /1000.0
        target_pose[2] = target_pose[2] # /1000.0

        T_c_t = self.__get_matrix_from_pose(target_pose)
        T_r_tcp = self.__get_matrix_from_pose(robot_tcp_pose)

        self.T_r_tcp = T_r_tcp

        self.T_r_c = np.matmul(T_r_tcp, self.T_tcp_c)

        self.T_r_t = np.matmul(self.T_r_c , T_c_t)

        return self.T_r_t


