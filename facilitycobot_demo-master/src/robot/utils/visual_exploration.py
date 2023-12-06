#!/usr/bin/env python3
import os
import time
import sys
import numpy as np
import math
from scipy.spatial.transform import Rotation
from numpy.linalg import inv
import time

# ROS related 
import rospy
from vision_msgs.msg import ObjectHypothesisWithPose

# custom
from robot.utils.robot_vision import object_pose_estimation
from robot.utils.method_logger import log_method

@log_method()
class visual_exploration:
    """[summary]
    This class is used to explore with the ur arm with the purpose of finding objects
    """    

    @log_method()
    def __init__(self, robot_arm, robot_vision):
        """[summary]
        Visual exploration initalizer

        args:
            robot_arm [description]: Is the ur class passed into the class
            robot_vision [description]: Is the vision class passed into the class
        """

        self.__robot_vision = robot_vision
        self.__robot_arm = robot_arm

        self.__default_joint_config = np.deg2rad([[-180, -84, 104, -118, -94, -175]])

    @log_method()
    def __get_visual_exploration_motion_trajectory(self):
        #trajectory_path = os.path.join('/media/lakshadeep/PhD/Codes/ros_ws/src/facilitycobot_demo/config/','visual_exploration_motion.npz') 
        trajectory_path = os.path.join('/home/peterduc/catkin_ws/src/MR_lab_work/facilitycobot_demo/config/','visual_exploration_motion.npz') 


        joints_positions = []
        with np.load(trajectory_path) as files:
            joints_positions = files['joints_positions']
            print("\njoint positions: ", joints_positions, "\n")
        return joints_positions
    
    @log_method()
    def explore(self, object_id, threshold=0.9):
        """[summary]
        is used to explore with the ur arm with the purpose of finding objects

        args:
            threshold [description]: Is the predictive confidence threshold before the pose is returned

        returns:
            pose [description]: Is the pose of the object found
        """

        # movement into the exploration zone
        start_velocity = 0.1
        start_acceleration = 0.1

        velocity_in_exploration = 0.8    
        acceleration_in_exploration = 0.3 

        time_delay = 20

        # middle position of ur exploration zone
        j_pos = [1.5815809965133667, -2.579487463037008, 2.6276100317584437, -2.4617630443968714, -1.6271784941302698, 3.2549710273742676]

        self.__pre_grasp_joint_motions = self.__get_path_planning(j_pos=j_pos, x_dist=math.pi/3, destination_amounts=7)

        size = self.__pre_grasp_joint_motions.__len__()

        self.__pre_grasp_joint_motions = np.array(self.__pre_grasp_joint_motions)

        velocity = start_velocity
        acceleration = start_acceleration

        for i in range(0, size):

            print("\nmoving to joint pose: ", self.__pre_grasp_joint_motions[i], "\n")
            self.__robot_arm.move_in_joint_space([self.__pre_grasp_joint_motions[i]], 
                                                    velocity=velocity,
                                                    acceleration=acceleration,
                                                    sync=False)

            velocity = velocity_in_exploration
            acceleration = acceleration_in_exploration

            pose = self.__get_predictive_pose_estimation(object_id=object_id, time_delay=time_delay, threshold=0.9)

            if pose != None:
                print("\nobject pose found, press 1 to abort movement\n")
                temp = input()

                if temp == "1":
                    return None
                else:
                    # update it, before returning
                    updated_pose = self.__robot_vision.get_object_pose()

                    if updated_pose == None:
                        return pose
                    else:
                        return updated_pose



            # If object is found, but pose was not reached, do small adjustment movements
            detectron_detection = self.__robot_vision.get_detectron_object_detection(object_id=object_id, score_thresholds=[0.95,0.95,0.95,0.90,0.95])

            if detectron_detection != None:
                print("\nobject has been detected, adjusting robot position to get try to get better pose estimation\n")

                curr_pos = self.__pre_grasp_joint_motions[i]
                curr_pos[3] += 0.2028

                path_positions = self.__get_path_planning(j_pos=curr_pos, x_dist=math.pi/24, destination_amounts=3)

                path_size = path_positions.__len__()

                path_positions = np.array(path_positions)


                for j in range(1, path_size): # skip the first move in the path planning, as it is the same as the current position
                    print("\nadjusting moving by going to: ", path_positions[j], "\n")
                    self.__robot_arm.move_in_joint_space([path_positions[j]], 
                                                            velocity=velocity,
                                                            acceleration=acceleration,
                                                            sync=False)
            

                    pose = self.__get_predictive_pose_estimation(object_id=object_id, time_delay=time_delay+5, threshold=0.9)

                    if pose != None:
                        print("\nobject pose found, press 1 to abort movement\n")
                        temp = input()
                            
                        # update it, before returning
                        updated_pose = self.__robot_vision.get_object_pose()

                        if updated_pose == None:
                            return pose
                        else:

                            if updated_pose[1] == object_id:
                                return updated_pose
                            else:
                                return pose


        return None

    @log_method()
    def __focus_on_object(self, pose, focus_time = 10):
        """[summary]
        is used to wait for new pose updates which yield more precise values

        args:
            focus time [description]: Is the time in seconds that the system waits
            pose [description]: If the focus time does not yield an updated pose, then the start poseis used.

        returns:
            [pose] [description]: is the updated pose after the focus time
        """
       
        rospy.sleep(focus_time)

        pose_package = self.__robot_vision.get_object_pose()

        if pose_package == None:
            return pose
        else:
            if pose_package[1] == pose[1]:
                return pose_package
            else:
                return pose
    
    @log_method()
    def __get_path_planning(self, j_pos, x_dist, destination_amounts):
        """[summary]
        is used to explore with the ur arm with the purpose of finding objects

        args:
            j_pos [description]: Is the base ur arm position that will be moved around at
            x_dist [description]: Is the distance from the base position which the robot will see
            destination_amounts [description]: Is the amount of stops the path must have. Must be an odd positive number as the 
                                                ur base position should be directly in the middle

        returns:
            path [description]: Is the path created based on the arm movements
        """

        if destination_amounts%2 == 0: # if the destination amounts is even, add 1 to make it odd
            destination_amounts += 1

        # variables used to look at both sides of the give j_pos
        change_sign_delay = 0 
        sign = 1

        planned_path =  []

        for i in np.linspace(0, x_dist, int(destination_amounts - (destination_amounts - 1) / 2)):
            
            shift = i*sign
            planned_path.append([j_pos[0] + shift, j_pos[1], j_pos[2], j_pos[3], j_pos[4], j_pos[5]])
            
            if change_sign_delay == 1:
                sign = -sign
                shift = i*sign
                change_sign_delay = 0

                planned_path.append([j_pos[0] + shift, j_pos[1], j_pos[2], j_pos[3], j_pos[4], j_pos[5]])
                change_sign_delay += 1

            else:
                change_sign_delay += 1
        

        return planned_path
    
    @log_method()
    def __get_predictive_pose_estimation(self, object_id, time_delay, threshold):
        """[summary]
        is used to detect if the pose estimator has found an object

        args:
            time_delay [description]: Is the time delay at which the method stops trying to find the pose estimation
            threshold [description]: Is the predictive confidence threshold, which must be reached before returning

        returns:
            pose/None [description]: returns a pose if the threshold criteria is met, or else returns None 
        """

        self.__robot_vision.delete_object_pose_estimation()
        self.__robot_vision.delete_detectron_object_detection()

        rospy.sleep(time_delay)
        
        predictive_confidence = self.__robot_vision.get_predictive_confidence()
        pose_and_id = self.__robot_vision.get_object_pose()

        if (predictive_confidence > threshold and pose_and_id != None):

            if pose_and_id[1] == object_id:
                return self.__focus_on_object(pose_and_id)
            else:
                return None
    
        return None

