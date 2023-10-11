#!/usr/bin/env python3

import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

from person_manager.msg import PoseArray
from person_manager.msg import Pose
from person_manager.msg import Landmark

from src.people_detection.detector import Detector

import rospy
import cv2
import numpy as np
import os


file_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.getcwd()


def publish_poses(result: vision.PoseLandmarkerResult, output_image: np.ndarray, timestamp_ms: int = None):
        # number_of_poses: int = len(result.pose_landmarks)
        number_of_poses: int = len(result.pose_world_landmarks)
        number_of_landmarks: int = 33

        # Create WorldLandmark message
        
        pose_array = PoseArray()
        for p in range(number_of_poses):
            pose = Pose()
            for l in range(number_of_landmarks):
                landmark = Landmark(
                    x = result.pose_world_landmarks[p][l].x,
                    y = result.pose_world_landmarks[p][l].y,
                    z = result.pose_world_landmarks[p][l].z,
                    visibility = result.pose_world_landmarks[p][l].visibility,
                    presence = result.pose_world_landmarks[p][l].presence
                )
                pose.landmarks.append(landmark)
            pose_array.poses.append(pose)

        pose_publisher.publish(pose_array)


def process_poses(pose_array: PoseArray):
    # print("Data: ", pose_array)
    print("type:",type(pose_array.poses[0].landmarks[0].x))
              


if __name__ == "__main__":
    rospy.init_node("person_detector_node")

    # Pose publisher
    pose_publisher = rospy.Publisher(
        name="persons/poses",
        queue_size=10,
        data_class=PoseArray
        )
    
    # Pose subscriber test
    pose_subsriber = rospy.Subscriber(
         name="persons/poses",
         data_class=PoseArray,
         callback=process_poses
    )




    ### IMAGE ###
    # detector = Detector(
    #   picture_file_name="scripts/src/people_detection/videos_and_images/a i-pose 1.jpg"
    #   )
    
    ### WEB CAM ###
    # detector = Detector(
    #   VideoCapture=cv2.VideoCapture(0),
    #   callback_function=publish_poses
    #   )
    # detector.webcam_begin_detect()
    # detector.webcam_stop_detect()
    
    ## ROS TOPIC ##
    detector = Detector(
        ROS_image_topic="/camera_floor/driver/color/image_raw"
        )



    rospy.spin()
    
