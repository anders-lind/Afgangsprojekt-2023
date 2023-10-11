import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

from sensor_msgs.msg import Image

import rospy
import numpy as np
import cv2
import time
import threading
import os


file_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.getcwd()


class Detector():
    def __init__(self, ROS_image_topic: str = None, VideoCapture: cv2.VideoCapture = None, picture_file_name: str = None, callback_function: callable = None):        
        # Member variables
        self.image = None
        self.detector = None
        self.detection_result = None
        self.ROS_image_topic_subscriber = None
        self.webcam_running = False
        self.video_capture = None
        self.ROS_image_topic = None
        self.callback_function = None
        # Threads
        self.webcam_detect_thread: threading.Thread = threading.Thread(target=self.__webcam_detect)
        self.show_landmarks_thread: threading.Thread = threading.Thread(target=self.__show_landmarks_image_async_internal)
        self.ROS_topic_detect_thread: threading.Thread = threading.Thread(target=self.__ROS_topic_detect)
        max_poses = 10


        # Image setup
        if picture_file_name != None:
            # Create options
            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=file_path+'/setup_files/pose_landmarker.task'),
                num_poses = max_poses,
                output_segmentation_masks=True,
            )
            # Create detector from options
            self.detector = vision.PoseLandmarker.create_from_options(options)

            # Load the input image.
            self.image = mp.Image.create_from_file(picture_file_name)

            # Detect pose landmarks from the input image.
            self.detection_result = self.detector.detect(self.image)

            print("Image detector created")


        # Stream setup
        if VideoCapture != None:
            self.video_capture = VideoCapture
            self.callback_function = callback_function

            # Create options
            options = vision.PoseLandmarkerOptions(
                base_options = python.BaseOptions(model_asset_path='scripts/src/people_detection/setup_files/pose_landmarker.task'),
                num_poses = max_poses,
                running_mode = vision.RunningMode.LIVE_STREAM,
                result_callback=self.stream_detection_callback
            )
            # Create detector from options
            self.detector = vision.PoseLandmarker.create_from_options(options)

            print("Stream detector created")
            

        # ROS topic setup
        if ROS_image_topic != None:
            self.ROS_image_topic = ROS_image_topic

            # Create options
            options = vision.PoseLandmarkerOptions(
                base_options = python.BaseOptions(model_asset_path=file_path+'/setup_files/pose_landmarker.task'),
                num_poses = max_poses,
                # running_mode = vision.RunningMode.VIDEO
                running_mode = vision.RunningMode.IMAGE
            )
            # Create detector from options
            self.detector = vision.PoseLandmarker.create_from_options(options)
            
            # Create subsriber
            self.ROS_image_topic_subscriber = rospy.Subscriber(
                name=self.ROS_image_topic,
                data_class=Image,
                callback=self.__ROS_new_image_callback
            )

            print("ROS topic image detector created")



    def __ROS_new_image_callback(self, image: Image):
        # Convert ROS Image.data to NumPy array
        data_array = np.frombuffer(image.data, np.uint8)
        data_array = data_array.reshape((image.height, image.width, 3))

        # Create mp image
        self.image = mp.Image(image_format=mp.ImageFormat.SRGB, data=data_array)

        # Run detection
        frame_timestamp_ms = round(time.time()*1000)
        self.detector.detect(data_array)
        


    def stream_detection_callback(self, result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.detection_result = result
        # Call user-specified callback
        if self.callback_function != None:
            self.callback_function(result, output_image.numpy_view(), timestamp_ms)

    
    def webcam_begin_detect(self):
        self.webcam_running = True
        self.webcam_detect_thread.start()
        print("Webcam detection started")


    def webcam_stop_detect(self):
        self.webcam_running = False
        self.webcam_detect_thread.join()
        print("Webcam detection stopped")


    def __webcam_detect(self):
        while self.webcam_running:
            cam = self.video_capture.read()[1]
            cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
            self.image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cam)
            frame_timestamp_ms = round(time.time()*1000)
            self.detector.detect_async(self.image, frame_timestamp_ms)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
                # break

    def __ROS_topic_detect(self):
        print("Detecting ROS topic image")


    def show_landmarks_image(self):
        while True:
            
            # Check if variables are set
            if type(self.detection_result) == type(None) or type(self.image) == type(None):
                continue
    
            pose_landmarks_list = self.detection_result.pose_landmarks
            annotated_image = np.copy(self.image.numpy_view())

            # Loop through the detected poses to visualize.
            for idx in range(len(pose_landmarks_list)):
                pose_landmarks = pose_landmarks_list[idx]

                # Draw the pose landmarks.
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    pose_landmarks_proto,
                    solutions.pose.POSE_CONNECTIONS,
                    solutions.drawing_styles.get_default_pose_landmarks_style())

            image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            # Resize and rotate image
            resized_image = self.__resizeWithAspectRatio(image, 800,800)
            # resized_image = cv2.rotate(resized_image, cv2.ROTATE_90_CLOCKWISE)
            
            cv2.imshow("imge", resized_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



    # TODO: DOES NOT WORK (stops detection callback)
    def show_landmarks_image_async_start(self):
        self.show_landmarks_thread.start()



    def __show_landmarks_image_async_internal(self):
        while True:
            # Check if variables are set
            if type(self.detection_result) == type(None) or type(self.image) == type(None):
                continue
    
            pose_landmarks_list = self.detection_result.pose_landmarks
            annotated_image = np.copy(self.image.numpy_view())

            # Loop through the detected poses to visualize.
            for idx in range(len(pose_landmarks_list)):
                pose_landmarks = pose_landmarks_list[idx]

                # Draw the pose landmarks.
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    pose_landmarks_proto,
                    solutions.pose.POSE_CONNECTIONS,
                    solutions.drawing_styles.get_default_pose_landmarks_style())

            image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            # Resize and rotate image
            resized_image = self.__resizeWithAspectRatio(image, 800,800)
            # resized_image = cv2.rotate(resized_image, cv2.ROTATE_90_CLOCKWISE)
            
            cv2.imshow("imge", resized_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    def show_segmentation_mask(self):
        segmentation_mask = self.detection_result.segmentation_masks[0].numpy_view()
        visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
        cv2.imshow("mask", visualized_mask)
        cv2.waitKey(0)

    
    def get_pose_landmarks(self):
        pose_landmarks_list = self.detection_result.pose_landmarks
        pose_landmarks = pose_landmarks_list[0]
        return pose_landmarks
    

    def get_pose_world_landmarks(self):
        pose_landmarks_list = self.detection_result.pose_world_landmarks
        pose_world_landmarks = pose_landmarks_list[0]
        return pose_world_landmarks
    

    def __resizeWithAspectRatio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)