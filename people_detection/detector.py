from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


class Detector():
    def __init__(self, picture_file_name):
        # STEP 2: Create an PoseLandmarker object.
        base_options = python.BaseOptions(model_asset_path='people_detection/setup_files/pose_landmarker.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # STEP 3: Load the input image.
        self.image = mp.Image.create_from_file(picture_file_name)

        # STEP 4: Detect pose landmarks from the input image.
        self.detection_result = self.detector.detect(self.image)


    def show_landmarks(self):
        # STEP 5: Process the detection result. In this case, visualize it.
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
        resized_image = cv2.rotate(resized_image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow("imge", resized_image)
        cv2.waitKey(0)


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