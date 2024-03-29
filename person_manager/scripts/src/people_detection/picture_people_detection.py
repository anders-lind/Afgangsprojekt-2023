from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import math


file_path = os.path.dirname(os.path.abspath(__file__))


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

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
        
    return annotated_image


def picture_detection(picture_file_name):
    #GET FILEPATH AND SHOW IMAGE

    img = cv2.imread(file_path + "/videos_and_images/" + picture_file_name)
    #img = cv2.resize(img, (0, 0), fx = 0.2, fy = 0.2)
    cv2.imshow("image", img)
    cv2.waitKey(0)


    # DETECT 3D AND 2D POINTS OF LANDMARKS IN IMAGE
    base_options = python.BaseOptions(model_asset_path=file_path + '/setup_files/pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(file_path + "/videos_and_images/" + picture_file_name)

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    #annotated_image = cv2.resize(annotated_image, (0, 0), fx = 0.2, fy = 0.2)
    #annotated_image = cv2.rotate(annotated_image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow("landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

    segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
    visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
    #visualized_mask = cv2.rotate(visualized_mask, cv2.ROTATE_90_CLOCKWISE)
    #visualized_mask = cv2.resize(visualized_mask, (0, 0), fx = 0.2, fy = 0.2)
    cv2.imshow("mask", visualized_mask)
    cv2.waitKey(0)

if __name__ == "__main__":
    picture_detection("anders_cropped.jpg")