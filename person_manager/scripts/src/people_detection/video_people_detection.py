from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


file_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.getcwd()


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

        print("Pose_landmarks_proto: ", pose_landmarks_proto)

        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())

        print("solutions.pose.POSE_CONNECTIONS: ", solutions.pose.POSE_CONNECTIONS)

    return annotated_image


def video_detection(video_file_name):
  video_file_path = file_path + "/videos_and_images/" + video_file_name
  
  base_options = python.BaseOptions(model_asset_path=file_path + '/setup_files/pose_landmarker.task')
  options = vision.PoseLandmarkerOptions(
      base_options=base_options,
      output_segmentation_masks=True)
  detector = vision.PoseLandmarker.create_from_options(options)


  cap = cv2.VideoCapture(video_file_path)

  while(True):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if type(frame) != None:
          frame = cv2.resize(frame, (400, 400))
          cv2.imwrite("frame.jpg", frame)
          

          # STEP 3: Load the input image.
          image = mp.Image.create_from_file("frame.jpg")

          # STEP 4: Detect pose landmarks from the input image.
          detection_result = detector.detect(image)

          # STEP 5: Process the detection result. In this case, visualize it.
          annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
          cv2.imshow("imge", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
          if cv2.waitKey(1) & 0xFF == ord('q'):
              os.system("rm frame.jpg")
              break

  # When everything done, release the capture
  cap.release()
  
  # finally, close the window
  cv2.destroyAllWindows()
  cv2.waitKey(1)

if __name__ == "__main__":
  video_detection("src/people_detection/videos_and_images/video_walking.mp4")
