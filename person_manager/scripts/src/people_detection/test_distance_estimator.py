from detector import Detector
import math
import os


file_path = os.path.dirname(os.path.abspath(__file__))


def dst(landmark1, landmark2):
    return math.sqrt(
        math.pow(landmark1.x - landmark2.x,2) +
        math.pow(landmark1.y - landmark2.y,2) + 
        math.pow(landmark1.z - landmark2.z,2)
    )


def test_distance_estimator(landmarks):
    # Shoulder to wrist
    shoulder_to_wrist_right = dst(landmarks[12], landmarks[16])
    shoulder_to_wrist_right_real = 0.51
    print("{:26}{:5}{:<6.2f}{:>12}{:4.2f}".format("shoulder_to_wrist_right:", "Real=", shoulder_to_wrist_right_real, "calculated=", shoulder_to_wrist_right))

    shoulder_to_wrist_left = dst(landmarks[11], landmarks[15])
    shoulder_to_wrist_left_real = 0.50
    print("{:26}{:5}{:<6.2f}{:>12}{:4.2f}".format("shoulder_to_wrist_left:", "Real=", shoulder_to_wrist_left_real, "calculated=", shoulder_to_wrist_left))

    # Hip to knee
    hip_to_knee_right = dst(landmarks[24], landmarks[26])
    hip_to_knee_right_real = 0.40
    print("{:26}{:5}{:<6.2f}{:>12}{:4.2f}".format("hip_to_knee_right:", "Real=", hip_to_knee_right_real, "calculated=", hip_to_knee_right))

    hip_to_knee_left = dst(landmarks[23], landmarks[25])
    hip_to_knee_left_real = 0.40
    print("{:26}{:5}{:<6.2f}{:>12}{:4.2f}".format("hip_to_knee_left:", "Real=", hip_to_knee_left_real, "calculated=", hip_to_knee_left))

    # Knee to ankle
    knee_to_ankle_right = dst(landmarks[26], landmarks[28])
    knee_to_ankle_right_real = 0.43
    print("{:26}{:5}{:<6.2f}{:>12}{:4.2f}".format("knee_to_ankle_right:", "Real=", knee_to_ankle_right_real, "calculated=", knee_to_ankle_right))

    knee_to_ankle_left = dst(landmarks[25], landmarks[27])
    knee_to_ankle_left_real = 0.42
    print("{:26}{:5}{:<6.2f}{:>12}{:4.2f}".format("knee_to_ankle_left:", "Real=", knee_to_ankle_left_real, "calculated=", knee_to_ankle_left))



if __name__ == "__main__":
    print("---- Test: T-pose 1 ----")
    detector1 = Detector(picture_file_name= file_path+"/videos_and_images/a T-pose 1.jpg")
    test_distance_estimator(detector1.get_pose_world_landmarks())
    detector1.show_landmarks_image()

    print("---- Test: T-pose 2 ----")
    detector2 = Detector(picture_file_name=file_path+"/videos_and_images/a T-pose 2.jpg")
    test_distance_estimator(detector2.get_pose_world_landmarks())
    detector2.show_landmarks_image()

    print("---- Test: i-pose 1 ----")
    detector3 = Detector(picture_file_name=file_path+"/videos_and_images/a i-pose 1.jpg")
    test_distance_estimator(detector3.get_pose_world_landmarks())
    detector3.show_landmarks_image()

    print("---- Test: i-pose 2 ----")
    detector4 = Detector(picture_file_name=file_path+"/videos_and_images/a i-pose 2.jpg")
    test_distance_estimator(detector4.get_pose_world_landmarks())
    detector4.show_landmarks_image()

    print("---- Test: l-pose 1 ----")
    detector5 = Detector(picture_file_name=file_path+"/videos_and_images/a l-pose 1.jpg")
    test_distance_estimator(detector5.get_pose_world_landmarks())
    detector5.show_landmarks_image()

    print("---- Test: l-pose 2 ----")
    detector6 = Detector(picture_file_name=file_path+"/videos_and_images/a l-pose 2.jpg")
    test_distance_estimator(detector6.get_pose_world_landmarks())
    detector6.show_landmarks_image()


