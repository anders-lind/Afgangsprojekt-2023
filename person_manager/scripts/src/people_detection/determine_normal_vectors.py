from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def get_tangentplane_and_normalvector(three_points_on_plane):

    #three_points_on_plane : [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2]]

    r1 = np.subtract(three_points_on_plane[0], three_points_on_plane[1])
    r2 = np.subtract(three_points_on_plane[2], three_points_on_plane[1])


    normalvector = np.cross(r1, r2)
    normalvector = normalvector / np.linalg.norm(normalvector)

    x0 = three_points_on_plane[0][0]
    y0 = three_points_on_plane[0][1]
    z0 = three_points_on_plane[0][2]
    
    a = normalvector[0]
    b = normalvector[1]
    c = normalvector[2]

    d = a*x0 +b*y0 +c*z0

    print("The equation of the plane is: ", str(round(a, 3)) + "x + " + str(round(b, 3)) + "y + " + str(round(c, 3)) + "z = "  + str(round(d, 3)))

    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)

    X, Y = np.meshgrid(x, y)
    Z = (d-a*X - b*Y)/c

    return X, Y, Z, normalvector


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    world_coordinate_list = detection_result.pose_world_landmarks
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
        
        points2d = []
        points3d = []

        for i in range(len(pose_landmarks)):
            points2d.append([int(math.floor(pose_landmarks[i].x*annotated_image.shape[1])), int(math.floor(pose_landmarks[i].y*annotated_image.shape[0]))])

        for i in range(len(world_coordinate_list[0])):
            points3d.append([world_coordinate_list[0][i].x, world_coordinate_list[0][i].y, world_coordinate_list[0][i].z])

    return annotated_image, points2d, points3d


def draw_human_landmarks(ax, points3d_np):
    x = []
    y = []
    z = []    
 
    for point in points3d_np:
        x.append(point[0])
        y.append(point[1])
        z.append(point[2])

    ax.scatter3D(x, y, z, c=z, cmap='hsv')

    ax.plot([x[11], x[12]], [y[11], y[12]], [z[11], z[12]])
    ax.plot([x[11], x[23]], [y[11], y[23]], [z[11], z[23]])
    ax.plot([x[24], x[12]], [y[24], y[12]], [z[24], z[12]])
    ax.plot([x[16], x[12]], [y[16], y[12]], [z[16], z[12]])
    ax.plot([x[11], x[15]], [y[11], y[15]], [z[11], z[15]])
    ax.plot([x[23], x[24]], [y[23], y[24]], [z[23], z[24]])
    ax.plot([x[28], x[24]], [y[28], y[24]], [z[28], z[24]])
    ax.plot([x[23], x[27]], [y[23], y[27]], [z[23], z[27]])
    ax.plot([x[11] + (x[12]-x[11])/2, x[0]], [y[11] + (y[12]-y[11])/2, y[0]], [z[11] + (z[12]-z[11])/2, z[0]])


def draw_plane_for_face(ax, points3d_np):
    right_ear = points3d_np[8]
    left_ear = points3d_np[7]
    mid_neck = points3d_np[11] + (points3d_np[12] - points3d_np[11])/2
    
    points = [right_ear, left_ear, mid_neck]
    averagePoint = []

    for i in range(3):
        sum = 0
        for j in range(3):
            sum += points[j][i]
        sum /= 3
        averagePoint.append(sum)

    X, Y, Z, normalvector = get_tangentplane_and_normalvector(points)


    # DRAW NORMAL VECTOR FROM MIDPOINT OF FACE
    ax.quiver(averagePoint[0], averagePoint[1], averagePoint[2], normalvector[0] + averagePoint[0], normalvector[1] + averagePoint[1], normalvector[2]+ averagePoint[2], color='b')
    #DRAW TANGENT PLANE FOR FACE
    ax.plot_surface(X, Y, Z, color='red', alpha=0.5)


def draw_plane_for_body(ax, points3d_np):

    right_shoulder = points3d_np[12]
    left_shoulder = points3d_np[11]
    mid_hip = points3d_np[23] + (points3d_np[24]- points3d_np[23])/2

    points = [right_shoulder, left_shoulder, mid_hip]
    averagePoint = []
    for i in range(3):
        sum = 0
        for j in range(3):
            sum += points[j][i]
        sum /= 3
        averagePoint.append(sum)

    X, Y, Z, normalvector = get_tangentplane_and_normalvector(points)

    # DRAW NORMAL VECTOR FROM MIDPOINT OF BODY
    ax.quiver(averagePoint[0], averagePoint[1], averagePoint[2], normalvector[0] + averagePoint[0], normalvector[1] + averagePoint[1], normalvector[2]+ averagePoint[2], color='b')
    #DRAW TANGENT PLANE FOR FACE
    ax.plot_surface(X, Y, Z, color='red', alpha=0.5)


def picture_detection(picture_file_name):

    #GET FILEPATH AND SHOW IMAGE
    file_path = os.getcwd()
    img = cv2.imread(file_path + "/" + picture_file_name)
    # cv2.imshow("image", img)
    # cv2.waitKey(0)


    # DETECT 3D AND 2D POINTS OF LANDMARKS IN IMAGE
    base_options = python.BaseOptions(model_asset_path='src/people_detection/setup_files/pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)
    image = mp.Image.create_from_file(picture_file_name)
    detection_result = detector.detect(image)
    annotated_image, points2d, points3d = draw_landmarks_on_image(image.numpy_view(), detection_result)

    points3d_np = np.array(points3d)

    #SHOW ANNOTATED IMAGE
    # cv2.imshow("annotated img", annotated_image)
    # cv2.waitKey(0)
    cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)


    #DRAW 3D POINTS IN PLOT
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.xlim([-1, 3])
    plt.ylim([-1, 3])


    # DRAW LANDMARKS OF HUMAN
    draw_human_landmarks(ax, points3d_np)

    # MAKE A 3D PLANE FOR FACE
    draw_plane_for_face(ax, points3d_np)

    # MAKE A 3D PLANE FOR BODY
    draw_plane_for_body(ax, points3d_np)

    plt.show()


if __name__ == "__main__":
    picture_detection("src/people_detection/videos_and_images/real_person_tpose.jpeg")