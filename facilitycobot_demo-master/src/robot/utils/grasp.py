#!/usr/bin/env python3

# Math calculation tools
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation

from robot.utils.method_logger import log_method



# 0 - cracker box
# 2 - bleach cleanser
# 3 - bowl
@log_method()
class grasp_pose_calculator:
    """[summary]
    This class is used to calculate the grasp pose of the object, 
    given the pose of the object in camera frame and the pose of the robot in base frame
    """
    @log_method()
    def __init__(self):
        """[summary]
        Initalizer for the grasp_pose_calculator class. Is used to calculate all poses for the robot to move.
        """

        # Sept 2022 tranformation from tcp to camera
        self.camera_extrinsics_rotation = np.array([[-0.99997757290510081, 0.0010000841051864507, 0.006622198925269749],
                                                    [-0.00094713229105471521, -0.99996759555892567, 0.0079944213376867568],
                                                    [0.0066299794303250814, 0.0079879699476008356, 0.99994611640271402]])

        self.camera_extrinsics_translation = np.array([0.025676673660282003, 0.067042556659101638, 0.025839870701497545])


        self.T_tcp_c = np.eye(4)
        self.T_tcp_c[:3,:3] = self.camera_extrinsics_rotation
        self.T_tcp_c[:3,3] = self.camera_extrinsics_translation

        self.object_pose_in_robot_frame = None
    
    @log_method()
    def get_object_pose_in_robot_frame(self, object_pose, robot_pose):
        """[summary]
        Calculates the pose of the object in robot frame, 
        given the pose of the object in camera frame and the pose of the robot in base frame

        args:
            obj_pose: [description] pose of the object in camera frame
            robot_pose: [description] pose of the robot in base frame


        Returns:
            obj_pose_in_robot_frame: [description] pose of the object in robot frame
        """

        T_r_tcp = self.__get_matrix_from_pose(robot_pose)

        T_r_c = np.matmul(T_r_tcp, self.T_tcp_c)

        T_r_o = np.matmul(T_r_c, self.__get_matrix_from_pose(object_pose))

        self.object_pose_in_robot_frame = self.__get_pose_from_transformation_matrix(T_r_o)

        return self.object_pose_in_robot_frame
    
    @log_method()
    def get_grasp_pose(self, obj_pose, obj_id):
        """[summary]
        computes the grasp pose of the object. 

        args:
            obj_pose: [description] pose of the object in robot frame
            obj_id: [description] id of the object    

        Returns:
            pose: [description] target pose.
        """

        if self.object_pose_in_robot_frame == None:
            print("\nError: object_pose_in_robot_frame is not set. Please set before calling this method\n")
            return None

        grasp_pose = None
        if obj_id == 0:
            grasp_pose = self.__compute_grasp_pose([0.0, 0, 0.24], [0, np.pi, 0])
        elif obj_id == 2:
            grasp_pose = self.__compute_grasp_pose([0.0, 0, 0.29], [-2.2214415, -2.2214415, 0])
        elif obj_id == 3:
            grasp_pose = self.__compute_grasp_pose([0.08, 0, 0.20], [np.pi, 0, 0])
        elif obj_id == 4:
            grasp_pose = self.__compute_grasp_pose([-0.04, 0.0, 0.22], [np.pi, 0, 0])

        Tog = self.__get_matrix_from_pose(grasp_pose)
        Tro = self.__get_matrix_from_pose(obj_pose)

        Trg_pose = self.__get_pose_from_transformation_matrix(np.matmul(Tro, Tog))
    

        return Trg_pose

    @log_method()
    def get_pregrasp_pose(self, obj_pose, obj_id):
        """[summary]
        computes the pre-grasp pose of the object

        args:
            obj_pose: [description] pose of the object in robot frame
            obj_id: [description] id of the object    

        Returns:
            pose: [description] target pose.
        """
        if self.object_pose_in_robot_frame == None:
            print("\nError: object_pose_in_robot_frame is not set. Please set before calling this method\n")
            return None

        pregrasp = None
        if obj_id == 0:
            pregrasp = self.__compute_grasp_pose([0.0, 0, 0.45], [0, np.pi, 0])
        elif obj_id == 2:
            pregrasp = self.__compute_grasp_pose([0.0, 0, 0.45], [-2.2214415, -2.2214415, 0])
        elif obj_id == 3:
            pregrasp = self.__compute_grasp_pose([0.08, 0, 0.32], [np.pi, 0, 0])
        elif obj_id == 4:
            pregrasp = self.__compute_grasp_pose([-0.04, 0.0, 0.34], [np.pi, 0, 0])

        Tog = self.__get_matrix_from_pose(pregrasp)
        Tro = self.__get_matrix_from_pose(obj_pose)

        Trg_pose = self.__get_pose_from_transformation_matrix(np.matmul(Tro, Tog))


        return Trg_pose

    # Private functions

    @log_method()
    def __compute_grasp_pose(
        self, 
        t: np.ndarray,
        r: np.ndarray 
    ) -> np.ndarray:
        """[summary]
        Computes grasp pose in object frame based on input position pointing towards the object center
        Args:
        t (np.ndarray): [descrittion] position
        r (np.ndarray): [description] direction vector
        Returns:
            np.ndarray: [description] grasp pose as x,y,z (position) & x,y,z,w (orientation in quaternions)
        """

        grasp_pose = np.zeros((7,))
        
        grasp_pose[0:3] = t
        grasp_pose[3:7] = Rotation.from_rotvec(r).as_quat()
        # grasp_pose[3:7] = Rotation.from_euler('xyz', r, degrees=False).as_quat()
        return grasp_pose

    @log_method()
    def __get_matrix_from_pose(self, pose):
        """[summary]
        Transforms a pose to a transformation matrix

        Returns:
            tranformation matrix: [description] returns a matrix presented as nested lists.
        """

        T = np.identity(4)
        T[:3,:3] = Rotation.from_quat(pose[3:7]).as_matrix()
        T[0:3,3] = pose[0:3]
        return T

    @log_method()
    def __get_pose_from_transformation_matrix(self, T):
        """[summary]
        Transforms a pose to a transformation matrix

        args:
            T: [description] transformation matrix

        Returns:
            pose: [description] returns a pose presented as (x,y,z, qx, qy, qz, qw)
        """

        quat = Rotation.from_matrix(T[:3,:3]).as_quat()
        return [T[0,3],T[1,3],T[2,3],quat[0],quat[1],quat[2],quat[3]]
