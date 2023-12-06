#!/usr/bin/env python3


# custom modules
from robot.ur import ur
from robot.utils.grasp import grasp_pose_calculator
from robot.utils.robot_vision import object_pose_estimation
from robot.utils.visual_exploration import visual_exploration

# utils
from robot.utils.method_logger import log_method


# 0 - cracker box
# 2 - bleach cleanser
# 3 - bowl
@log_method()
class object_manipulator_with_ur:
    
    @log_method()
    def __init__(self):
        """[summary]
        Is used to solve the task of picking up object, and placing it in bin.
        Uses three classes. The grasp_pose_calulator gets the poses of the objects in robot frame,
        The ur class communicates and controls the ur robot. The robot vision class gets the objects pose from camera frame.
        """

        self.__ur = ur("192.168.11.40")
        self.__grasp_pose_calculator = grasp_pose_calculator()
        self.__robot_vision = object_pose_estimation()
        self.__visual_exploration = visual_exploration(self.__ur, self.__robot_vision)

    @log_method()
    def search_for_object(self, object_id):
        """[summary]
        Makes the robot move based on the visual exploration movement, and stops when the robot has found the object it is searching for

        returns:
        [object pose, object id] [description]: returns a list containing the pose in camera frame, and the if of the object
        """
        
        object_pose_and_id = self.__visual_exploration.explore(object_id)
        return object_pose_and_id
        
    @log_method()
    def move_to_object(self, object_pose_and_id):
        """[summary]
        Moves the robot to the object, from pregrasp pose then to grasp pose.
        arg:
            object_pose_and_id [description]: list containing the pose of the object in camera frame, and the id of the object
        """

        # get object pose in camera frame
        object_pose = object_pose_and_id[0]
        object_id = object_pose_and_id[1]
        robot_pose = self.__ur.get_tcp_pose()

        # Convert object pose from camera frame to robot frame
        object_pose_in_robot_frame = self.__grasp_pose_calculator.get_object_pose_in_robot_frame(object_pose, robot_pose)

        # Get pregrasp and grasp pose
        pregrasp_pose = self.__grasp_pose_calculator.get_pregrasp_pose(object_pose_in_robot_frame, object_id)
        grasp_pose = self.__grasp_pose_calculator.get_grasp_pose(object_pose_in_robot_frame, object_id)

        self.__ur.move_in_cartesian_space(pregrasp_pose)
        self.__ur.move_in_cartesian_space(grasp_pose)

    @log_method()
    def move_to_object_and_pick(self, object_pose_and_id):
        """[summary]
        Moves the robot to the object, going to pregrasp then grasp pose. Then picks up the object.
        """

        self.move_to_object(object_pose_and_id)
        self.__ur.gripper.close()

    @log_method()
    def move_to_start_pose(self):
        """[summary]
        Moves the robot to the joint positions of the start pose.
        """

        self.__ur.set_joint_positions(1.4214767217636108, -2.2918912372984828, 2.5344913641559046, -2.3186546764769496, -1.599729363118307, 3.140084743499756)

    @log_method()
    def move_to_bin_pose(self):
        """[summary]
        Moves the robot to the pose where the bin object is placed.
        """

        pose = [0.4644097773179391, 0.05941475429070929, 0.5465010920860061, -0.6964246117221837, -0.7173326157068847, -0.017407681064603015, 0.011119859292174626]
        self.__ur.move_in_cartesian_space(pose)
    
    @log_method()
    def pick_up_object(self):
        """[summary]
        closes the gripper on the ur robot
        """

        self.__ur.gripper.close()

    @log_method()
    def put_down_object(self):
        """[summary]
        opens the gripper on the ur robot
        """

        self.__ur.gripper.open()

    @log_method()
    def move_to_home(self):
        self.__ur.set_joint_positions(3.1473498344421387, -1.570796628991598, 2.3308056036578577, -3.257125040093893, -1.596511189137594, 3.1366167068481445)

