#!/usr/bin/env python3
import numpy as np
import math
import sys
import rospy

from robot.mir import mir
from robot.ur import ur
from robot.task.pick_and_place import object_manipulator_with_ur

from fc_msgs.srv import GetOptimalPlan

# for transforming between quaternion and euler
from scipy.spatial.transform import Rotation

# utils
from robot.utils.method_logger import log_method

@log_method()
def wait(mir_obj, goal_name = False):
    """[summary]
    waits until the the state is either 3 (completed) or the traversing failed
    Args:
        dist (float32): [description] distance in m
        angle (float32): [description] turn angle in radians  
        goal_name (string): [description] name of the position, used to print that the specific position with the name is reached
    Returns:
        bool: [description] status
    """

    state = mir_obj.get_goal_status()
    while(state != 3):
        new_state = mir_obj.get_goal_status()

        if new_state != state:
            state = new_state

        if state == 2 or state == 3 or state == 4 or state ==  5 or state == 9:
            break
        rospy.sleep(1)

        
    if goal_name:
        name = goal_name
    else:
        name = "goal"

    if (state == 3):
        print("--------------------------- ",name, " reached: robot state ", state, " ---------------------------\n\n\n")
    elif (state == 9):
        print("--------------------------- ERROR: goal lost, robot state is ", state," ---------------------------\n\n\n")
    elif (state == 4):
        print("--------------------------- ERROR: goal aborted, robot state is ", state, " ---------------------------\n\n\n")
    else:
        print("--------------------------- ERROR: mir stopped for unknown reason, robot state is ", state, " ---------------------------\n\n\n")

    return state

@log_method()
def move_relative_wait(mir_obj, dist, angle, goal_name = False):
        """[summary]
        sends the goal, and waits in the method until the mir has reached its destination or failed to do so
        Args:
            dist (float32): [description] distance in m
            angle (float32): [description] turn angle in radians  
            goal_name (string): [description] name of the position, used to print that the specific position with the name is reached
        Returns:
            bool: [description] status
        """

        mir_obj.move_relative(dist=dist, angle=angle)
        wait(mir_obj=mir_obj, goal_name = goal_name)
        return mir_obj.client.get_state()

@log_method()
def send_goal_wait(mir_obj, x, y, theta, goal_name = False,  frame=None):
        """[summary]
        sends the goal, and waits in the method until the mir has reached its destination or failed to do so
        This 
        Args:
            mir_obj (type): [description] is the controller class for the mir robot
            x (float32): [description] x coordinate in m
            y (float32): [description] y coordinate in m
            theta (float32): [description] theta angle in degrees
            goal_name (string): [description] name of the position, used to print that the specific position with the name is reached
            frame (string): [description] goal frame, if not specified use map frame
        """        

        max_tries = 3

        movement_success = False

        for i in range(0, max_tries):
            mir_obj.send_goal(x=x, y=y, theta=theta)

            movement_accomplished = wait(mir_obj=mir_obj, goal_name=goal_name)

            if movement_accomplished == 3: # successfully moved to location
                movement_success = True
                break
            else:
                print("\nmovement failed, retrying movement: retry ", i,"\n")
                rospy.sleep(3)
        
        if movement_success == True:
            print("\ngoal reached, startig adjust movement before stopping\n")
            mir_obj.send_goal(x=x, y=y, theta=theta)


        return mir_obj.client.get_state()


@log_method()
def user_confirmation_delay():
    """[summary]
    waits for user input to continue or abort the test
    Returns:
        int: [description] 1 if the test should be aborted, 2 if the test should continue
    """

    print("\npress 'enter' to continue, press 1 to abort\n")

    var = input()
    if (var == "1"):
        return 1
    else:
        return 2

@log_method()
def get_optimizer_goals():
    """[summary]
    gets the goals from the optimizer topic
    Returns:
        [type]: [description] list of poses
    """

    rospy.wait_for_service('/optimization/get_hardcoded_plan')
    try:

        optimizer_client = rospy.ServiceProxy('/optimization/get_hardcoded_plan', GetOptimalPlan)
        res = optimizer_client([])
        return res.base_poses
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

@log_method()
def get_planned_goals():
    """[summary]
    gets the goals from the optimizer topic
    Returns:
        [list]: [description] list of poses
    """

    goals = get_optimizer_goals()
    goal_list = []
    
    for poses in goals:
        x = poses.pose.position.x
        y = poses.pose.position.y

        rx = poses.pose.orientation.x
        ry = poses.pose.orientation.y
        rz = poses.pose.orientation.z
        rw = poses.pose.orientation.w

        euler = Rotation.from_quat([rx, ry, rz, rw]).as_euler("xyz")

        theta = euler[2]*360 / (2*math.pi)

        goal_list.append([x,y,theta, "no name"])

    if False:
        goal_list.append([6.894, 5.120, -131.292, "Waypoint_Bowl_pickup"])
        goal_list.append([5.972, 4.709, -179.082, "Waypoint_mug_pickup"])
        goal_list.append([5.369, 5.265, 149.098, "Way point_cheezits_pickup"])
        goal_list.append([4.968, 5.588, 131.245, "Waypoint_Bleach_cleaner_pickup"])

    return goal_list

@log_method()
def execute_move_and_pick_task(goal_list, object_id_at_goals):
    """[summary]
    executes the move and pick task
    Args:
        goal_list ([list]): [description] list of poses
        object_id_at_goals [list]: [description] is the objects at the given locations.
    """

    task_obj = object_manipulator_with_ur()
    mir_obj = mir()

    continue_test = 2

    test_aborted = False

    for i in range(0,goal_list.__len__()):

            #Moving to the designated position
            if test_aborted == False:
                continue_test = user_confirmation_delay()

            if (continue_test == 2 and test_aborted == False):
                x = goal_list[i][0]
                y = goal_list[i][1]
                theta = goal_list[i][2]
                goal_name = goal_list[i][3]

                send_goal_wait(mir_obj=mir_obj, x=x, y=y, theta=theta, goal_name=goal_name)
            else:
                test_aborted = True
            
            for object_id in object_id_at_goals[i]:
                #picking object at position
                if test_aborted == False:
                    continue_test = user_confirmation_delay()
                
                if (continue_test == 2 and test_aborted == False):
                    
                    task_obj.move_to_home()
                    object_pose_and_id = task_obj.search_for_object(object_id)

                    if object_pose_and_id != None:
                        task_obj.move_to_object_and_pick(object_pose_and_id)
                        task_obj.move_to_start_pose()
                        task_obj.move_to_bin_pose()
                        task_obj.put_down_object()

                    task_obj.move_to_home()
                    
                else:
                    test_aborted = True
                
@log_method()
def get_test_goals():
    """ [summary]
    list of positions of objects while the goal optimizer is not present.
    
    """

    cheeze = [6.070, 5.969, -87.213, "cheeze"]
    bleach = [6.025, 5.389, -94.463, "bleach"]
    bowl = [4.340, 5.203, 123.902, "bowl"]
    mug = [4.146, 6.010, 91.602, "mug"]

    return [[cheeze, bleach, bowl, mug], [[0], [2], [3], [4]]]

if __name__ == "__main__":
    try:
        print("Starting new node: mir_ur")
        rospy.init_node('mir_ur')

        #goal_list = get_planned_goals()
        goal_list = get_test_goals()

        goals = goal_list[0]
        objects_id_at_goals = goal_list[1]

        execute_move_and_pick_task(goals, objects_id_at_goals)

        if True:
            rospy.loginfo("Goal execution done!")
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")