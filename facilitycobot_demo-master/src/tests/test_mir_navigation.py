#!/usr/bin/env python3
import numpy as np
import math
import sys
import rospy

from robot.mir import mir

#utils
from robot.utils.method_logger import log_method

# define 6 waypoints around the FacilityCobot table using map created by the MiR 
# make robot move from one waypoint to next so it encircles the entire table

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
    print("MIR STATUS: ", state)
    while(True):
        new_state = mir_obj.get_goal_status()

        if new_state != state:
            print("Status changed too: ", new_state)
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
        movement_accomplished = True
    elif (state == 9):
        print("--------------------------- ERROR: goal lost, robot state is ", state," ---------------------------\n\n\n")
        movement_accomplished = False
    elif (state == 4):
        print("--------------------------- ERROR: goal aborted, robot state is ", state, " ---------------------------\n\n\n")
        movement_accomplished = False
    else:
        print("--------------------------- ERROR: mir stopped for unknown reason, robot state is ", state, " ---------------------------\n\n\n")
        movement_accomplished = False

    return movement_accomplished

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


if __name__ == "__main__":
    try:
        print("Starting new node: mir")
        rospy.init_node('mir')

        mir_obj = mir()
    
        send_goal_wait(mir_obj=mir_obj, x=5.540, y=5.380, theta=134.635, goal_name="waypoint 1")

        if True:
            rospy.loginfo("Goal execution done!")
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")