#!/usr/bin/env python3
import rospy 
from geometry_msgs.msg import PoseArray, Pose, TwistStamped
from sensor_msgs.msg import PointCloud

from math import cos, sin, atan2, sqrt, pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Process

from mir import mir as MIR
from dynamic_window_approach import DWA
from safe_artificial_potential_fields import SAPF


class Local_Planner:
    def __init__(self, rate = 10):
        rospy.init_node('obstacle_detector', anonymous=True)
        
        # self.obstacles = []
        # self.humans = []
         
        # self.obstacle_subscriber = rospy.Subscriber('/obstacles_without_people',PointCloud, self.update_obstacles_callback)
        # self.human_subscriber = rospy.Subscriber('/people',PoseArray, self.update_humans_callback)
        
        
        # self.x = 0
        # self.y = 0
        # self.theta = 0
        
        # self.v = 0
        # self.w = 0
        
        # self.velocity_subscriber = rospy.Subscriber('/cmd_vel',TwistStamped, self.update_velocities_callback)
        # self.pose_subscriber = rospy.Subscriber('/robot_pose',Pose, self.update_pose_callback)
        
        # self.dt = rate/1000.0
        # self.rate = rospy.Rate(rate)
        
        # self.dwa = DWA()
        # self.sapf = SAPF()

        # # Logs
        # self.x_log = []
        # self.y_log = []
        # self.v_log = []
        # self.w_log = []

    
    def get_pos(self):
        return self.x, self.y
                  
        
    
    def update_obstacles_callback(self, data:PointCloud):
        self.obstacles.clear()
        for i in range(len(data.points)):
            point = [data.points[i].x, data.points[i].y]
            self.obstacles.append(point)
            
            
    def update_humans_callback(self, data:PoseArray):
        self.humans.clear()
        for i in range(len(data.poses)):
            orientation = data.poses[i].orientation
            position = data.poses[i].position
            
            point = [position.x, position.y]
            
            t3 = +2.0 * (orientation.w * orientation.z)
            t4 = +1.0 - 2.0 * (orientation.z * orientation.z)
            angle = atan2(t3, t4)
            
            orient = [cos(angle), sin(angle)]
            person = [point, orient] 
            
            self.humans.append(person)
            
            
    def update_velocities_callback(self, data:TwistStamped):
        self.v = data.twist.linear.x
        self.w = data.twist.angular.z
    
    
    
    def update_pose_callback(self, data:Pose):
        self.x = data.position.x
        self.y = data.position.y
        
        t3 = +2.0 * (data.orientation.w * data.orientation.z)
        t4 = +1.0 - 2.0 * (data.orientation.z * data.orientation.z)
        self.theta = atan2(t3, t4)

    
    def log(self):
        self.x_log.append(self.x)
        self.y_log.append(self.y)
        self.v_log.append(self.v)
        self.w_log.append(self.w)
    

    def init_plot(self):
        self.line_plot.set_data([], [])

        return self.line_plot,



    def animate(self, x_log, y_log):
        self.line_plot.set_data(x_log, y_log)

        return self.line_plot, 
    

    def show_log(self, v=False, w=False):
        # plot pos and goal and obstacles
        print("print")
        plt.figure()
        plt.title("pos")
        plt.grid()
        plt.plot(self.goal[0], self.goal[1], marker='o')
        plt.plot(self.x_log, self.y_log)
        for o in range(len(self.obstacles)):
            plt.plot(self.obstacles[o][0], self.obstacles[o][1], marker='o')

        # Plot v
        if v:
            fig = plt.figure()
            plt.grid()
            plt.title("v")
            plt.plot(self.v_log)

        # Plot w
        if w:
            fig = plt.figure()
            plt.grid()
            plt.title("w")
            plt.plot(self.w_log)

        plt.show()


    def continuois_plot(self, i: int, fig):
        if (i % 100 == 0):
            fig.clear()
            plt.plot(self.x_log, self.y_log, color="b")
            plt.plot(self.x, self.y, marker='o', color="r")
            plt.plot(self.goal[0], self.goal[1], marker="o", color="g")
            plt.draw()
            plt.pause(0.0001)


            
            
    def move_robot(self, v, w, simulation):
        # Simulation of movement
        if simulation:
                self.theta = self.theta + w * (self.dt)
                self.x = self.x + v * (self.dt) * cos(self.theta)
                self.y = self.y + v * (self.dt) * sin(self.theta)
        
        # Real movement
        else:
            pass


    def planner(self, use_sapf = True):
        
        # Using SAPF
        if use_sapf == True:
            # Update map and robot state
            self.sapf.update_map(obstacles=np.array(self.obstacles), humans=np.array(self.humans), goal=np.array(self.goal))
            self.sapf.update_robot_state(x=self.x, y=self.y, theta=self.theta)

            # Get linear and angular velocities
            v, w = self.sapf.calc_step()

            return v, w
        

        # Using DWA
        else:
            #TODO: Do DWA stuff here
            pass



    
    def main(self):
        # Init values
        self.goal = [10,10]
        self.obstacles = [[5,5], [6,5]]
        self.max_i = 10000
        i = 0
        fig = plt.figure()
        

        
        while not rospy.is_shutdown():
            self.v, self.w = self.planner(use_sapf=True)
            self.move_robot(self.v, self.w, simulation=True)
            self.log()

            # Continuois plot
            self.continuois_plot(i, fig)
            


            # If at goal
            if sqrt((self.goal[0] - self.x)**2 + (self.goal[1] - self.y)**2) < 0.1:
                print("At goal")
                break

            # If at max iterations
            i+=1
            if i > self.max_i:
                print("Max iterations reached")
                break

            # self.rate.sleep()

        self.show_log()
    

    def test_robot_movement(self):
        mir = MIR()
        mir.send_goal(x=1, y=1, theta=0)
        # mir.move_relative(dist=0.1, angle=0)



if __name__ == "__main__":
    try:   
        LOC = Local_Planner()
        # LOC.main()
        LOC.test_robot_movement()
    except rospy.ROSInterruptException:
        print("Error")
    