#!/usr/bin/env python3
import tf
import rospy
from geometry_msgs.msg import Twist
from math import *
from turtlesim.msg import Pose
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Transform, Pose, PointStamped
from math import pow, atan2, sqrt, asin, cos, sin, pi
from sensor_msgs.msg import LaserScan
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d.axes3d import get_test_data
from matplotlib import cm
from scipy.stats import (multivariate_normal as mvn,
                           norm)
from human_cost import Human_cost as HC
from scipy.stats._multivariate import _squeeze_output
from math import *

def euler_from_quaternion(x, y, z, w):
    
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians
    
    
def sigmoid(x):
    return 1/(1 + exp(-x))


def dist_homemade(x: list, y: list):
    if len(x) != len(y):
        raise Exception
    dist_sum = 0
    
    for i in range(len(x)):
        dist_sum += (x[i] - y[i])**2
    
    return sqrt(dist_sum)

class DWA:
    def __init__(self, dT = 0.1, simT = 1.0, vPrec = 0.05, wPrec = 0.05, goal_th = 0.3, with_people = False):
        
        #Making a square map
        
        self.with_people = with_people
        
        
        self.map_y_cent = 0
        self.map_x_cent = 0
        self.map_width = 20
        
        
        self.simT = simT
        self.vPrec = vPrec
        self.wPrec = wPrec
        self.dT = dT
        self.N = int(floor(self.simT/self.dT))
        
        self.heading_score = 0
        self.distance_score = 0
        self.velocity_score = 0
        self.human_score = 0
        
        self.obj_alpha = 0.7  # 0.8   (values with old heading score)
        self.obj_beta = -0.2   #-0.04
        self.obj_gamma = 0.1  #0.1
        
        
        if self.with_people == True:
            self.obj_alpha = 0.1  # 0.01
            self.obj_beta = -0.2  
            self.obj_gamma = 0.2 
            self.obj_eta = 1.0
            
            self.cost = HC()
        
        self.max_iterations= 1000
        self.goal_th = goal_th
               
        self.theta = 0
        self.v = 0
        self.w = 0
        
        self.x = -8
        self.y = 8
        
        self.goal_x = 8
        self.goal_y = 8
        
        #Udkommenter dette hvis tilfældige start og slut ikke ønskes:
        
        self.x = (random.random()- 0.5)*self.map_width*0.95 + self.map_x_cent
        self.y = (random.random()- 0.5)*self.map_width*0.95 + self.map_y_cent  
        
        self.goal_x = (random.random()- 0.5)*self.map_width*0.95 + self.map_x_cent 
        self.goal_y = (random.random()- 0.5)*self.map_width*0.95 + self.map_y_cent
        
        while dist([self.x, self.y], [self.goal_x, self.goal_y]) < self.map_width*0.9:
            self.x = (random.random()- 0.5)*self.map_width*0.95 + self.map_x_cent
            self.y = (random.random()- 0.5)*self.map_width*0.95 + self.map_y_cent  
            
            self.goal_x = (random.random()- 0.5)*self.map_width*0.95 + self.map_x_cent 
            self.goal_y = (random.random()- 0.5)*self.map_width*0.95 + self.map_y_cent
        
        
        
        
        self.obstacles = [
            [5.15, 5.22],
            [7.01, 3.55],
            [5.0, 3.0],
            [2, 1],
            [-3, 3],
            [-4, -6],
            [-2, 5]
        ]
        
        self.people = [
            [[-5, 7.9],[0, 1]] ,
            [[-3, 8.1],[0, -1]],
            [[3, 7.9],[0, 1]],
        [[6, 8.1], [0, -1]],
        ]
        

        # udkommeter dette hvis tilfældige forhindringer IKKE ønskes:
        
        num_obstacles = 10
        num_people = 10
        
        random.seed(125)
        
        self.obstacles.clear()
        for i in range(num_obstacles):
            self.obstacles.append([(random.random()- 0.5)*self.map_width*0.8 + self.map_x_cent, (random.random()- 0.5)*self.map_width*0.8 + self.map_y_cent])
            
        self.people.clear()
        for i in range(num_people):
            pos = [(random.random()- 0.5)*self.map_width*0.8 + self.map_x_cent, (random.random()- 0.5)*self.map_width*0.8 + self.map_y_cent]
            
            dirx = (random.random()-0.5)
            diry = (random.random()-0.5)
            size = sqrt(dirx**2 + diry**2)
            dir = [(1/size) * dirx, (1/size)*diry]

            self.people.append([pos, dir])
        
        
        
        
        self.a_max = 1.5 #m/s²
        self.alpha_max = 1 #rad/s²
        
        self.v_max = 1.5 #m/s
        self.v_min = 0 #m/s
        self.w_max = 1.0 #rad/s
        self.w_min = - self.w_max #rad/s    
        
        self.stop = False
        
        
    def dwa(self):        
        multi_array = {}
        
        #Dynamic window
        vr_min = max(self.v_min, self.v-self.a_max*self.dT)
        vr_max = min(self.v_max, self.v + self.a_max*self.dT)
        wr_min = max(self.w_min, self.w - self.alpha_max*self.dT)
        wr_max = min(self.w_max, self.w + self.alpha_max*self.dT)
        
        #Determine poses fo each v,w

        for v in np.arange(vr_min, vr_max, self.vPrec):
            for w in np.arange(wr_min, wr_max, self.wPrec):
                
                theta_vals = []
                x_vals = []
                y_vals = []
                
                #Determine theta
                
                for i in range  (0, self.N):
                    theta_i = self.theta + self.dT*w*i
                    theta_i = atan2(sin(theta_i), cos(theta_i))
                    theta_vals.append(theta_i)
            
                #Determine x and y
                
                for i in range(0, self.N):
                    cosSum = 0
                    sinSum = 0
                    for j in range(0, i):
                        cosSum += cos(theta_vals[i])
                        sinSum += sin(theta_vals[i])

                    y_i = self.y + self.dT*i*v*sinSum
                    x_i = self.x + self.dT*i*v*cosSum
                    
                    x_vals.append(x_i)
                    y_vals.append(y_i)
                
                
                #creating pose_array
                
                poses = []
                
                for i in range(0, self.N):
                    pose = (x_vals[i], y_vals[i], theta_vals[i])  
                    poses.append(pose)
                    
                pose_and_score = {"Score": 0, "Poses": poses}
                
                multi_array[(v, w)] = pose_and_score
                
        
        winner_v, winner_w, winner_poses = self.determine_scores(multi_array)
        
        self.x = winner_poses[1][0]
        self.y = winner_poses[1][1]
        self.theta = winner_poses[1][2]
        
        self.w = winner_w
        self.v = winner_v
    
    

    def determine_scores(self, multi_array):
        self.stop = False
        
        max_score = -10000
        
        winner_v = None
        winner_w = None
        winner_poses = None
        
        for vel in multi_array:
            v = vel[0]
            w = vel[1]
            poses = multi_array[(v, w)]["Poses"]
            
            score = 0
                       
            min_dist = 100000
            change_in_distance_to_goal = - 100000
            
            for x,y,theta in poses:                 
                for obstacle in self.obstacles:
                    obs_dist = dist([x, y], [obstacle[0], obstacle[1]])
                    if obs_dist < min_dist:
                        min_dist = obs_dist
    
                dist_goal = dist([self.x, self.y], [self.goal_x, self.goal_y]) - dist([x, y], [self.goal_x, self.goal_y])
                                    
                if dist_goal > change_in_distance_to_goal:
                    change_in_distance_to_goal = dist_goal
            
            if dist([poses[1][0], poses[1][1]], [self.goal_x, self.goal_y]) < self.goal_th:
                self.stop = True
            
            
            self.distance_score = self.obj_beta*((1/min_dist)**2)
            
            self.heading_score = self.obj_alpha*(dist_goal)
            
            self.velocity_score = self.obj_gamma*v

            human_score = 0
            
            if self.with_people == True:
                worst_score = float("inf")
                worst_pose = None

                for p in range(len(poses)):
                    x,y,theta = poses[p]
                    worst_score = float("inf")
                    worst_pose = None
                    
                    for h in range(len(self.people)):
                        hum_to_rob = [x - self.people[h][0][0], y - self.people[h][0][1]]
                        hum_ori = [self.people[h][1][0], self.people[h][1][1]]
                        price = self.cost.get_cost_xy(hum_to_rob[0], hum_to_rob[1], hum_ori[0], hum_ori[1])
                        if (price < worst_score):
                            worst_score = price
                            worst_pose = p
                            human_score = price

            self.human_score = self.obj_eta*( human_score )
            
            score = self.heading_score + self.distance_score + self.velocity_score + self.human_score

            if score > max_score:
                max_score = score
                winner_v = v
                winner_w = w
                winner_poses = poses

            multi_array[(v, w)]["Score"] = score        
        
        return winner_v, winner_w, winner_poses
            
            
            
    def simulate_dwa(self):

        print("Start pos: (", self.x, self.y, ")")
        print("Goal pos: (", self.goal_x, self.goal_y, ")")
        
        ### Draw obstacles ###
        figure, axes = plt.subplots()
        for i in range(len(self.obstacles)):    
            drawing_circles = plt.Circle( (self.obstacles[i][0], self.obstacles[i][1]), 0.2, fill = False )
            axes.add_artist(drawing_circles)
        
        if self.with_people == True:
            for i in range(len(self.people)):                 
                drawing_circles = plt.Circle( (self.people[i][0][0], self.people[i][0][1]), 0.1, fill = True, color = (1, 0, 0) )
                axes.add_artist(drawing_circles)
                
                plt.quiver(self.people[i][0][0], self.people[i][0][1], self.people[i][1][0], self.people[i][1][1], scale=5, scale_units="inches", minshaft=2, headlength=5)
        
                
            

        plt.plot(self.goal_x, self.goal_y, 'g*')
        plt.plot(self.x, self.y, 'r*')
        
        goal_circles = plt.Circle( (self.goal_x, self.goal_y), 0.2, color=(0, 1, 0) ,fill = True )
        axes.add_artist(goal_circles)    
            
        
        # Simulate movement    
        i = 0
        
        x = np.zeros(1)
        y = np.zeros(1)
        x[i] = self.x
        y[i] = self.y
        time = [0]
        vels = [self.v]
        wels = [self.w]
        
        heading_scores = [0]
        velocity_scores = [0]
        distance_scores = [0]
        total_score =[0]
        
        human_scores = [0]
        

        
        while i < self.max_iterations and self.stop == False:
            i += 1

            self.dwa()

            x = np.append(x, self.x)
            y = np.append(y, self.y)
            
            vels.append(self.v)
            wels.append(self.w)
            
            
            time.append(time[-1] + self.dT)
            
            heading_scores.append(self.heading_score)
            velocity_scores.append(self.velocity_score)
            distance_scores.append(self.distance_score)
                 
            if self.with_people == True:
                human_scores.append(self.human_score)
            
            total_score.append(self.heading_score + self.velocity_score + self.distance_score)       
            
            # If at goal
            if dist([self.x, self.y], [self.goal_x, self.goal_y]) < self.goal_th:
                print("At goal!")
                print("i =", i)

        

        # Plot settings
        plt.xlim([self.map_x_cent - 0.5*self.map_width, self.map_x_cent + self.map_width*0.5])
        plt.ylim([self.map_y_cent - 0.5*self.map_width, self.map_y_cent + self.map_width*0.5])
        axes.set_aspect(1)
        plt.title("Map")
        plt.grid()
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        
        ### Plot path ###
        plt.plot(x,y)
        
        
        ### Plot x,y (t) ###
        fig2 = plt.figure("Paths")
        plt.plot(time, x, color='g', label='x(t)')
        plt.plot(time, y, color='r', label='y(t)')
        plt.legend()
        plt.title("Paths")
        plt.grid()
        plt.xlabel("t [sek]")
        plt.ylabel("Dist [m]")


        ### Plot Scores ###
        fig3 = plt.figure("Scores")
        plt.plot(time, velocity_scores, color='g', label='Vel Score')
        plt.plot(time, distance_scores, color='r', label='Dist Score')
        plt.plot(time, heading_scores, color='b', label='Head Score')
        
        if self.with_people == True:
            plt.plot(time, human_scores, color='c', label='Human Score')
            
        plt.plot(time, total_score, color ='y', label='Total Score')
        plt.legend()
        plt.title("Scores")
        plt.grid()
        plt.xlabel("t [sek]")
        plt.ylabel("Score []")
        
        ### Plot velocities ###
        fig4 = plt.figure("Velocities")
        plt.plot(time, wels, color='g', label='Ang. Vel [rad/s]')
        plt.plot(time, vels, color='r', label='Lin. Vel [m/s]')
        plt.legend()
        plt.title("Velocities [rad/s] and [m/s]")
        plt.grid()
        plt.xlabel("t [sek]")
        plt.ylabel("Velocity")
        
        # Show plot
        plt.show()
        


if __name__ == '__main__':
    try:
        dwa = DWA(with_people=True)
        dwa.simulate_dwa()
        #plot_skew_gauss()
    except rospy.ROSInterruptException:
        print("Error")
        