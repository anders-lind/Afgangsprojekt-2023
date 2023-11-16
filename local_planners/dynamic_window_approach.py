#!/usr/bin/env python3
import tf
import rospy
from geometry_msgs.msg import Twist
from math import floor, exp
from turtlesim.msg import Pose
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Transform, Pose, PointStamped
from math import pow, atan2, sqrt, asin, cos, sin, pi
from sensor_msgs.msg import LaserScan
import sys
import numpy as np
import matplotlib.pyplot as plt
import random

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


class DWA:
    def __init__(self, dT = 0.1, simT = 0.5, vPrec = 0.05, wPrec = 0.05):
        
        self.simT = simT
        self.vPrec = vPrec
        self.wPrec = wPrec
        self.dT = dT
        self.N = int(floor(self.simT/self.dT))
        
        self.obj_alpha = 0.8
        self.obj_beta = -0.04
        self.obj_gamma = 0.1
        
        self.heading_score = 0
        self.distance_score = 0
        self.velocity_score = 0
        
        self.max_iterations= 1000
        self.goal_th = 0.5
        
        
        self.x = -8
        self.y = -8
        self.theta = 0
        self.v = 0
        self.w = 0
        
        self.goal_x = 8
        self.goal_y = 8
        
        self.obstacles = [
            [5.15, 5.22],
            [7.01, 3.55],
            [5.0, 3.0],
            [2, 1],
            [-3, 3],
            [-4, -6],
            [-2, 5]
        ]
        
        self.a_max = 10 #m/s²
        self.alpha_max = 5 #rad/s²
        
        self.v_max = 3 #m/s
        self.v_min = 0 #m/s
        self.w_max = 1.0 #rad/s
        self.w_min = - self.w_max #rad/s    
    
    def dwa(self):        
        multi_array = {}
        
        #Dynamic window
        vr_min = max(0, self.v-self.a_max*self.dT)
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
        
        
    def sigmoid(self, x):
        return 1/(1 + exp(-x))
    
    

    def determine_scores(self, multi_array):
        
        #Hyper parameters for objective function
        
        
        max_score = -10000
        
        winner_v = None
        winner_w = None
        winner_poses = None
        
        for vel in multi_array:
            v = vel[0]
            w = vel[1]
            poses = multi_array[(v, w)]["Poses"]
            
            score = 0
            
            #Determine heading score
            self.heading_score = self.obj_alpha*(1/(sqrt((self.goal_x - poses[-1][0])**2 + (self.goal_y - poses[-1][1])**2)))

            print("Score: ", self.heading_score, " goal: (", self.goal_x, self.goal_y, ")  pose: (", poses[-1][0], poses[-1][1], ") sqrt: ", sqrt((self.goal_x - poses[-1][0])**2 + (self.goal_y - poses[-1][1])**2))
                
            #Determine distance score            
            min_dist = 100000
            
            for x,y,theta in poses:
                for obstacle in self.obstacles:
                    dist = sqrt((x-obstacle[0])**2 +(y-obstacle[1])**2)
                    if dist < min_dist:
                        min_dist = dist
            
                            
            self.distance_score = self.obj_beta*((1/min_dist)**3)
            
            #Determine velocity score
            self.velocity_score = self.obj_gamma*v
            
            # Sum the scores           
            score = self.heading_score + self.distance_score + self.velocity_score

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
        
        self.obstacles.clear()
        for i in range(18):
            self.obstacles.append([random.randint(-7, 7), random.randint(-7, 7)])
        
        # Simulate movement    
        i = 0
        
        x = np.zeros(1)
        y = np.zeros(1)
        x[i] = self.x
        y[i] = self.y
        time = [0]
        
        heading_scores = [0]
        velocity_scores = [0]
        distance_scores = [0]
        total_score =[0]

        
        while i < self.max_iterations and sqrt((self.goal_x-self.x)**2 + (self.goal_y - self.y)**2) > self.goal_th:
            i += 1

            self.dwa()

            x = np.append(x, self.x)
            y = np.append(y, self.y)
            
            time.append(time[-1] + self.dT)
            heading_scores.append(self.heading_score)
            velocity_scores.append(self.velocity_score)
            distance_scores.append(self.distance_score)     
            
            total_score.append(self.heading_score + self.velocity_score + self.distance_score)       
            
            # If at goal
            if sqrt((self.goal_x-self.x)**2 + (self.goal_y - self.y)**2) < self.goal_th:
                print("At goal!")
                print("i =", i)

        
        
        ### Draw obstacles ###
        figure, axes = plt.subplots()
        for i in range(len(self.obstacles)):    
            obs_x = self.obstacles[i][0]
            obs_y = self.obstacles[i][1]
            drawing_circles = plt.Circle( (obs_x, obs_y), 0.2, fill = False )
            goal_circles = plt.Circle( (self.goal_x, self.goal_y), 0.2, color=(0, 1, 0) ,fill = True )
            axes.add_artist(drawing_circles)
            axes.add_artist(goal_circles)
        
        # Plot settings
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        axes.set_aspect(1)
        plt.title("Path")
        plt.grid()
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        
        ### Plot path ###
        plt.plot(x,y)
        
        
        ### Plot x,y (t) ###
        fig1 = plt.figure("Figure 1")
        plt.plot(time, x, color='g', label='x(t)')
        plt.plot(time, y, color='r', label='y(t)')
        plt.legend()
        plt.title("Paths")
        plt.grid()
        plt.xlabel("t [sek]")
        plt.ylabel("Dist [m]")


        ### Plot Scores ###
        fig2 = plt.figure("Figure 2")
        plt.plot(time, velocity_scores, color='g', label='Vel Score')
        plt.plot(time, distance_scores, color='r', label='Dist Score')
        plt.plot(time, heading_scores, color='b', label='Head Score')
        plt.plot(time, total_score, color ='y', label='Total Score')
        plt.legend()
        plt.title("Scores")
        plt.grid()
        plt.xlabel("t [sek]")
        plt.ylabel("Score []")
        
        # Show plot
        plt.show()
        


if __name__ == '__main__':
    try:
        dwa = DWA()
        dwa.simulate_dwa()
    except rospy.ROSInterruptException:
        print("Error")
        