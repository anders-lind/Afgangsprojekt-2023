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

class DWA:
    def __init__(self, dT = 0.1, simT = 1.0, vPrec = 0.05, wPrec = 0.05, goal_th = 0.3):        
        
        self.map_y_cent = 0
        self.map_x_cent = 0
        self.map_width = 20
        
        self.simT = simT
        self.vPrec = vPrec
        self.wPrec = wPrec
        self.dT = dT
        self.N = int(floor(self.simT/self.dT))
        
        self.obj_alpha = 0.2
        self.obj_beta = -0.2
        self.obj_gamma = 0.2 
        self.obj_eta = 1.0
            
        self.cost = HC()
        
        self.max_iterations= 1000
        self.goal_th = goal_th
        
        self.a_max = 1.5 #m/s²
        self.alpha_max = 1 #rad/s²
        
        self.v_max = 1.5 #m/s
        self.v_min = 0 #m/s
        self.w_max = 1.0 #rad/s
        self.w_min = - self.w_max #rad/s    
        
        self.stop = False
        
    
    def dwa(self, vcur, wcur, xcur, ycur, thetacur, xgoal, ygoal, obstacles, people):
        vels_and_poses = []
        
        #Dynamic window
        vr_min = max(self.v_min, vcur - self.a_max*self.dT)
        vr_max = min(self.v_max, vcur + self.a_max*self.dT)
        wr_min = max(self.w_min, wcur - self.alpha_max*self.dT)
        wr_max = min(self.w_max, wcur + self.alpha_max*self.dT)
        print("a: ", self.a_max, ", dt: ",  self.dT, ", alpha : ", self.alpha_max)
        print("w: ", wcur, ", v: ", vcur)
        print("v: ", vr_min, vr_max, ", w: ", wr_min, wr_max)
        #Determine poses fo each v,w
        for v in np.arange(vr_min, vr_max, self.vPrec):
            for w in np.arange(wr_min, wr_max, self.wPrec):
                
                theta_vals = []
                x_vals = []
                y_vals = []
                
                #Determine theta
                for i in range  (0, self.N):
                    theta_i = thetacur + self.dT*w*i
                    theta_i = atan2(sin(theta_i), cos(theta_i))
                    theta_vals.append(theta_i)
            
                #Determine x and y
                for i in range(0, self.N):
                    cosSum = 0
                    sinSum = 0
                    for j in range(0, i):
                        cosSum += cos(theta_vals[j])
                        sinSum += sin(theta_vals[j])

                    y_i = ycur + self.dT*i*v*sinSum
                    x_i = xcur + self.dT*i*v*cosSum
                    
                    x_vals.append(x_i)
                    y_vals.append(y_i)
                
                
                #creating array of poses
                poses = []
                
                for i in range(0, self.N):
                    pose = (x_vals[i], y_vals[i], theta_vals[i])  
                    poses.append(pose)
                    
                vels_and_poses.append({"v": v, "w":w, "p":poses})
        
        
        #Determining scores for velocities
        
        max_score = float("-inf")
        
        winner_v = None
        winner_w = None
        winner_poses = None
        for i in range(len(vels_and_poses)):
            v = vels_and_poses[i]["v"]
            w = vels_and_poses[i]["w"]
            
            poses = vels_and_poses[i]["p"]
            
            score = 0 
            human_score = 0
            min_dist = 100000
            change_in_distance_to_goal = -100000
            
            with_people = False
        
            if len(people) > 0:
                with_people = True
            
            with_obstacles = False
            
            if len(obstacles) > 0:
                with_obstacles = True
                
                
                
            for i in range(len(poses)):
                
                x = poses[i][0]
                y = poses[i][1]
                
                if with_obstacles:                 
                    for i in range(len(obstacles)):
                        obs_dist = dist([x, y], [obstacles[i][0], obstacles[i][1]])
                        
                        if obs_dist < min_dist:
                            min_dist = obs_dist

                if with_people:
                    for i in range(len(people)):
                        vec = [x - people[i][0][0], y - people[i][0][1]]
                        people_direc = [people[i][1][0], people[i][1][1]]
                        
                        price = self.cost.get_cost_xy(vec[0], vec[1], people_direc[0], people_direc[1])
                        human_score -= price

                dist_goal = dist([xcur, ycur], [xgoal, ygoal]) - dist([x, y], [xgoal, ygoal])
                                    
                if dist_goal > change_in_distance_to_goal:
                    change_in_distance_to_goal = dist_goal
            
            if dist([poses[1][0], poses[1][1]], [xgoal, ygoal]) < self.goal_th:
                self.stop = True

            
            heading_score = self.obj_alpha*dist_goal
            
            velocity_score = self.obj_gamma*abs(v)
            
            distance_score = 0
            
            if with_obstacles:
                distance_score = self.obj_beta/min_dist**2
            
            people_score = 0
            
            if with_people:
                people_score = self.obj_eta*human_score
                
            total_score = distance_score + heading_score + velocity_score + people_score

            scores = [distance_score, heading_score, velocity_score, people_score, total_score]
            
            if total_score > max_score:
                max_score = score
                winner_v = v
                winner_w = w
                winner_poses = poses    
     
        return winner_v, winner_w, winner_poses, scores  
            
            
    def simulate_dwa(self, num_obstacles = 10, num_people = 10):
                
        xstart = (random.random()- 0.5)*self.map_width*0.95 + self.map_x_cent
        ystart = (random.random()- 0.5)*self.map_width*0.95 + self.map_y_cent  
        
        xgoal = (random.random()- 0.5)*self.map_width*0.95 + self.map_x_cent 
        ygoal = (random.random()- 0.5)*self.map_width*0.95 + self.map_y_cent
        
        while dist([xstart, ystart], [xgoal, ygoal]) < self.map_width*0.9:
            xstart = (random.random()- 0.5)*self.map_width*0.95 + self.map_x_cent
            ystart = (random.random()- 0.5)*self.map_width*0.95 + self.map_y_cent  
            
            xgoal = (random.random()- 0.5)*self.map_width*0.95 + self.map_x_cent 
            ygoal = (random.random()- 0.5)*self.map_width*0.95 + self.map_y_cent
        
        thetastart = atan2((ygoal - ystart), (xgoal-xstart))
        
        
        figure, axes = plt.subplots()
        
        obstacles = []
        people = []
        
        with_obstacles = False
        with_people = False
        
        if num_obstacles > 0:    
            for i in range(num_obstacles):
                obstacles.append([(random.random()- 0.5)*self.map_width*0.8 + self.map_x_cent, (random.random()- 0.5)*self.map_width*0.8 + self.map_y_cent])
            
            for i in range(len(obstacles)):    
                drawing_circles = plt.Circle( (obstacles[i][0], obstacles[i][1]), 0.2, fill = False )
                axes.add_artist(drawing_circles)
            
            with_obstacles = True
            
            

        if num_people > 0:
            for i in range(num_people):
                pos = [(random.random()- 0.5)*self.map_width*0.8 + self.map_x_cent, (random.random()- 0.5)*self.map_width*0.8 + self.map_y_cent]
                
                dirx = (random.random()-0.5)
                diry = (random.random()-0.5)
                size = sqrt(dirx**2 + diry**2)
                dir = [(1/size) * dirx, (1/size)*diry]

                people.append([pos, dir])
                
            for i in range(len(people)):                 
                drawing_circles = plt.Circle( (people[i][0][0], people[i][0][1]), 0.1, fill = True, color = (1, 0, 0) )
                axes.add_artist(drawing_circles)
                
                plt.quiver(people[i][0][0], people[i][0][1], people[i][1][0], people[i][1][1], scale=5, scale_units="inches", minshaft=2, headlength=5)
            
            with_people = True
        
        
        print("Start pos: (", xstart, ystart, ")")
        print("Goal pos: (", xgoal, ygoal, ")")
        


        plt.plot(xgoal, ygoal, 'g*')
        plt.plot(xstart, ystart, 'r*')
        
        goal_circles = plt.Circle( (xgoal, ygoal), 0.2, color=(0, 1, 0) ,fill = True )
        axes.add_artist(goal_circles)    
            
        # Simulate movement    
        i = 0
        
        x = [xstart]
        y = [ystart]
        theta = [thetastart]
        
        vels = [0]
        wels = [0]
        
        time = [0]
        
        heading_scores = [0]
        velocity_scores = [0]
        distance_scores = [0]
        people_scores = [0]
        total_score =[0]

        while i < self.max_iterations and self.stop == False:
            i += 1

            v, w, poses, scores = self.dwa(vcur=vels[-1], wcur=wels[-1], xcur=x[-1], ycur=y[-1], thetacur=theta[-1], 
                                           xgoal=xgoal, ygoal=ygoal, obstacles=obstacles, people=people)
            
            vels.append(v)
            wels.append(w)
            
            x.append(poses[1][0])
            y.append(poses[1][1])
            theta.append(poses[1][2])
            
            time.append(time[-1] + self.dT)


            heading_scores.append(scores[1])
            velocity_scores.append(scores[2])
            
            if with_obstacles:
                distance_scores.append(scores[0])
                 
            if with_people: 
                people_scores.append(scores[3])
            
            total_score.append(scores[4])      
            
            # If at goal
            if dist([poses[1][0], poses[1][1]], [xgoal, ygoal]) < self.goal_th:
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
        plt.plot(time, heading_scores, color='b', label='Head Score')
        
        if with_people:
            plt.plot(time, people_scores, color='c', label='Human Score')
        
        if with_obstacles:
            plt.plot(time, distance_scores, color='r', label='Dist Score')
            
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
        dwa = DWA()
        dwa.simulate_dwa(num_obstacles=0, num_people=0)
        
    except rospy.ROSInterruptException:
        print("Error")
        
        
        

# vector_1 = [0, 1]
# vector_2 = [1, 0]

# unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
# unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
# dot_product = np.dot(unit_vector_1, unit_vector_2)
# angle = np.arccos(dot_product)
# score = -1*(angle/pi)
# print(angle)
