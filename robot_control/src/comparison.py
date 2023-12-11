#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import random
from dynamic_window_approach import DWA
from math import atan2, sqrt, sin, cos, radians, atan2, dist, pi, floor
from numpy.linalg import norm
from safe_artificial_potential_fields import SAPF

def determine_path_length(x, y):
    path_length = 0
    for i in range(len(x)):
        path_length += sqrt(x[i]**2 + y[i]**2)
    return path_length

def determine_total_duration(t):
    return t[-1]

def closest_distance_to_an_object(x, y, obstacles):
    min_dist = 10000
    for i in range(len(obstacles)):
        for j in range(len(x)):
            distance = sqrt((obstacles[i][0] - x[j])**2 + (obstacles[i][1] - y[j])**2)
            if distance < min_dist:
                min_dist = distance
    
    return min_dist

def closest_distance_to_a_human(x,y,humans):
    min_dist = 10000
    for i in range(len(humans)):
        for j in range(len(x)):
            distance = sqrt((humans[i][0][0] - x[j])**2 + (humans[i][0][1] - y[j])**2)
            if distance < min_dist:
                min_dist = distance
    
    return min_dist
    


def simulate(num_simulations):
    
    map_x_cent = 0
    map_y_cent = 0
    
    map_width = 20  
    
    num_humans = 10
    num_obstacles = 20
    
    max_iterations = 500
    goal_th = 0.6
    
    
    for i in range(num_simulations):
        
        xstart = (random.random()- 0.5)*map_width*0.95 + map_x_cent
        ystart = (random.random()- 0.5)*map_width*0.95 + map_y_cent  
        
        xgoal = (random.random()- 0.5)*map_width*0.95 + map_x_cent 
        ygoal = (random.random()- 0.5)*map_width*0.95 + map_y_cent
        
        while dist([xstart, ystart], [xgoal, ygoal]) < map_width*0.95:
            xstart = (random.random()- 0.5)*map_width*0.95 + map_x_cent
            ystart = (random.random()- 0.5)*map_width*0.95 + map_y_cent  
            
            xgoal = (random.random()- 0.5)*map_width*0.95 + map_x_cent 
            ygoal = (random.random()- 0.5)*map_width*0.95 + map_y_cent
        
        thetastart = atan2((ygoal - ystart), (xgoal-xstart))


        obstacles = []
        people = []

        if num_obstacles > 0:    
            for j in range(num_obstacles):
                obstacles.append([(random.random()- 0.5)*map_width*0.8 + map_x_cent, (random.random()- 0.5)*map_width*0.8 + map_y_cent])
            
        if num_humans > 0:
            for j in range(num_humans):
                pos = [(random.random()- 0.5)*map_width*0.8 + map_x_cent, (random.random()- 0.5)*map_width*0.8 + map_y_cent]
                
                dirx = (random.random()-0.5)
                diry = (random.random()-0.5)
                size = sqrt(dirx**2 + diry**2)
                dir = [(1/size) * dirx, (1/size)*diry]

                people.append([pos, dir])
                

        dwa = DWA()
        
        dwa_iter = 0
        
        dwa_x = [xstart]
        dwa_y = [ystart]
        dwa_theta = [thetastart]
        
        dwa_vels = [0]
        dwa_wels = [0]
        
        dwa_time = [0]

        while dwa_iter < max_iterations:
            dwa_iter += 1

            v, w, poses, scores = dwa.dwa(vcur=dwa_vels[-1], wcur=dwa_wels[-1], xcur=dwa_x[-1], ycur=dwa_y[-1], thetacur=dwa_theta[-1], 
                                           xgoal=xgoal, ygoal=ygoal, obstacles=obstacles, people=people)
            
            dwa_vels.append(v)
            dwa_wels.append(w)
            
            dwa_x.append(poses[1][0])
            dwa_y.append(poses[1][1])
            dwa_theta.append(poses[1][2])
            
            dwa_time.append(dwa_time[-1] + dwa.dT)
            
            if dist([poses[1][0], poses[1][1]], [xgoal, ygoal]) < goal_th:
                break    
        
        print(dwa_iter)
        
        dwa_got_to_goal = True
        if dwa_iter == max_iterations:
            dwa_got_to_goal = False
        
        
        
        # sapf = SAPF()
        
        # sapf_iter = 0
        
        # sapf_x = [xstart]
        # sapf_y = [ystart]
        # sapf_theta = [thetastart]
        
        # sapf_vels = [0]
        # sapf_wels = [0]
        
        # sapf_time = [0]

        # while sapf_iter < max_iterations and dwa.stop == False:
        #     sapf_iter += 1

        #     v, w, poses, scores = dwa.dwa(vcur=dwa_vels[-1], wcur=dwa_wels[-1], xcur=dwa_x[-1], ycur=dwa_y[-1], thetacur=dwa_theta[-1], 
        #                                    xgoal=xgoal, ygoal=ygoal, obstacles=obstacles, people=people)
            
        #     sapf_vels.append(v)
        #     sapf_wels.append(w)
            
        #     sapf_x.append(poses[1][0])
        #     sapf_y.append(poses[1][1])
        #     sapf_theta.append(poses[1][2])
            
        #     sapf_time.append(dwa_time[-1] + dwa.dT)

        #     # If at goal
        #     if dist([poses[1][0], poses[1][1]], [xgoal, ygoal]) < goal_th:
        #         dwa.stop = True
        #         print("At goal!")
        #         print("i =", i)
        

if __name__ == "__main__":
    simulate(3)