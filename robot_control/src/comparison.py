#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import random
from dynamic_window_approach import DWA
from math import atan2, sqrt, sin, cos, radians, atan2, dist, pi, floor
from numpy.linalg import norm
from safe_artificial_potential_fields import SAPF
from human_cost import Human_cost as HC

def determine_path_length(x, y):
    path_length = 0
    for i in range(1, len(x)):
        path_length += sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
    return round(path_length, 2)


def determine_total_duration(t):
    return round(t[-1], 2)


def closest_distance_to_an_object(x, y, obstacles):
    min_dist = 10000
    for i in range(len(obstacles)):
        for j in range(len(x)):
            distance = sqrt((obstacles[i][0] - x[j])**2 + (obstacles[i][1] - y[j])**2)
            if distance < min_dist:
                min_dist = distance
    
    return round(min_dist, 2)


def closest_distance_to_a_human(x,y,humans):
    min_dist = 10000
    for i in range(len(humans)):
        for j in range(len(x)):
            distance = sqrt((humans[i][0][0] - x[j])**2 + (humans[i][0][1] - y[j])**2)
            if distance < min_dist:
                min_dist = distance
    
    return round(min_dist, 2)


def time_in_social_shapes(x, y, humans, dT):
    
    hc = HC()
    time_in_intimate = 0
    time_in_personal = 0
    time_in_social = 0
    
    for i in range(len(humans)):
        for j in range(len(x)):
            if dist((x[j], y[j]), (humans[i][0][0], humans[i][0][1])) < hc.intimate:
                time_in_intimate += dT
            elif dist((x[j], y[j]), (humans[i][0][0], humans[i][0][1])) < hc.personal:
                time_in_personal += dT
            elif dist((x[j], y[j]), (humans[i][0][0], humans[i][0][1])) < hc.social:
                time_in_social += dT
    
    return round(time_in_intimate, 2), round(time_in_personal, 2), round(time_in_social, 2)


def completion(iter, max_iter):
    if iter >= max_iter:
        return False
    return True


def simulate(num_simulations):
    
    map_x_cent = 0
    map_y_cent = 0
    
    map_width = 20  
    
    num_humans = 10
    num_obstacles = 20
    
    max_iterations = 500
    goal_th = 0.6
    dT = 0.1
    
    
    dwa_stats = {}
    sapf_stats = {}
    
    
    dwa_breaks = 0
    sapf_breaks = 0
    
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
                

        ### DWA ###
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
        
        complete = completion(dwa_iter, max_iterations)
        
        time_in_intimate, time_in_personal, time_in_social = time_in_social_shapes(dwa_x, dwa_y, people, dT)
            
        path_length = determine_path_length(dwa_x, dwa_y)
        
        total_duration = determine_total_duration(dwa_time)
        
        closest_dist_human = closest_distance_to_a_human(dwa_x, dwa_y, people)
        
        closest_dist_obstacle = closest_distance_to_an_object(dwa_x, dwa_y, obstacles)
    
    
        if complete and closest_dist_human > 0.3 and closest_dist_obstacle > 0.05:
            
            dwa_stats[i] = [path_length, total_duration, closest_dist_human, closest_dist_obstacle, time_in_intimate, time_in_personal, time_in_social]
        else:
            dwa_breaks += 1





        ### SAPF ###
        sapf = SAPF(goal=np.array([xgoal,ygoal]), start_pos=np.array([xstart, ystart]), start_theta=thetastart,
                    obstacles=np.array(obstacles), humans=np.array(people), goal_th=goal_th)
        
        sapf_iter = 0
        
        sapf_x = [xstart]
        sapf_y = [ystart]
        sapf_theta = [thetastart]
        
        sapf_vels = [0]
        sapf_wels = [0]
        
        sapf_time = [0]

        while sapf_iter < max_iterations:
            sapf_iter += 1
            
            # Calculate potentials and move robot
            pot = sapf.calc_potential(pos=sapf.pos, goal=sapf.goal)
            v_ref, theta_ref = sapf.calc_ref_values(pot)
            sapf.simulate_robot_step(v_ref=v_ref, theta_ref=theta_ref)
            
            sapf_vels.append(sapf.vel)
            sapf_wels.append(sapf.omega)
            
            sapf_x.append(sapf.pos[0])
            sapf_y.append(sapf.pos[1])
            sapf_theta.append(sapf.theta)
            
            sapf_time.append(sapf_time[-1] + sapf.time_step_size)
            
            # If at goal
            if dist(sapf.pos, [xgoal, ygoal]) < goal_th:
                break
            
            
        complete = completion(sapf_iter, max_iterations)
        
        time_in_intimate, time_in_personal, time_in_social = time_in_social_shapes(sapf_x, sapf_y, people, dT)
            
        path_length = determine_path_length(sapf_x, sapf_y)
        
        total_duration = determine_total_duration(sapf_time)
        
        closest_dist_human = closest_distance_to_a_human(sapf_x, sapf_y, people)
        
        closest_dist_obstacle = closest_distance_to_an_object(sapf_x, sapf_y, obstacles)
    
    
        if complete and closest_dist_human > 0.3 and closest_dist_obstacle > 0.05:
            
            sapf_stats[i] = [path_length, total_duration, closest_dist_human, closest_dist_obstacle, time_in_intimate, time_in_personal, time_in_social]
        else:
            sapf_breaks += 1

        
        
    print("Sapf stats: ", sapf_stats)
    print("Num of sapf_breaks: ", sapf_breaks)

    print("Dwa stats: ", dwa_stats)
    print("Num of dwa_breaks: ", dwa_breaks)

        

if __name__ == "__main__":
    simulate(3)