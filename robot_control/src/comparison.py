#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import random
from dynamic_window_approach import DWA
from math import atan2, sqrt, sin, cos, radians, atan2, dist, pi, floor, ceil
from numpy.linalg import norm
from safe_artificial_potential_fields import SAPF
from human_cost import Human_cost as HC
import csv
import time


def plot_map_and_save_figure(dwa_x, dwa_y, sapf_x, sapf_y, obstacles, humans, start, goal, x_cent, y_cent, map_width, iter, goalth):
    # Plot settings
    figure, axes = plt.subplots()
    plt.xlim([x_cent - 0.55*map_width, x_cent + map_width*0.55])
    plt.ylim([y_cent - 0.55*map_width, y_cent + map_width*0.55])
    axes.set_aspect(1)
    plt.title("Map")
    plt.grid()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    
    ### Plot path ###
    plt.plot(dwa_x,dwa_y, label = "DWA")
    plt.plot(sapf_x,sapf_y, label="SAPF")
    
    for i in range(len(obstacles)):    
        drawing_circles = plt.Circle( (obstacles[i][0], obstacles[i][1]), 0.2, fill = False )
        axes.add_artist(drawing_circles)
    
    for i in range(len(humans)):                 
        drawing_circles = plt.Circle( (humans[i][0][0], humans[i][0][1]), 0.3,  fill = False, color = (1, 0, 0) )
        axes.add_artist(drawing_circles)        
        plt.quiver(humans[i][0][0], humans[i][0][1], humans[i][1][0], humans[i][1][1], scale=5, scale_units="inches", minshaft=2, headlength=5)
    

    plt.plot(goal[0], goal[1], 'g')
    plt.plot(start[0], start[1], 'r')
    
    goal_circles = plt.Circle( (goal[0], goal[1]), goalth, alpha =1.0, color=(0, 1, 0) ,fill = True )
    axes.add_artist(goal_circles)
    start_circles = plt.Circle( (start[0], start[1]), goalth, alpha =1.0, color=(1, 0, 0) ,fill = True )
    axes.add_artist(start_circles) 
    
    axes.legend(framealpha = 0.5)
    
    plt.savefig(f'robot_control/src/comparison_data/map_images/{iter}.png', bbox_inches='tight')

    plt.close('all')
    


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
    
    for j in range(len(x)):
        in_intimate = False
        in_personal = False
        in_social = False    
        for i in range(len(humans)):
            if dist((x[j], y[j]), (humans[i][0][0], humans[i][0][1])) < hc.intimate and in_intimate == False:
                in_intimate = True
            
            elif dist((x[j], y[j]), (humans[i][0][0], humans[i][0][1])) < hc.personal and in_personal == False:
                in_personal = True
                
            elif dist((x[j], y[j]), (humans[i][0][0], humans[i][0][1])) < hc.social and in_social == False:
                in_social = True
            
        if in_intimate:
            time_in_intimate += dT
        elif in_personal:
            time_in_personal += dT
        elif in_social:
            time_in_social += dT
                
                
    return round(time_in_intimate, 2), round(time_in_personal, 2), round(time_in_social, 2)


def completion(iter, max_iter):
    if iter >= max_iter:
        return False
    return True


def simulate(end_i, start_i=0):
    
    header = ['Simulation Number []', 'Path Length [m]', 'Total Duration [s]', 'Smallest Distance to Person [m]', 'Smallest Distance to Obstacle [m]', 
              "Time in Intimate Space [s]", "Time in Personal Space [s]", "Time in Social-Consultive Space [s]", "Average Execution Time [s]"]
    
    with open('robot_control/src/comparison_data/sim_data/dwa.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

    with open('robot_control/src/comparison_data/sim_data/sapf.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)
    
    map_x_cent = 0
    map_y_cent = 0
    
    map_width = 20  
    
    num_humans = 10
    num_obstacles = 20

    goal_th = 0.6
    
    max_time = 60    

    dwa_dT = 0.25
    sapf_dT = 0.1

    dwa_max_iterations = max_time/dwa_dT
    sapf_max_iterations = max_time/sapf_dT
    
    
    dwa_stats = {}
    sapf_stats = {}
    total_path = 0
    
    
    dwa_breaks = 0
    sapf_breaks = 0
    
    for i in range(start_i, end_i):
        print("Simulation", i+1 ,"of", end_i)
        
        xstart = (random.random()- 0.5)*map_width*0.95 + map_x_cent
        ystart = (random.random()- 0.5)*map_width*0.95 + map_y_cent  
        
        xgoal = (random.random()- 0.5)*map_width*0.95 + map_x_cent 
        ygoal = (random.random()- 0.5)*map_width*0.95 + map_y_cent
        
        while dist([xstart, ystart], [xgoal, ygoal]) < map_width*0.95:
            xstart = (random.random()- 0.5)*map_width*0.95 + map_x_cent
            ystart = (random.random()- 0.5)*map_width*0.95 + map_y_cent  
            
            xgoal = (random.random()- 0.5)*map_width*0.95 + map_x_cent 
            ygoal = (random.random()- 0.5)*map_width*0.95 + map_y_cent
        
        total_path += dist([xgoal,ygoal],[xstart,ystart])
        
        thetastart = (random.random()-0.5)*2*pi

        # Hardcoded start and goal
        # xstart, ystart = -9,-9
        # xgoal, ygoal = 9,9
        # thetastart = atan2(ygoal-ystart, xgoal-xstart)

        empty_list = []
        obstacles = []
        people = []
        obstacles_and_people = []

        # Walls        
        # for wx in np.arange(floor(map_x_cent - 0.5*map_width), ceil(map_x_cent + map_width*0.5), 0.1):
        #     obstacles.append([wx, map_y_cent - 0.5*map_width])
        #     obstacles.append([wx, map_y_cent + 0.5*map_width])
        # for wy in np.arange(floor(map_y_cent - 0.5*map_width), ceil(map_y_cent + map_width*0.5), 0.1):
        #     obstacles.append([map_x_cent - 0.5*map_width, wy])
        #     obstacles.append([map_x_cent + 0.5*map_width, wy])


        # Obstacles
        if num_obstacles > 0:    
            for j in range(num_obstacles):
                x_obs = (random.random()- 0.5)*map_width*0.8 + map_x_cent
                y_obs = (random.random()- 0.5)*map_width*0.8 + map_y_cent
                while dist([x_obs, y_obs], [xgoal, ygoal]) < 3 or dist([x_obs, y_obs], [xstart, ystart]) < 3:
                    x_obs = (random.random()- 0.5)*map_width*0.8 + map_x_cent
                    y_obs = (random.random()- 0.5)*map_width*0.8 + map_y_cent
                obstacles.append([x_obs,  y_obs])
                obstacles_and_people.append([x_obs,  y_obs])
        # obstacles = [[3.4620780011110384, -6.022094011188084], [-4.009974079787394, -0.6485367921078513], [-7.36096762363364, 7.216330086697337], [5.608744401297672, 1.850811724550809], [3.8458299368470623, 3.357341808539937], [-4.778073368033663, -5.501141866315759], [4.723132838747985, 0.3845932731672015], [-2.229574295791508, 2.036716162637534], [4.171285482646521, 3.7728185792673266], [-6.218284379822736, 5.97907738914949], [4.154923208102396, 1.8794276841810884], [-2.9544228765998177, 0.4480890828980719], [0.8345912954350894, 0.9778563383034644], [-6.743270413377292, 0.4883006991849541], [-5.667493154441951, -7.785785428113856], [-0.35529269417974874, -1.6953325996094204], [-6.258261636260494, -7.750274910776324], [5.673472479630764, -1.7661319837998537], [-1.8415642436138828, 6.773339898825298], [3.224003003417552, 3.289856053257889]]
            
        # Humans
        if num_humans > 0:
            for j in range(num_humans):
                
                x_hum = (random.random()- 0.5)*map_width*0.8 + map_x_cent
                y_hum = (random.random()- 0.5)*map_width*0.8 + map_y_cent
                while dist([x_hum, y_hum], [xgoal, ygoal]) < 3 or dist([x_hum, y_hum], [xstart, ystart]) < 3:
                    x_hum = (random.random()- 0.5)*map_width*0.8 + map_x_cent
                    y_hum = (random.random()- 0.5)*map_width*0.8 + map_y_cent
                
                pos = [x_hum, y_hum]
                
                dirx = (random.random()-0.5)
                diry = (random.random()-0.5)
                size = sqrt(dirx**2 + diry**2)
                dir = [(1/size) * dirx, (1/size)*diry]

                people.append([pos, dir])
                obstacles_and_people.append(pos)
        # people = [[[-7.628100894388067, 3.4154098195069467], [0.8049132248720982, 0.5933925348586712]], [[-5.466481775607566, -6.896318233696028], [-0.010661367814394045, -0.9999431660031114]], [[-3.312477282810234, -2.0009274781740807], [-0.6587384015591221, -0.7523720610916734]], [[3.5389702000507945, 0.05233796970694371], [0.9495008281317766, -0.31376452536427735]], [[4.927280407180085, 5.4040645428380785], [-0.41194351355772246, -0.9112093840812432]], [[-2.6019992670325376, -6.628660260019831], [-0.9304019911509825, 0.3665407683495615]], [[3.8934643935681468, -3.7715344243867435], [0.8467107583147311, -0.5320534670069291]], [[-4.49657147274171, -2.0257417935565734], [0.9215614591575776, 0.3882325037852399]], [[7.054064494277485, -6.948301300913004], [-0.9998381819074005, -0.01798916340755726]], [[6.089431096668327, -3.785125599146481], [0.9005975797225737, 0.43465388459996807]]]

        
                

        ### DWA ###
        dwa = DWA(simT=1, dT=dwa_dT)
        
        dwa_iter = 0
        
        dwa_x = [xstart]
        dwa_y = [ystart]
        dwa_theta = [thetastart]
        
        dwa_vels = [0]
        dwa_wels = [0]
        
        dwa_time = [0]
               
        complete = False
        
        time_start = time.time()

        while dwa_iter < dwa_max_iterations:
            dwa_iter += 1


            # Using people
            v, w, poses, scores = dwa.dwa(vcur=dwa_vels[-1], wcur=dwa_wels[-1], xcur=dwa_x[-1], ycur=dwa_y[-1], thetacur=dwa_theta[-1], 
                                           xgoal=xgoal, ygoal=ygoal, obstacles=obstacles, people=people)
            # Using people as obstacles
            # v, w, poses, scores = dwa.dwa(vcur=dwa_vels[-1], wcur=dwa_wels[-1], xcur=dwa_x[-1], ycur=dwa_y[-1], thetacur=dwa_theta[-1], 
            #                                xgoal=xgoal, ygoal=ygoal, obstacles=obstacles_and_people, people=empty_list)
            
            
            dwa_vels.append(v)
            dwa_wels.append(w)
            
            dwa_x.append(poses[1][0])
            dwa_y.append(poses[1][1])
            dwa_theta.append(poses[1][2])
            
            dwa_time.append(dwa_time[-1] + dwa.dT)
            
            if dist([poses[1][0], poses[1][1]], [xgoal, ygoal]) < goal_th:
                complete = True
                break
                
        dwa_average_elapsed_time = (time.time() - time_start)/dwa_iter
        print("DWA time: ", dwa_average_elapsed_time*dwa_iter)
        
        time_in_intimate, time_in_personal, time_in_social = time_in_social_shapes(dwa_x, dwa_y, people, dwa_dT)
            
        path_length = determine_path_length(dwa_x, dwa_y)
        
        total_duration = determine_total_duration(dwa_time)
        
        closest_dist_human = closest_distance_to_a_human(dwa_x, dwa_y, people)
        
        closest_dist_obstacle = closest_distance_to_an_object(dwa_x, dwa_y, obstacles)
    
    
        if complete:
            with open('robot_control/src/comparison_data/sim_data/dwa.csv', 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                
                row = [i, path_length, total_duration, closest_dist_human, closest_dist_obstacle, time_in_intimate, time_in_personal, time_in_social, dwa_average_elapsed_time]
                writer.writerow(row)
        else:
            sapf_breaks += 1



        ### SAPF ###
        sapf = SAPF()
        sapf.dT = sapf_dT
        
        # Using people
        sapf.init(new_goal=np.array([xgoal,ygoal]), new_start_pos=np.array([xstart, ystart]), new_start_theta=thetastart, new_obstacles=np.array(obstacles), new_humans=np.array(people)) 
        
        # Using people as obstacles
        # sapf.init(new_goal=np.array([xgoal,ygoal]), new_start_pos=np.array([xstart, ystart]), new_start_theta=thetastart, new_obstacles=np.array(obstacles_and_people), new_humans=np.array(empty_list)) 
        
        
        sapf_iter = 0

        sapf_x = [xstart]
        sapf_y = [ystart]
        sapf_theta = [thetastart]
        
        sapf_vels = [0]
        sapf_wels = [0]
        
        sapf_time = [0]

        
        complete = False
        
        time_start = time.time()
                
        while sapf_iter < sapf_max_iterations:
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
            
            sapf_time.append(sapf_time[-1] + sapf.dT)
            
            # If at goal
            if dist(sapf.pos, [xgoal, ygoal]) < goal_th:
                complete = True
                break
        
        sapf_average_elapsed_time = (time.time() - time_start)/sapf_iter
        print("SAPF time:", sapf_average_elapsed_time*sapf_iter)
        
        time_in_intimate, time_in_personal, time_in_social = time_in_social_shapes(sapf_x, sapf_y, people, sapf_dT)
            
        path_length = determine_path_length(sapf_x, sapf_y)
        
        total_duration = determine_total_duration(sapf_time)
        
        closest_dist_human = closest_distance_to_a_human(sapf_x, sapf_y, people)
        
        closest_dist_obstacle = closest_distance_to_an_object(sapf_x, sapf_y, obstacles)
    
    
        if complete:
            with open('robot_control/src/comparison_data/sim_data/sapf.csv', 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)

                row = [i, path_length, total_duration, closest_dist_human, closest_dist_obstacle, time_in_intimate, time_in_personal, time_in_social, sapf_average_elapsed_time]
                writer.writerow(row)
        else:
            sapf_breaks += 1
        
        plot_map_and_save_figure(dwa_x, dwa_y, sapf_x, sapf_y, obstacles,people, [xstart, ystart], [xgoal, ygoal], map_x_cent, map_y_cent, map_width, i, goal_th)

        
    print("Avg shortest path length:", total_path/(end_i - start_i))
        
    with open('robot_control/src/comparison_data/sim_data/dwa.txt', 'w', encoding='UTF8', newline='') as f:
        f.write(f"Number of breaks: {dwa_breaks} \n")
        f.write(f"Total number of simulations: {end_i} \n")
               
        
    with open('robot_control/src/comparison_data/sim_data/sapf.txt', 'w', encoding='UTF8', newline='') as f:
        f.write(f"Number of breaks: {sapf_breaks} \n")
        f.write(f"Total number of simulations: {end_i} \n")
        

if __name__ == "__main__":
    simulate(start_i=0, end_i=500)