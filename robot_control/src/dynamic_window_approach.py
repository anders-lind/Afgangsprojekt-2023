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
    def __init__(self, dT = 0.25, simT = 2.0, vPrec = 0.1, wPrec = 0.1, goal_th = 0.6):        
        
        self.map_y_cent = 0
        self.map_x_cent = 0
        self.map_width = 30
        
        self.simT = simT
        self.vPrec = vPrec
        self.wPrec = wPrec
        self.dT = dT
        self.N = int(floor(self.simT/self.dT))
        
        self.obj_heading =   0.3
        self.obj_speed = 1.3
        self.obj_obstacle = 0.6
        
        self.obj_people =  4
        
        self.obj_th = 0.5
            
        self.cost = HC()
        
        self.max_iterations= 600
        self.goal_th = goal_th
        
        self.a_max = 5 #m/s²
        self.alpha_max = pi #rad/s²
        
        self.v_max = 1.5 #m/s
        self.v_min = 0 #m/s
        self.w_max = pi/2 #rad/s
        self.w_min = -self.w_max #rad/s    
        
        self.stop = False
        
    
    def dwa(self, vcur, wcur, xcur, ycur, thetacur, xgoal, ygoal, obstacles, people):
        vels_and_poses = []
        
        #Dynamic window
        vr_min = max(self.v_min, vcur - self.a_max*self.dT)
        vr_max = min(self.v_max, vcur + self.a_max*self.dT)
        wr_min = max(self.w_min, wcur - self.alpha_max*self.dT)
        wr_max = min(self.w_max, wcur + self.alpha_max*self.dT)

        #Determine poses fo each v,w
        for v in np.arange(vr_min, vr_max, self.vPrec):
            for w in np.arange(wr_min, wr_max, self.wPrec):
                
                theta_vals = [thetacur]
                x_vals = [xcur]
                y_vals = [ycur]
                
                #Determine theta
                for i in range  (1, self.N):
                    theta_i = thetacur + self.dT*w*i
                    theta_vals.append(theta_i)
            
                #Determine x and y
                for i in range(1, self.N):
                    x_i = x_vals[i-1] + self.dT*v*cos(theta_vals[0] + i*w*self.dT)
                    y_i = y_vals[i-1] + self.dT*v*sin(theta_vals[0] + i*w*self.dT)
                    
                    x_vals.append(x_i)
                    y_vals.append(y_i)
                
                #creating array of poses
                poses = []
                
                for i in range(0, self.N):
                    pose = (x_vals[i], y_vals[i], theta_vals[i])  
                    poses.append(pose)
    
                vels_and_poses.append({"v": v, "w":w, "p":poses})
        
        
        #Determining scores for velocities
        
        max_score = 100000
        
        winner_v = None
        winner_w = None
        winner_poses = None
        
        for i in range(len(vels_and_poses)):
            v = vels_and_poses[i]["v"]
            w = vels_and_poses[i]["w"]
            poses = vels_and_poses[i]["p"]
        
            
            ###    OBSTACLE SCORE    ###
            
            obstacle_score = 0
            
            if len(obstacles) > 0:
                minr = float("Inf")
                
                for j in range(len(obstacles)):
                    for h in range(self.N):
                        dx = poses[h][0] - obstacles[j][0]
                        dy = poses[h][1] - obstacles[j][1]
                        
                        r = sqrt(dx**2 + dy**2)
                        
                        if r < minr:
                            minr = r
                
                obstacle_score = self.obj_obstacle*(1/minr**2)
            
            
            ###   PEOPLE SCORE  ###
            
            people_score = 0
            
            if len(people) > 0:
                human_score = 0
                for j in range(len(people)):
                    for h in range(self.N):
                        vec = [poses[h][0] - people[j][0][0], poses[h][1] - people[j][0][1]]
                        people_direc = [people[j][1][0], people[j][1][1]]
                        
                        price = self.cost.get_cost_xy(vec[0], vec[1], people_direc[0], people_direc[1])
                        if price > human_score:
                            human_score = price
                
                people_score = self.obj_people*human_score      
                            
                
            
            angle_error = poses[self.N-1][2] - atan2(ygoal- poses[self.N-1][1], xgoal - poses[self.N-1][0])
            heading = abs(atan2(sin(angle_error), cos(angle_error)))
            heading_score =  self.obj_heading*heading
            
            
            velocity_score = self.obj_speed*(self.v_max - v)



            total_score = velocity_score + heading_score + obstacle_score + people_score

            scores = [obstacle_score, heading_score, velocity_score, people_score, total_score]
            
            if total_score < max_score:
                max_score = total_score
                winner_v = v
                winner_w = w
                winner_poses = poses
                scores = scores

        return winner_v, winner_w, winner_poses, scores  
            
            
    def simulate_dwa(self, num_obstacles = 10, num_people = 10):
                
        xstart = (random.random()- 0.5)*self.map_width*0.95 + self.map_x_cent
        ystart = (random.random()- 0.5)*self.map_width*0.95 + self.map_y_cent  
        
        xgoal = (random.random()- 0.5)*self.map_width*0.95 + self.map_x_cent 
        ygoal = (random.random()- 0.5)*self.map_width*0.95 + self.map_y_cent
        
        while dist([xstart, ystart], [xgoal, ygoal]) < self.map_width*0.95:
            xstart = (random.random()- 0.5)*self.map_width*0.95 + self.map_x_cent
            ystart = (random.random()- 0.5)*self.map_width*0.95 + self.map_y_cent  
            
            xgoal = (random.random()- 0.5)*self.map_width*0.95 + self.map_x_cent 
            ygoal = (random.random()- 0.5)*self.map_width*0.9 + self.map_y_cent
        
        
        xstart = -9
        ystart = -9
        
        xgoal = 9
        ygoal = 9
        
        # thetastart = atan2((ygoal - ystart), (xgoal-xstart))
        thetastart = (random.random() - 0.5)*2*pi
        
        print("theta: ", thetastart)
        print("xgoal: ", xgoal)
        print("ygoal: ", ygoal)
        print("xstart: ", xstart)
        print("ystart: ", ystart)
        
        
        
        figure, axes = plt.subplots()
        
        obstacles = []
        people = []
        
        with_obstacles = False
        with_people = False
        
        
        # people = [[[-7.628100894388067, 3.4154098195069467], [0.8049132248720982, 0.5933925348586712]], [[-5.466481775607566, -6.896318233696028], [-0.010661367814394045, -0.9999431660031114]], [[-3.312477282810234, -2.0009274781740807], [-0.6587384015591221, -0.7523720610916734]], [[3.5389702000507945, 0.05233796970694371], [0.9495008281317766, -0.31376452536427735]], [[4.927280407180085, 5.4040645428380785], [-0.41194351355772246, -0.9112093840812432]], [[-2.6019992670325376, -6.628660260019831], [-0.9304019911509825, 0.3665407683495615]], [[3.8934643935681468, -3.7715344243867435], [0.8467107583147311, -0.5320534670069291]], [[-4.49657147274171, -2.0257417935565734], [0.9215614591575776, 0.3882325037852399]], [[7.054064494277485, -6.948301300913004], [-0.9998381819074005, -0.01798916340755726]], [[6.089431096668327, -3.785125599146481], [0.9005975797225737, 0.43465388459996807]]]
        # obstacles = [[3.4620780011110384, -6.022094011188084], [-4.009974079787394, -0.6485367921078513], [-7.36096762363364, 7.216330086697337], [5.608744401297672, 1.850811724550809], [3.8458299368470623, 3.357341808539937], [-4.778073368033663, -5.501141866315759], [4.723132838747985, 0.3845932731672015], [-2.229574295791508, 2.036716162637534], [4.171285482646521, 3.7728185792673266], [-6.218284379822736, 5.97907738914949], [4.154923208102396, 1.8794276841810884], [-2.9544228765998177, 0.4480890828980719], [0.8345912954350894, 0.9778563383034644], [-6.743270413377292, 0.4883006991849541], [-5.667493154441951, -7.785785428113856], [-0.35529269417974874, -1.6953325996094204], [-6.258261636260494, -7.750274910776324], [5.673472479630764, -1.7661319837998537], [-1.8415642436138828, 6.773339898825298], [3.224003003417552, 3.289856053257889]]
            

        if num_obstacles > 0:  
            with_obstacles = True  
            for j in range(num_obstacles):
                x_obs = (random.random()- 0.5)*self.map_width*0.8 + self.map_x_cent
                y_obs = (random.random()- 0.5)*self.map_width*0.8 + self.map_y_cent
                while dist([x_obs, y_obs], [xgoal, ygoal]) < 3 or dist([x_obs, y_obs], [xstart, ystart]) < 3:
                    x_obs = (random.random()- 0.5)*self.map_width*0.8 + self.map_x_cent
                    y_obs = (random.random()- 0.5)*self.map_width*0.8 + self.map_y_cent
                obstacles.append([x_obs,  y_obs])
        
        for i in range(len(obstacles)):    
            drawing_circles = plt.Circle( (obstacles[i][0], obstacles[i][1]), 0.2, fill = False )
            axes.add_artist(drawing_circles)
        
        if num_people > 0:
            with_people = True
            for j in range(num_people):
                
                x_hum = (random.random()- 0.5)*self.map_width*0.8 + self.map_x_cent
                y_hum = (random.random()- 0.5)*self.map_width*0.8 + self.map_y_cent
                while dist([x_hum, y_hum], [xgoal, ygoal]) < 3 or dist([x_hum, y_hum], [xstart, ystart]) < 3:
                    x_hum = (random.random()- 0.5)*self.map_width*0.8 + self.map_x_cent
                    y_hum = (random.random()- 0.5)*self.map_width*0.8 + self.map_y_cent
                
                pos = [x_hum, y_hum]
                
                dirx = (random.random()-0.5)
                diry = (random.random()-0.5)
                size = sqrt(dirx**2 + diry**2)
                dir = [(1/size) * dirx, (1/size)*diry]

                people.append([pos, dir])
            
            for i in range(len(people)):                 
                drawing_circles = plt.Circle( (people[i][0][0], people[i][0][1]), self.cost.person_size,  fill = False, color = (1, 0, 0) )
                axes.add_artist(drawing_circles)
                plt.quiver(people[i][0][0], people[i][0][1], people[i][1][0], people[i][1][1], scale=5, scale_units="inches", minshaft=2, headlength=5)
    
        print("Start pos: (", xstart, ystart, ")")
        print("Goal pos: (", xgoal, ygoal, ")")
        

        plt.plot(xgoal, ygoal, 'g')
        plt.plot(xstart, ystart, 'r')
        
        goal_circles = plt.Circle( (xgoal, ygoal), self.goal_th, alpha =1.0, color=(0, 1, 0) ,fill = True )
        axes.add_artist(goal_circles)
        start_circles = plt.Circle( (xstart, ystart), self.goal_th, alpha =1.0, color=(1, 0, 0) ,fill = True )
        axes.add_artist(start_circles)        
            
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
                self.stop = True
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
        
        
        # ### Plot x,y (t) ###
        # fig2 = plt.figure("Paths")
        # plt.plot(time, x, color='g', label='x(t)')
        # plt.plot(time, y, color='r', label='y(t)')
        # plt.legend()
        # plt.title("Paths")
        # plt.grid()
        # plt.xlabel("t [sek]")
        # plt.ylabel("Dist [m]")


        # ### Plot Scores ###
        fig3 = plt.figure("Scores")
        plt.plot(time, velocity_scores, color='g', label='Vel Score')
        plt.plot(time, heading_scores, color='b', label='Head Score')
        
        if with_people:
            plt.plot(time, people_scores, color='c', label='Human Score')
        
        if with_obstacles:
            plt.plot(time, distance_scores, color='r', label='Dist Score')
            
        #plt.plot(time, total_score, color ='y', label='Total Score')
        plt.legend()
        plt.title("Scores")
        plt.grid()
        plt.xlabel("t [sek]")
        plt.ylabel("Score []")
        
        # ### Plot velocities ###
        # fig4 = plt.figure("Velocities")
        # plt.plot(time, wels, color='g', label='Ang. Vel [rad/s]')
        # plt.plot(time, vels, color='r', label='Lin. Vel [m/s]')
        # plt.legend()
        # plt.title("Velocities [rad/s] and [m/s]")
        # plt.grid()
        # plt.xlabel("t [sek]")
        # plt.ylabel("Velocity")
        
        # Show plot
        plt.show()
        


if __name__ == '__main__':
    try:
        dwa = DWA()
        dwa.simulate_dwa(num_obstacles=50, num_people=50)
        
    except rospy.ROSInterruptException:
        print("Error")
