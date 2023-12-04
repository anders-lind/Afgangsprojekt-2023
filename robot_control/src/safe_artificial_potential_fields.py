#!/usr/bin/env python3

import rospy
from math import atan2, sqrt, sin, cos, radians, atan2, dist, pi, floor
import numpy as np
from numpy.linalg import norm
import random
import matplotlib.pyplot as plt
from human_cost import Human_cost as hc


class SAPF:
    def __init__(self,
            # start orientation
            start_pos = [0,0],
            start_theta = 0, 
            # goal
            goal = [10,10],
            # obstacles
            obstacles = [],
            humans = [],
            human_point_is_middle = True,
            use_human_cost = False,      # Use homemade human function?
            # SAPF parameters
            d_star_obs = 0.3,
            d_safe_obs = 0.2,
            d_vort_obs = 0.35,
            zeta_obs = 2.1547,
            eta_obs = 0.732,
            Q_star_obs = 1.0,
            alpha_th_obs = radians(5),
            theta_max_error_obs = radians(135),
            d_star_hum = 2.0,         # Remove human version
            d_safe_hum = 0.2,
            d_vort_hum = 2.0,
            zeta_hum = 0.3,           # Remove human version
            eta_hum = 3.8,
            Q_star_hum = 5.0,
            alpha_th_hum = radians(5),
            theta_max_error_hum = radians(135),
            # robot
            robot_size = 0.55,
            v_max = 1.5,
            omega_max = 1.0,
            lin_a_max = 1.0,
            rot_a_max = 1.0,
            # Simulation #
            Kp_lin_a = 0.2,
            Kp_omega = 1.6,
            time_step_size = 0.1,
            goal_th = 0.2,
            max_iterations = 10000,
            noise_mag = 0.001,
            crash_dist = 0.1
        ):

        ### SAPF parameters ###
        # obstacle parameters
        self.d_star_obs = d_star_obs
        self.d_safe_obs = d_safe_obs
        self.d_vort_obs = d_vort_obs
        self.zeta_obs = zeta_obs
        self.eta_obs = eta_obs
        self.Q_star_obs = Q_star_obs
        self.alpha_th_obs = alpha_th_obs
        self.theta_max_error_obs = theta_max_error_obs
        # Human parameters
        self.d_star_hum = d_star_hum
        self.d_safe_hum = d_safe_hum
        self.d_vort_hum = d_vort_hum
        self.zeta_hum = zeta_hum
        self.eta_hum = eta_hum
        self.Q_star_hum = Q_star_hum
        self.alpha_th_hum = alpha_th_hum
        self.theta_max_error_hum = theta_max_error_hum
        self.human_point_is_middle = human_point_is_middle
        self.hc = hc()
        self.use_human_cost = use_human_cost

        ### Robot dynamics ###
        self.robot_size = robot_size
        self.v_max = v_max
        self.omega_max = omega_max
        self.lin_a_max = lin_a_max
        self.rot_a_max = rot_a_max

        ### Simulation ###
        self.Kp_lin_a = Kp_lin_a
        self.Kp_omega = Kp_omega
        self.time_step_size = time_step_size
        self.goal_th = goal_th
        self.max_iterations = max_iterations
        self.noise_mag = noise_mag
        self.crash_dist = crash_dist

        ### Start configruation ###
        self.start = start_pos.copy()   # Required, otherwise memory leek
        self.pos = start_pos
        self.start_theta = start_theta
        self.theta = start_theta
        self.omega = 0.0
        self.vel = 0.0

        ### Goal ###
        self.goal = goal

        ### Obstacles ###
        self.obstacles = obstacles
        self.humans = humans
        


    def reset(self):
        self.pos = self.start
        self.theta = self.start_theta
        self.omega = 0.0
        self.vel = 0.0

    
    def update_map(self, obstacles: [float, float], humans: [[[float, float], [float, float]]], goal=[float, float]):
        self.obstacles = obstacles
        self.humans = humans
        self.goal = goal
    

    def update_robot_state(self, x: float, y: float, theta: float):
        self.pos = [x, y]
        self.theta = theta

    
    def calc_step(self) -> [float, float]:
        """
        This method uses SAPF to calculate a linear- and angular velocity for the current state of the robot and map
        """

        # vel_ref
        pot = self.calc_potential(pos=self.pos, goal=self.goal, robot_size=self.robot_size)
        vel_ref, theta_ref = self.calc_ref_values(pot)

        # omega_ref
        theta_error = theta_ref - self.theta
        theta_error = atan2(sin(theta_error), cos(theta_error))
        omega_ref = self.Kp_omega * theta_error


        return vel_ref, omega_ref



    def simulate_path(self, plot_path=False, plot_more=False, debug=False):
        reached_goal = False
        hit_obstacle = False
        hit_human = False
        
        if plot_path:
            # Init
            x = np.zeros(1)
            y = np.zeros(1)
            # Log
            x[0] = self.pos[0]
            y[0] = self.pos[1]
        if plot_more:
            # Init
            pot_tot_x = np.zeros(1)
            pot_tot_y = np.zeros(1)
            pot_tot_mag = np.zeros(1)
            pot_att_x = np.zeros(1)
            pot_att_y = np.zeros(1)
            pot_att_mag = np.zeros(1)
            pot_rep_x = np.zeros(1)
            pot_rep_y = np.zeros(1)
            pot_rep_mag = np.zeros(1)
            tot,att,rep = self.calc_potential_full(pos=self.pos, goal=self.goal)
            theta_error = np.zeros(1)
            omega = np.zeros(1)
            vel_mag = np.zeros(1)
            tot,att,rep = self.calc_potential_full(pos=self.pos, goal=self.goal)
            # Log
            pot_tot_x[0] = tot[0]
            pot_tot_y[0] = tot[1]
            pot_tot_mag = norm(tot)
            pot_att_x[0] = att[0]
            pot_att_y[0] = att[1]
            pot_att_mag = norm(att)
            pot_rep_x[0] = rep[0]
            pot_rep_y[0] = rep[1]
            pot_rep_mag = norm(rep)
            v_ref, theta_ref = self.calc_ref_values(tot)
            theta_error[0] = atan2(sin(theta_ref - self.theta), cos(theta_ref - self.theta))
            omega[0] = self.omega
            vel_mag[0] = norm(self.vel)
        
        if debug:
            print("Start pos:", self.start)
            print("Goal:     ", self.goal)


        ### Simulate movement ###
        i = 0
        while i < self.max_iterations and norm(self.goal - self.pos) > self.goal_th:
            i += 1
            
            # Calculate potentials and move robot
            pot = self.calc_potential(pos=self.pos, goal=self.goal, robot_size=self.robot_size)
            v_ref, theta_ref = self.calc_ref_values(pot)
            self.simulate_robot_step(v_ref=v_ref, theta_ref=theta_ref)

            # Log position:
            if plot_path:
                x = np.append(x, self.pos[0])
                y = np.append(y, self.pos[1])
            if plot_more:
                # Log potential
                tot,att,rep = self.calc_potential_full(pos=self.pos, goal=self.goal)
                pot_tot_x = np.append(pot_tot_x, tot[0])
                pot_tot_y = np.append(pot_tot_y, tot[1])
                pot_tot_mag = np.append(pot_tot_mag, norm(tot))
                pot_att_x = np.append(pot_att_x, att[0])
                pot_att_y = np.append(pot_att_y, att[1])
                pot_att_mag = np.append(pot_att_mag, norm(att))
                pot_rep_x = np.append(pot_rep_x, rep[0])
                pot_rep_y = np.append(pot_rep_y, rep[1])
                pot_rep_mag = np.append(pot_rep_mag, norm(rep))
                # Log robot state
                theta_error = np.append(theta_error, atan2(sin(theta_ref - self.theta), cos(theta_ref - self.theta)))
                omega = np.append(omega, self.omega)
                vel_mag = np.append(vel_mag, norm(self.vel))


            # If hit obstacle
            for o in range(len(self.obstacles)):
                if norm(self.obstacles[o] - self.pos) < self.crash_dist + self.robot_size:
                    hit_obstacle = True
                    if debug:
                        print("Hit obstacle:", self.pos)
            
            # If hit human
            for h in range(len(self.humans)):
                if norm(self.humans[h][0] - self.pos) < self.crash_dist + self.robot_size:
                    hit_human = True
                    if debug:
                        print("Hit human!")

            # If at goal
            if norm(self.goal - self.pos) < self.goal_th:
                reached_goal = True
                if debug:
                    print("At goal!")
                    print("Speed:", norm(self.vel))

        # Print iterations
        if debug: print("i =", i)
        time_axis = np.linspace(start=0, stop=i*self.time_step_size, num=i+1)


        if plot_path:
            figure, axes = plt.subplots()

            ### Plot path ###
            plt.title("Path")
            plt.plot(x,y, 'r')
            plt.grid()
            axes.set_aspect(1)
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            # # Plot settings
            # if lim_start[0] < lim_stop[0]:
            #     plt.xlim(lim_start[0], lim_stop[0])
            # else:
            #     plt.xlim(lim_stop[0], lim_start[0])
            # if lim_start[1] < lim_stop[1]:
            #     plt.ylim(lim_start[1], lim_stop[1])
            # else:
            #     plt.ylim(lim_stop[1], lim_start[1])

            ### Draw obstacles ###
            for o in range(len(self.obstacles)):    
                obs_x = self.obstacles[o][0]
                obs_y = self.obstacles[o][1]
                drawing_obs = plt.Circle( (obs_x, obs_y), self.crash_dist, fill = True, color=(1,0,1) )
                drawing_d_safe_obs = plt.Circle( (obs_x, obs_y), self.d_safe_obs, fill = False, color=(1,0,1) )
                drawing_d_vort_obs = plt.Circle( (obs_x, obs_y), self.d_vort_obs, fill = False, color=(1,0,1) )
                drawing_Q_star_obs = plt.Circle( (obs_x, obs_y), self.Q_star_obs, fill = False, color=(1,0,1) )
                axes.add_artist(drawing_obs)
                if plot_more:
                    axes.add_artist(drawing_d_safe_obs)
                    axes.add_artist(drawing_d_vort_obs)
                    axes.add_artist(drawing_Q_star_obs)
        
            ### Draw humans ###
            for h in range(len(self.humans)):    
                hum_x = self.humans[h][0][0]
                hum_y = self.humans[h][0][1]
                drawing_hum = plt.Circle( (hum_x, hum_y), self.human_size, fill = True, color=(1,0.5,0) )
                drawing_d_safe_hum = plt.Circle( (hum_x, hum_y), self.d_safe_hum, fill = False, color=(1,0.5,0) )
                drawing_d_vort_hum = plt.Circle( (hum_x, hum_y), self.d_vort_hum, fill = False, color=(1,0.5,0) )
                drawing_Q_star_hum = plt.Circle( (hum_x, hum_y), self.Q_star_hum, fill = False, color=(1,0.5,0) )
                axes.add_artist(drawing_hum)
                plt.quiver(self.humans[h][0][0], self.humans[h][0][1], self.humans[h][1][0], self.humans[h][1][1], scale=4, scale_units="inches")
                if plot_more:
                    axes.add_artist(drawing_d_safe_hum)
                    axes.add_artist(drawing_d_vort_hum)
                    axes.add_artist(drawing_Q_star_hum)
                
            

        
            ### Draw goal ###
            drawing_goal = plt.Circle( (self.goal[0], self.goal[1]), self.goal_th, fill = True, color=(0,1,0))
            axes.add_artist(drawing_goal)
        

        ### Draw potential field ###
        # TODO: Make dynaimc with changes in plot viewer
        
        if plot_more:
            # Size limits
            lim_start = self.start
            lim_stop = self.goal*1.1
            # Nr. of arrows
            field_x_arrows = 75
            field_y_arrows = 75
            # Init arrays
            x_line = np.linspace(start=lim_start[0], stop=lim_stop[0], num=field_x_arrows)
            y_line = np.linspace(start=lim_start[1], stop=lim_stop[1], num=field_y_arrows)
            x_field, y_field = np.meshgrid(x_line, y_line, sparse=False)
            field_potential_x = np.zeros((field_y_arrows, field_x_arrows))
            field_potential_y = np.zeros((field_y_arrows, field_x_arrows))
            for i_x in range(field_x_arrows):
                for i_y in range(field_y_arrows):
                    # Calc pot
                    pos = [x_line[i_x], y_line[i_y]]
                    pot = self.calc_potential(goal=self.goal, pos=pos, debug=debug)
                    # Limit arrow size
                    if norm(pot) > 1:
                        pot = pot/norm(pot)
                    # save pot
                    field_potential_x[i_y][i_x] = pot[0]
                    field_potential_y[i_y][i_x] = pot[1]

            # Apply arrows
            plt.quiver(x_field, y_field, field_potential_x, field_potential_y, scale=5, scale_units="inches", minshaft=2, headlength=5)
        

            # Plot potentials
            fig = plt.figure()
            plt.title("Potentials")
            plt.xlabel("time [s]")
            plt.ylabel("Potential magnitude")
            plt.plot(time_axis, pot_tot_mag, color = 'b', label="tot")
            plt.plot(time_axis, pot_att_mag, color = 'g', label="att")
            plt.plot(time_axis, pot_rep_mag, color = 'r', label="rep")
            plt.legend()
            plt.grid()
        
            # Plot robot state
            fig = plt.figure()
            plt.title("Robot state")
            plt.xlabel("time [s]")
            plt.plot(time_axis, theta_error, color='b', label="theta_error [radians]")
            plt.plot(time_axis, omega, color='g', label="omega [radians/s]")
            plt.plot(time_axis, vel_mag, color='r', label="velocity_mag [m/s]")
            plt.legend()
            plt.grid()
        
            # Plot position
            fig = plt.figure()
            plt.title("Robot position")
            plt.plot(time_axis, x, label="x")
            plt.plot(time_axis, y, label="y")
            plt.legend()
            plt.grid()
        

        # Show plots
        if plot_path or plot_more:
            plt.show()


        return reached_goal, hit_obstacle, hit_human, i*self.time_step_size


    def draw_arrows(self, text: str):
        print("something changed:", text)



    def simulate_robot_step(self, v_ref: float, theta_ref: float):
        # Calculate rotational acceleration
        # Calculate rotational speed
        # Calculate rotation
        # Calculate linear acceleration
        # Calculate linear velocity
        # Calculate position

        # Add noise
        self.theta += (random.random()-0.5) * self.noise_mag
        self.pos[0] += (random.random()-0.5) * self.noise_mag
        self.pos[1] += (random.random()-0.5) * self.noise_mag

        # Calculate rotational acceleration
        theta_error = theta_ref - self.theta
        theta_error = atan2(sin(theta_error), cos(theta_error))
        omega_ref = self.Kp_omega * theta_error
        omega_error = omega_ref - self.omega
        
        rot_acc = omega_error
        if abs(rot_acc) > self.rot_a_max:
            if rot_acc < 0:
                rot_acc = -self.rot_a_max
            else:
                rot_acc = self.rot_a_max
        
        # Calculate rotational speed
        self.omega = self.omega + rot_acc*self.time_step_size
        if (self.omega > self.omega_max):
            self.omega = self.omega_max

        # Calculate rotation
        self.theta = self.theta + self.omega*self.time_step_size


        # Calculate linear acceleration
        v_error = v_ref - self.vel
        lin_a = min(self.Kp_lin_a * v_error, self.lin_a_max)

        # Calculate linear velocity
        self.vel = self.vel + lin_a*self.time_step_size
        if self.vel > self.v_max:
            self.vel = self.v_max

        # Calculate position
        x = self.pos[0] + cos(self.theta)*self.vel*self.time_step_size
        y = self.pos[1] + sin(self.theta)*self.vel*self.time_step_size
        self.pos = np.array([x, y])



    # Returns the reference linear velocity and the reference direction
    def calc_ref_values(self, potential):
        theta_ref = atan2(potential[1], potential[0])
        theta_error = theta_ref - self.theta
        theta_error = atan2(sin(theta_error), cos(theta_error)) # wrap [-pi, pi]

        beta = (self.theta_max_error_obs - theta_error) / (self.theta_max_error_obs)

        if abs(theta_error) < self.theta_max_error_obs:
            v_ref = min(beta*norm(potential), self.v_max)
        else:
            v_ref = 0

        return v_ref, theta_ref



    def calc_potential(self, pos, goal, robot_size=0, debug=False):
        t,a,r = self.calc_potential_full(pos=pos, goal=goal, robot_size=robot_size, debug=debug)
        return t

    def calc_potential_full(self, pos, goal, robot_size=0, debug=False):
        # Calculate attractive potential
        # For every obstacle and human
        #   Calculate repulsive potential
        #   Calculate values used for vortex
        #   Rotate repulsive potentiel according to vortex
        #   Add repulsive potential to total repulsive potential
        # Calculate full potential as sum of attractive and repulsive potentials
        
        nablaU_att = np.zeros(2)
        nablaU_rep = np.zeros(2)
        nablaU_rep_obs = np.zeros(2)
        nablaU_rep_hum = np.zeros(2)


        # Calculate attractive potential
        if dist(pos, goal) < self.d_star_obs:
            nablaU_att = self.zeta_obs * (goal - pos)
        else:
            nablaU_att = self.d_star_obs * self.zeta_obs * ((goal - pos) / norm(goal - pos))


        # For every obstacle
        for i in range(len(self.obstacles)):
            # Distance to object
            d_O_i = dist(pos, self.obstacles[i]) - robot_size
            if d_O_i <= 0: d_O_i = 0.0000001

            # Calculate repulsive potential for each object
            if (d_O_i < self.Q_star_obs):
                nablaU_repObs_i = self.eta_obs * (1/d_O_i - 1/self.Q_star_obs) * (1.0 / (pow(d_O_i,2))) * (pos - self.obstacles[i])
            else:
                nablaU_repObs_i = np.zeros(2)

            # Calculate values used for vortex
            alpha = self.theta - atan2(pos[1] - self.obstacles[i][1],pos[0] - self.obstacles[i][0])
            alpha = atan2(sin(alpha), cos(alpha))
            
            if alpha <= self.alpha_th_obs:
                D_alpha = +1
            else:
                D_alpha = -1
            
            if d_O_i < self.d_safe_obs:
                d_rel_O_i = 0
            elif d_O_i > 2*self.d_vort_obs - self.d_safe_obs:
                d_rel_O_i = 1
            else:
                d_rel_O_i = (d_O_i - self.d_safe_obs)/(2*(self.d_vort_obs - self.d_safe_obs))

            if d_rel_O_i <= 0.5:
                gamma = pi*D_alpha*d_rel_O_i
            else:
                gamma = pi*D_alpha*(1-d_rel_O_i)

            R_gamma = np.array([
                [cos(gamma), -sin(gamma)],
                [sin(gamma), cos(gamma)] 
            ])
            nablaU_repObs_i = np.matmul(nablaU_repObs_i, R_gamma)            


            # Add repulsive potential to total repulsive potential
            nablaU_rep_obs += nablaU_repObs_i
        

        # For every human
        for i in range(len(self.humans)):
            if self.human_point_is_middle:
                d_O_i = dist(pos, self.humans[i][0]) - robot_size - self.human_size
            else:
                d_O_i = dist(pos, self.humans[i][0]) - robot_size
            if d_O_i <= 0: d_O_i = 0.0000001

            # Calculate repulsive potential for each human
            # Using homemade function
            if self.use_human_cost:
                human_relative_pos = self.pos - self.humans[i][0]
                nablaU_rep_hum_i = np.array(self.hc.get_cost_xy(x=human_relative_pos[0], y=human_relative_pos[1]))
                nablaU_rep_hum_i = nablaU_rep_hum_i * (self.pos - self.humans[i][0])/norm(self.pos - self.humans[i][0])
                # TODO: Make sure homemade works!
                print(self.hc.get_cost_xy(x=human_relative_pos[0], y=human_relative_pos[1]))
            # Using potential fields
            else:
                if (d_O_i < self.Q_star_hum):
                    nablaU_rep_hum_i = self.eta_hum * (1/d_O_i - 1/self.Q_star_hum) * (1.0 / (pow(d_O_i,2))) * (pos - self.humans[i][0])
                else:
                    nablaU_rep_hum_i = np.zeros(2)

            # Calculate values used for vortex
            alpha = self.theta - atan2(pos[1] - self.humans[i][0][1],pos[0] - self.humans[i][0][0])
            alpha = atan2(sin(alpha), cos(alpha))
            
            if alpha <= self.alpha_th_hum:
                D_alpha = +1
            else:
                D_alpha = -1
            
            if d_O_i < self.d_safe_hum:
                d_rel_O_i = 0
            elif d_O_i > 2*self.d_vort_hum - self.d_safe_hum:
                d_rel_O_i = 1
            else:
                d_rel_O_i = (d_O_i - self.d_safe_hum)/(2*(self.d_vort_hum - self.d_safe_hum))

            if d_rel_O_i <= 0.5:
                gamma = pi*D_alpha*d_rel_O_i
            else:
                gamma = pi*D_alpha*(1-d_rel_O_i)

            R_gamma = np.array([
                [cos(gamma), -sin(gamma)],
                [sin(gamma), cos(gamma)] 
            ])
            nablaU_rep_hum_i = np.matmul(nablaU_rep_hum_i, R_gamma)            


            # Add repulsive potential to total repulsive potential
            nablaU_rep_hum += nablaU_rep_hum_i


        # Calculate full potential as sum of attractive and all repulsive
        nabla_U = nablaU_att + nablaU_rep_obs + nablaU_rep_hum

        # if debug: print("nablaU_att:", nablaU_att)
        # if debug: print("nablaU_rep:", nablaU_rep)

        return nabla_U, nablaU_att, nablaU_rep



# def create_potential_field():



if __name__ == "__main__":
    # Good seeds
    #20
    #537
    #492525103  
    #36821076
    seed = random.randrange(1000000000)
    seed = 36821076
    random.seed(seed)
    print("seed:", seed)
    
    # Start, goal and obstacles
    start = [-10.0, -10.0]
    start_theta = 0
    goal = [10.0, 10.0]

    obstacles = []
    for i in range(0):
        obstacles.append([(random.random()-0.5)*20, (random.random()-0.5)*20])

    humans = []
    for i in range(1):
        humans.append([[(random.random()-0.5)*14, (random.random()-0.5)*14], [random.random()-0.5, random.random()-0.5]])
        scale = sqrt(humans[i][1][0]**2 + humans[i][1][1]**2)        
        humans[i][1][0] = humans[i][1][0] / scale
        humans[i][1][1] = humans[i][1][1] / scale
    
    # humans = [[[8, 5], [1, 0]]]
        

    sapf = SAPF (
        # # SAPF obstacle parameters
        # d_star_obs=0.3,
        # Q_star_obs=1.0,
        # d_safe_obs=0.2,
        # d_vort_obs=0.35,
        # zeta_obs=2.1547,
        # eta_obs=0.732,
        # alpha_th_obs=radians(5),
        # theta_max_error_obs=radians(135),
        # # SAPF human parameters
        # d_star_hum=2.0,  # Remove human version
        # Q_star_hum=5.0,
        # d_safe_hum=0.2,
        # d_vort_hum=2.0,
        # zeta_hum=0.3,   # Remove human version
        # eta_hum=3.8,
        # alpha_th_hum=radians(5),
        # theta_max_error_hum=radians(135),
        # # Robot limits
        # robot_size=0.55,
        # v_max=1.5,
        # lin_a_max=1.0,
        # omega_max=1.0,
        # rot_a_max=1.0,
        # # Start
        # start_pos=np.array(start),
        # start_theta=start_theta,
        # # Goal
        # goal=np.array(goal),
        # # Obstacles
        # obstacles=np.array(obstacles),
        # # Humans
        # humans=np.array(humans),
        # human_point_is_middle=True,
        # use_human_cost = False,       # Use homemade human function?
    )

    # Single test
    if True:
        result = sapf.simulate_path(plot_path=True, plot_more=True, debug=True)
        print(result)

        # sapf.use_human_cost = False
        # sapf.reset()

        # result = sapf.simulate_path(plot_path=True, plot_more=False, debug=True)
        # print(result)
        pass

    
    # Multiple test
    if False:
        time = []
        goals = 0
        crashes = 0
        fails = 0
        for t in range(20):
            print("test:", t)
            # Obstacles
            obstacles = []
            for o in range(9):
                obstacles.append([(random.random()-0.5)*14, (random.random()-0.5)*14])
            sapf_humans.obstacles=np.array(obstacles)
            sapf.obstacles=np.array(obstacles)
            sapf.reset()
            sapf_humans.reset()
            
            # Simulate
            # g,c,t = sapf_humans.simulate_path(debug=False, plot_path=True, plot_pot_field=False, plot_pots=False, plot_state=False)
            g,o,h,t = sapf.simulate_path(debug=False, plot_path=True, plot_pot_field=False, plot_pots=False, plot_state=False)
            print(g,o,h,t)
            if o or h:
                crashes += 1
            elif g:
                goals += 1
                time.append(t)
            else:
                fails += 1
            
        print("goals:", goals)
        print("crashes:", crashes)
        print("fails:", fails)
        print("time average:", np.average(time))