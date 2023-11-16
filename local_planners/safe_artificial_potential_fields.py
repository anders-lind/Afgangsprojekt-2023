#!/usr/bin/env python3

import rospy
from math import atan2, sqrt, sin, cos, radians, atan2, dist, pi, floor
import numpy as np
from numpy.linalg import norm
import random
import matplotlib.pyplot as plt

class SAPF:
    def __init__(self,
            # start orientation
            start_pos,
            start_theta,
            # goal
            goal,
            # obstacles
            obstacles,
            # SAPF parameters
            d_star,
            d_safe,
            d_vort,
            zeta,
            eta,
            Q_star,
            alpha_th,
            theta_max_error,
            # robot
            v_max,
            omega_max,
            lin_a_max,
            rot_a_max,
            # Simulation #
            Kp_lin_a = 0.3,
            Kp_omega = 0.3,
            time_step_size = 0.1,
            goal_th = 0.1,
            max_iterations = 1000,
            noise_mag = 0.001
        ):

        ### SAPF parameters ###
        self.d_star = d_star
        self.d_safe = d_safe
        self.d_vort = d_vort
        self.zeta = zeta
        self.eta = eta
        self.Q_star = Q_star
        self.alpha_th = alpha_th
        self.theta_max_error = theta_max_error

        ### Robot dynamics ###
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

        ### Start configruation ###
        self.start = start_pos.copy()   # Required, otherwise memory leek
        self.pos = start_pos
        self.theta = start_theta
        self.omega = 0.0
        self.vel = 0.0

        ### Goal ###
        self.goal = goal

        ### Obstacles ###
        self.obstacles = obstacles
        



    def simulate_path(self, plot_path=True, plot_pots=True, plot_state=True, plot_pot_field=True):
        if plot_path:
            # Init
            x = np.zeros(1)
            y = np.zeros(1)
            # Log
            x[0] = self.pos[0]
            y[0] = self.pos[1]
        if plot_pots:
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
        if plot_state:
            # init
            theta_error = np.zeros(1)
            omega = np.zeros(1)
            vel_mag = np.zeros(1)
            tot,att,rep = self.calc_potential_full(pos=self.pos, goal=self.goal,debug=True)
            # log
            v_ref, theta_ref = self.calc_ref_values(tot)
            theta_error[0] = atan2(sin(theta_ref - self.theta), cos(theta_ref - self.theta))
            omega[0] = self.omega
            vel_mag[0] = norm(self.vel)
        

        print("Start pos:", self.start)
        print("Goal:", self.goal)

        # Simulate movement
        i = 0
        while i < self.max_iterations and norm(self.goal - self.pos) > self.goal_th:
            i += 1
            
            # Calculate potentials and move robot
            pot = self.calc_potential(pos=self.pos, goal=self.goal)
            v_ref, theta_ref = self.calc_ref_values(pot)
            self.simulate_robot(v_ref=v_ref, theta_ref=theta_ref)

            # Log position:
            if plot_path:
                x = np.append(x, self.pos[0])
                y = np.append(y, self.pos[1])
            # Log potential
            if plot_pots:
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
            if plot_state:
                theta_error = np.append(theta_error, atan2(sin(theta_ref - self.theta), cos(theta_ref - self.theta)))
                omega = np.append(omega, self.omega)
                vel_mag = np.append(vel_mag, norm(self.vel))

            # If hit obstacle
            for o in range(len(obstacles)):
                if norm(obstacles[o] - self.pos) < 0.05:
                    print("Hit obstacle!")


            # If at goal
            if norm(self.goal - self.pos) < self.goal_th:
                print("At goal!")

        # Print iterations
        print("i =", i)
        time_axis = np.linspace(start=0, stop=i*self.time_step_size, num=i+1)


        ### Draw obstacles ###
        if plot_path:
            figure, axes = plt.subplots()
            for i in range(len(self.obstacles)):    
                obs_x = self.obstacles[i][0]
                obs_y = self.obstacles[i][1]
                drawing_d_safe = plt.Circle( (obs_x, obs_y), self.d_safe, fill = False )
                drawing_d_vort = plt.Circle( (obs_x, obs_y), self.d_vort, fill = False )
                drawing_Q_star = plt.Circle( (obs_x, obs_y), self.Q_star, fill = False )
                axes.add_artist(drawing_d_safe)
                axes.add_artist(drawing_d_vort)
                axes.add_artist(drawing_Q_star)

        
        ### Draw goal ###
        if plot_path:
            drawing_goal = plt.Circle( (self.goal[0], self.goal[1]), self.goal_th, fill = True, color=(0,1,0))
            axes.add_artist(drawing_goal)
        

        ### Draw potential field ###
        if plot_pot_field:
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
                    pot = self.calc_potential(goal=self.goal, pos=pos)
                    # Limit arrow size
                    if norm(pot) > 1:
                        pot = pot/norm(pot)
                    # save pot
                    field_potential_x[i_y][i_x] = pot[0]
                    field_potential_y[i_y][i_x] = pot[1]

            # Apply arrows
            plt.quiver(x_field, y_field, field_potential_x, field_potential_y, scale=5, scale_units="inches", minshaft=2, headlength=5)
        
        ### Plot path ###
        if plot_path:
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
            
        
        


        # Plot potentials
        if plot_pots:
            fig = plt.figure()
            plt.title("Potentials")
            plt.xlabel("time [s]")
            plt.ylabel("Potential magnitude")
            plt.plot(time_axis, pot_tot_mag, color = 'b', label="tot")
            plt.plot(time_axis, pot_att_mag, color = 'g', label="att")
            plt.plot(time_axis, pot_rep_mag, color = 'r', label="rep")
            plt.legend()
        


        # Plot robot state
        if plot_state:
            fig = plt.figure()
            plt.title("Robot state")
            plt.xlabel("time [s]")
            plt.plot(time_axis, theta_error, color='b', label="theta_error [radians]")
            plt.plot(time_axis, omega, color='g', label="omega [radians/s]")
            plt.plot(time_axis, vel_mag, color='r', label="velocity_mag [m/s]")
            plt.legend()
        

        
        # Show plots
        if plot_path:
            plt.show()




    def simulate_robot(self, v_ref: float, theta_ref: float):
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
        
        # rot_acc = omega_error
        rot_acc = omega_ref
        if abs(rot_acc) > self.rot_a_max:
            if rot_acc < 0:
                rot_acc = -self.rot_a_max
            else:
                rot_acc = self.rot_a_max
        
        # Calculate rotational speed
        # self.omega = self.omega + rot_acc*self.time_step_size
        self.omega = self.omega + omega_error
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

        if abs(theta_error) < self.theta_max_error:
            v_ref = min(norm(potential)*(self.theta_max_error - theta_error) / (self.theta_max_error), self.v_max)
        else:
            v_ref = 0


        return v_ref, theta_ref



    def calc_potential(self, pos, goal, debug=False):
        t,a,r = self.calc_potential_full(pos, goal, debug)
        return t

    def calc_potential_full(self, pos, goal, debug=False):
        # Calculate attractive potential
        # For every obstacle
        #   Calculate repulsive potential
        #   Calculate values used for vortex
        #   Rotate repulsive potentiel according to vortex
        #   Add repulsive potential to total repulsive potential
        # Calculate full potential as sum of attractive and repulsive potentials
        
        nablaU_att = np.zeros(2)
        nablaU_rep = np.zeros(2)

        # Calculate attractive potential
        if dist(pos, goal) < self.d_star:
            nablaU_att = self.zeta * (goal - pos)
        else:
            nablaU_att = self.d_star * self.zeta * ((goal - pos) / norm(goal - pos))

        # For every obstacle
        for i in range(len(self.obstacles)):
            d_O_i = dist(pos, self.obstacles[i])

            # Calculate repulsive potential for each object
            if (d_O_i < self.Q_star):
                nablaU_repObj_i = self.eta * (1/d_O_i - 1/self.Q_star) * (1.0 / (pow(d_O_i,2))) * (pos - self.obstacles[i])
            else:
                nablaU_repObj_i = np.zeros(2)

            # Calculate values used for vortex
            alpha = self.theta - atan2(self.obstacles[i][1] - pos[1],self.obstacles[i][0] - pos[0])
            alpha = atan2(sin(alpha), cos(alpha))
            
            if alpha <= self.alpha_th:
                D_alpha = +1
            else:
                D_alpha = -1
            
            if d_O_i < self.d_safe:
                d_rel_O_i = 0
            elif d_O_i > 2*self.d_vort - self.d_safe:
                d_rel_O_i = 1
            else:
                d_rel_O_i = (d_O_i - self.d_safe)/(2*(self.d_vort - self.d_safe))

            if d_rel_O_i <= 0.5:
                gamma = pi*D_alpha*d_rel_O_i
            else:
                gamma = pi*D_alpha*(1-d_rel_O_i)

            R_gamma = np.array([
                [cos(gamma), -sin(gamma)],
                [sin(gamma), cos(gamma)] 
            ])
            nablaU_repObj_i = np.matmul(nablaU_repObj_i, R_gamma)            


        #   Add repulsive potential to total repulsive potential
            nablaU_rep += nablaU_repObj_i
        # nablaU_rep = np.zeros(2)

        # Calculate full potential as sum of attractive and all repulsive
        nabla_U = nablaU_att + nablaU_rep

        if debug: print("nablaU_att:", nablaU_att)
        if debug: print("nablaU_rep:", nablaU_rep)

        return nabla_U, nablaU_att, nablaU_rep


def create_potential_field():




if __name__ == "__main__":
    seed = int(random.random()*1000)
    print("seed =", seed)
    random.seed(seed)
    
    # Start, goal and obstacles
    start = [-10.0, -10.0]
    start_theta = 0
    goal = [10.0, 10.0]
    obstacles = []
    for i in range(9):
        obstacles.append([(random.random()-0.5)*14, (random.random()-0.5)*14])
    

    sapf_obstacles = SAPF(
        # SAPF parameters
        d_star=0.3,
        Q_star=1.0,
        d_safe=0.2,
        d_vort=0.35,
        zeta=1.1547,
        # zeta=2.1547,
        # eta=0.0732,
        eta=0.732,
        alpha_th=radians(5),
        theta_max_error=radians(135),
        # Robot limits
        v_max=2.5,
        lin_a_max=2.0,
        omega_max=2.0,
        rot_a_max=2.0,
        # Start
        start_pos=np.array(start),
        start_theta=start_theta,
        # Goal
        goal=np.array(goal),
        # Obstacles
        obstacles=np.array(obstacles),
        # Simulation
        goal_th=0.2,
        Kp_lin_a=0.2,
        Kp_omega=0.15,
        time_step_size=0.1,
        max_iterations=10000,
        noise_mag=0.001
    )

    sapf_humans = SAPF(
        # SAPF parameters
        d_star=2.0,
        Q_star=5.0,
        d_safe=0.2,
        d_vort=1.2,
        zeta=0.3,
        # zeta=2.1547,
        # eta=0.0732,
        eta=3.8,
        alpha_th=radians(5),
        theta_max_error=radians(135),
        # Robot limits
        v_max=1.5,
        lin_a_max=2.0,
        omega_max=2.0,
        rot_a_max=2.0,
        # Start
        start_pos=np.array(start),
        start_theta=start_theta,
        # Goal
        goal=np.array(goal),
        # Obstacles
        obstacles=np.array(obstacles),
        # Simulation
        goal_th=0.2,
        Kp_lin_a=0.2,
        Kp_omega=0.15,
        time_step_size=0.1,
        max_iterations=10000,
        noise_mag=0.001
    )


    # sapf_obstacles.simulate_path()
    sapf_humans.simulate_path()

    # Good seeds
    #20
    #537