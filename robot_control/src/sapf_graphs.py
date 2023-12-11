import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from math import pi, cos, sin, sqrt, atan2
from human_cost import Human_cost as HC
from numpy.linalg import norm




def att_graph():
    d_star = 0.3
    zeta = 2

    nablaU_att = []
    x = []
    d = 0
    while d < 0.7:
        if d < d_star:
            nablaU_att.append(zeta * d)
        else:
            nablaU_att.append(d_star * zeta)

        x.append(d)
        d += 0.1



    # Text size
    linewidth = 4
    plt.rcParams.update({'font.size': 22})

    # Plots
    plt.plot(x, nablaU_att, linewidth=linewidth)

    # Dashed line
    dashed_line_x = [0,d_star, d_star]
    dashed_line_y = [zeta*d_star, zeta*d_star, 0]
    plt.plot(dashed_line_x,dashed_line_y,linestyle="dashed",linewidth=linewidth, color=(0,0,0))

    # Arrow
    plt.arrow(0.5, zeta*d_star, 0.1, 0, head_width=0.015, head_length=0.012, color=(31/255.0, 119/255.0, 180/255.0), linewidth=linewidth-1)
    
    # Labels and title
    plt.title("Attractive gradient")
    plt.xlabel("Distance to goal")
    plt.ylabel("$|\\vec{U}_{att}(\\vec{P}_{robot})|$")

    # Limits
    plt.ylim(0,zeta*d_star*1.3)

    # Ticks
    plt.xticks(ticks=[0,d_star], labels=["0", "$d^{*}_{g}$"])
    plt.yticks(ticks=[d_star*zeta], labels=["$\zeta \cdot d^{*}_{g}$"])

    plt.show()





def rep_graph():
    eta = 1
    Q_star = 1

    nablaU_repObs_i = []
    x = []
    d = 0.1
    while d < 1.5:
        if (d < Q_star):
            nablaU_repObs_i.append(eta * (1/d - 1/Q_star) * (1.0 / (pow(d,2))) * (d))
        else:
            nablaU_repObs_i.append(0)

        x.append(d)
        d += 0.01
    
    linewidth = 4
    plt.rcParams.update({'font.size': 22})

    plt.plot([0,Q_star],[0,0],linestyle="dashed", color=(0,0,0), linewidth=linewidth-0.1)
    plt.plot(x, nablaU_repObs_i, linewidth=linewidth)
    

    plt.title("Repulsive gradient")
    plt.ylabel("$|\\vec{U}_{rep}^{O_i}(P_{robot})|$")
    plt.xlabel("Distance to obstacle")

    plt.xticks(ticks=[0,Q_star],labels=["0", "$d_{O_i}^{*}$"])
    plt.yticks(ticks=[0],labels=["0"])

    plt.xlim(0,1.5)
    plt.ylim(0,80)

    

    plt.show()


def switch_graph():
    Q_star = 4
    d_vort = 2
    d_safe = 1
    D_alpha = 1

    y_rep = []
    y_vor = []
    d_O_i_ = []
    d_O_i = 0
    while d_O_i < Q_star:
        d_rel_O_i = None
        
        # Calc d_rel
        if d_O_i < d_safe:
            d_rel_O_i = 0
        elif d_O_i > 2*d_vort - d_safe:
            d_rel_O_i = 1
        else:
            d_rel_O_i = (d_O_i - d_safe)/(2*(d_vort - d_safe))

        # Calc gamma
        if d_rel_O_i <= 0.5:
            gamma = math.pi*D_alpha*d_rel_O_i
        else:
            gamma = math.pi*D_alpha*(1-d_rel_O_i)

        y_rep.append(1-math.sin(gamma))
        y_vor.append(math.sin(gamma))
        d_O_i_.append(d_O_i)

        d_O_i += 0.01
    

    # Plot settings
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure()
    ax = fig.add_subplot()
    

    plt.plot(d_O_i_,y_vor, label="Vortex")
    plt.plot(d_O_i_,y_rep, label="Repulsive")
    
    plt.xticks(ticks=[0, 1, 2, 3], labels=["0", "$d_{safe}$", "$d_{vort}$", "$2 d_{vort}-d_{safe}$"])
    plt.yticks(ticks=[0.0,0.5,1.0], labels=["0%", "50%", "100%"])
    plt.legend()
    plt.title("Switching function")
    plt.xlabel("Distance from obstacle")
    plt.ylabel("Influence")
    
    plt.show()



def overview_graph():
    d_safe = 1
    d_vort = 2
    Q_star = 3

    goal_color = (0, 0.8, 0)
    rep_color = (255/255,190/255,99/255)
    vor_color = (44/255,126/255,184/255)
    att_color = (0.0,0.4,0)

    obs = [0.0,0.0]
    
    goal = [4,3]
    robot0 = [obs[0]+2.5*math.cos(math.pi/2.5), obs[1]+2.5*math.sin(math.pi/2.5)]
    robot1 = [d_safe+0.2, 0.2]
    robot2 = [2*d_vort,0.0]
    robots = [robot0, robot1, robot2]

    plt.rcParams.update({'font.size': 15})
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect(1)
    
    # Object
    ax.plot(obs[0], obs[1], marker='o', color=(1,0,0), label="Obstacle")

    # Circles
    safe = plt.Circle( (obs[0], obs[1]), d_safe, fill = False, color=rep_color )
    vort = plt.Circle( (obs[0], obs[1]), d_vort, fill = False, color=vor_color )
    star = plt.Circle( (obs[0], obs[1]), Q_star, fill = False, color=rep_color )
    ax.add_artist(safe)
    ax.add_artist(vort)
    ax.add_artist(star)

    # Goal
    ax.plot(goal[0], goal[1], marker = "o", linestyle=None, color=goal_color, label="Goal")
    
    for i in range(len(robots)):
        robot = robots[i]

        # Robot
        ax.plot(robot[0], robot[1], marker='o', color=(0,0,0), label="Robot")

        # Goal arrow
        arrow_dir = 1.2 * (np.array(goal) - np.array(robot)) / (math.dist(goal, robot))
        ax.arrow(robot[0], robot[1], arrow_dir[0], arrow_dir[1], head_width=0.15, head_length=0.12, color=att_color, linewidth=2)
        if i == 2:
            ax.text(robot[0] + arrow_dir[0] +0.0, robot[1] + arrow_dir[1]+0.2, s="$U_{att}$")
        else:
            ax.text(robot[0] + arrow_dir[0] +0.1, robot[1] + arrow_dir[1]+0.1, s="$U_{att}$")
        
        # Repulsive arrow
        if i == 0 or i == 1:
            arrow_dir = 1.2 * (np.array(robot) - np.array(obs)) / (math.dist(obs, robot))
            ax.arrow(robot[0], robot[1], arrow_dir[0], arrow_dir[1], head_width=0.15, head_length=0.12, color=rep_color, linewidth=2)
            ax.text(robot[0] + arrow_dir[0] +0.1, robot[1] + arrow_dir[1]+0.1, s="$U_{rep}$")
        
        # Vortex arrow
        if i == 0 or i == 1:
            if i == 1:
                arrow_dir = 0.4 * (np.array(obs) - np.array(robot)) / (math.dist(obs, robot))
            else:
                arrow_dir = 1.2 * (np.array(obs) - np.array(robot)) / (math.dist(obs, robot))
            ax.arrow(robot[0], robot[1], -arrow_dir[1], arrow_dir[0], head_width=0.15, head_length=0.12, color=vor_color, linewidth=2)
            if i == 0:
                ax.text(robot[0] - arrow_dir[1] + 0.0, robot[1] + arrow_dir[0]-0.4, s="$U_{vort}$")
            else:
                ax.text(robot[0] - arrow_dir[1] + 0.1, robot[1] + arrow_dir[0]+0.1, s="$U_{vort}$")


    limit = 4.5
    plt.xlim(-0.5, limit)
    plt.ylim(-0.5, limit)

    plt.yticks(ticks=[d_safe, d_vort, Q_star], labels=["$d_{safe}$", "$d_{vort}$", "$2d_{vort}-d_{safe}$"])
    plt.xticks([], [])

    matplotlib.rcParams['legend.handlelength'] = 0
    matplotlib.rcParams['legend.numpoints'] = 1

    # https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.2, 1.1), framealpha=1)


    plt.show()

def att_graph():
    d_star = 0.3
    zeta = 2

    nablaU_att = []
    x = []
    d = 0
    while d < 0.7:
        if d < d_star:
            nablaU_att.append(zeta * d)
        else:
            nablaU_att.append(d_star * zeta)

        x.append(d)
        d += 0.1



    # Text size
    linewidth = 4
    plt.rcParams.update({'font.size': 22})

    # Plots
    plt.plot(x, nablaU_att, linewidth=linewidth)

    # Dashed line
    dashed_line_x = [0,d_star, d_star]
    dashed_line_y = [zeta*d_star, zeta*d_star, 0]
    plt.plot(dashed_line_x,dashed_line_y,linestyle="dashed",linewidth=linewidth, color=(0,0,0))

    # Arrow
    plt.arrow(0.5, zeta*d_star, 0.1, 0, head_width=0.015, head_length=0.012, color=(31/255.0, 119/255.0, 180/255.0), linewidth=linewidth-1)
    
    # Labels and title
    plt.title("Attractive gradient")
    plt.xlabel("Distance to goal")
    plt.ylabel("$|\\vec{U}_{att}(\\vec{P}_{robot})|$")

    # Limits
    plt.ylim(0,zeta*d_star*1.3)

    # Ticks
    plt.xticks(ticks=[0,d_star], labels=["0", "$d^{*}_{g}$"])
    plt.yticks(ticks=[d_star*zeta], labels=["$\zeta \cdot d^{*}_{g}$"])

    plt.show()


def sapf_graph():
    xend = 32
    xstart = -1

    yend = 32
    ystart = -1
    
    jump = 1.0
    
    obstacle = np.array([[14, 10], [7, 23]] )
    goal = np.array([20, 20])
    
    Q_star_obs = 15
    eta = 15.0
    d_star_obs = 2
    zeta_obs = 0.1
    
    d_safe_obs = 1
    d_vort_obs = 3
    
    
    numx = int((xend-xstart)/jump)
    numy = int((yend-ystart)/jump)
    
    x = np.arange(xstart, xend, jump)
    y = np.arange(ystart, yend, jump)
    
    ux = np.zeros(shape=(numx, numy))
    uy = np.zeros(shape=(numx, numy))
    
    
    
    figure, axes = plt.subplots()
    axes.set_aspect(1)
    
    arrow_max = 0.15
    
    for i in range(numx):
        for j in range(numy):
            dist_goal = sqrt((x[i] - goal[0])**2 + (y[j] - goal[1])**2)
            if  dist_goal< d_star_obs:
                nablaU_att = zeta_obs * (goal - np.array([x[i], y[j]]))
            else:
                nablaU_att = d_star_obs * zeta_obs * ((goal - np.array([x[i], y[j]])) /dist_goal)

            ux[j][i] = nablaU_att[0]
            uy[j][i] = nablaU_att[1]
            
            for h in range(len(obstacle)):
                alpha = atan2(goal[1] - y[j],goal[0] - x[i]) - atan2(y[j] - obstacle[h][1],x[i] - obstacle[h][0])
                alpha = atan2(sin(alpha), cos(alpha))
                
                d_O_i = sqrt((x[i] - obstacle[h][0])**2 + (y[j] - obstacle[h][1])**2)
                if d_O_i <= 0: d_O_i = 0.0000001

                nablaU_repObs_i = np.zeros(2)
                
                # # Calculate repulsive potential for each object
                D_alpha = 0
                          
                if alpha <= 0:
                    D_alpha = +1
                else:
                    D_alpha = -1
                
                if d_O_i < d_safe_obs:
                    d_rel_O_i = 0
                elif d_O_i > 2*d_vort_obs - d_safe_obs:
                    d_rel_O_i = 1
                else:
                    d_rel_O_i = (d_O_i - d_safe_obs)/(2*(d_vort_obs - d_safe_obs))

                if d_rel_O_i <= 0.5:
                    gamma = pi*D_alpha*d_rel_O_i
                else:
                    gamma = pi*D_alpha*(1-d_rel_O_i)

                R_gamma = np.array([
                    [cos(gamma), -sin(gamma)],
                    [sin(gamma), cos(gamma)] 
                ])
                
                if (d_O_i < Q_star_obs):
                    nablaU_repObs_i = eta * (1/d_O_i - 1/Q_star_obs) * (1.0 / (pow(d_O_i,2))) * (np.array([x[i], y[j]]) - obstacle[h])
                    nablaU_repObs_i = np.matmul(nablaU_repObs_i, R_gamma)
                else:
                    pass
                #nablaU_repObs_i = np.matmul(nablaU_repObs_i, R_gamma)            


                # # Add repulsive potential to total repulsive potential
                # nablaU_rep_obs += nablaU_repObs_i
                    
                ux[j][i] += nablaU_repObs_i[0]
                uy[j][i] += nablaU_repObs_i[1]
            
            siz = sqrt(ux[j][i]**2 + uy[j][i]**2)
            if siz > arrow_max: 
                ux[j][i] = ux[j][i]/(siz/arrow_max)
                uy[j][i] = uy[j][i]/(siz/arrow_max)


    plt.quiver(x, y, ux, uy, scale=1.5, scale_units="inches",width = 0.005,  minshaft=2, headlength=4, color=(0.2, 0.2, 0.2))

    
    
    # perimeter = plt.Circle( (obstacle[0][0], obstacle[0][1]), Q_star_obs,  fill = False, color = (1, 0, 0) )
    # axes.add_artist(perimeter)
    for i in range(len(obstacle)):
        if i == 0:
            plt.plot( obstacle[i][0],obstacle[i][1], color = (1, 0, 0), linestyle= "", marker = "o", label="Obstacle")
        plt.plot( obstacle[i][0],obstacle[i][1], color = (1, 0, 0), linestyle= "", marker = "o")
        
    plt.plot( goal[0],goal[1], color = (0, 1, 0), linestyle= "", marker = "o", label="Goal")
    
    
    axes.legend(framealpha = 1)
    

    plt.title("SAPF Map")
    plt.xlim(xstart + 1, xend - 1)
    plt.ylim(ystart + 1, yend -1)   
    
    # disabling xticks by Setting xticks to an empty list
    plt.xticks([])  
    
    # disabling yticks by setting yticks to an empty list
    plt.yticks([])  

    plt.show()
    


def sapf_graph_with_people():
    xend = 32
    xstart = -1

    yend = 32
    ystart = -1
    
    jump = 0.25
    
    obstacle = np.array([[14, 10], [7, 23]] )
    humans = np.array([[[23, 4], [-1/sqrt(2), 1/sqrt(2)]]] )
    goal = np.array([20, 20])
    
    Q_star_obs = 15
    eta = 15.0
    d_star_obs = 2
    zeta_obs = 0.1
    
    d_safe_obs = 1
    d_vort_obs = 3
    
    hc = HC()
    
    numx = int((xend-xstart)/jump)
    numy = int((yend-ystart)/jump)
    
    x = np.arange(xstart, xend, jump)
    y = np.arange(ystart, yend, jump)
    
    ux = np.zeros(shape=(numx, numy))
    uy = np.zeros(shape=(numx, numy))
    
    Kh = 1
    
    
    
    figure, axes = plt.subplots()
    axes.set_aspect(1)
    
    arrow_max = 0.15
    
    for i in range(numx):
        for j in range(numy):
            dist_goal = sqrt((x[i] - goal[0])**2 + (y[j] - goal[1])**2)
            if  dist_goal< d_star_obs:
                nablaU_att = zeta_obs * (goal - np.array([x[i], y[j]]))
            else:
                nablaU_att = d_star_obs * zeta_obs * ((goal - np.array([x[i], y[j]])) /dist_goal)

            ux[j][i] = nablaU_att[0]
            uy[j][i] = nablaU_att[1]
            
            # for h in range(len(obstacle)):
            #     alpha = atan2(goal[1] - y[j],goal[0] - x[i]) - atan2(y[j] - obstacle[h][1],x[i] - obstacle[h][0])
            #     alpha = atan2(sin(alpha), cos(alpha))
                
            #     d_O_i = sqrt((x[i] - obstacle[h][0])**2 + (y[j] - obstacle[h][1])**2)
            #     if d_O_i <= 0: d_O_i = 0.0000001

            #     nablaU_repObs_i = np.zeros(2)
                
            #     # # Calculate repulsive potential for each object
            #     D_alpha = 0
                          
            #     if alpha <= 0:
            #         D_alpha = +1
            #     else:
            #         D_alpha = -1
                
            #     if d_O_i < d_safe_obs:
            #         d_rel_O_i = 0
            #     elif d_O_i > 2*d_vort_obs - d_safe_obs:
            #         d_rel_O_i = 1
            #     else:
            #         d_rel_O_i = (d_O_i - d_safe_obs)/(2*(d_vort_obs - d_safe_obs))

            #     if d_rel_O_i <= 0.5:
            #         gamma = pi*D_alpha*d_rel_O_i
            #     else:
            #         gamma = pi*D_alpha*(1-d_rel_O_i)

            #     R_gamma = np.array([
            #         [cos(gamma), -sin(gamma)],
            #         [sin(gamma), cos(gamma)] 
            #     ])
                
            #     if (d_O_i < Q_star_obs):
            #         nablaU_repObs_i = eta * (1/d_O_i - 1/Q_star_obs) * (1.0 / (pow(d_O_i,2))) * (np.array([x[i], y[j]]) - obstacle[h])
            #         nablaU_repObs_i = np.matmul(nablaU_repObs_i, R_gamma)
            #     else:
            #         pass
            #     #nablaU_repObs_i = np.matmul(nablaU_repObs_i, R_gamma)            


            #     # # Add repulsive potential to total repulsive potential
            #     # nablaU_rep_obs += nablaU_repObs_i
                    
            #     ux[j][i] += nablaU_repObs_i[0]
            #     uy[j][i] += nablaU_repObs_i[1]

            
            # For every human
            for h in range(len(humans)):
                
                d_O_i = sqrt((x[i] - humans[h][0][0])**2 + (y[j] - humans[h][0][1])**2)
                if d_O_i <= 0: d_O_i = 0.0000001

                # Calculate repulsive potential for each human
                # Using homemade function
                nablaU_rep_hum_i = np.zeros(2)
                human_relative_pos = np.array([x[i], y[j]]) - humans[h][0]
                nablaU_rep_hum_i = np.array(hc.get_cost_xy(human_relative_pos[0], human_relative_pos[1], humans[h][1][0], humans[h][1][1]))
                nablaU_rep_hum_i = Kh*nablaU_rep_hum_i * (np.array([x[i], y[j]]) - humans[h][0])/norm(np.array([x[i], y[j]]) - humans[h][0])
                    # TODO: Make sure homemade works!

                # Calculate values used for vortex
                alpha = atan2(goal[1] - y[j],goal[0] - x[i]) - atan2(y[j] - humans[h][0][1],x[i] - humans[h][0][0])
                alpha = atan2(sin(alpha), cos(alpha))

                # # Calculate repulsive potential for each object
                D_alpha = 0
                          
                if alpha <= 0:
                    D_alpha = +1
                else:
                    D_alpha = -1
                d_rel_O_i = 0
                
                if d_O_i < d_safe_obs:
                    d_rel_O_i = 0
                elif d_O_i > 2*d_vort_obs - d_safe_obs:
                    d_rel_O_i = 1
                else:
                    d_rel_O_i = (d_O_i - d_safe_obs)/(2*(d_vort_obs - d_safe_obs))

                if d_rel_O_i <= 0.5:
                    gamma = pi*D_alpha*d_rel_O_i
                else:
                    gamma = pi*D_alpha*(1-d_rel_O_i)

                R_gamma = np.array([
                    [cos(gamma), -sin(gamma)],
                    [sin(gamma), cos(gamma)] 
                ])
                nablaU_rep_hum_i = np.matmul(nablaU_rep_hum_i, R_gamma)
                
                ux[j][i] += nablaU_rep_hum_i[0]
                uy[j][i] += nablaU_rep_hum_i[1]                

            siz = sqrt(ux[j][i]**2 + uy[j][i]**2)
            if siz > arrow_max: 
                ux[j][i] = ux[j][i]/(siz/arrow_max)
                uy[j][i] = uy[j][i]/(siz/arrow_max)


    plt.quiver(x, y, ux, uy, scale=1.5, scale_units="inches",width = 0.005,  minshaft=2, headlength=4, color=(0.2, 0.2, 0.2))

    
    
    # perimeter = plt.Circle( (obstacle[0][0], obstacle[0][1]), Q_star_obs,  fill = False, color = (1, 0, 0) )
    # axes.add_artist(perimeter)
    for i in range(len(obstacle)):
        if i == 0:
            plt.plot( obstacle[i][0],obstacle[i][1], color = (1, 0, 0), linestyle= "", marker = "o", label="Obstacle")
        else:
            plt.plot( obstacle[i][0],obstacle[i][1], color = (1, 0, 0), linestyle= "", marker = "o")
        
    
    for i in range(len(humans)):
        if i == 0:
            plt.plot( humans[i][0][0],humans[i][0][1], color = (1, 0, 1), linestyle= "", marker = "o", label="Human")
        else:
            plt.plot( humans[i][0][0],humans[i][0][1], color = (1, 0, 1), linestyle= "", marker = "o")
        plt.quiver(humans[i][0][0], humans[i][0][1], humans[i][1][0], humans[i][1][1], scale=1.5, scale_units="inches",width = 0.005,  minshaft=2, headlength=4, color=(0.2, 0.2, 0.2))
        
    plt.plot( goal[0],goal[1], color = (0, 1, 0), linestyle= "", marker = "o", label="Goal")
    
    
    axes.legend(framealpha = 1)
    

    plt.title("SAPF Map")
    plt.xlim(xstart + 1, xend - 1)
    plt.ylim(ystart + 1, yend -1)   
    
    # disabling xticks by Setting xticks to an empty list
    plt.xticks([])  
    
    # disabling yticks by setting yticks to an empty list
    plt.yticks([])  

    plt.show()
    


def vortex_and_attract():
    xend = 32
    xstart = -1

    yend = 32
    ystart = -1
    
    jump = 1.5
    
    obstacle = np.array([[14, 10], [7, 23]] )
    goal = np.array([20, 20])
    
    Q_star_obs = 15
    eta = 15.0
    d_star_obs = 2
    zeta_obs = 0.1
    
    numx = int((xend-xstart)/jump)
    numy = int((yend-ystart)/jump)
    
    x = np.arange(xstart, xend, jump)
    y = np.arange(ystart, yend, jump)
    
    ux = np.zeros(shape=(numx, numy))
    uy = np.zeros(shape=(numx, numy))
    
    
    # Distance to object
    gamma = pi/2
    
    R_gamma = np.array([
                [cos(gamma), -sin(gamma)],
                [sin(gamma), cos(gamma)] 
            ])
    
    figure, axes = plt.subplots()
    axes.set_aspect(1)
    
    arrow_max = 0.25
    
    for i in range(numx):
        for j in range(numy):
            dist_goal = sqrt((x[i] - goal[0])**2 + (y[j] - goal[1])**2)
            if  dist_goal< d_star_obs:
                nablaU_att = zeta_obs * (goal - np.array([x[i], y[j]]))
            else:
                nablaU_att = d_star_obs * zeta_obs * ((goal - np.array([x[i], y[j]])) /dist_goal)

            ux[j][i] = nablaU_att[0]
            uy[j][i] = nablaU_att[1]
            
            for h in range(len(obstacle)):
                alpha = atan2(y[j] - obstacle[h][1],x[i] - obstacle[h][0])
                alpha = atan2(sin(alpha), cos(alpha))
                gamma = pi/2
                R_gamma = np.array([
                                [cos(gamma), -sin(gamma)],
                                [sin(gamma), cos(gamma)] 
                            ])
                # if alpha <= 5*pi/180:
                #     gamma = pi/2
                #     R_gamma = np.array([
                #                 [cos(gamma), -sin(gamma)],
                #                 [sin(gamma), cos(gamma)] 
                #             ])
                # else:
                #     gamma = -pi/2
                #     R_gamma = np.array([
                #                 [cos(gamma), -sin(gamma)],
                #                 [sin(gamma), cos(gamma)] 
                #             ])
                
                nablaU_repObs_i = np.zeros(2)
                d_O_i = sqrt((x[i] - obstacle[h][0])**2 + (y[j] - obstacle[h][1])**2)
                if d_O_i <= 0: d_O_i = 0.0000001

                # Calculate repulsive potential for each object
                if (d_O_i < Q_star_obs):
                    nablaU_repObs_i = eta * (1/d_O_i - 1/Q_star_obs) * (1.0 / (pow(d_O_i,2))) * (np.array([x[i], y[j]]) - obstacle[h])
                    nablaU_repObs_i = np.matmul(nablaU_repObs_i, R_gamma)
                else:
                    nablaU_repObs_i = np.zeros(2)
                    
                ux[j][i] += nablaU_repObs_i[0]
                uy[j][i] += nablaU_repObs_i[1]
            
            siz = sqrt(ux[j][i]**2 + uy[j][i]**2)
            if siz > arrow_max: 
                ux[j][i] = ux[j][i]/(siz/arrow_max)
                uy[j][i] = uy[j][i]/(siz/arrow_max)


    plt.quiver(x, y, ux, uy, scale=1.5, scale_units="inches",width = 0.005,  minshaft=2, headlength=4, color=(0.2, 0.2, 0.2))

    
    
    # perimeter = plt.Circle( (obstacle[0][0], obstacle[0][1]), Q_star_obs,  fill = False, color = (1, 0, 0) )
    # axes.add_artist(perimeter)
    for i in range(len(obstacle)):
        if i == 0:
            plt.plot( obstacle[i][0],obstacle[i][1], color = (1, 0, 0), linestyle= "", marker = "o", label="Obstacle")
        plt.plot( obstacle[i][0],obstacle[i][1], color = (1, 0, 0), linestyle= "", marker = "o")
        
    plt.plot( goal[0],goal[1], color = (0, 1, 0), linestyle= "", marker = "o", label="Goal")
    
    
    axes.legend(framealpha = 1)
    

    plt.title("VAPF Map")
    plt.xlim(xstart + 1, xend - 1)
    plt.ylim(ystart + 1, yend -1)   
    
    # disabling xticks by Setting xticks to an empty list
    plt.xticks([])  
    
    # disabling yticks by setting yticks to an empty list
    plt.yticks([])  

    plt.show()
    

def gamma_drel():
    d_vort = 6
    d_safe = 2
    
    start = 0
    end = 2*d_vort
    jump = 0.01
    
    num_x = int((end-start)/jump)
    x = np.arange(start, end, jump)
    y = np.zeros(num_x)
    
    figure, axes = plt.subplots()
    
    plt.plot([d_safe, d_vort], [0, pi/2], color = (0, 0, 1))
    plt.plot([d_vort, 2*d_vort - d_safe], [pi/2, 0], color = (0, 0, 1))
    plt.plot([0, d_safe], [0, 0], color = (0, 0, 1))
    plt.plot([ 2*d_vort - d_safe, 2*d_vort ], [0, 0], color = (0, 0, 1))
    
    
    plt.title("The Rotation Function $\gamma$", fontsize=25)
    plt.xlim(start, end)
    plt.ylim(-0.1, 1.7)
    axes.grid() 

    
    plt.xlabel("$d_{O_i}$", fontsize=20)
    plt.ylabel("|$\gamma(H(d_{O_i})$|", fontsize=20)   
    
    # disabling xticks by Setting xticks to an empty list
    plt.xticks([d_safe, d_vort, 2*d_vort-d_safe], ['$d_{safe}$', '$d_{vort}$', '$2 \cdot d_{vort} - d_{safe}$'], fontsize=12)  
    
    # disabling yticks by setting yticks to an empty list
    plt.yticks([0, pi/2], ['$0$', '$\\frac{\pi}{2}$'], fontsize=18)  

    figure2, axes2 = plt.subplots()

    for i in range(num_x):
        y[i] = 0
        if x[i] <= d_safe:
            y[i] = 0
        elif x[i] >=2*d_vort-d_safe:
            y[i] = 1
        else:
            y[i] = (x[i] - d_safe)/(2*(d_vort-d_safe)) 
    
    plt.plot(x, y, color= (0, 0, 1))
    
    plt.title("The Infuence Function $H(d_{O_i})$", fontsize=25)
    plt.xlim(start, end)
    plt.ylim(-0.1, 1.1)
    axes2.grid() 

    
    plt.xlabel("$d_{O_i} $", fontsize=20)
    plt.ylabel("$H(d_{O_i}) $", fontsize=20)   
    
    # disabling xticks by Setting xticks to an empty list
    plt.xticks([d_safe, 2*d_vort-d_safe], ['$d_{safe}$', '$2 \cdot d_{vort} - d_{safe}$'], fontsize=12)  
    
    # disabling yticks by setting yticks to an empty list
    plt.yticks([0, 1], ['$0$', '1'], fontsize=16)  

    plt.show()
    
    
    

def apf_full():
    xend = 32
    xstart = -1

    yend = 32
    ystart = -1
    
    jump = 1.5
    
    obstacle = np.array([[14, 10], [7, 23]] )
    goal = np.array([20, 20])
    
    Q_star_obs = 15
    eta = 15.0
    d_star_obs = 2
    zeta_obs = 0.1
    
    numx = int((xend-xstart)/jump)
    numy = int((yend-ystart)/jump)
    
    x = np.arange(xstart, xend, jump)
    y = np.arange(ystart, yend, jump)
    
    ux = np.zeros(shape=(numx, numy))
    uy = np.zeros(shape=(numx, numy))
    
    
    # Distance to object
    gamma = 0
    
    R_gamma = np.array([
                [cos(gamma), -sin(gamma)],
                [sin(gamma), cos(gamma)] 
            ])
    
    figure, axes = plt.subplots()
    axes.set_aspect(1)
    
    arrow_max = 0.25
    
    for i in range(numx):
        for j in range(numy):
            dist_goal = sqrt((x[i] - goal[0])**2 + (y[j] - goal[1])**2)
            if  dist_goal< d_star_obs:
                nablaU_att = zeta_obs * (goal - np.array([x[i], y[j]]))
            else:
                nablaU_att = d_star_obs * zeta_obs * ((goal - np.array([x[i], y[j]])) /dist_goal)

            ux[j][i] = nablaU_att[0]
            uy[j][i] = nablaU_att[1]
            
            for h in range(len(obstacle)):
                alpha = atan2(y[j] - obstacle[h][1],x[i] - obstacle[h][0])
                alpha = atan2(sin(alpha), cos(alpha))
                # gamma = pi/2
                # R_gamma = np.array([
                #                 [cos(gamma), -sin(gamma)],
                #                 [sin(gamma), cos(gamma)] 
                #             ])
                # if alpha <= 5*pi/180:
                #     gamma = pi/2
                #     R_gamma = np.array([
                #                 [cos(gamma), -sin(gamma)],
                #                 [sin(gamma), cos(gamma)] 
                #             ])
                # else:
                #     gamma = -pi/2
                #     R_gamma = np.array([
                #                 [cos(gamma), -sin(gamma)],
                #                 [sin(gamma), cos(gamma)] 
                #             ])
                
                nablaU_repObs_i = np.zeros(2)
                d_O_i = sqrt((x[i] - obstacle[h][0])**2 + (y[j] - obstacle[h][1])**2)
                if d_O_i <= 0: d_O_i = 0.0000001

                # Calculate repulsive potential for each object
                if (d_O_i < Q_star_obs):
                    nablaU_repObs_i = eta * (1/d_O_i - 1/Q_star_obs) * (1.0 / (pow(d_O_i,2))) * (np.array([x[i], y[j]]) - obstacle[h])
                    nablaU_repObs_i = np.matmul(nablaU_repObs_i, R_gamma)
                else:
                    nablaU_repObs_i = np.zeros(2)
                    
                ux[j][i] += nablaU_repObs_i[0]
                uy[j][i] += nablaU_repObs_i[1]
            
            siz = sqrt(ux[j][i]**2 + uy[j][i]**2)
            if siz > arrow_max: 
                ux[j][i] = ux[j][i]/(siz/arrow_max)
                uy[j][i] = uy[j][i]/(siz/arrow_max)


    plt.quiver(x, y, ux, uy, scale=1.5, scale_units="inches",width = 0.005,  minshaft=2, headlength=4, color=(0.2, 0.2, 0.2))

    
    
    # perimeter = plt.Circle( (obstacle[0][0], obstacle[0][1]), Q_star_obs,  fill = False, color = (1, 0, 0) )
    # axes.add_artist(perimeter)
    for i in range(len(obstacle)):
        if i == 0:
            plt.plot( obstacle[i][0],obstacle[i][1], color = (1, 0, 0), linestyle= "", marker = "o", label="Obstacle")
        plt.plot( obstacle[i][0],obstacle[i][1], color = (1, 0, 0), linestyle= "", marker = "o")
        
    plt.plot( goal[0],goal[1], color = (0, 1, 0), linestyle= "", marker = "o", label="Goal")
    
    
    axes.legend(framealpha = 1)
    

    plt.title("APF Map")
    plt.xlim(xstart + 1, xend - 1)
    plt.ylim(ystart + 1, yend -1)   
    
    # disabling xticks by Setting xticks to an empty list
    plt.xticks([])  
    
    # disabling yticks by setting yticks to an empty list
    plt.yticks([])  

    plt.show()



def vortex_graph():
    xend = 5
    xstart = -5

    yend = 5
    ystart = -5
    
    jump = 0.5
    
    obstacle = np.array([0, 0])
    Q_star_obs = 3
    eta = 3.0
    
    numx = int((xend-xstart)/jump)
    numy = int((yend-ystart)/jump)
    
    x = np.arange(xstart, xend, jump)
    y = np.arange(ystart, yend, jump)
    
    # Distance to object
    gamma = pi/2
    
    R_gamma = np.array([
                [cos(gamma), -sin(gamma)],
                [sin(gamma), cos(gamma)] 
            ])
    
    figure, axes = plt.subplots()
    axes.set_aspect(1)
    
    arrow_max = 0.32
    
    for i in range(numx):
        for j in range(numy):
            
            nablaU_repObs_i = np.zeros(2)
            d_O_i = sqrt((x[i] - obstacle[0])**2 + (y[j] - obstacle[1])**2)
            if d_O_i <= 0: d_O_i = 0.0000001

            # Calculate repulsive potential for each object
            if (d_O_i < Q_star_obs):
                nablaU_repObs_i = eta * (1/d_O_i - 1/Q_star_obs) * (1.0 / (pow(d_O_i,2))) * (np.array([x[i], y[j]]) - obstacle)
                siz = sqrt(nablaU_repObs_i[0]**2 + nablaU_repObs_i[1]**2)
                if siz > arrow_max: 
                    nablaU_repObs_i = nablaU_repObs_i/(siz/arrow_max)
                nablaU_repObs_i = np.matmul(nablaU_repObs_i, R_gamma)
                plt.quiver(x[i], y[j], nablaU_repObs_i[0], nablaU_repObs_i[1], scale=1.5, scale_units="inches",width = 0.005,  minshaft=2, headlength=4, color=(0.2, 0.2, 0.2))
                
            else:
                pass

    
    perimeter = plt.Circle( (obstacle[0], obstacle[1]), Q_star_obs,  fill = False, color = (0, 0, 0) )
    axes.add_artist(perimeter)
    
    plt.plot( obstacle[0],obstacle[1], color = (1, 0, 0), linestyle= "", marker = "o", label="Obstacle")
    
    
    theta = pi/8
    x1, y1 = [obstacle[0], obstacle[0] + cos(theta)*Q_star_obs], [obstacle[1], obstacle[1] + sin(theta)*Q_star_obs]
    plt.plot(x1, y1 , color = (1, 0, 0))
    
    axes.text( obstacle[0] + cos(theta)*Q_star_obs*0.60, obstacle[1] + sin(theta)*Q_star_obs*0.95, "$d_{O_i}^*$",
                         size="x-large", rotation=20, color = (0, 0, 1))
    axes.legend(framealpha = 1)

    plt.title("Vortex gradient 2D")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)   
    
    # disabling xticks by Setting xticks to an empty list
    plt.xticks([])  
    
    # disabling yticks by setting yticks to an empty list
    plt.yticks([])  

    plt.show()


def rep_graph():
    eta = 1
    Q_star = 1

    nablaU_repObs_i = []
    x = []
    d = 0.1
    while d < 1.5:
        if (d < Q_star):
            nablaU_repObs_i.append(eta * (1/d - 1/Q_star) * (1.0 / (pow(d,2))) * (d))
        else:
            nablaU_repObs_i.append(0)

        x.append(d)
        d += 0.01
    
    linewidth = 4
    plt.rcParams.update({'font.size': 22})

    plt.plot([0,Q_star],[0,0],linestyle="dashed", color=(0,0,0), linewidth=linewidth-0.1)
    plt.plot(x, nablaU_repObs_i, linewidth=linewidth)
    

    plt.title("Repulsive gradient")
    plt.ylabel("$|\\vec{U}_{rep}^{O_i}(P_{robot})|$")
    plt.xlabel("Distance to obstacle")

    plt.xticks(ticks=[0,Q_star],labels=["0", "$d_{O_i}^{*}$"])
    plt.yticks(ticks=[0],labels=["0"])

    plt.xlim(0,1.5)
    plt.ylim(0,80)

    

    plt.show()


if __name__ == "__main__":
    # att_graph()
    # rep_graph()
    #vortex_and_attract()
    #apf_full()
    #gamma_drel()
    #sapf_graph()
    sapf_graph_with_people()