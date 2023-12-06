import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math



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
    rep_color = (255/255,130/255,19/255)
    vor_color = (44/255,126/255,184/255)

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
    ax.plot(obs[0], obs[1], marker='o', color=(0,0,0), label="Obstacle")

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
        ax.plot(robot[0], robot[1], marker='o', color=(1,0,0), label="Robot")

        # Goal arrow
        arrow_dir = 1.2 * (np.array(goal) - np.array(robot)) / (math.dist(goal, robot))
        ax.arrow(robot[0], robot[1], arrow_dir[0], arrow_dir[1], head_width=0.15, head_length=0.12, color=goal_color, linewidth=2)
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




if __name__ == "__main__":
    # att_graph()
    # rep_graph()
    # switch_graph()
    # overview_graph()
    pass