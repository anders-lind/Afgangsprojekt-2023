import numpy as np
import matplotlib.pyplot as plt



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


if __name__ == "__main__":
    att_graph()
    rep_graph()