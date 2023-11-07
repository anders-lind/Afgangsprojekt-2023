#!/usr/bin/env python3

import rospy
import math
import numpy as np



def norm(pos):
    return math.sqrt(pos[0]*pos[0] + pos[1]+pos[1])



if __name__ == "__main__":

    start = [0.0,0.0]
    x = start[0]
    y = start[1]
    theta = 0

    goal = [5.0,5.0]
    x_goal = goal[0]
    y_goal = goal[1]

    obs = np.array([
        [1,1],
        [3,2],
        [1,3],
        [2,2],
    ])


    dstar = 1.0
    zeta = 0.3
    Qstar = 1.0



    

    ### calculate attractive force ###
    if norm([x-x_goal, y-y_goal]) <= dstar:
        print("case 1")
        nablaU_att =  zeta*np.array([x, y]-[x_goal, y_goal])
    else:
        print("case 2")
        nablaU_att = dstar/norm([x-x_goal, y-y_goal]) * zeta*np.array([x-x_goal, y-y_goal])

    print("nablaU_att:", nablaU_att)


    # Find distance to all objects
    obst_dist = np.zeros((len(obs)))
    for i in range(len(obs)):
        obst_dist[i] = norm(np.array([x,y]) - obs[i])

    
    for i in range(len(obs)):
        print("obs:", obs[i])

        if obst_dist(i) <= Qstar or  abs(alpha) < deg2rad(150):
            nablaU_rep_Oi = (SAPF.eta*(1/SAPF.Qstar - 1/obst_dist(i)) * 1/obst_dist(i)^2) * ([x y] - [obstacle(i,obst_idx(i),1) obstacle(i,obst_idx(i),2)]);

        


