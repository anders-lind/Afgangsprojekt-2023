import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from math import *

class Human_cost:
    def __init__(self, mu_x = 0, var_x = 3, mu_y = 0, var_y = 12):
        self.mu_x = mu_x
        self.var_x = var_x
        self.mu_y = mu_y
        self.var_y = var_y
        
        self.cost_func = None
        
        self.person_size = 0.25
        self.intimate = 1.5*0.3 + self.person_size
        self.personal = 4*0.3 + self.person_size
        self.social = 10*0.3 + self.person_size
        
        
        self.x = np.linspace(-7,7,500)
        self.y = np.linspace(-7,7,500)
        
        self.create_graph(visualize=False)
    
    def create_graph(self, visualize = True):
        #Create grid and multivariate normal
        self.x  = np.linspace(-7,7,500)
        self.y  = np.linspace(-7,7,500)
        Y, X = np.meshgrid(self.x , self.y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X; pos[:, :, 1] = Y
        rv = multivariate_normal([self.mu_x, self.mu_y], [[self.var_x, 0], [0, self.var_y]])

        pdf = rv.pdf(pos)
        skew = rv.pdf(pos)
        skewpdf = rv.pdf(pos)
        scaling = rv.pdf(pos)
        
        self.cost_func = rv.pdf(pos)

        b = 1
        c = 3
        a = b/(c**2)

        max = pdf[250,250]



        for i in range(np.shape(self.x)[0]):
            for j in range(np.shape(self.y)[0]):
                # Unit
                pdf[j,i] = pdf[j,i] * 1/max

                # only asym
                if self.y[j] > 0:
                    skew[j,i] = -a*self.y[j]**2+b
                if self.y[j] > c:
                    skew[j,i] = 0
                elif self.y[j] <= 0:
                    skew[j,i] = 1
                

                # Only scale
                if dist((self.x[i],self.y[j]), (0,0)) < self.person_size:
                    scaling[j,i] = 1.1
                elif dist((self.x[i],self.y[j]), (0,0)) < self.intimate:
                    scaling[j,i] = 1.0
                elif dist((self.x[i],self.y[j]), (0,0)) < self.personal:
                    scaling[j,i] = 0.5
                elif dist((self.x[i],self.y[j]), (0,0)) < self.social:
                    scaling[j,i] = 0.25
                elif dist((self.x[i],self.y[j]), (0,0)) >= self.social:
                    scaling[j,i] = 0.0
                
                # skew pdf
                skewpdf[j,i] = pdf[j,i] * skew[j,i]

                # Asym skew
                self.cost_func[j,i] = pdf[j,i] * skew[j,i] * scaling[j,i]
                
        if visualize == True:
            #Make a 3D plot
            
            ax1 = plt.figure("Total").add_subplot(projection='3d')
            plt.title("total")
            ax1 = plt.axes(projection='3d')
            ax1.plot_surface(X, Y, self.cost_func,cmap='viridis',linewidth=0)
            ax1.set_xlabel('X axis')
            ax1.set_ylabel('Y axis')
            ax1.set_zlabel('Z axis')

            ax2 = plt.figure("skew").add_subplot(projection='3d')
            ax2.plot_surface(X, Y, skew,cmap='viridis',linewidth=0)
            ax2.set_xlabel('X axis')
            ax2.set_ylabel('Y axis')
            ax2.set_zlabel('Z axis')

            ax3 = plt.figure("Scaling").add_subplot(projection='3d')
            ax3.plot_surface(X, Y, scaling,cmap='viridis',linewidth=0)
            ax3.set_xlabel('X axis')
            ax3.set_ylabel('Y axis')
            ax3.set_zlabel('Z axis')

            ax4 = plt.figure("pdf").add_subplot(projection='3d')
            plt.title("pdf")
            ax4.plot_surface(X, Y, pdf,cmap='viridis',linewidth=0)
            ax4.set_xlabel('X axis')
            ax4.set_ylabel('Y axis')
            ax4.set_zlabel('Z axis')

            ax5 = plt.figure("skew pdf").add_subplot(projection='3d')
            ax5.plot_surface(X, Y, skewpdf,cmap='viridis',linewidth=0)
            ax5.set_xlabel('X axis')
            ax5.set_ylabel('Y axis')
            ax5.set_zlabel('Z axis')

            plt.show()
                    

    def get_cost_xy(self, x, y):
            if sqrt(x**2 + y**2) >= self.personal:
                return 0
            
            # # method to find closest value
            #https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
            
            idx_x = (np.abs(self.x - x)).argmin()
            idx_y = (np.abs(self.y - y)).argmin()

            return self.cost_func[idx_x, idx_y]
        
    def get_cost_angledistance(self, angle, distance):        
        #polar coordinates:
        if distance >= self.personal:
            return 0
        
        x = distance*cos(angle)
        y = distance*sin(angle)
        
        return self.get_cost_xy(x, y)
        
        
if __name__ == "__main__":
    cost = Human_cost(mu_x = 0, var_x = 3, mu_y = 0, var_y = 12)
    
    x = 1  #m
    y = 1  #m
    cost_xy = cost.get_cost_xy(x, y)
    print("The value of the cost-function at (", x, ", ", y, ") is: ", cost_xy)
    
    angle = pi/4 #rad
    distance = sqrt(2) #m
    cost_angledistance = cost.get_cost_angledistance(angle, distance)
    print("The value of the cost-function at angle: ", angle, ", distance: ", distance, " is: ", cost_angledistance)
  
    
    cost.create_graph(visualize=True)