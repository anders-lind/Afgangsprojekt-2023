import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from math import log, dist

#Parameters to set
mu_x = 0
variance_x = 3

mu_y = 0
variance_y = 12

#Create grid and multivariate normal
x = np.linspace(-7,7,500)
y = np.linspace(-7,7,500)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])

pdf = rv.pdf(pos)
skew = rv.pdf(pos)
skewpdf = rv.pdf(pos)
scaling = rv.pdf(pos)
total = rv.pdf(pos)


total = np.zeros(np.shape(pdf))




# Old version
# x1 = 0
# y1 = 1
# x2 = 4
# y2 = 0

# b = 1
# c = 3
# a = b/(c**2)

# max = pdf[250,250]


# for i in range(np.shape(x)[0]):
#     for j in range(np.shape(y)[0]):
#         pdf[j,i] = pdf[j,i] * 1/max

#         skewed_pdf[j,i] = pdf[j,i]
#         if y[j] > 0:
#             skewed_pdf[j,i] = pdf[j,i]*(-a*y[j]**2+b)
#         if y[j] > c:
#             skewed_pdf[j,i] = 0


x1 = 0
y1 = 1
x2 = 4
y2 = 0

b = 1
c = 3
a = b/(c**2)

max = pdf[250,250]

person_size = 0.3
intimate = 1.5*0.3 + person_size
personal = 4*0.3 + person_size
social = 10*0.3 + person_size


for i in range(np.shape(x)[0]):
    for j in range(np.shape(y)[0]):
        # Unit
        pdf[j,i] = pdf[j,i] * 1/max

        # only asym
        if y[j] > 0:
            skew[j,i] = -a*y[j]**2+b
        if y[j] > c:
            skew[j,i] = 0
        elif y[j] <= 0:
            skew[j,i] = 1
        

        # Only scale
        if dist((x[i],y[j]), (0,0)) < person_size:
            scaling[j,i] = 1.1
        elif dist((x[i],y[j]), (0,0)) < intimate:
            scaling[j,i] = 1.0
        elif dist((x[i],y[j]), (0,0)) < personal:
            scaling[j,i] = 0.5
        elif dist((x[i],y[j]), (0,0)) < social:
            scaling[j,i] = 0.25
        elif dist((x[i],y[j]), (0,0)) >= social:
            scaling[j,i] = 0.0
        
        # skew pdf
        skewpdf[j,i] = pdf[j,i] * skew[j,i]

        # Asym skew
        total[j,i] = pdf[j,i] * skew[j,i] * scaling[j,i]


# TODO
# method to find closest value
# make class we both can import
x_index = np.where(x == A)[0][0]
y_index = np.where(y == B)[0][0]

return total[x_index, y_index]





#Make a 3D plot
fig = plt.figure("Total")
plt.title("total")
ax1 = plt.axes(projection='3d')
ax1.plot_surface(X, Y, total,cmap='viridis',linewidth=0)
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')

fig = plt.figure("skew")
ax2 = fig.gca(projection='3d')
ax2.plot_surface(X, Y, skew,cmap='viridis',linewidth=0)
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.set_zlabel('Z axis')

fig = plt.figure("Scaling")
ax3 = fig.gca(projection='3d')
ax3.plot_surface(X, Y, scaling,cmap='viridis',linewidth=0)
ax3.set_xlabel('X axis')
ax3.set_ylabel('Y axis')
ax3.set_zlabel('Z axis')

fig = plt.figure("pdf")
plt.title("pdf")
ax4 = fig.gca(projection='3d')
ax4.plot_surface(X, Y, pdf,cmap='viridis',linewidth=0)
ax4.set_xlabel('X axis')
ax4.set_ylabel('Y axis')
ax4.set_zlabel('Z axis')

fig = plt.figure("skew pdf")
ax5 = fig.gca(projection='3d')
ax5.plot_surface(X, Y, skewpdf,cmap='viridis',linewidth=0)
ax5.set_xlabel('X axis')
ax5.set_ylabel('Y axis')
ax5.set_zlabel('Z axis')

plt.show()
