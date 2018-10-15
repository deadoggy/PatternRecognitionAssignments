#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import sys
from matplotlib import cm

data_path = sys.path[0] + '/../dataset/%s.out'
data_name = 'iris'

with open(data_path%data_name, 'r') as data_f:
    result_json = json.load(data_f)

X = result_json['X']
truth_y = result_json['truth_y']
beta = result_json['beta'][0]
rate = result_json['rate']
average_rate = result_json['average_rate']

X_l = [[],[]]
for i in xrange(len(truth_y)):
    if truth_y[i] == 0:
        X_l[0].append(X[i])
    else:
        X_l[1].append(X[i])
X_l[0] = np.array(X_l[0])
X_l[1] = np.array(X_l[1])

fig = plt.figure()
axe = fig.gca(projection='3d')

 
axe.scatter(X_l[0][:,0], X_l[0][:,1],X_l[0][:,2], marker='x')
axe.scatter(X_l[1][:,0], X_l[1][:,1],X_l[1][:,2], marker='o', c='r')

# Make data.
X = np.arange(0., 1., 0.05)
Y = np.arange(0., 1., 0.05)
X, Y = np.meshgrid(X, Y)
Z = (-beta[0]*X -beta[1]*Y - beta[3])/beta[2]

axe.plot_surface(X, Y, Z,linewidth=0, antialiased=False,alpha=0.3)

axe.set_title('iris')
plt.show()

