#coding:utf-8

import numpy as np
import json
from fully_connnected_nn import FullyConnectedNN, sigmod, derivative_sigmod
import sys
from sklearn.preprocessing import scale, MinMaxScaler

#load data and scale
with open(sys.path[0] + "/../dataset/wine.json") as wine_f:
    wine_data = json.load(wine_f)
train_X = []
train_Y = []
test_X = []
test_Y = []
for key in wine_data.keys():
    X = wine_data[key]
    for index, x in enumerate(X):
        y = [0,0,0]
        y[int(key)-1] = 1
        if index < 0.8*len(X):
            train_X.append(x)
            train_Y.append(y)
        else:
            test_X.append(x)
            test_Y.append(y)
train_len = len(train_X)
minmaxscaler = MinMaxScaler()
scale_array = minmaxscaler.fit_transform(np.array(train_X + test_X))
train_X = scale_array[0:train_len]
test_X = scale_array[train_len:]


#train
NN = FullyConnectedNN(np.array([13, 20, 3]), sigmod, derivative_sigmod)

NN.fit(train_X, train_Y, 0.5, 0.)

predict_Y = NN.predict(test_X)
print predict_Y
        