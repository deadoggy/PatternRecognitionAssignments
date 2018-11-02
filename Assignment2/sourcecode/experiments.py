#coding:utf-8

import numpy as np
import json
from fully_connnected_nn import FullyConnectedNN, sigmod, derivative_sigmod
import sys
from sklearn.preprocessing import scale, MinMaxScaler


def run(dataset_name, layer_sizes, tol, scale, learning_rate, regularization_lambda):
    #load data and scale
    with open(sys.path[0] + "/../dataset/%s.json"%dataset_name) as wine_f:
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
    NN = FullyConnectedNN(layer_sizes, sigmod, derivative_sigmod, tol=tol, normal_random_scale=scale)

    NN.fit(train_X, train_Y, learning_rate, regularization_lambda)

    predict_Y = NN.predict(test_X)
    result_Y = []
    for y in predict_Y:
        max_index = -1
        max_v = -np.inf
        for i in xrange(0,3):
            if y[i]>max_v:
                max_v = y[i]
                max_index = i
        ry = [0,0,0]
        ry[max_index] = 1
        result_Y.append(ry)

    count = 0.
    for i, y in enumerate(result_Y):
        if y != test_Y[i]:
            count += 1
    return 1 - count / len(test_Y)

dataset_name = ['wine', 'iris']
tols = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
scales = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1]
learning_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5]
lambdas = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

last_result = {}

for dn in dataset_name:
    experiment_result = {
        'tol':[],
        'scale':[],
        'learning_rate':[],
        'lambdas':[],
        'hidden_layer_size':[],
        'hidden_layers':[]
    }
    if dn=='wine':
        layers = np.array([13, 15, 3])
    else:
        layers = np.array([4, 6, 3])
    print dn
    #tols experiments
    #### scales: 0.4;  learning_rate: 0.9; lambda: 0.
    print '======tol======'
    for tol in tols:
        print tol
        rate = run(dn, layers, tol, 0.4, 0.9,0.)
        experiment_result['tol'].append(rate)
    #scales experiments
    #### tol: 0.01; learning_rate: 0.9; lambda: 0.
    print '======scale======'
    for scale in scales:
        print scale
        rate = run(dn, layers, 0.01, scale, 0.9, 0.)
        experiment_result['scale'].append(rate)
    #learning_rate experiments
    #### tol: 0.01; scale: 0.4; lambda: 0.
    print '======learning rate======'
    for lr in learning_rate:
        print lr
        rate = run(dn, layers, 0.01, 0.4, lr, 0.)
        experiment_result['learning_rate'].append(rate)
    #lambda experiments
    #### tol: 0.01; scale: 0.4, learning_rate: 0.9,
    print '======lambda======' 
    for l in lambdas:
        print l
        rate = run(dn, layers, 0.01, 0.4, 0.9, l)
        experiment_result['lambdas'].append(rate)
        
    #hidden layer sizes
    #### tol:0.01; scale: 0.1; learning_rate: 0.9; lambda:0.
    print '======hidden layer sizes======'
    for hls in xrange(layers[0]-2, layers[0] + 5):
        layers[1] = hls
        print layers
        rate = run(dn, layers, 0.01, 0.4, 0.9, 0.)
        experiment_result['hidden_layer_size'].append(rate)
    
    ##hidden layers: 1/2/3
    #### tol:0.01; scale: 0.4; learning_rate: 0.9; lambda:0.
    print '======hidden layers======'
    for i in xrange(3):
        if i > 0:
            tmp_layer = layers.tolist()
            tmp_layer.insert(1, tmp_layer[1])
            layers = np.array(tmp_layer)
        print layers
        rate = run(dn, layers, 0.01, 0.4, 0.9, 0.)
        experiment_result['hidden_layers'].append(rate)
    last_result[dn] = experiment_result

with open(sys.path[0] + "/../experiments_out.json", 'w') as out_f:
    json.dump(last_result, out_f)

