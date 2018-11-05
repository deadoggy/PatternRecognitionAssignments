#coding:utf-8

import json
from matplotlib import pyplot as plt
import sys

with open(sys.path[0] + '/../experiments_out.json') as result_in:
    result_json = json.load(result_in)

datasets = ['iris', 'wine']

tols = ['0.01', '0.02', '0.03', '0.04', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5']
scales = ['0.01', '0.02', '0.03', '0.04', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.', '1.1']
learning_rate = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.', '1.1', '1.2', '1.3', '1.4', '1.5']
lambdas = ['0.', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7']


folder = sys.path[0]+'/../figures/'
for dsn in datasets:
    data = result_json[dsn]
    #learning rate
    axe = plt.subplot()
    lr_rate = data['learning_rate']
    axe.plot(learning_rate,lr_rate, marker='x')
    axe.grid()
    axe.set_title('%s learning rate experiments'%dsn.upper())
    axe.set_xlabel('learning rate')
    axe.set_ylabel('accurency')
    plt.savefig(folder+'%s_learning_rate.png'%dsn.upper())
    plt.cla()
    #tol
    axe = plt.subplot()
    tol_rate = data['tol']
    axe.plot(tols,tol_rate, marker='x')
    axe.grid()
    axe.set_title('%s tolerance experiments'%dsn.upper())
    axe.set_xlabel('tolerance')
    axe.set_ylabel('accurency')
    plt.savefig(folder+'%s_tol.png'%dsn.upper())
    plt.cla()
    #scale
    axe = plt.subplot()
    scale_rate = data['scale']
    axe.plot(scales,scale_rate, marker='x')
    axe.grid()
    axe.set_title('%s scale experiments'%dsn.upper())
    axe.set_xlabel('scale')
    axe.set_ylabel('accurency')
    plt.savefig(folder+'%s_scale.png'%dsn.upper())
    plt.cla()
    #lambda
    axe = plt.subplot()
    lamb_rate = data['lambdas']
    axe.plot(lambdas,lamb_rate, marker='x')
    axe.grid()
    axe.set_title('%s lambdas experiments'%dsn.upper())
    axe.set_xlabel('lambdas')
    axe.set_ylabel('accurency')
    plt.savefig(folder+'%s_lambdas.png'%dsn.upper())
    plt.cla()
    #hidden layer sizes
    baselayer = 4 if dsn=='iris' else 13
    x_range = [ str(i) for i in range(baselayer-2, baselayer+5)]
    axe = plt.subplot()
    hls_rate = data['hidden_layer_size']
    axe.plot(x_range,hls_rate, marker='x')
    axe.grid()
    axe.set_title('%s hidden layer size experiments'%dsn.upper())
    axe.set_xlabel('hidden layer size')
    axe.set_ylabel('accurency')
    plt.savefig(folder+'%s_hidden_layer_size.png'%dsn.upper())
    plt.cla()
    # hidden layers
    x_range = [ str(i) for i in range(1, 4) ]
    axe = plt.subplot()
    hl_rate = data['hidden_layers']
    axe.plot(x_range,hl_rate, marker='x')
    axe.grid()
    axe.set_title('%s hidden layers\' number experiments'%dsn.upper())
    axe.set_xlabel('hidden layers\' number')
    axe.set_ylabel('accurency')
    plt.savefig(folder+'%s_hidden_layers_number.png'%dsn.upper())
    
