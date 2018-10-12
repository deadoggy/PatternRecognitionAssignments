#coding:utf-8

import sys
import numpy as np
import json
from lr import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

data_path = sys.path[0] + '/../dataset/%s.json'
data_name = 'bank'
train_scale = 0.8
cross_valid_scale = 0.1
test_scale = 1 - train_scale - cross_valid_scale

with open(data_path%data_name) as data_f:
    data = json.load(data_f)

#generate trainset, cross validation set and test set
################################################################################
train_X = []
train_y = []
cross_valid_X = []
cross_valid_y = []
test_X = []
test_y = []
for i, label in enumerate(['X_0', 'X_1']):
    entire_set = data[label]
    train_sup = int(len(entire_set) * train_scale)
    cross_sup = int(len(entire_set) * (train_scale + cross_valid_scale))
    
    for j, vec in enumerate(entire_set):
        if j >=0 and j < train_sup:
            train_X.append(vec)
            train_y.append(i)
        elif j >= train_sup and j < cross_sup:
            cross_valid_X.append(vec)
            cross_valid_y.append(i)
        else:
            test_X.append(vec)
            test_y.append(i)

train_X = np.array(train_X)
train_y = np.array(train_y)
cross_valid_X = np.array(cross_valid_X)
cross_valid_y = np.array(cross_valid_y)
test_X = np.array(test_X)
test_y = np.array(test_y)

#dimensionality reduction using LDA
###############################################################################
dim_reducer = LDA(n_components=2)
train_X = dim_reducer.fit_transform(train_X, train_y)
cross_valid_X = dim_reducer.fit_transform(cross_valid_X, cross_valid_y)
test_X = dim_reducer.fit_transform(test_X, test_y)

#run experiments
###############################################################################
max_itr = 20
best_model = None
best_rate = -0.1
for i in xrange(max_itr):
    lr = LogisticRegression()
    lr.fit(train_X, train_y)
    exp_cross_y = lr.predict(cross_valid_X)
    tmp_rate = 1.0 - (cross_valid_y^exp_cross_y).sum()/float(cross_valid_X.shape[0])
    if tmp_rate > best_rate:
        best_rate = tmp_rate
        best_model = lr

exp_test_y = best_model.predict(test_X)
rate = 1.0 - (test_y^exp_test_y).sum()/float(test_X.shape[0])
bete = best_model._beta

print rate


